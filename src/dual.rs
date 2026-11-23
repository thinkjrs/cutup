//! The min-max linear program behind [`MinCvar`](crate::MinCvar) and
//! [`MinMad`](crate::MinMad): the portfolio within the bounds, groups and any target return
//! whose worst case is least, where the worst case is the most a weighting `z` of some
//! scenario vectors `dₜ`, within bounds of its own, can make of `Σₜ zₜdₜᵀw`.
//!
//! For CVaR the scenarios are the losses `−rₜ`, weighted by a probability `z` over the tail;
//! for the mean absolute deviation they're the deviations `rₜ − r̄`, weighted by `±1/T`.
//! Either way, minimizing over the portfolio is a min-max. Swapping the two and writing the
//! minimum over the portfolio as its own dual gives a maximization over `z` and that dual's
//! multipliers: `ν` for the budget, `η` for the target return, `λ⁺`, `λ⁻` for the bounds, and
//! `γ⁺`, `γ⁻` for the groups' floors and caps, subject to
//! `Dᵀz = ν1 + ημ + λ⁺ − λ⁻ + G(γ⁺ − γ⁻)`, one row per asset, where column `k` of `G` marks
//! group `k`'s members. The weights are those rows' multipliers, and the worst case is the
//! optimum. A group limit the bounds already meet has no multiplier. That program has a row
//! per asset rather than per scenario, and a row more if the weighting must sum to 1.

use crate::bounds::{ReturnRange, feasible_start};
use crate::{CutupError, Returns, WeightBounds, lp, qp};
use nalgebra::{DMatrix, DVector};

/// The portfolios a min-max method chooses from: within its bounds and groups, with its
/// target return if it has one, checked before any program is built.
pub(crate) struct Portfolios {
    bounds: qp::Bounds,
    /// The target return and the sample mean it's reached by.
    target: Option<(f64, DVector<f64>)>,
}

impl Portfolios {
    /// Resolves `bounds` for `data`, and checks some portfolio within them meets every group
    /// and, with one, reaches `target_return`.
    ///
    /// Groups that can't all be met leave the dual unbounded rather than infeasible, so they're
    /// found first, from the portfolios themselves: by the range's ends with a target,
    /// otherwise by the simplex solver's start. Without groups, neither can fail.
    ///
    /// # Errors
    ///
    /// The bounds' errors from [`WeightBounds`], [`CutupError::InfeasibleGroups`],
    /// [`CutupError::InvalidConfig`] for a target return that isn't finite, and
    /// [`CutupError::TargetReturnOutOfRange`].
    pub(crate) fn new(
        data: &Returns,
        bounds: &WeightBounds,
        target_return: Option<f64>,
    ) -> Result<Self, CutupError> {
        let bounds = bounds.resolve(data.labels())?;
        let target = match target_return {
            Some(target) if !target.is_finite() => {
                return Err(CutupError::InvalidConfig {
                    field: "target_return",
                    value: target,
                });
            }
            Some(target) => {
                let mu = data.returns.row_mean().transpose();
                Some((ReturnRange::new(&mu, &bounds)?.reachable(target)?, mu))
            }
            None => {
                feasible_start(&bounds)?;
                None
            }
        };
        Ok(Self { bounds, target })
    }

    /// The portfolio whose worst case over the scenarios is least, and that worst case.
    ///
    /// The columns of `scenarios` are the `dₜ`, already multiplied by `unit`, the power of two
    /// the method scales its returns by so that they're near 1 next to the bounds and the
    /// budget, which don't scale. The target return is scaled by it here, and the worst case
    /// unscaled. Each `zₜ` lies within `weighting`, and with `budget` the `zₜ` sum to 1. The
    /// scenarios in `at_upper` start their `zₜ` at its upper bound: see [`lp::solve_from`].
    ///
    /// # Errors
    ///
    /// [`CutupError::NotConverged`] if the linear program stops early, which only extreme inputs
    /// should cause.
    pub(crate) fn min_max(
        &self,
        scenarios: &DMatrix<f64>,
        unit: f64,
        weighting: (f64, f64),
        budget: bool,
        at_upper: &[usize],
    ) -> Result<(Vec<f64>, f64), CutupError> {
        let bounds = &self.bounds;
        let n = scenarios.nrows();
        let mut dual = Columns::new(budget);
        // The dual's columns, in order: z (one per scenario), ν, then η with a target, λ⁺ (n),
        // λ⁻ (n), and a γ for each finite group limit. Resolved bounds are all finite, so every
        // asset has a λ⁻. Maximizing ν + ρη + lᵀλ⁺ − uᵀλ⁻ + Lᵀγ⁺ − Uᵀγ⁻ is minimizing its
        // negative.
        for scenario in scenarios.column_iter() {
            dual.push(scenario.iter().copied(), 1.0, 0.0, weighting);
        }
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        dual.push(std::iter::repeat_n(-1.0, n), 0.0, -1.0, free);
        if let Some((rho, mu)) = &self.target {
            dual.push(mu.iter().map(|m| -m * unit), 0.0, -rho * unit, free);
        }
        let nonnegative = (0.0, f64::INFINITY);
        let unit_column =
            |i: usize, sign: f64| (0..n).map(move |j| if j == i { sign } else { 0.0 });
        for i in 0..n {
            dual.push(unit_column(i, -1.0), 0.0, -bounds.lower[i], nonnegative);
        }
        for i in 0..n {
            dual.push(unit_column(i, 1.0), 0.0, bounds.upper[i], nonnegative);
        }
        let rows = &bounds.rows;
        for (k, members) in rows.a.column_iter().enumerate() {
            // A limit the bounds already meet is infinite, and has no multiplier.
            for (sign, cost) in [(-1.0, -rows.lower[k]), (1.0, rows.upper[k])] {
                if cost.is_finite() {
                    dual.push(members.iter().map(|g| sign * g), 0.0, cost, nonnegative);
                }
            }
        }
        let a = DMatrix::from_columns(&dual.a);
        let mut b = DVector::zeros(a.nrows());
        if budget {
            b[n] = 1.0;
        }
        let (c, lower, upper) = (
            DVector::from_vec(dual.cost),
            DVector::from_vec(dual.lower),
            DVector::from_vec(dual.upper),
        );

        // The portfolio is feasible and the worst case is bounded below on it, so the dual has
        // an optimum.
        let lp::Outcome::Optimal(solution) = lp::solve_from(&c, &a, &b, &lower, &upper, at_upper)?
        else {
            return Err(CutupError::NotConverged);
        };
        let weights = within_bounds(&solution.y.as_slice()[..n], bounds);
        Ok((weights, -solution.objective / unit))
    }

    /// The portfolio whose largest `cutᵀw` over `cuts` is least, and that largest value, from
    /// scratch: see [`Master`]. For the tests, which check it against [`Portfolios::min_max`].
    ///
    /// # Errors
    ///
    /// [`CutupError::NotConverged`] if the linear program stops early.
    #[cfg(test)]
    pub(crate) fn min_of_max(
        &self,
        cuts: &[DVector<f64>],
        unit: f64,
    ) -> Result<(Vec<f64>, f64), CutupError> {
        let mut master = self.master(unit);
        for cut in cuts {
            master.add(cut.clone());
        }
        master.solve()
    }

    /// The program for the portfolio whose largest `cutᵀw` over some cuts is least, with no
    /// cuts yet. The cuts it's given must already be multiplied by `unit`, as the scenarios
    /// are; `unit` scales the target to match.
    pub(crate) fn master(&self, unit: f64) -> Master<'_> {
        Master {
            portfolios: self,
            unit,
            cuts: Vec::new(),
            basis: None,
        }
    }
}

/// The portfolio whose largest `cutᵀw` over the cuts so far is least, for cutting planes,
/// which add a cut a round.
///
/// The same min-max as [`Portfolios::min_max`], with the cuts as the scenarios and their
/// weighting any probability, but solved as the primal, `min η` subject to `cutᵀw ≤ η` for
/// each cut, a row per cut rather than per asset: the bounds are the weights' own, so with
/// few cuts the program is small however many assets there are.
///
/// Each cut is a row and a slack variable, both last, so the program after a new cut is the
/// last one with a row more, whose slack is basic in it. Each solve starts from the basis the
/// last ended at, the new slack added, with [`lp::solve_warm`]: a few dual pivots repair the
/// cut the last optimum violates, rather than solving from scratch.
pub(crate) struct Master<'a> {
    portfolios: &'a Portfolios,
    unit: f64,
    cuts: Vec<DVector<f64>>,
    /// Where the last solve ended, with a row for each cut added since, if it can be a start.
    basis: Option<lp::Basis>,
}

impl Master<'_> {
    /// Adds `cut`, a row the next solve has to meet.
    pub(crate) fn add(&mut self, cut: DVector<f64>) {
        self.cuts.push(cut);
        if let Some(basis) = &mut self.basis {
            basis.add_row();
        }
    }

    /// Whether `cut` is one the program already has.
    pub(crate) fn contains(&self, cut: &DVector<f64>) -> bool {
        self.cuts.contains(cut)
    }

    /// The portfolio whose largest `cutᵀw` is least, and that largest value, in the units of
    /// the cuts: unscaling it could take it out of the normal range, and the caller compares it
    /// with scaled values.
    ///
    /// # Errors
    ///
    /// [`CutupError::NotConverged`] if the linear program stops early.
    pub(crate) fn solve(&mut self) -> Result<(Vec<f64>, f64), CutupError> {
        let portfolios = self.portfolios;
        let bounds = &portfolios.bounds;
        let n = bounds.lower.len();
        let groups = bounds.rows.a.ncols();
        let target = usize::from(portfolios.target.is_some());
        // Columns: w (n), η, each group's total, then a slack for each cut. Rows: the budget,
        // the target, each group, then each cut. The cuts come last, so a new one adds a row and
        // a column after the others.
        let (eta, totals) = (n, n + 1);
        let slacks = totals + groups;
        let cols = slacks + self.cuts.len();
        let first_cut = 1 + target + groups;
        let rows = first_cut + self.cuts.len();
        let mut a = DMatrix::zeros(rows, cols);
        let mut b = DVector::zeros(rows);
        a.view_mut((0, 0), (1, n)).fill(1.0);
        b[0] = 1.0;
        if let Some((rho, mu)) = &portfolios.target {
            a.view_mut((1, 0), (1, n))
                .copy_from(&(mu * self.unit).transpose());
            b[1] = rho * self.unit;
        }
        for (g, members) in bounds.rows.a.column_iter().enumerate() {
            // Gᵀw − t = 0, with t within the group's limits.
            let row = 1 + target + g;
            a.view_mut((row, 0), (1, n)).copy_from(&members.transpose());
            a[(row, totals + g)] = -1.0;
        }
        for (k, cut) in self.cuts.iter().enumerate() {
            // cutᵀw − η + s = 0, s ≥ 0.
            let row = first_cut + k;
            a.view_mut((row, 0), (1, n)).copy_from(&cut.transpose());
            a[(row, eta)] = -1.0;
            a[(row, slacks + k)] = 1.0;
        }
        let mut c = DVector::zeros(cols);
        c[eta] = 1.0;
        let lower = DVector::from_fn(cols, |j, _| match j {
            j if j < n => bounds.lower[j],
            j if j == eta => f64::NEG_INFINITY,
            j if j < slacks => bounds.rows.lower[j - totals],
            _ => 0.0,
        });
        let upper = DVector::from_fn(cols, |j, _| match j {
            j if j < n => bounds.upper[j],
            j if j == eta => f64::INFINITY,
            j if j < slacks => bounds.rows.upper[j - totals],
            _ => f64::INFINITY,
        });
        let (outcome, basis) = lp::solve_warm(&c, &a, &b, &lower, &upper, self.basis.as_ref())?;
        self.basis = basis;
        let lp::Outcome::Optimal(solution) = outcome else {
            return Err(CutupError::NotConverged);
        };
        let weights = within_bounds(&solution.x.as_slice()[..n], bounds);
        Ok((weights, solution.objective))
    }
}

/// The program's weights `x`, clamped to `bounds` and summing to 1. The simplex method meets
/// the budget's row only to its tolerance, so `x` can sum to a little more or less than 1, and
/// scaling it to 1 could move a weight at its cap past it, or one at its floor below. Instead,
/// what the clamped weights miss of 1 goes to those with the most room toward it, each only as
/// far as its bounds. Bounds that a portfolio meets have room for all of it. Each weight then
/// holds its own bounds exactly. Groups and a target hold to the rounding of their sums, as
/// they did when the weights were scaled: what moves is rounding, about `1e-15`.
fn within_bounds(x: &[f64], bounds: &qp::Bounds) -> Vec<f64> {
    let (lower, upper) = (&bounds.lower, &bounds.upper);
    let mut weights: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, w)| w.clamp(lower[i], upper[i]))
        .collect();
    let mut missing = 1.0 - weights.iter().sum::<f64>();
    let up = missing > 0.0;
    let room = |i: usize, w: f64| if up { upper[i] - w } else { w - lower[i] };
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&i, &j| room(j, weights[j]).total_cmp(&room(i, weights[i])));
    for i in order {
        if missing == 0.0 {
            break;
        }
        let moved = (weights[i] + missing).clamp(lower[i], upper[i]);
        missing -= moved - weights[i];
        weights[i] = moved;
    }
    // A weight of -0.0, from the program or the clamp, is 0.0, which prints without a sign:
    // adding 0.0 makes it so and leaves every other weight as it is.
    weights.iter().map(|w| w + 0.0).collect()
}

/// The columns of the dual linear program, each with its cost and bounds, in the order
/// they're pushed. Built a column at a time rather than at computed offsets, so that no
/// column's place depends on arithmetic.
struct Columns {
    /// Whether the scenarios' weights sum to 1, a row after the asset rows.
    budget: bool,
    a: Vec<DVector<f64>>,
    cost: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Columns {
    fn new(budget: bool) -> Self {
        Self {
            budget,
            a: Vec::new(),
            cost: Vec::new(),
            lower: Vec::new(),
            upper: Vec::new(),
        }
    }

    /// A column with `assets` in the asset rows and, if there's one, `budget` in the row that
    /// sums the scenarios' weights.
    fn push(
        &mut self,
        assets: impl Iterator<Item = f64>,
        budget: f64,
        cost: f64,
        (lower, upper): (f64, f64),
    ) {
        let entries: Vec<f64> = if self.budget {
            assets.chain([budget]).collect()
        } else {
            assets.collect()
        };
        self.a.push(DVector::from_vec(entries));
        self.cost.push(cost);
        self.lower.push(lower);
        self.upper.push(upper);
    }
}

/// The power of two that scales `largest`, the largest return in size, to between ½ and 1, or
/// 1 if it's 0. Within the exponents an `f64` can scale by in one step, so that a subnormal
/// `largest` is scaled up as far as that goes.
pub(crate) fn unit_scale(largest: f64) -> f64 {
    if largest > 0.0 {
        2f64.powi(-(largest.log2().floor() as i32 + 1).clamp(-1022, 1022))
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::methods::normalized;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn the_worst_case_is_the_optimum_unscaled() {
        // Returns of a few percent scale by 16, so the optimum has to be divided back by it.
        let data = Returns::returns_matrix(crate::test_support::factor_returns(4, 30, 3)).unwrap();
        let unit = unit_scale(data.returns.amax());
        assert_eq!(unit, 16.0);
        let beta = 0.8;
        let cap = crate::cvar::tail_weight(beta, 30).unwrap();
        let losses = (&data.returns * unit).transpose().map(|r| -r);
        let portfolios = Portfolios::new(&data, &WeightBounds::default(), None).unwrap();
        let (weights, worst) = portfolios
            .min_max(&losses, unit, (0.0, cap), true, &[])
            .unwrap();
        let cvar = data.cvar(&weights.into(), beta).unwrap();
        assert!(cvar > 0.0);
        assert!((worst - cvar).abs() <= 1e-15, "{worst} vs {cvar}");
    }

    #[test]
    fn the_primal_and_the_dual_agree() {
        // The same min-max over some cuts, by the program with a row per cut and the one with
        // a row per asset, within bounds, two groups and a target. Returns that scale by 8.
        let returns = crate::test_support::factor_returns(5, 30, 9);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let unit = unit_scale(data.returns.amax());
        assert_eq!(unit, 8.0);
        let cuts: Vec<DVector<f64>> = (0..8)
            .map(|k| returns.row(k).transpose() * unit - returns.row(k + 8).transpose() * unit)
            .collect();
        let mu = data.returns.row_mean();
        let bounds = WeightBounds::uniform(0.05, 0.4)
            .group([0, 1], 0.3, 0.5)
            .group([2, 3, 4], 0.0, 0.6);
        for target in [None, Some(0.4 * mu[0] + 0.3 * mu[2] + 0.3 * mu[4])] {
            let portfolios = Portfolios::new(&data, &bounds, target).unwrap();
            let (primal, worst) = portfolios.min_of_max(&cuts, unit).unwrap();
            let (dual, worst_dual) = portfolios
                .min_max(
                    &DMatrix::from_columns(&cuts),
                    unit,
                    (0.0, f64::INFINITY),
                    true,
                    &[],
                )
                .unwrap();
            assert!(
                (worst / unit - worst_dual).abs() <= 1e-15,
                "{worst} vs {worst_dual}"
            );
            // Each one's worst case is its largest cut, unscaled.
            for weights in [&primal, &dual] {
                let w = DVector::from_column_slice(weights);
                let largest = cuts.iter().map(|c| c.dot(&w)).fold(f64::MIN, f64::max) / unit;
                assert!(
                    (largest - worst / unit).abs() <= 1e-15,
                    "{largest} vs {worst}"
                );
                assert!(
                    weights
                        .iter()
                        .all(|&x| (0.05 - 1e-15..=0.4 + 1e-15).contains(&x))
                );
                assert!((0.3 - 1e-15..=0.5 + 1e-15).contains(&(weights[0] + weights[1])));
                assert!(weights[2] + weights[3] + weights[4] <= 0.6 + 1e-15);
                if let Some(target) = target {
                    let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
                    assert!((achieved - target).abs() <= 1e-17, "{achieved} vs {target}");
                }
            }
        }
    }

    #[test]
    fn each_round_from_the_last_matches_solving_from_scratch() {
        // Cuts added one at a time, each solve starting from the last, within bounds, a group
        // and a target, against the same cuts solved from scratch each time.
        let returns = crate::test_support::factor_returns(5, 30, 4);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let unit = unit_scale(data.returns.amax());
        let mu = data.returns.row_mean();
        let bounds = WeightBounds::uniform(0.0, 0.5).group([1, 2], 0.2, 0.6);
        let target = Some(0.3 * mu[0] + 0.4 * mu[1] + 0.3 * mu[3]);
        let portfolios = Portfolios::new(&data, &bounds, target).unwrap();
        let mut master = portfolios.master(unit);
        let mut cuts = Vec::new();
        for k in 0..12 {
            let cut = (returns.row(k).transpose() - returns.row(k + 12).transpose()) * unit;
            assert!(!master.contains(&cut));
            master.add(cut.clone());
            assert!(master.contains(&cut));
            cuts.push(cut);
            let (weights, worst) = master.solve().unwrap();
            // Each solve leaves a basis for the next to start from, so the rounds after the
            // first are warm starts.
            assert!(master.basis.is_some(), "{k}");
            // Where the program ties, the two can land on different vertices, whose values
            // round differently, and differently again on wasm.
            let (_, scratch) = portfolios.min_of_max(&cuts, unit).unwrap();
            assert!(
                (worst - scratch).abs() <= 1e-12 * scratch.abs().max(1.0),
                "{k}: {worst} vs {scratch}"
            );
            assert!(weights.iter().all(|&w| (-1e-15..=0.5 + 1e-15).contains(&w)));
        }
    }

    #[test]
    fn the_worst_case_can_be_below_0() {
        // Early cuts can lie below 0 everywhere: here −0.1 and −0.2 on every asset, so −0.1 at
        // any portfolio. The largest has no floor at 0.
        let data = Returns::returns_matrix(crate::test_support::factor_returns(3, 20, 1)).unwrap();
        let portfolios = Portfolios::new(&data, &WeightBounds::default(), None).unwrap();
        let cuts = [
            DVector::from_element(3, -0.1),
            DVector::from_element(3, -0.2),
        ];
        let (_, worst) = portfolios.min_of_max(&cuts, 1.0).unwrap();
        assert!((worst + 0.1).abs() <= 1e-15, "{worst}");
    }

    #[test]
    fn the_unit_scale_takes_the_largest_return_to_between_a_half_and_1() {
        for largest in [0.5, 0.75, 1.0, 3.0, 1e100, 1e-100, f64::MIN_POSITIVE] {
            let scaled = largest * unit_scale(largest);
            assert!((0.5..1.0).contains(&scaled), "{largest}: {scaled}");
        }
        assert_eq!(unit_scale(1.0), 0.5);
        // Past what one power of two can reach, as far as it goes.
        assert_eq!(unit_scale(f64::MAX), 2f64.powi(-1022));
        assert_eq!(unit_scale(5e-324), 2f64.powi(1022));
        assert_eq!(unit_scale(0.0), 1.0);
    }

    /// Bounds `lower ≤ w ≤ upper` with no groups, for [`within_bounds`].
    fn plain(lower: &[f64], upper: &[f64]) -> qp::Bounds {
        let labels: Vec<String> = (0..lower.len()).map(|i| i.to_string()).collect();
        WeightBounds::per_asset(lower.to_vec(), upper.to_vec())
            .resolve(&labels)
            .unwrap()
    }

    #[test]
    fn weights_a_little_short_of_one_keep_a_weight_at_its_cap() {
        let bounds = plain(&[0.0; 3], &[0.5, 1.0, 1.0]);
        let x = [0.5, 0.3, 0.2 - 3e-15];
        // Scaling them to 1 moves the first past its cap.
        assert!(normalized(DVector::from_column_slice(&x))[0] > 0.5);
        let weights = within_bounds(&x, &bounds);
        assert_eq!(weights[0], 0.5);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 2.0 * f64::EPSILON);
        // The last, with the most room below its cap, takes what's missing.
        assert!(weights[1] == 0.3 && weights[2] > 0.2 - 3e-15);
    }

    #[test]
    fn weights_a_little_over_one_keep_a_weight_at_its_floor() {
        let bounds = plain(&[0.2, 0.0, 0.0], &[f64::INFINITY; 3]);
        let x = [0.2, 0.5, 0.3 + 3e-15];
        assert!(normalized(DVector::from_column_slice(&x))[0] < 0.2);
        let weights = within_bounds(&x, &bounds);
        assert_eq!(weights[0], 0.2);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 2.0 * f64::EPSILON);
        // The middle, with the most room above its floor, gives up what's over.
        assert!(weights[1] < 0.5 && weights[2] == 0.3 + 3e-15);
    }

    #[test]
    fn what_one_weight_has_no_room_for_goes_to_the_next() {
        // 0.375 short: the last has room for 0.3125 and the first for the other 0.0625, taken
        // in that order. Each is a binary fraction, so the sums are exact.
        let bounds = plain(&[0.0; 3], &[0.5, 0.25, 0.4375]);
        let weights = within_bounds(&[0.25, 0.25, 0.125], &bounds);
        assert_eq!(weights, [0.3125, 0.25, 0.4375]);
        // A weight past its bounds comes back to them first, and then the rest makes up 1.
        let weights = within_bounds(&[0.625, 0.25, 0.125], &bounds);
        assert_eq!(weights, [0.5, 0.25, 0.25]);
    }

    #[test]
    fn room_is_the_distance_to_the_bound() {
        // Short of 1, the first has 0.5 below its cap and the second 0.375, so the first takes
        // the 0.375: by the ratio to the cap, 2 against 4, the second would.
        let bounds = plain(&[0.0; 2], &[1.0, 0.5]);
        assert_eq!(within_bounds(&[0.5, 0.125], &bounds), [0.875, 0.125]);
        // Over 1, the first has 0.5 above its floor and the second 0.375, so the first gives up
        // the 0.125: by the sum with the floor, 0.5 against 0.875, the second would.
        let bounds = plain(&[0.0, 0.25], &[f64::INFINITY; 2]);
        assert_eq!(within_bounds(&[0.5, 0.625], &bounds), [0.375, 0.625]);
        // And by the ratio to the floor, 3 against 6, the second would here.
        let bounds = plain(&[0.25, 0.0625], &[f64::INFINITY; 2]);
        assert_eq!(within_bounds(&[0.75, 0.375], &bounds), [0.625, 0.375]);
    }

    #[test]
    fn weights_that_sum_to_one_within_their_bounds_are_left_as_they_are() {
        let bounds = plain(&[0.0; 3], &[1.0; 3]);
        assert_eq!(
            within_bounds(&[0.5, 0.25, 0.25], &bounds),
            [0.5, 0.25, 0.25]
        );
        // But a weight of -0.0 comes back as 0.0.
        let weights = within_bounds(&[1.0, -0.0, 0.0], &bounds);
        assert!(weights.iter().all(|w| w.is_sign_positive()), "{weights:?}");
    }
}
