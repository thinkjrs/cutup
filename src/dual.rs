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
use crate::methods::normalized;
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
        let weights = DVector::from_fn(n, |i, _| {
            solution.y[i].clamp(bounds.lower[i], bounds.upper[i])
        });
        Ok((normalized(weights), -solution.objective / unit))
    }

    /// The portfolio whose largest `cutᵀw` over `cuts` is least, and that largest value.
    ///
    /// The same min-max as [`Portfolios::min_max`], with the cuts as the scenarios and their
    /// weighting any probability, but solved as the primal, `min η` subject to `cutᵀw ≤ η` for
    /// each cut, a row per cut rather than per asset: the bounds are the weights' own, so
    /// with few cuts the program is small however many assets there are. The cuts are
    /// multiplied by `unit`, as the scenarios are, and so is the largest value: unscaling it
    /// could take it out of the normal range, and the caller compares it with scaled values.
    ///
    /// # Errors
    ///
    /// [`CutupError::NotConverged`] if the linear program stops early.
    pub(crate) fn min_of_max(
        &self,
        cuts: &[DVector<f64>],
        unit: f64,
    ) -> Result<(Vec<f64>, f64), CutupError> {
        let bounds = &self.bounds;
        let n = bounds.lower.len();
        let groups = bounds.rows.a.ncols();
        let target = usize::from(self.target.is_some());
        // Columns: w (n), η, a slack for each cut, and one for each group's total. Rows: each
        // cut, the budget, the target, then each group.
        let (eta, slacks) = (n, n + 1);
        let totals = slacks + cuts.len();
        let rows = cuts.len() + 1 + target + groups;
        let mut a = DMatrix::zeros(rows, totals + groups);
        let mut b = DVector::zeros(rows);
        for (k, cut) in cuts.iter().enumerate() {
            // cutᵀw − η + s = 0, s ≥ 0.
            a.view_mut((k, 0), (1, n)).copy_from(&cut.transpose());
            a[(k, eta)] = -1.0;
            a[(k, slacks + k)] = 1.0;
        }
        let budget = cuts.len();
        a.view_mut((budget, 0), (1, n)).fill(1.0);
        b[budget] = 1.0;
        if let Some((rho, mu)) = &self.target {
            a.view_mut((budget + 1, 0), (1, n))
                .copy_from(&(mu * unit).transpose());
            b[budget + 1] = rho * unit;
        }
        let first = budget + 1 + target;
        for (g, members) in bounds.rows.a.column_iter().enumerate() {
            // Gᵀw − t = 0, with t within the group's limits.
            a.view_mut((first + g, 0), (1, n))
                .copy_from(&members.transpose());
            a[(first + g, totals + g)] = -1.0;
        }
        let cols = totals + groups;
        let mut c = DVector::zeros(cols);
        c[eta] = 1.0;
        let lower = DVector::from_fn(cols, |j, _| match j {
            j if j < n => bounds.lower[j],
            j if j == eta => f64::NEG_INFINITY,
            j if j < totals => 0.0,
            j => bounds.rows.lower[j - totals],
        });
        let upper = DVector::from_fn(cols, |j, _| match j {
            j if j < n => bounds.upper[j],
            j if j >= totals => bounds.rows.upper[j - totals],
            _ => f64::INFINITY,
        });
        let lp::Outcome::Optimal(solution) = lp::solve(&c, &a, &b, &lower, &upper)? else {
            return Err(CutupError::NotConverged);
        };
        let weights = DVector::from_fn(n, |i, _| {
            solution.x[i].clamp(bounds.lower[i], bounds.upper[i])
        });
        Ok((normalized(weights), solution.objective))
    }
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
}
