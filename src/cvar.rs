//! Conditional value at risk: measuring it, and the long-only portfolio that minimizes it.

use crate::bounds::{ReturnRange, feasible_start};
use crate::methods::normalized;
use crate::{Allocation, CutupError, Returns, WeightBounds, Weights, lp};
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "serde")]
mod serde_impl;

/// The long-only portfolio with the lowest conditional value at risk (CVaR, or expected
/// shortfall): the average loss over the worst `1 − β` of the return scenarios, for a
/// confidence `β`, 0.95 by default (Rockafellar and Uryasev, 2000).
///
/// The scenarios are the returns in [`Returns`], one per row, each equally likely, so CVaR is in
/// their units and period: with daily returns, it's daily CVaR. When `(1 − β)T` isn't a whole
/// number of the `T` scenarios, the last one in the tail counts in part; when it's under one,
/// CVaR is the worst loss.
///
/// Unlike the variance-based methods it looks only at the scenarios' losses, not at a
/// covariance, so it can spread weight differently when returns are skewed or fat-tailed. It
/// needs enough scenarios to have a tail worth averaging: at 95% that's at least 20.
///
/// ```
/// use cutup::{MinCvar, Returns, WeightBounds};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
/// ];
/// let data = Returns::new(&returns)?;
///
/// let method = MinCvar::default().confidence(0.8).bounds(WeightBounds::uniform(0.0, 0.6));
/// let weights = data.allocate(&method)?;
/// let cvar = data.cvar(&weights, 0.8)?;
///
/// // No asset alone does better.
/// for asset in 0..3 {
///     let mut alone = vec![0.0; 3];
///     alone[asset] = 1.0;
///     assert!(cvar <= data.cvar(&alone.into(), 0.8)? + 1e-12);
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the confidence isn't in `(0, 1)`, or the target return
///   isn't finite.
/// * [`CutupError::InvalidAssetValue`] if an asset's lower bound is negative or not finite, or
///   its upper bound is below its lower bound or NaN. For [uniform](WeightBounds::uniform)
///   bounds, it's [`CutupError::InvalidConfig`].
/// * [`CutupError::DimensionMismatch`] if per-asset bounds don't have one entry per asset.
/// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if a setting names an
///   asset that isn't in the data, or names one twice.
/// * [`CutupError::InfeasibleBounds`] if the bounds can't sum to 1.
/// * [`CutupError::InfeasibleGroups`] if no portfolio within the bounds meets every
///   [group](WeightBounds::group).
/// * [`CutupError::TargetReturnOutOfRange`] if no portfolio within the bounds and groups has
///   the target return, by the sample mean of the returns.
/// * [`CutupError::NotConverged`] if the linear program stops early, which only extreme inputs
///   should cause.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(as = "serde_impl::Wire"))]
#[non_exhaustive]
pub struct MinCvar {
    confidence: f64,
    bounds: WeightBounds,
    target_return: Option<f64>,
}

impl Default for MinCvar {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            bounds: WeightBounds::default(),
            target_return: None,
        }
    }
}

impl MinCvar {
    /// Sets the confidence `β`, in `(0, 1)`: CVaR averages the worst `1 − β` of the
    /// scenarios. Defaults to 0.95.
    pub fn confidence(mut self, beta: f64) -> Self {
        self.confidence = beta;
        self
    }

    /// Keeps each weight within `bounds`. Defaults to long-only with no caps.
    pub fn bounds(mut self, bounds: WeightBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Requires an expected portfolio return, by the sample mean of the returns, of
    /// `target`.
    pub fn target_return(mut self, target: f64) -> Self {
        self.target_return = Some(target);
        self
    }
}

impl Allocation for MinCvar {
    /// Solves the linear program's dual, which has a row per asset rather than per scenario.
    ///
    /// CVaR is the most a probability weighting `z` of the scenarios can make of the loss,
    /// `max zᵀ(−Rw)` over `0 ≤ zₜ ≤ 1/((1 − β)T)` with `Σzₜ = 1`, so minimizing it is a
    /// min-max. Swapping the two and writing the minimum over the portfolio as its own dual
    /// gives a maximization over `z` and that dual's multipliers: `ν` for the budget, `η` for
    /// the target return, `λ⁺`, `λ⁻` for the bounds, and `γ⁺`, `γ⁻` for the groups' floors and
    /// caps, subject to `−Rᵀz = ν1 + ημ + λ⁺ − λ⁻ + G(γ⁺ − γ⁻)`, one row per asset, where
    /// column `k` of `G` marks group `k`'s members. The weights are those rows' multipliers,
    /// and CVaR is its optimum. A group limit the bounds already meet has no multiplier.
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let returns = &data.returns;
        let (t, n) = returns.shape();
        let cap = tail_weight(self.confidence, t)?;
        let bounds = self.bounds.resolve(data.labels())?;
        // Groups that can't all be met leave the dual unbounded rather than infeasible, so
        // they're found first, from the portfolios themselves: by the range's ends with a
        // target, otherwise by the simplex solver's start. Without groups, neither can fail.
        let target = match self.target_return {
            Some(target) if !target.is_finite() => {
                return Err(CutupError::InvalidConfig {
                    field: "target_return",
                    value: target,
                });
            }
            Some(target) => {
                let mu = returns.row_mean().transpose();
                Some((ReturnRange::new(&mu, &bounds)?.reachable(target)?, mu))
            }
            None => {
                feasible_start(&bounds)?;
                None
            }
        };
        // CVaR scales with the returns, and its minimizer doesn't change, so the program is
        // solved on returns scaled to at most 1 in size: next to the bounds and the budget's
        // entries, which don't scale, returns near 1e100 or 1e-100 leave the simplex method's
        // tolerances meaningless. A power of two, so that the scaling is exact.
        let unit = unit_scale(returns.amax());
        let returns = &(returns * unit);
        let target = target.map(|(rho, mu)| (rho * unit, mu * unit));

        // The dual's columns, in order: z (t), ν, then η with a target, λ⁺ (n), λ⁻ (n), and a γ
        // for each finite group limit. Resolved bounds are all finite, so every asset has a
        // λ⁻. Maximizing ν + ρη + lᵀλ⁺ − uᵀλ⁻ + Lᵀγ⁺ − Uᵀγ⁻ is minimizing its negative.
        let mut dual = Columns::default();
        for row in returns.row_iter() {
            dual.push(row.iter().map(|r| -r), 1.0, 0.0, (0.0, cap));
        }
        let free = (f64::NEG_INFINITY, f64::INFINITY);
        dual.push(std::iter::repeat_n(-1.0, n), 0.0, -1.0, free);
        if let Some((rho, mu)) = &target {
            dual.push(mu.iter().map(|m| -m), 0.0, -rho, free);
        }
        let nonnegative = (0.0, f64::INFINITY);
        let unit = |i: usize, sign: f64| (0..n).map(move |j| if j == i { sign } else { 0.0 });
        for i in 0..n {
            dual.push(unit(i, -1.0), 0.0, -bounds.lower[i], nonnegative);
        }
        for i in 0..n {
            dual.push(unit(i, 1.0), 0.0, bounds.upper[i], nonnegative);
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
        let mut b = DVector::zeros(n + 1);
        b[n] = 1.0;
        let (c, lower, upper) = (
            DVector::from_vec(dual.cost),
            DVector::from_vec(dual.lower),
            DVector::from_vec(dual.upper),
        );

        // The portfolio is feasible and CVaR is bounded below on it, so the dual has an optimum.
        let start = starting_tail(returns, cap);
        let lp::Outcome::Optimal(solution) = lp::solve_from(&c, &a, &b, &lower, &upper, &start)?
        else {
            return Err(CutupError::NotConverged);
        };
        let weights = DVector::from_fn(n, |i, _| {
            solution.y[i].clamp(bounds.lower[i], bounds.upper[i])
        });
        Ok(data.weights(normalized(weights)))
    }
}

/// The columns of [`MinCvar`]'s dual linear program, each with its cost and bounds, in the
/// order they're pushed. Built a column at a time rather than at computed offsets, so that no
/// column's place depends on arithmetic.
#[derive(Default)]
struct Columns {
    a: Vec<DVector<f64>>,
    cost: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl Columns {
    /// A column with `assets` in the asset rows and `budget` in the budget row `Σzₜ = 1`.
    fn push(
        &mut self,
        assets: impl Iterator<Item = f64>,
        budget: f64,
        cost: f64,
        (lower, upper): (f64, f64),
    ) {
        self.a
            .push(DVector::from_vec(assets.chain([budget]).collect()));
        self.cost.push(cost);
        self.lower.push(lower);
        self.upper.push(upper);
    }
}

impl Returns {
    /// The conditional value at risk of `weights` at confidence `beta`: the average loss over
    /// the worst `1 − β` of the return scenarios, each row of returns one equally likely
    /// scenario. When `(1 − β)T` isn't a whole number of the `T` scenarios, the last one in
    /// the tail counts in part; when it's under one, this is the worst loss. A gain counts as
    /// a negative loss, so CVaR can be negative.
    ///
    /// This is what [`MinCvar`] minimizes.
    ///
    /// # Errors
    ///
    /// * [`CutupError::InvalidConfig`] if `beta` isn't in `(0, 1)`.
    /// * [`CutupError::DimensionMismatch`] if there isn't one weight per asset.
    /// * [`CutupError::LabelMismatch`] if the weights are labeled with other assets than the
    ///   data, or in another order. Weights labeled by column, such as `Weights::from(vec)`,
    ///   are taken in column order.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// // Losses of 4%, 2%, 1% and a 1% gain: the worst half averages 3%.
    /// let returns = [[-0.04], [-0.02], [-0.01], [0.01]];
    /// let data = Returns::new(&returns)?;
    /// let cvar = data.cvar(&vec![1.0].into(), 0.5)?;
    /// assert!((cvar - 0.03).abs() < 1e-15);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn cvar(&self, weights: &Weights, beta: f64) -> Result<f64, CutupError> {
        let cap = tail_weight(beta, self.returns.nrows())?;
        self.check_weights(weights)?;
        let w = DVector::from_column_slice(weights.as_slice());
        Ok(tail_loss(
            (-(&self.returns * w)).iter().copied().collect(),
            cap,
        ))
    }
}

/// The average of the worst `losses`, each weighing `cap` up to a total of 1: their CVaR, for
/// `cap` from [`tail_weight`].
pub(crate) fn tail_loss(mut losses: Vec<f64>, cap: f64) -> f64 {
    let t = losses.len();
    // Only the worst ⌈1/cap⌉ losses are in the tail, one more for rounding: those go first, and
    // only they're sorted.
    let tail = ((1.0 / cap).ceil() as usize + 1).min(t);
    if tail < t {
        losses.select_nth_unstable_by(tail, |a, b| b.total_cmp(a));
    }
    losses[..tail].sort_by(|a, b| b.total_cmp(a));
    let mut left = 1.0f64;
    let mut total = 0.0;
    for &loss in &losses[..tail] {
        let share = cap.min(left);
        total += share * loss;
        left -= share;
        if left <= 0.0 {
            break;
        }
    }
    total
}

/// The power of two that scales `largest`, the largest return in size, to between ½ and 1, or
/// 1 if it's 0. Within the exponents an `f64` can scale by in one step, so that a subnormal
/// `largest` is scaled up as far as that goes.
fn unit_scale(largest: f64) -> f64 {
    if largest > 0.0 {
        2f64.powi(-(largest.log2().floor() as i32 + 1).clamp(-1022, 1022))
    } else {
        1.0
    }
}

/// The scenarios whose `zₜ` start at `cap` in [`MinCvar`]'s LP: those where the equal-weight
/// portfolio loses most, worst first, as many as the whole number nearest `1/cap`, and at least
/// one.
///
/// Every asset row of the LP has `b = 0`, so starting every `zₜ` at 0 leaves all those rows
/// without a residual: a degenerate start, which the LP leaves only by widening its bounds,
/// and more slowly. Starting these at the cap leaves asset `i`'s row a residual unless its
/// returns over them sum to zero, and starts `z` near a tail the optimum is likely to share.
/// From 200 assets that's faster than starting the first scenarios there: 0.31 s against
/// 0.48 s at 200, and 1.4 s against 2.7 s at 300.
///
/// The nearest whole number, not the floor: `1/cap` is `(1 − β)T`, which rounding can leave
/// just under a whole number, as for `β = 0.9` and `T = 10`. At least one, so that a cap over 1,
/// under one scenario's worth of tail, doesn't bring back the start at 0. Then `Σzₜ` can start
/// over 1, and the budget row's residual is negative, which phase 1 handles like any other.
fn starting_tail(returns: &DMatrix<f64>, cap: f64) -> Vec<usize> {
    let losses: Vec<f64> = returns.row_iter().map(|row| -row.sum()).collect();
    let mut scenarios: Vec<usize> = (0..returns.nrows()).collect();
    scenarios.sort_by(|&p, &q| losses[q].total_cmp(&losses[p]));
    let count = (1.0 / cap).round().max(1.0) as usize;
    scenarios.truncate(count.min(returns.nrows()));
    scenarios
}

/// The weight `1/((1 − β)T)` of each of `t` scenarios in the tail at confidence `beta`.
pub(crate) fn tail_weight(beta: f64, t: usize) -> Result<f64, CutupError> {
    if !(beta > 0.0 && beta < 1.0) {
        return Err(CutupError::InvalidConfig {
            field: "confidence",
            value: beta,
        });
    }
    Ok(1.0 / ((1.0 - beta) * t as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn allocator() -> Returns {
        Returns::returns_matrix(dmatrix![
             0.010, -0.020,  0.005;
            -0.004,  0.015,  0.002;
             0.007,  0.003, -0.006;
             0.002, -0.001,  0.004;
            -0.003,  0.006,  0.001;
             0.004, -0.008,  0.003;
            -0.006,  0.011, -0.002;
             0.001,  0.002, -0.004
        ])
        .unwrap()
    }

    #[test]
    fn tail_loss_counts_the_scenario_rounding_leaves_in_the_tail() {
        // At β = 0.5 over 10 scenarios each of the worst 5 weighs 0.2, which leaves about
        // 5.6e-17 of the tail for the sixth.
        let cap = tail_weight(0.5, 10).unwrap();
        let mut losses = vec![0.0; 5];
        losses.extend([-1e300; 5]);
        assert!(tail_loss(losses, cap) < -1e283);
    }

    #[test]
    fn cvar_rejects_weights_labeled_for_other_data() {
        let returns = dmatrix![
            -0.04, 0.01, 0.02;
            -0.02, 0.03, -0.01;
            0.01, -0.02, 0.00
        ];
        let data = |labels: [&str; 3]| {
            Returns::returns_matrix(returns.clone())
                .unwrap()
                .labeled(labels)
                .unwrap()
        };
        let allocator = data(["SPY", "TLT", "GLD"]);
        let weights = allocator.allocate(&crate::EqualWeight).unwrap();
        assert!(allocator.cvar(&weights, 0.5).is_ok());
        // Weights labeled by column are taken in column order.
        let by_column = Weights::from(vec![1.0 / 3.0; 3]);
        assert_eq!(
            allocator.cvar(&by_column, 0.5),
            allocator.cvar(&weights, 0.5)
        );
        assert_eq!(
            data(["SPY", "GLD", "TLT"]).cvar(&weights, 0.5),
            Err(CutupError::LabelMismatch {
                column: 1,
                expected: "GLD".into(),
                actual: "TLT".into()
            })
        );
    }

    #[test]
    fn cvar_averages_the_tail() {
        let allocator = Returns::returns_matrix(dmatrix![-0.04; -0.02; -0.01; 0.01]).unwrap();
        let one = Weights::from(vec![1.0]);
        // Half of four scenarios: the two worst, 4% and 2%.
        assert!((allocator.cvar(&one, 0.5).unwrap() - 0.03).abs() < 1e-15);
        // 30% of four is 1.2 scenarios: all of the 4% and a fifth of the 2%, over 1.2.
        let expected = (0.04 + 0.2 * 0.02) / 1.2;
        assert!((allocator.cvar(&one, 0.7).unwrap() - expected).abs() < 1e-15);
        // Under one scenario, the worst loss.
        assert!((allocator.cvar(&one, 0.9).unwrap() - 0.04).abs() < 1e-15);
    }

    #[test]
    fn the_start_is_the_equal_weight_tail_worst_first() {
        let returns = dmatrix![
            0.01, 0.02;
            -0.03, 0.01;
            -0.01, -0.04;
            0.00, -0.01;
            0.02, 0.01
        ];
        // At a cap of 1/2, two scenarios: the 5% loss, then the 2%.
        assert_eq!(starting_tail(&returns, 0.5), vec![2, 1]);
        // At 1/3, a third: the 1%.
        assert_eq!(starting_tail(&returns, 1.0 / 3.0), vec![2, 1, 3]);
        // Under one scenario's worth, still the worst, rather than every zₜ at 0.
        assert_eq!(starting_tail(&returns, 2.0), vec![2]);
        // A tail of 0.9 × 10 scenarios gives 1/cap just under 1, which still counts as one.
        let cap = 1.0 / ((1.0 - 0.9) * 10.0);
        assert!(1.0 / cap < 1.0);
        assert_eq!(starting_tail(&returns, cap), vec![2]);
    }

    /// Every scenario weight used to start at 0, which left the LP on a degenerate vertex
    /// that it didn't leave before the iteration cap, from about 200 assets. Unoptimized,
    /// this takes about 40 seconds, so `tests.yml` runs it with `--release`.
    #[test]
    #[ignore = "about 40 s unoptimized; tests.yml runs it with --release --ignored"]
    fn converges_on_two_hundred_assets() {
        let returns = crate::test_support::factor_returns(200, 400, 7);
        let allocator = Returns::returns_matrix(returns).unwrap();
        let weights = allocator.allocate(&MinCvar::default()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w >= 0.0));
        let cvar = allocator.cvar(&weights, 0.95).unwrap();
        let equal = allocator.allocate(&crate::EqualWeight).unwrap();
        assert!(cvar < allocator.cvar(&equal, 0.95).unwrap());
    }

    #[test]
    fn the_optimum_matches_its_own_cvar_and_beats_every_vertex() {
        let allocator = allocator();
        for beta in [0.5, 0.75, 0.9] {
            let weights = allocator
                .allocate(&MinCvar::default().confidence(beta))
                .unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(weights.iter().all(|&w| w >= 0.0));
            let cvar = allocator.cvar(&weights, beta).unwrap();
            for asset in 0..3 {
                let mut alone = vec![0.0; 3];
                alone[asset] = 1.0;
                let single = allocator.cvar(&alone.into(), beta).unwrap();
                assert!(cvar <= single + 1e-15, "{beta}: {cvar} > {single}");
            }
        }
    }

    #[test]
    fn bounds_and_targets_hold() {
        let allocator = allocator();
        let mu = allocator.returns.row_mean();
        let target = (mu[0] + mu[1] + mu[2]) / 3.0 + 1e-4;
        let weights = allocator
            .allocate(
                &MinCvar::default()
                    .bounds(WeightBounds::uniform(0.1, 0.5))
                    .target_return(target),
            )
            .unwrap();
        assert!(
            weights
                .iter()
                .all(|&w| (0.1 - 1e-15..=0.5 + 1e-15).contains(&w))
        );
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - target).abs() < 1e-15, "{achieved} vs {target}");
    }

    #[test]
    fn groups_hold_a_hedge_at_their_limits() {
        // A and B hedge each other, and C loses 1% in both scenarios, so at 50% the tail is
        // the worse scenario, and CVaR is 0.02|A − B| + 0.01C: least at A = B, as much of
        // them as the groups allow.
        let allocator = Returns::returns_matrix(dmatrix![
            -0.02,  0.02, -0.01;
             0.02, -0.02, -0.01
        ])
        .unwrap();
        let method = MinCvar::default().confidence(0.5);
        for (bounds, expected, cvar) in [
            (WeightBounds::default(), [0.5, 0.5, 0.0], 0.0),
            (
                WeightBounds::default().group(["0"], 0.0, 0.2),
                [0.2, 0.2, 0.6],
                0.006,
            ),
            (
                WeightBounds::default().group([2], 0.5, 1.0),
                [0.25, 0.25, 0.5],
                0.005,
            ),
        ] {
            let weights = allocator.allocate(&method.clone().bounds(bounds)).unwrap();
            for (w, e) in weights.iter().zip(expected) {
                assert!((w - e).abs() < 1e-15, "{weights:?}");
            }
            let achieved = allocator.cvar(&weights, 0.5).unwrap();
            assert!((achieved - cvar).abs() < 1e-15, "{achieved}");
        }
    }

    #[test]
    fn groups_that_cant_all_be_met_are_found_first() {
        let grouped = WeightBounds::default()
            .group([0], 0.6, 1.0)
            .group([1], 0.6, 1.0);
        assert_eq!(
            allocator().allocate(&MinCvar::default().bounds(grouped)),
            Err(CutupError::InfeasibleGroups { group: None })
        );
    }

    #[test]
    fn a_target_that_isnt_finite_is_found_before_the_groups() {
        let grouped = WeightBounds::default()
            .group([0], 0.6, 1.0)
            .group([1], 0.6, 1.0);
        let method = MinCvar::default().target_return(f64::INFINITY);
        assert_eq!(
            allocator().allocate(&method.bounds(grouped)),
            Err(CutupError::InvalidConfig {
                field: "target_return",
                value: f64::INFINITY,
            })
        );
    }

    #[test]
    fn groups_that_cant_all_be_met_are_found_with_a_target() {
        let grouped = WeightBounds::default()
            .group([0], 0.6, 1.0)
            .group([1], 0.6, 1.0);
        let method = MinCvar::default().target_return(0.001);
        assert_eq!(
            allocator().allocate(&method.bounds(grouped)),
            Err(CutupError::InfeasibleGroups { group: None })
        );
    }

    #[test]
    fn groups_and_targets_hold() {
        let allocator = allocator();
        let mu = allocator.returns.row_mean();
        // Without the group, A and B hold 0.634 at this target; with it, its cap.
        let target = 0.0009;
        let plain = MinCvar::default().target_return(target);
        let free = allocator.allocate(&plain).unwrap();
        assert!(free[0] + free[1] > 0.63);
        let weights = allocator
            .allocate(&plain.bounds(WeightBounds::default().group([0, 1], 0.0, 0.6)))
            .unwrap();
        assert!((weights[0] + weights[1] - 0.6).abs() < 1e-15, "{weights:?}");
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - target).abs() < 1e-15, "{achieved} vs {target}");
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

    #[test]
    fn the_scale_of_the_returns_changes_nothing() {
        // From the fuzzer: at 1e96, the program's returns dwarfed its other entries, and the
        // simplex method's answer put everything on an asset its group capped at 0. Every
        // portfolio ties here, at the worst scenario's loss, so the answer only has to hold.
        let rows: [[f64; 5]; 15] = [
            [-1.0, -5.0, -170.0, -1.0, -1.0],
            [-1.0; 5],
            [-1.0; 5],
            [-1.0; 5],
            [-1.0; 5],
            [-1.0; 5],
            [-1.0; 5],
            [-1.0, -1.0, -1.0, -1.0, -250.0],
            [-1.0, -9729.0, -9767.0, -9767.0, -9767.0],
            [-9767.0; 5],
            [-1.0, -766.0, -1.0, -1.0, 5397.0],
            [5397.0; 5],
            [5397.0; 5],
            [5397.0, -1.0, -1.0, -17.0, 6143.0],
            [5911.0, 5911.0, 5911.0, -1.0, 0.0],
        ];
        for scale in [1e96, 1e-300] {
            let returns: Vec<[f64; 5]> = rows.iter().map(|row| row.map(|r| r * scale)).collect();
            let grouped = MinCvar::default().bounds(WeightBounds::default().group([0], 0.0, 0.0));
            let weights = Returns::new(&returns).unwrap().allocate(&grouped).unwrap();
            assert_eq!(weights[0], 0.0, "{scale}: {weights:?}");
        }
        // Where the answer is unique, it's the same at any scale.
        let base = allocator();
        let method = MinCvar::default().confidence(0.75);
        let expected = base.allocate(&method).unwrap();
        for scale in [1e100, 1e-100] {
            let scaled = Returns::returns_matrix(&base.returns * scale).unwrap();
            let weights = scaled.allocate(&method).unwrap();
            for (w, e) in weights.iter().zip(expected.iter()) {
                assert!(
                    (w - e).abs() < 1e-12,
                    "{scale}: {weights:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn lower_bounds_summing_to_one_fix_the_weights() {
        // 1 − 0.9 rounds to just under 0.1, the first asset's room above its floor.
        let allocator = Returns::returns_matrix(dmatrix![
             0.010, -0.020;
            -0.004,  0.015;
             0.007,  0.003
        ])
        .unwrap();
        let weights = allocator
            .allocate(
                &MinCvar::default().bounds(WeightBounds::per_asset(vec![0.1, 0.9], vec![1.0, 1.0])),
            )
            .unwrap();
        assert!((weights[0] - 0.1).abs() < 1e-15 && (weights[1] - 0.9).abs() < 1e-15);
    }

    #[test]
    fn a_single_asset_takes_the_whole_budget() {
        let one = Returns::returns_matrix(dmatrix![0.01; -0.02; 0.005; 0.003]).unwrap();
        for allocator in [
            MinCvar::default(),
            MinCvar::default().target_return(-0.0005),
        ] {
            assert_eq!(one.allocate(&allocator).unwrap().as_slice(), &[1.0]);
        }
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let allocator = allocator();
        for beta in [0.0, 1.0, f64::NAN] {
            assert!(matches!(
                allocator.allocate(&MinCvar::default().confidence(beta)),
                Err(CutupError::InvalidConfig {
                    field: "confidence",
                    ..
                })
            ));
        }
        assert!(matches!(
            allocator.allocate(&MinCvar::default().target_return(1.0)),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        for target in [f64::INFINITY, f64::NAN] {
            assert!(matches!(
                allocator.allocate(&MinCvar::default().target_return(target)),
                Err(CutupError::InvalidConfig {
                    field: "target_return",
                    ..
                })
            ));
        }
        assert_eq!(
            allocator.allocate(&MinCvar::default().bounds(WeightBounds::uniform(0.0, 0.2))),
            Err(CutupError::InfeasibleBounds {
                lower_sum: 0.0,
                upper_sum: 0.6000000000000001
            })
        );
        assert_eq!(
            allocator.cvar(&vec![1.0].into(), 0.5),
            Err(CutupError::DimensionMismatch {
                field: "weights",
                expected: 3,
                actual: 1
            })
        );
    }

    #[test]
    fn degenerate_scenarios_converge() {
        // A property test's shrunk input: most returns exactly zero, so the program is highly
        // degenerate. A reduced cost of pure rounding once entered here, moved nothing, and
        // left the basis singular.
        #[rustfmt::skip]
        let data = vec![
            0.049046206985969405, 0.0, 0.0, 0.018918236666771126, 0.0, 0.0, 0.0,
            0.02331762851401267, 0.0, -0.026903675628981525, 0.04253505321714205, 0.0,
            0.0361087756727277, 0.0, 0.034043311064955735, 0.0, 0.022568977423950948, 0.0,
            0.013270785074508322, 0.0, 0.0, -0.022042427374501777, 0.036451221073154974,
            0.0473428014604323, 0.0, -0.007958233019630202, -0.008144504154924737,
            -0.024854928967824912, 0.0, -0.035089569764353054, 0.049698982240741575, 0.0,
            0.02755697787157803, 0.0, 0.02358022931946681, -0.004670622996365131, 0.0, 0.0,
            0.0, 0.02242619203429917, 0.0, 0.0, 0.0, 0.039747808049717774,
            -0.01926202201906651, 0.001966764898914833, 0.0, 0.0, -0.025971216114690443, 0.0,
            0.0, -0.02382851848035964, 0.0013794705644694543, -0.004727636380476252,
        ];
        let allocator = Returns::returns_matrix(DMatrix::from_vec(9, 6, data)).unwrap();
        let beta = 0.8549681742164362;
        let weights = allocator
            .allocate(&MinCvar::default().confidence(beta))
            .unwrap();
        let cvar = allocator.cvar(&weights, beta).unwrap();
        let even = allocator.cvar(&vec![1.0 / 6.0; 6].into(), beta).unwrap();
        assert!(cvar <= even + 1e-15, "{cvar} > {even}");
    }
}
