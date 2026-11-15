//! The portfolio with the lowest mean absolute deviation, or the lowest mean shortfall below a
//! threshold.

use crate::dual::{Portfolios, unit_scale};
use crate::{Allocation, CutupError, Returns, WeightBounds, Weights};
use nalgebra::DMatrix;

#[cfg(feature = "serde")]
mod serde_impl;

/// The long-only portfolio with the lowest mean absolute deviation (MAD) of its return in each
/// period around their mean, `(1/T) Σₜ |rₜ − r̄|` (Konno and Yamazaki, 1991). With
/// [`downside`](Self::downside), the lowest mean shortfall below a threshold instead,
/// `(1/T) Σₜ max(τ − rₜ, 0)`, the first lower partial moment.
///
/// It's mean-variance without the squares: [`Returns::mad`] in place of the variance, so a
/// few large deviations count for less, and it looks only at the returns themselves, not a
/// covariance. Like [`MinCvar`](crate::MinCvar), it's a linear program, solved exactly.
///
/// ```
/// use cutup::{MinMad, Returns, WeightBounds};
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
/// let weights = data.allocate(&MinMad::default().bounds(WeightBounds::uniform(0.0, 0.6)))?;
/// let mad = data.mad(&weights)?;
/// // No asset alone does better.
/// for asset in 0..3 {
///     let mut alone = vec![0.0; 3];
///     alone[asset] = 1.0;
///     assert!(mad <= data.mad(&alone.into())? + 1e-12);
/// }
///
/// // The least shortfall below a loss of 0.1% a period.
/// let weights = data.allocate(&MinMad::default().downside(-0.001))?;
/// assert!(data.downside_mad(&weights, -0.001)? <= data.downside_mad(&vec![1.0 / 3.0; 3].into(), -0.001)?);
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the threshold or the target return isn't finite.
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
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(as = "serde_impl::Wire"))]
#[non_exhaustive]
pub struct MinMad {
    bounds: WeightBounds,
    target_return: Option<f64>,
    downside: Option<f64>,
}

impl MinMad {
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

    /// Minimizes the mean shortfall below `threshold`, a minimum acceptable return per period,
    /// rather than the deviation either side of the mean: only returns below the threshold
    /// count, by how far below it they are. See [`Returns::downside_mad`].
    ///
    /// The threshold is fixed. Below the portfolio's own mean, the shortfalls add up to the
    /// excesses above it, so they'd be half the MAD and the weights the same as without this.
    pub fn downside(mut self, threshold: f64) -> Self {
        self.downside = Some(threshold);
        self
    }
}

impl Allocation for MinMad {
    /// Solves the linear program's dual, which has a row per asset rather than per period, as
    /// for [`MinCvar`](crate::MinCvar).
    ///
    /// The MAD is the most `Σₜ zₜ(rₜ − r̄)ᵀw` can be over `|zₜ| ≤ 1/T`. Since the weights sum
    /// to 1, the shortfall below `τ` in period `t` is `(τ1 − rₜ)ᵀw` where it's positive, so the
    /// mean shortfall is the most `Σₜ zₜ(τ1 − rₜ)ᵀw` can be over `0 ≤ zₜ ≤ 1/T`. Either way,
    /// minimizing it is a min-max, and the dual of the minimum over the portfolio turns it into
    /// a maximization with one row per asset, whose multipliers are the weights.
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        if let Some(threshold) = self.downside.filter(|t| !t.is_finite()) {
            return Err(CutupError::InvalidConfig {
                field: "threshold",
                value: threshold,
            });
        }
        let portfolios = Portfolios::new(data, &self.bounds, self.target_return)?;
        let returns = &data.returns;
        let (t, n) = returns.shape();
        // The program is solved on returns, and a threshold, scaled to at most 1 in size, as
        // for `MinCvar`: neither the MAD's minimizer nor the shortfall's changes with it.
        let largest = returns.amax().max(self.downside.map_or(0.0, f64::abs));
        let unit = unit_scale(largest);
        let returns = returns * unit;
        // Each zₜ within 1 rather than 1/T: that scales the dual's optimum by T, but not its
        // multipliers, the weights, and keeps its entries near 1.
        let (scenarios, weighting) = match self.downside {
            None => {
                let mean = returns.row_mean();
                let deviations = DMatrix::from_fn(n, t, |i, k| returns[(k, i)] - mean[i]);
                (deviations, (-1.0, 1.0))
            }
            Some(threshold) => {
                let tau = threshold * unit;
                let shortfalls = DMatrix::from_fn(n, t, |i, k| tau - returns[(k, i)]);
                (shortfalls, (0.0, 1.0))
            }
        };
        let start = rising(&scenarios);
        let (weights, _) = portfolios.min_max(&scenarios, unit, weighting, false, &start)?;
        Ok(data.weights(weights))
    }
}

/// The periods whose `zₜ` start at their upper bound in [`MinMad`]'s program: those where
/// equal weights deviate upward, or fall short of the threshold, the column of `scenarios`
/// summing to more than 0.
///
/// Every asset row has a right-hand side of 0, so starting every `zₜ` at its lower bound, 0
/// for the shortfall, leaves those rows without a residual: a degenerate start, which the
/// linear program leaves only by widening its bounds, and more slowly. For the MAD, with
/// `zₜ` at `±1`, the signs of equal weights' deviations start `z` near where the optimum's are.
fn rising(scenarios: &DMatrix<f64>) -> Vec<usize> {
    (0..scenarios.ncols())
        .filter(|&k| scenarios.column(k).sum() > 0.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp;
    use crate::test_support::factor_returns;
    use crate::{EqualWeight, MinVariance};
    use nalgebra::DVector;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    /// The least MAD, or the least mean shortfall below `threshold`, of a long-only portfolio
    /// with weights in `[lower, upper]`, solved as the primal program directly: minimize
    /// `(1/T) Σeₜ` subject to `eₜ ≥ dₜᵀw`, and for the MAD `eₜ ≥ −dₜᵀw`, with `eₜ ≥ 0`. A
    /// formulation of its own, with a row per period, to check the dual against.
    fn primal(data: &Returns, threshold: Option<f64>, (lower, upper): (f64, f64)) -> f64 {
        let returns = &data.returns;
        let (t, n) = returns.shape();
        let mean = returns.row_mean();
        let d = |k: usize, i: usize| match threshold {
            None => returns[(k, i)] - mean[i],
            Some(tau) => tau - returns[(k, i)],
        };
        let sides: &[f64] = if threshold.is_some() {
            &[1.0]
        } else {
            &[1.0, -1.0]
        };
        // Columns: w (n), e (t), a slack for each row. Rows: each side of each period, the budget.
        let rows = sides.len() * t + 1;
        let cols = n + t + sides.len() * t;
        let mut a = DMatrix::zeros(rows, cols);
        let mut row = 0;
        for &side in sides {
            for k in 0..t {
                // eₖ − side·dₖᵀw − s = 0.
                for i in 0..n {
                    a[(row, i)] = -side * d(k, i);
                }
                a[(row, n + k)] = 1.0;
                a[(row, n + t + row)] = -1.0;
                row += 1;
            }
        }
        for i in 0..n {
            a[(row, i)] = 1.0;
        }
        let mut b = DVector::zeros(rows);
        b[row] = 1.0;
        let c = DVector::from_fn(cols, |j, _| {
            if (n..n + t).contains(&j) {
                1.0 / t as f64
            } else {
                0.0
            }
        });
        let lower = DVector::from_fn(cols, |j, _| if j < n { lower } else { 0.0 });
        let upper = DVector::from_fn(cols, |j, _| if j < n { upper } else { f64::INFINITY });
        let lp::Outcome::Optimal(solution) = lp::solve(&c, &a, &b, &lower, &upper).unwrap() else {
            panic!("the primal has an optimum");
        };
        solution.objective
    }

    fn data() -> Returns {
        Returns::returns_matrix(factor_returns(6, 40, 5)).unwrap()
    }

    #[test]
    fn the_start_is_where_equal_weights_rise() {
        // Columns summing to 2, −2, 0, 0.5 and −0.5: only those above 0 start at the upper
        // bound, a column that sums to exactly 0 not among them.
        let scenarios = DMatrix::from_column_slice(
            2,
            5,
            &[1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 0.25, 0.25, -0.25, -0.25],
        );
        assert_eq!(rising(&scenarios), vec![0, 3]);
    }

    #[test]
    fn the_optimum_is_the_primal_programs() {
        let data = data();
        for (lower, upper) in [(0.0, 1.0), (0.0, 0.3), (0.05, 0.4)] {
            let bounds = WeightBounds::uniform(lower, upper);
            let weights = data
                .allocate(&MinMad::default().bounds(bounds.clone()))
                .unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(
                weights
                    .iter()
                    .all(|&w| (lower - 1e-15..=upper + 1e-15).contains(&w))
            );
            let mad = data.mad(&weights).unwrap();
            let best = primal(&data, None, (lower, upper));
            assert!((mad - best).abs() <= 1e-13, "{upper}: {mad} vs {best}");

            for threshold in [0.0, -0.004, 0.003] {
                let method = MinMad::default().bounds(bounds.clone()).downside(threshold);
                let weights = data.allocate(&method).unwrap();
                let shortfall = data.downside_mad(&weights, threshold).unwrap();
                let best = primal(&data, Some(threshold), (lower, upper));
                assert!(
                    (shortfall - best).abs() <= 1e-13,
                    "{upper}, {threshold}: {shortfall} vs {best}"
                );
            }
        }
    }

    #[test]
    fn the_optimum_beats_other_methods() {
        let data = data();
        let weights = data.allocate(&MinMad::default()).unwrap();
        let mad = data.mad(&weights).unwrap();
        for other in [
            data.allocate(&EqualWeight).unwrap(),
            data.allocate(&MinVariance::default()).unwrap(),
        ] {
            assert!(mad <= data.mad(&other).unwrap() + 1e-15);
        }
        // Minimizing the MAD isn't minimizing the variance.
        let variance = data.allocate(&MinVariance::default()).unwrap();
        assert!(mad < data.mad(&variance).unwrap());
    }

    #[test]
    fn a_threshold_no_period_falls_short_of_costs_nothing() {
        let data = data();
        let floor = data.returns.min();
        let weights = data
            .allocate(&MinMad::default().downside(floor - 0.01))
            .unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(data.downside_mad(&weights, floor - 0.01), Ok(0.0));
    }

    #[test]
    fn bounds_groups_and_targets_hold() {
        let data = data();
        let mu = data.returns.row_mean();
        let target = mu.mean() + 2e-5;
        let method = MinMad::default()
            .bounds(WeightBounds::uniform(0.0, 0.5).group([0, 1, 2], 0.2, 0.4))
            .target_return(target);
        for method in [method.clone(), method.downside(0.0)] {
            let weights = data.allocate(&method).unwrap();
            assert!(weights.iter().all(|&w| (-1e-15..=0.5 + 1e-15).contains(&w)));
            let group: f64 = weights.iter().take(3).sum();
            assert!((0.2 - 1e-12..=0.4 + 1e-12).contains(&group), "{group}");
            let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
            assert!((achieved - target).abs() < 1e-15, "{achieved} vs {target}");
        }
    }

    #[test]
    fn the_scale_of_the_returns_changes_nothing() {
        let base = data();
        for method in [MinMad::default(), MinMad::default().downside(0.0)] {
            let expected = base.allocate(&method).unwrap();
            // Powers of two scale the program exactly.
            for scale in [2f64.powi(300), 2f64.powi(-300)] {
                let scaled = Returns::returns_matrix(&base.returns * scale).unwrap();
                assert_eq!(scaled.allocate(&method).unwrap(), expected);
            }
        }
        // A threshold scales with the returns too.
        let method = MinMad::default().downside(-0.004);
        let expected = base.allocate(&method).unwrap();
        let scale = 2f64.powi(200);
        let scaled = Returns::returns_matrix(&base.returns * scale).unwrap();
        let weights = scaled
            .allocate(&MinMad::default().downside(-0.004 * scale))
            .unwrap();
        assert_eq!(weights, expected);
    }

    #[test]
    fn a_threshold_past_every_return_takes_the_highest_mean() {
        // Every period falls short, by the threshold less the portfolio's return, so the least
        // mean shortfall is at the highest mean return: here all on the best asset.
        let data = data();
        let weights = data.allocate(&MinMad::default().downside(0.5)).unwrap();
        let mu = data.returns.row_mean();
        let best = mu.iter().copied().fold(f64::MIN, f64::max);
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - best).abs() <= 1e-15, "{weights:?}");
        // Far past the returns, each shortfall rounds to the threshold itself, so every
        // portfolio ties, but the answer is still one.
        let weights = data.allocate(&MinMad::default().downside(1e200)).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let data = data();
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                data.allocate(&MinMad::default().downside(threshold)),
                Err(CutupError::InvalidConfig {
                    field: "threshold",
                    ..
                })
            ));
        }
        assert!(matches!(
            data.allocate(&MinMad::default().target_return(f64::NAN)),
            Err(CutupError::InvalidConfig {
                field: "target_return",
                ..
            })
        ));
        assert!(matches!(
            data.allocate(&MinMad::default().target_return(1.0)),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        assert!(matches!(
            data.allocate(&MinMad::default().bounds(WeightBounds::uniform(0.0, 0.1))),
            Err(CutupError::InfeasibleBounds { .. })
        ));
        assert!(matches!(
            data.allocate(
                &MinMad::default().bounds(
                    WeightBounds::default()
                        .group([0, 1], 0.7, 1.0)
                        .group([1, 2], 0.0, 0.0)
                        .group([0], 0.0, 0.2)
                )
            ),
            Err(CutupError::InfeasibleGroups { .. })
        ));
    }

    /// As for `MinCvar`, the dual's rows start with residuals, so it doesn't stall on a
    /// degenerate vertex. Unoptimized, this takes about 35 seconds, so `tests.yml` runs it with
    /// `--release`, where it takes about 0.2.
    #[test]
    #[ignore = "about 35 s unoptimized; tests.yml runs it with --release --ignored"]
    fn converges_on_two_hundred_assets() {
        let data = Returns::returns_matrix(factor_returns(200, 400, 7)).unwrap();
        for method in [MinMad::default(), MinMad::default().downside(0.0)] {
            let weights = data.allocate(&method).unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            let equal = data.allocate(&EqualWeight).unwrap();
            assert!(data.mad(&weights).unwrap() < data.mad(&equal).unwrap());
        }
    }
}
