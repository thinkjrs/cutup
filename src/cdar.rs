//! The portfolio with the lowest conditional drawdown at risk.

use crate::cvar::tail_weight;
use crate::dual::{Portfolios, unit_scale};
use crate::{Allocation, CutupError, Returns, WeightBounds, Weights};
use nalgebra::{DMatrix, DVector};

#[cfg(feature = "serde")]
mod serde_impl;

/// The most rounds of cuts [`MinCdar`] takes before [`CutupError::NotConverged`]. Each adds a
/// cut its program didn't have, and there are finitely many, so it stops; in practice within
/// about 100, even at 500 assets and 1,000 periods.
const MAX_ROUNDS: usize = 1000;

/// How far toward the best portfolio so far [`MinCdar`] takes each cut, from the master's
/// minimizer: an in-out stabilization (Ben-Ameur and Neto, 2007). Kelley's cuts at the
/// minimizer alone zig-zag, and halfway takes about half as many rounds at 500 assets.
const STABILITY: f64 = 0.5;

/// The long-only portfolio with the lowest conditional drawdown at risk (CDaR): the average of
/// the worst `1 − β` of its drawdowns, for a confidence `β`, 0.95 by default (Chekhlov, Uryasev
/// and Zabarankin, 2005). See [`Returns::cdar`].
///
/// A drawdown is how far the running sum of the portfolio's returns, from 0 before the first
/// period, is below its highest so far, one after each period. The returns are summed, not
/// compounded, the form that keeps the program linear: with returns of a few percent a period,
/// the two differ only in the second order. When `(1 − β)T` isn't a whole number of the `T`
/// periods, the last drawdown in the tail counts in part; when it's under one, CDaR is the
/// maximum drawdown, so a `β` near 1 minimizes that.
///
/// Unlike every other method, it depends on the order of the rows of [`Returns`], which must
/// be in time order: losses in a row are a deeper drawdown than the same losses spread out. It
/// looks at the path of the returns rather than their spread, so it suits a budget that must
/// not fall far behind its best, such as ad spend whose results come in streaks.
///
/// ```
/// use cutup::{MinCdar, Returns, WeightBounds};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
///      [-0.006, -0.002,  0.003],
/// ];
/// let data = Returns::new(&returns)?;
///
/// let method = MinCdar::default().confidence(0.5).bounds(WeightBounds::uniform(0.0, 0.6));
/// let weights = data.allocate(&method)?;
/// let cdar = data.cdar(&weights, 0.5)?;
///
/// // No asset alone does better.
/// for asset in 0..3 {
///     let mut alone = vec![0.0; 3];
///     alone[asset] = 1.0;
///     assert!(cdar <= data.cdar(&alone.into(), 0.5)? + 1e-12);
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
/// * [`CutupError::NotConverged`] if a linear program stops early, or the cuts don't meet the
///   optimum within 1,000 rounds, which only extreme inputs should cause.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(as = "serde_impl::Wire"))]
#[non_exhaustive]
pub struct MinCdar {
    confidence: f64,
    bounds: WeightBounds,
    target_return: Option<f64>,
}

impl Default for MinCdar {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            bounds: WeightBounds::default(),
            target_return: None,
        }
    }
}

impl MinCdar {
    /// Sets the confidence `β`, in `(0, 1)`: CDaR averages the worst `1 − β` of the
    /// drawdowns. Defaults to 0.95.
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

impl Allocation for MinCdar {
    /// Solves by cutting planes (Kelley, 1960), with a program that grows with the rounds
    /// rather than the periods.
    ///
    /// CDaR is convex in the weights, and at any portfolio it's `gᵀw` for a cut `g`: the
    /// running sums' gap from each worst drawdown's peak, weighted by the tail. Every cut is
    /// below CDaR everywhere, so the most of the cuts so far is a lower bound on it. Each round
    /// minimizes that bound over the portfolios, a linear program with a row per cut, and adds
    /// a cut: at a point halfway from the best portfolio so far to the minimizer if that cut
    /// rules the minimizer out, otherwise at the minimizer, which always does. It stops when
    /// the best portfolio's CDaR meets the bound, to rounding, which makes it the optimum.
    ///
    /// The full program, with rows for every period's peak and drawdown, takes about 90
    /// seconds at 1,000 periods. This way it's milliseconds there for tens of assets, and a
    /// few seconds for 500.
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let returns = &data.returns;
        let (t, n) = returns.shape();
        let cap = tail_weight(self.confidence, t)?;
        let portfolios = Portfolios::new(data, &self.bounds, self.target_return)?;
        // CDaR scales with the returns, and its minimizer doesn't change, so the program is
        // solved on returns scaled to at most 1 in size, as for `MinCvar`.
        let unit = unit_scale(returns.amax());
        let scaled = returns * unit;
        let drawdowns = Drawdowns::new(&scaled, cap);
        // Everything below is in the scaled returns' units: the cuts, the CDaRs and the bound.
        let mut cuts = vec![drawdowns.cut(&vec![1.0 / n as f64; n]).0];
        let mut best: Option<(Vec<f64>, f64)> = None;
        for _ in 0..MAX_ROUNDS {
            let (master, bound) = portfolios.min_of_max(&cuts, unit)?;
            let (master_cut, master_cdar) = drawdowns.cut(&master);
            let (best_weights, best_cdar) = match best.take() {
                Some((weights, cdar)) if cdar <= master_cdar => (weights, cdar),
                _ => (master.clone(), master_cdar),
            };
            // Within rounding of the bound: the cuts and the CDaRs are sums over the same
            // scaled returns, to about 1e-15 of the largest drawdown. The scaled returns are at
            // most 1 in size, so a sum of `t` of them rounds by up to about `tε`: below that, a
            // CDaR of 0 and a bound of −1e-17 agree.
            let tolerance = (1e-12 * best_cdar.abs()).max(t as f64 * f64::EPSILON);
            if best_cdar - bound <= tolerance {
                return Ok(data.weights(best_weights));
            }
            // The cut at a point between the best so far and the minimizer, if it cuts the
            // minimizer off and the bound doesn't have it yet; otherwise the minimizer's own,
            // which always does.
            let query: Vec<f64> = best_weights
                .iter()
                .zip(&master)
                .map(|(b, m)| STABILITY * b + (1.0 - STABILITY) * m)
                .collect();
            let (query_cut, query_cdar) = drawdowns.cut(&query);
            let separates = query_cut.dot(&DVector::from_column_slice(&master)) > bound + tolerance;
            let cut = if separates && !cuts.contains(&query_cut) {
                query_cut
            } else {
                master_cut
            };
            best = Some(if query_cdar < best_cdar {
                (query, query_cdar)
            } else {
                (best_weights, best_cdar)
            });
            if cuts.contains(&cut) {
                // The minimizer's cut is one the bound already has, so the bound is its CDaR,
                // and the best so far is no worse.
                let (weights, _) = best.unwrap();
                return Ok(data.weights(weights));
            }
            cuts.push(cut);
        }
        Err(CutupError::NotConverged)
    }
}

/// The running sums of a portfolio's scaled returns, to find its drawdowns and its cut.
struct Drawdowns<'a> {
    returns: &'a DMatrix<f64>,
    /// Row `k` is the sum of the first `k` periods' returns, each asset's: row 0 is 0.
    sums: DMatrix<f64>,
    cap: f64,
}

impl<'a> Drawdowns<'a> {
    fn new(returns: &'a DMatrix<f64>, cap: f64) -> Self {
        let (t, n) = returns.shape();
        let mut sums = DMatrix::zeros(t + 1, n);
        for k in 0..t {
            let next = sums.row(k) + returns.row(k);
            sums.set_row(k + 1, &next);
        }
        Self { returns, sums, cap }
    }

    /// The cut at `weights`, and their CDaR, in the scaled returns' units. The cut is
    /// `Σₜ qₜ(Sₚ₍ₜ₎ − Sₜ)`: `S` the running sums, `p(t)` the peak before period `t`, and `q` the
    /// tail's weight on each drawdown, the worst first, up to `cap` each and 1 in all. Its
    /// product with any weights is at most their CDaR, and with `weights` it's theirs.
    fn cut(&self, weights: &[f64]) -> (DVector<f64>, f64) {
        let w = DVector::from_column_slice(weights);
        let portfolio = self.returns * &w;
        let (mut sum, mut peak, mut at) = (0.0f64, 0.0f64, 0);
        let mut falls: Vec<(f64, usize, usize)> = Vec::with_capacity(portfolio.len());
        for (k, r) in portfolio.iter().enumerate() {
            sum += r;
            if sum > peak {
                peak = sum;
                at = k + 1;
            }
            falls.push((peak - sum, at, k + 1));
        }
        // The worst first, the earlier of equal ones first. Only the worst ⌈1/cap⌉ are in the
        // tail, one more for rounding, as in `tail_loss`: only those are sorted.
        let worst_first = |a: &(f64, usize, usize), b: &(f64, usize, usize)| {
            b.0.total_cmp(&a.0).then(a.2.cmp(&b.2))
        };
        let tail = ((1.0 / self.cap).ceil() as usize + 1).min(falls.len());
        if tail < falls.len() {
            falls.select_nth_unstable_by(tail, worst_first);
        }
        falls[..tail].sort_by(worst_first);
        let mut cut = DVector::zeros(w.len());
        let (mut left, mut cdar) = (1.0f64, 0.0);
        for &(fall, from, to) in &falls[..tail] {
            let share = self.cap.min(left);
            for (i, c) in cut.iter_mut().enumerate() {
                *c += (self.sums[(from, i)] - self.sums[(to, i)]) * share;
            }
            cdar += share * fall;
            left -= share;
            if left <= 0.0 {
                break;
            }
        }
        (cut, cdar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp;
    use crate::test_support::factor_returns;
    use crate::{EqualWeight, MinCvar};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    /// The least CDaR within `[lower, upper]`, with `target` as the expected return if given,
    /// from the full program (Chekhlov, Uryasev and Zabarankin, 2005), solved directly: a row
    /// for each period's peak and each period's excess over the tail's threshold `α`, with `u`
    /// the peaks less the running sums, minimize `α + cap Σzₜ` subject to `uₜ − uₜ₋₁ + rₜᵀw ≥
    /// 0`, `zₜ − uₜ + α ≥ 0`, `1ᵀw = 1` and `w, u, z ≥ 0`. A formulation of its own, with no
    /// cuts, to check them against.
    fn primal(data: &Returns, beta: f64, (lower, upper): (f64, f64), target: Option<f64>) -> f64 {
        let returns = &data.returns;
        let (t, n) = returns.shape();
        let cap = 1.0 / ((1.0 - beta) * t as f64);
        let rows = 2 * t + 1 + usize::from(target.is_some());
        let cols = n + 2 * t + 1 + 2 * t;
        let (u0, z0, alpha, s0) = (n, n + t, n + 2 * t, n + 2 * t + 1);
        let mut a = DMatrix::zeros(rows, cols);
        for k in 0..t {
            for i in 0..n {
                a[(k, i)] = returns[(k, i)];
            }
            a[(k, u0 + k)] = 1.0;
            if k > 0 {
                a[(k, u0 + k - 1)] = -1.0;
            }
            a[(k, s0 + k)] = -1.0;
            a[(t + k, z0 + k)] = 1.0;
            a[(t + k, u0 + k)] = -1.0;
            a[(t + k, alpha)] = 1.0;
            a[(t + k, s0 + t + k)] = -1.0;
        }
        let mut b = DVector::zeros(rows);
        for i in 0..n {
            a[(2 * t, i)] = 1.0;
        }
        b[2 * t] = 1.0;
        if let Some(target) = target {
            let mu = returns.row_mean();
            for i in 0..n {
                a[(2 * t + 1, i)] = mu[i];
            }
            b[2 * t + 1] = target;
        }
        let c = DVector::from_fn(cols, |j, _| match j {
            j if j == alpha => 1.0,
            j if (z0..z0 + t).contains(&j) => cap,
            _ => 0.0,
        });
        let low = DVector::from_fn(cols, |j, _| match j {
            j if j < n => lower,
            j if j == alpha => f64::NEG_INFINITY,
            _ => 0.0,
        });
        let high = DVector::from_fn(cols, |j, _| if j < n { upper } else { f64::INFINITY });
        let lp::Outcome::Optimal(solution) = lp::solve(&c, &a, &b, &low, &high).unwrap() else {
            panic!("the primal has an optimum");
        };
        solution.objective
    }

    fn data() -> Returns {
        Returns::returns_matrix(factor_returns(5, 60, 11)).unwrap()
    }

    #[test]
    fn the_optimum_is_the_full_programs() {
        let data = data();
        for beta in [0.5, 0.8, 0.95, 0.99] {
            for (lower, upper) in [(0.0, 1.0), (0.0, 0.35), (0.1, 0.5)] {
                let method = MinCdar::default()
                    .confidence(beta)
                    .bounds(WeightBounds::uniform(lower, upper));
                let weights = data.allocate(&method).unwrap();
                assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(
                    weights
                        .iter()
                        .all(|&w| (lower - 1e-15..=upper + 1e-15).contains(&w))
                );
                let cdar = data.cdar(&weights, beta).unwrap();
                let best = primal(&data, beta, (lower, upper), None);
                assert!(
                    (cdar - best).abs() <= 1e-12 * best,
                    "{beta}, {upper}: {cdar} vs {best}"
                );
            }
        }
    }

    #[test]
    fn a_target_holds_at_the_least_cdar_that_reaches_it() {
        let data = data();
        let mu = data.returns.row_mean();
        let target = mu.mean() + 1e-4;
        let weights = data
            .allocate(&MinCdar::default().confidence(0.9).target_return(target))
            .unwrap();
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - target).abs() < 1e-15, "{achieved} vs {target}");
        let cdar = data.cdar(&weights, 0.9).unwrap();
        let best = primal(&data, 0.9, (0.0, 1.0), Some(target));
        assert!((cdar - best).abs() <= 1e-12 * best, "{cdar} vs {best}");
    }

    #[test]
    fn groups_hold() {
        let data = data();
        let method = MinCdar::default().confidence(0.9).bounds(
            WeightBounds::uniform(0.0, 0.6)
                .group([0, 1], 0.3, 0.5)
                .group([3], 0.0, 0.1),
        );
        let weights = data.allocate(&method).unwrap();
        let group = weights[0] + weights[1];
        assert!((0.3 - 1e-12..=0.5 + 1e-12).contains(&group), "{group}");
        assert!(weights[3] <= 0.1 + 1e-12);
        // A portfolio that meets them all does no better.
        let cdar = data.cdar(&weights, 0.9).unwrap();
        let other = Weights::from(vec![0.2, 0.2, 0.3, 0.1, 0.2]);
        assert!(data.cdar(&other, 0.9).unwrap() >= cdar - 1e-15);
    }

    #[test]
    fn the_order_of_the_periods_matters() {
        // The same returns, reversed in time: CVaR's optimum doesn't change, CDaR's does.
        let data = data();
        let (t, n) = data.returns.shape();
        let reversed =
            Returns::returns_matrix(DMatrix::from_fn(t, n, |k, i| data.returns[(t - 1 - k, i)]))
                .unwrap();
        let cvar = |d: &Returns| {
            d.cvar(
                &d.allocate(&MinCvar::default().confidence(0.8)).unwrap(),
                0.8,
            )
            .unwrap()
        };
        assert!((cvar(&data) - cvar(&reversed)).abs() <= 1e-15);
        let forward = data.allocate(&MinCdar::default().confidence(0.8)).unwrap();
        let backward = reversed
            .allocate(&MinCdar::default().confidence(0.8))
            .unwrap();
        assert_ne!(forward, backward);
    }

    #[test]
    fn rising_assets_have_no_drawdown() {
        // Every asset rises every period, so every portfolio's CDaR is 0.
        let data = Returns::new(&[[0.01, 0.02], [0.02, 0.01], [0.015, 0.03]]).unwrap();
        let weights = data.allocate(&MinCdar::default().confidence(0.5)).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert_eq!(data.cdar(&weights, 0.5), Ok(0.0));
    }

    #[test]
    fn the_scale_of_the_returns_changes_nothing() {
        let base = data();
        let method = MinCdar::default().confidence(0.9);
        let expected = base.allocate(&method).unwrap();
        // Powers of two scale every round exactly.
        for scale in [2f64.powi(300), 2f64.powi(-300)] {
            let scaled = Returns::returns_matrix(&base.returns * scale).unwrap();
            assert_eq!(scaled.allocate(&method).unwrap(), expected);
        }
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let data = data();
        for beta in [0.0, 1.0, f64::NAN] {
            assert!(matches!(
                data.allocate(&MinCdar::default().confidence(beta)),
                Err(CutupError::InvalidConfig {
                    field: "confidence",
                    ..
                })
            ));
        }
        assert!(matches!(
            data.allocate(&MinCdar::default().target_return(f64::INFINITY)),
            Err(CutupError::InvalidConfig {
                field: "target_return",
                ..
            })
        ));
        assert!(matches!(
            data.allocate(&MinCdar::default().target_return(1.0)),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        assert!(matches!(
            data.allocate(&MinCdar::default().bounds(WeightBounds::uniform(0.0, 0.1))),
            Err(CutupError::InfeasibleBounds { .. })
        ));
    }

    #[test]
    fn a_cut_measures_cdar_at_its_weights_and_bounds_it_elsewhere() {
        let data = data();
        let cap = tail_weight(0.8, 60).unwrap();
        let drawdowns = Drawdowns::new(&data.returns, cap);
        let weights = [0.4, 0.1, 0.2, 0.0, 0.3];
        let (cut, cdar) = drawdowns.cut(&weights);
        let want = data.cdar(&Weights::from(weights.to_vec()), 0.8).unwrap();
        assert!((cdar - want).abs() <= 1e-15, "{cdar} vs {want}");
        let at = cut.dot(&DVector::from_column_slice(&weights));
        assert!((at - want).abs() <= 1e-15, "{at} vs {want}");
        for other in [
            [0.2; 5],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.5, 0.0],
        ] {
            let below = cut.dot(&DVector::from_column_slice(&other));
            let actual = data.cdar(&Weights::from(other.to_vec()), 0.8).unwrap();
            assert!(below <= actual + 1e-15, "{below} > {actual}");
        }
    }

    /// The cuts' rounds stay few as the assets and periods grow. Unoptimized, this takes about
    /// 6 seconds, and longer on wasm, so `tests.yml` runs it with `--release`, where it takes
    /// under a tenth of one.
    #[test]
    #[ignore = "about 6 s unoptimized; tests.yml runs it with --release --ignored"]
    fn converges_on_a_hundred_assets_and_a_thousand_periods() {
        let data = Returns::returns_matrix(factor_returns(100, 1000, 7)).unwrap();
        let weights = data.allocate(&MinCdar::default()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let equal = data.allocate(&EqualWeight).unwrap();
        assert!(data.cdar(&weights, 0.95).unwrap() < data.cdar(&equal, 0.95).unwrap());
    }
}
