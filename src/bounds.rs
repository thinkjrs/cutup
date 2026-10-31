//! Weight bounds, and the expected returns a portfolio within them can reach.

use crate::{Asset, CutupError, PerAsset, asset, qp};
use nalgebra::{DMatrix, DVector};

/// Lower and upper limits on each asset's weight, for
/// [`MinVariance::bounds`](crate::MinVariance::bounds),
/// [`MeanVariance::bounds`](crate::MeanVariance::bounds) and
/// [`MinCvar::bounds`](crate::MinCvar::bounds).
///
/// Lower bounds must be finite and nonnegative, and each upper bound at least its lower bound.
/// An upper bound of `f64::INFINITY` means no cap. The default, `0 ≤ w` with no cap, is the
/// plain long-only constraint.
///
/// Caps that sum to just under 1, or floors to just over, by at most `1e-12`, are taken as
/// rounding: ten caps of 0.1 sum to 0.9999999999999999. No portfolio meets such bounds
/// exactly, so they're rescaled to sum to 1, which moves each by at most `1e-12` of itself.
///
/// ```
/// use cutup::WeightBounds;
///
/// // No asset above 30%.
/// let capped = WeightBounds::uniform(0.0, 0.3);
///
/// // At least 5% in each of three assets, and at most 50% in the first.
/// let mixed = WeightBounds::per_asset(vec![0.05; 3], vec![0.5, 1.0, 1.0]);
///
/// // No asset above 40%, except between 10% and 60% in TLT.
/// let labeled = WeightBounds::uniform(0.0, 0.4).asset("TLT", 0.1, 0.6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WeightBounds {
    lower: Limit,
    upper: Limit,
    assets: Vec<(Asset, f64, f64)>,
}

/// One side of [`WeightBounds`].
#[derive(Debug, Clone, PartialEq)]
enum Limit {
    Uniform(f64),
    PerAsset(PerAsset),
}

impl Limit {
    fn resolve(&self, field: &'static str, labels: &[String]) -> Result<DVector<f64>, CutupError> {
        let n = labels.len();
        match self {
            Limit::Uniform(value) => Ok(DVector::from_element(n, *value)),
            Limit::PerAsset(values) => {
                let values = values.resolve(field, labels, None)?;
                Ok(DVector::from_vec(values))
            }
        }
    }
}

impl Default for WeightBounds {
    fn default() -> Self {
        Self::uniform(0.0, f64::INFINITY)
    }
}

impl WeightBounds {
    /// The same bounds `lower ≤ wᵢ ≤ upper` for every asset.
    pub fn uniform(lower: f64, upper: f64) -> Self {
        Self {
            lower: Limit::Uniform(lower),
            upper: Limit::Uniform(upper),
            assets: Vec::new(),
        }
    }

    /// Bounds `lower[i] ≤ wᵢ ≤ upper[i]` for each asset `i`: [`PerAsset`] values, in column
    /// order or by asset, with a value for every asset.
    pub fn per_asset(lower: impl Into<PerAsset>, upper: impl Into<PerAsset>) -> Self {
        Self {
            lower: Limit::PerAsset(lower.into()),
            upper: Limit::PerAsset(upper.into()),
            assets: Vec::new(),
        }
    }

    /// Bounds `lower ≤ w ≤ upper` for `asset`, by label or by column, in place of its other
    /// bounds. Setting the same asset again replaces them.
    pub fn asset(mut self, asset: impl Into<Asset>, lower: f64, upper: f64) -> Self {
        self.assets.push((asset.into(), lower, upper));
        self
    }

    /// Checks the bounds for the assets `labels`, and returns them with each upper bound
    /// lowered to what the other assets' lower bounds leave room for, `1 − Σⱼ≠ᵢ lⱼ`. That
    /// doesn't change which portfolios are feasible, and makes every upper bound finite.
    pub(crate) fn resolve(&self, labels: &[String]) -> Result<qp::Bounds, CutupError> {
        let n = labels.len();
        // Uniform bounds aren't any one asset's, so they're checked as settings.
        if let Limit::Uniform(l) = self.lower {
            if !(l.is_finite() && l >= 0.0) {
                return Err(CutupError::InvalidConfig {
                    field: "lower_bound",
                    value: l,
                });
            }
        }
        if let Limit::Uniform(u) = self.upper {
            let below = matches!(self.lower, Limit::Uniform(l) if u < l);
            if u.is_nan() || below {
                return Err(CutupError::InvalidConfig {
                    field: "upper_bound",
                    value: u,
                });
            }
        }
        let mut lower = self.lower.resolve("lower_bound", labels)?;
        let mut upper = self.upper.resolve("upper_bound", labels)?;
        for (asset, l, u) in &self.assets {
            let i = asset.resolve(labels)?;
            lower[i] = *l;
            upper[i] = *u;
        }
        asset::check_values("lower_bound", labels, lower.as_slice(), |l| {
            l.is_finite() && l >= 0.0
        })?;
        if let Some(i) = (0..n).find(|&i| upper[i].is_nan() || upper[i] < lower[i]) {
            return Err(asset::invalid_value("upper_bound", labels, i, upper[i]));
        }
        let (lower_sum, upper_sum) = (lower.sum(), upper.sum());
        // Within rounding of a budget of 1, as when ten caps of 0.1 sum to 0.9999999999999999.
        if lower_sum > 1.0 + 1e-12 || upper_sum < 1.0 - 1e-12 {
            return Err(CutupError::InfeasibleBounds {
                lower_sum,
                upper_sum,
            });
        }
        // Within rounding but short of the budget, no portfolio meets the bounds exactly, so
        // they're rescaled onto it, which moves each by at most 1e-12 of itself.
        let (lower, lower_sum) = if lower_sum > 1.0 {
            (lower / lower_sum, 1.0)
        } else {
            (lower, lower_sum)
        };
        let upper = if upper_sum < 1.0 {
            upper / upper_sum
        } else {
            upper
        };
        // Rounding can put that room a hair below lᵢ when the lower bounds sum to 1.
        let upper = DVector::from_fn(n, |i, _| {
            upper[i].min(1.0 - (lower_sum - lower[i])).max(lower[i])
        });
        Ok(qp::Bounds { lower, upper })
    }
}

/// A point strictly inside resolved `bounds` with weights summing to 1, where they allow one:
/// `l + t(u − l)` for the `t` that sums to 1. With the default bounds that's equal weights.
pub(crate) fn interior_start(bounds: &qp::Bounds) -> DVector<f64> {
    let room = &bounds.upper - &bounds.lower;
    let t = match room.sum() {
        total if total > 0.0 => ((1.0 - bounds.lower.sum()) / total).clamp(0.0, 1.0),
        _ => 0.0,
    };
    &bounds.lower + room * t
}

/// A target-return problem from [`ReturnRange::target_problem`].
pub(crate) struct TargetProblem {
    pub(crate) e: DMatrix<f64>,
    pub(crate) b: DVector<f64>,
    pub(crate) bounds: qp::Bounds,
    pub(crate) start: DVector<f64>,
}

/// The lowest and highest expected return of any portfolio within resolved bounds, and the
/// portfolios that reach them: start every asset at its lower bound, then raise assets to
/// their upper bounds in order of expected return until the weights sum to 1.
pub(crate) struct ReturnRange {
    lowest: f64,
    highest: f64,
    lowest_weights: DVector<f64>,
    highest_weights: DVector<f64>,
    /// The expected return of the last asset raised above its lower bound, for the lowest
    /// and the highest, or `None` if the lower bounds already sum to 1.
    margins: (Option<f64>, Option<f64>),
}

impl ReturnRange {
    pub(crate) fn new(mu: &DVector<f64>, bounds: &qp::Bounds) -> Self {
        // Stable sorts, so that ties are filled in column order, as argmin and argmax pick.
        let mut ascending: Vec<usize> = (0..mu.len()).collect();
        ascending.sort_by(|&i, &j| mu[i].total_cmp(&mu[j]));
        let mut descending: Vec<usize> = (0..mu.len()).collect();
        descending.sort_by(|&i, &j| mu[j].total_cmp(&mu[i]));
        let fill = |order: &[usize]| {
            let mut weights = bounds.lower.clone();
            let mut left = 1.0 - bounds.lower.sum();
            let mut margin = None;
            for &i in order {
                if left <= 0.0 {
                    break;
                }
                let raise = (bounds.upper[i] - bounds.lower[i]).min(left);
                if raise > 0.0 {
                    weights[i] += raise;
                    left -= raise;
                    margin = Some(mu[i]);
                }
            }
            (mu.dot(&weights), weights, margin)
        };
        let (lowest, lowest_weights, low_margin) = fill(&ascending);
        let (highest, highest_weights, high_margin) = fill(&descending);
        ReturnRange {
            lowest,
            highest,
            lowest_weights,
            highest_weights,
            margins: (low_margin, high_margin),
        }
    }

    /// Returns `target` if some portfolio within the bounds reaches it: one within rounding
    /// of the lowest or highest is moved onto it.
    pub(crate) fn reachable(&self, target: f64) -> Result<f64, CutupError> {
        let (lowest, highest) = (self.lowest, self.highest);
        let tolerance = 1e-12 * lowest.abs().max(highest.abs());
        if target < lowest - tolerance || target > highest + tolerance {
            return Err(CutupError::TargetReturnOutOfRange {
                target_return: target,
                lowest,
                highest,
            });
        }
        // From either side: just inside an end, only a sliver of portfolios next to its face
        // reach the target, and the solvers resolve that to rounding no better than the face.
        // In a range narrower than that, to the nearer end.
        let (below, above) = (target - lowest, highest - target);
        Ok(if below.min(above) > tolerance {
            target
        } else if below < above {
            lowest
        } else {
            highest
        })
    }

    /// The constraints `eᵀw = b` and bounds of a long-only problem at expected return
    /// `target`, and a start that meets them. Only the portfolios that fill the bounds in
    /// order of expected return reach an end of the range, so there every asset but those
    /// tied at the margin is fixed, and the target needs no constraint of its own. Inside it,
    /// mixing the lowest and highest return portfolios reaches the target.
    pub(crate) fn target_problem(
        mu: &DVector<f64>,
        bounds: qp::Bounds,
        target: f64,
    ) -> Result<TargetProblem, CutupError> {
        let n = mu.len();
        let range = ReturnRange::new(mu, &bounds);
        let target = range.reachable(target)?;
        if target == range.lowest || target == range.highest {
            let bounds = range.face(mu, &bounds, target == range.highest);
            return Ok(TargetProblem {
                e: DMatrix::from_element(n, 1, 1.0),
                b: DVector::from_element(1, 1.0),
                start: interior_start(&bounds),
                bounds,
            });
        }
        Ok(TargetProblem {
            e: DMatrix::from_columns(&[DVector::from_element(n, 1.0), mu.clone()]),
            b: DVector::from_vec(vec![1.0, target]),
            start: range.mix(target),
            bounds,
        })
    }

    /// The mix of the lowest and highest return portfolios that reaches `target`, strictly
    /// between them.
    fn mix(&self, target: f64) -> DVector<f64> {
        // Dividing by the larger in magnitude first keeps the differences from overflowing.
        let scale = self.lowest.abs().max(self.highest.abs());
        let t = ((target / scale - self.lowest / scale)
            / (self.highest / scale - self.lowest / scale))
            .clamp(0.0, 1.0);
        &self.lowest_weights * (1.0 - t) + &self.highest_weights * t
    }

    /// The bounds of the portfolios with the highest expected return, or the lowest. Assets
    /// whose expected return is above the margin's (below it, for the lowest) are held at
    /// their upper bound, those on the other side at their lower bound, and only assets tied
    /// with the margin keep their bounds.
    fn face(&self, mu: &DVector<f64>, bounds: &qp::Bounds, highest: bool) -> qp::Bounds {
        let (weights, margin) = if highest {
            (&self.highest_weights, self.margins.1)
        } else {
            (&self.lowest_weights, self.margins.0)
        };
        let mut face = bounds.clone();
        for i in (0..mu.len()).filter(|&i| margin != Some(mu[i])) {
            face.lower[i] = weights[i];
            face.upper[i] = weights[i];
        }
        face
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_weights_close, full_rank_prices, uncorrelated_returns};
    use crate::{CovarianceConfig, MeanVariance, MinVariance, Returns, Weights};
    use nalgebra::dvector;

    #[test]
    fn test_a_range_narrower_than_rounding_snaps_to_the_nearer_end() {
        // Returns of 100, 100 + 5e-11 and 100: the whole range is within rounding, and a
        // target at its top is only reached by the second asset.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = MeanVariance::target_return(100.0 + 5e-11)
            .expected_returns(dvector![100.0, 100.0 + 5e-11, 100.0].as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-15,
        );
    }

    #[test]
    fn test_bounded_min_variance_water_fills() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let config = CovarianceConfig::default();
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().covariance(config.clone()))
                .unwrap(),
            &Weights::from(vec![4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]),
            1e-12,
        );

        // A cap of 50% on the first asset passes the rest to the others in proportion to
        // inverse variance, 1/3 and 1/6.
        let capped = WeightBounds::per_asset(vec![0.0; 3], vec![0.5, 1.0, 1.0]);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(capped.clone()),
            )
            .unwrap();
        assert_weights_close(
            &weights,
            &Weights::from(vec![0.5, 1.0 / 3.0, 1.0 / 6.0]),
            1e-12,
        );

        // A floor of 25% on the last asset leaves 75% to split 2 : 1.
        let floored = WeightBounds::per_asset(vec![0.0, 0.0, 0.25], vec![0.5, 1.0, 1.0]);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(floored.clone()),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.25, 0.25]), 1e-12);

        // Floors and caps of 1/3 leave equal weights as the only portfolio.
        let pinned = WeightBounds::uniform(1.0 / 3.0, 1.0 / 3.0);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(pinned.clone()),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![1.0 / 3.0; 3]), 1e-15);
    }

    #[test]
    fn test_bounds_that_dont_bind_change_nothing() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().shrinkage(0.2);
        let loose = WeightBounds::uniform(0.0, 1.0);
        assert_eq!(
            allocator
                .allocate(
                    &MinVariance::default()
                        .covariance(config.clone())
                        .bounds(loose.clone())
                )
                .unwrap(),
            allocator
                .allocate(&MinVariance::default().covariance(config.clone()))
                .unwrap()
        );
        for config in [
            MeanVariance::risk_aversion(5.0),
            MeanVariance::target_return(0.004),
        ] {
            let config = config.expected_returns(dvector![0.002, 0.006, 0.003].as_slice());
            assert_eq!(
                allocator
                    .allocate(&config.clone().bounds(loose.clone()))
                    .unwrap(),
                allocator.allocate(&config).unwrap()
            );
        }
    }

    #[test]
    fn test_bounded_mean_variance_targets_the_reachable_range() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let mu = dvector![0.003, 0.003, 0.001];
        let bounds = WeightBounds::uniform(0.0, 0.6);

        // The lowest return fills the last asset to 60% first, then the first to 40%.
        let config = MeanVariance::target_return(0.004).expected_returns(mu.as_slice());
        match allocator.allocate(&config.clone().bounds(bounds.clone())) {
            Err(CutupError::TargetReturnOutOfRange {
                target_return,
                lowest,
                highest,
            }) => {
                assert_eq!(target_return, 0.004);
                assert!((lowest - 0.0018).abs() < 1e-15, "{lowest}");
                assert_eq!(highest, 0.003);
            }
            other => panic!("{other:?}"),
        }

        // At the highest return the last asset holds nothing, and the two tied assets split
        // the weight by minimum variance, 2/3 and 1/3, until the 60% cap binds.
        let config = MeanVariance::target_return(0.003).expected_returns(mu.as_slice());
        let weights = allocator
            .allocate(&config.clone().bounds(bounds.clone()))
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.6, 0.4, 0.0]), 1e-12);

        // In between, the target holds and the weights stay within the bounds.
        let config = MeanVariance::target_return(0.0025).expected_returns(mu.as_slice());
        let weights = allocator
            .allocate(&config.clone().bounds(bounds.clone()))
            .unwrap();
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - 0.0025).abs() < 1e-15, "{achieved}");
        assert!(
            weights.iter().all(|&w| (0.0..=0.6 + 1e-15).contains(&w)),
            "{weights:?}"
        );
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_invalid_bounds_are_rejected() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let n = full_rank_prices().ncols();
        let config = CovarianceConfig::default();
        let solve = |bounds: WeightBounds| {
            allocator.allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(bounds.clone()),
            )
        };
        let infeasible = |lower_sum: f64, upper_sum: f64| {
            Err(CutupError::InfeasibleBounds {
                lower_sum,
                upper_sum,
            })
        };

        let caps = 0.9 / n as f64;
        assert_eq!(
            solve(WeightBounds::uniform(0.0, caps)),
            infeasible(0.0, caps * n as f64)
        );
        let floors = 1.1 / n as f64;
        assert_eq!(
            solve(WeightBounds::uniform(floors, 1.0)),
            infeasible(floors * n as f64, n as f64)
        );
        assert_eq!(
            solve(WeightBounds::uniform(-0.1, 1.0)),
            Err(CutupError::InvalidConfig {
                field: "lower_bound",
                value: -0.1
            })
        );
        assert_eq!(
            solve(WeightBounds::uniform(0.3, 0.2)),
            Err(CutupError::InvalidConfig {
                field: "upper_bound",
                value: 0.2
            })
        );
        assert!(matches!(
            solve(WeightBounds::uniform(0.0, f64::NAN)),
            Err(CutupError::InvalidConfig {
                field: "upper_bound",
                ..
            })
        ));
        assert!(matches!(
            solve(WeightBounds::uniform(f64::INFINITY, f64::INFINITY)),
            Err(CutupError::InvalidConfig {
                field: "lower_bound",
                ..
            })
        ));
        assert_eq!(
            solve(WeightBounds::per_asset(vec![0.0; n + 1], vec![1.0; n + 1])),
            Err(CutupError::DimensionMismatch {
                field: "lower_bound",
                expected: n,
                actual: n + 1
            })
        );

        // Caps or floors short of the budget by less than 1e-12 are rescaled onto it, giving
        // equal weights within 1e-12 of each bound.
        for (lower, upper) in [
            (0.0, (1.0 - 5e-13) / n as f64),
            (0.0, (1.0 - 9e-13) / n as f64),
            ((1.0 + 9e-13) / n as f64, 1.0),
        ] {
            let weights = solve(WeightBounds::uniform(lower, upper)).unwrap();
            assert_weights_close(&weights, &Weights::from(vec![1.0 / n as f64; n]), 1e-15);
            for &w in &weights {
                assert!(
                    w >= lower * (1.0 - 1e-12) && w <= upper * (1.0 + 1e-12),
                    "{w}"
                );
            }
        }

        // Within rounding of a budget of 1 is feasible: ten caps of 0.1 sum to just under it.
        let ten = Returns::returns_matrix(DMatrix::from_fn(12, 10, |i, j| {
            ((i * 7 + j * 3) % 11) as f64 * 0.001 - 0.005 + (i == j) as u8 as f64 * 0.01
        }))
        .unwrap();
        let weights = ten
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(WeightBounds::uniform(0.0, 0.1)),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.1; 10]), 1e-15);
    }
}
