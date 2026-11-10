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
    /// How far from the lowest and the highest a target is within rounding of them.
    tolerances: (f64, f64),
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
        let rounding = |w: &DVector<f64>| 1e-12 * mu.component_mul(w).abs().sum();
        ReturnRange {
            lowest,
            highest,
            tolerances: (rounding(&lowest_weights), rounding(&highest_weights)),
            lowest_weights,
            highest_weights,
            margins: (low_margin, high_margin),
        }
    }

    /// The highest expected return of any portfolio within the bounds.
    pub(crate) fn highest(&self) -> f64 {
        self.highest
    }

    /// Returns `target` if some portfolio within the bounds reaches it: one within rounding
    /// of the lowest or highest is moved onto it.
    ///
    /// Rounding is measured at each end on its own, as `1e-12` of `Σ|μᵢwᵢ|` for that end's
    /// portfolio: the size of the terms its expected return sums. A mean far from the rest
    /// widens only the tolerance of an end that holds its asset.
    pub(crate) fn reachable(&self, target: f64) -> Result<f64, CutupError> {
        let (lowest, highest) = (self.lowest, self.highest);
        let (low, high) = self.tolerances;
        if target < lowest - low || target > highest + high {
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
        Ok(match (below <= low, above <= high) {
            (false, false) => target,
            (true, false) => lowest,
            (false, true) => highest,
            (true, true) if below < above => lowest,
            (true, true) => highest,
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
            start: range.start(mu, &bounds, target),
            bounds,
        })
    }

    /// A portfolio within the bounds reaching `target`. With every asset at its lower bound,
    /// the budget left over is a window laid along the assets sorted by expected return, each
    /// taking up to its room above its lower bound. At one end the window holds the highest
    /// return portfolio and at the other the lowest, and between them its return falls
    /// linearly from one breakpoint, where it meets an asset's edge, to the next. So the target
    /// is reached exactly by holding only assets next to each other in expected return. An
    /// asset whose expected return is far from the rest is held only if the target needs it:
    /// a start holding a little of every asset would leave it free in the solver's first step,
    /// where its expected return swamps the target's constraint.
    fn start(&self, mu: &DVector<f64>, bounds: &qp::Bounds, target: f64) -> DVector<f64> {
        // Divided by the largest in magnitude, so that no sum below can overflow. The target
        // is strictly inside the range, so some expected return isn't 0.
        let scale = mu.amax();
        let (mu, target) = (mu / scale, target / scale);
        let budget = 1.0 - bounds.lower.sum();
        // Highest first, ties in column order, as the highest return portfolio fills them.
        let mut order: Vec<usize> = (0..mu.len()).collect();
        order.sort_by(|&i, &j| mu[j].total_cmp(&mu[i]));
        // Each asset's stretch of the line, no longer than the budget.
        let mut edges = vec![0.0];
        for &i in &order {
            let room = (bounds.upper[i] - bounds.lower[i]).min(budget);
            edges.push(edges[edges.len() - 1] + room);
        }
        // The window starting `s` along the line. l + (u − l) can round past u.
        let at = |s: f64| {
            let mut w = bounds.lower.clone();
            for (k, &i) in order.iter().enumerate() {
                let held = edges[k + 1].min(s + budget) - edges[k].max(s);
                w[i] = (w[i] + held.max(0.0)).min(bounds.upper[i]);
            }
            w
        };
        // Where either end of the window meets an edge, from the start of the line. The window
        // reaches the lowest return portfolio, at or below the target, at the last edge less
        // the budget, so the search stops by then.
        let mut breaks: Vec<f64> = edges
            .iter()
            .flat_map(|&edge| [edge, edge - budget])
            .filter(|&s| s >= 0.0)
            .collect();
        breaks.sort_by(f64::total_cmp);
        for pair in breaks.windows(2) {
            let (high, low) = (mu.dot(&at(pair[0])), mu.dot(&at(pair[1])));
            if low <= target {
                #[allow(
                    clippy::manual_clamp,
                    reason = "f64::max drops the NaN of a segment that doesn't move the return, \
                              which clamp keeps"
                )]
                let t = ((high - target) / (high - low)).max(0.0).min(1.0);
                return at(pair[0] + t * (pair[1] - pair[0]));
            }
        }
        self.lowest_weights.clone()
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
    use crate::{CovarianceConfig, MeanVariance, MinVariance, Returns, WeightBounds, Weights};
    use nalgebra::{DMatrix, dvector};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_a_range_narrower_than_rounding_snaps_to_the_nearer_end() {
        // Returns of 100, 100 + 5e-11 and 100: the whole range is within rounding, and a
        // target at its top is only reached by the second asset.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let mean_variance = MeanVariance::target_return(100.0 + 5e-11)
            .expected_returns(dvector![100.0, 100.0 + 5e-11, 100.0].as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance).unwrap(),
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
                    .bounds(capped),
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
                    .bounds(floored),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.25, 0.25]), 1e-12);

        // Floors and caps of 1/3 leave equal weights as the only portfolio.
        let pinned = WeightBounds::uniform(1.0 / 3.0, 1.0 / 3.0);
        let weights = allocator
            .allocate(&MinVariance::default().covariance(config).bounds(pinned))
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
                .allocate(&MinVariance::default().covariance(config))
                .unwrap()
        );
        for mean_variance in [
            MeanVariance::risk_aversion(5.0),
            MeanVariance::target_return(0.004),
        ] {
            let mean_variance =
                mean_variance.expected_returns(dvector![0.002, 0.006, 0.003].as_slice());
            assert_eq!(
                allocator
                    .allocate(&mean_variance.clone().bounds(loose.clone()))
                    .unwrap(),
                allocator.allocate(&mean_variance).unwrap()
            );
        }
    }

    #[test]
    fn test_bounded_mean_variance_targets_the_reachable_range() {
        let allocator = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let mu = dvector![0.003, 0.003, 0.001];
        let bounds = WeightBounds::uniform(0.0, 0.6);

        // The lowest return fills the last asset to 60% first, then the first to 40%.
        let mean_variance = MeanVariance::target_return(0.004).expected_returns(mu.as_slice());
        match allocator.allocate(&mean_variance.bounds(bounds.clone())) {
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
        let mean_variance = MeanVariance::target_return(0.003).expected_returns(mu.as_slice());
        let weights = allocator
            .allocate(&mean_variance.bounds(bounds.clone()))
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.6, 0.4, 0.0]), 1e-12);

        // In between, the target holds and the weights stay within the bounds.
        let mean_variance = MeanVariance::target_return(0.0025).expected_returns(mu.as_slice());
        let weights = allocator.allocate(&mean_variance.bounds(bounds)).unwrap();
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - 0.0025).abs() < 1e-15, "{achieved}");
        assert!(
            weights.iter().all(|&w| (0.0..=0.6 + 1e-15).contains(&w)),
            "{weights:?}"
        );
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-15);
    }

    fn three() -> Vec<String> {
        ["A", "B", "C"].map(String::from).to_vec()
    }

    #[test]
    fn test_resolved_caps_leave_room_for_the_other_floors() {
        let resolved = WeightBounds::default().resolve(&three()).unwrap();
        assert_eq!(resolved.upper, dvector![1.0, 1.0, 1.0]);
        let resolved = WeightBounds::per_asset(vec![0.2, 0.1, 0.0], vec![f64::INFINITY; 3])
            .resolve(&three())
            .unwrap();
        assert_weights_close(
            &Weights::from(resolved.upper.as_slice().to_vec()),
            &Weights::from(vec![0.9, 0.8, 0.7]),
            1e-15,
        );
    }

    #[test]
    fn test_bounds_exactly_at_the_rounding_allowance_are_feasible() {
        // Floors summing to exactly 1 + 1e-12 are rounding, and are rescaled to sum to 1.
        let resolved = WeightBounds::per_asset(vec![1.0 + 1e-12, 0.0, 0.0], vec![2.0; 3])
            .resolve(&three())
            .unwrap();
        assert!((resolved.lower.sum() - 1.0).abs() <= f64::EPSILON);
        // Caps summing to exactly 1 − 1e-12 are rounding too.
        assert!(
            WeightBounds::per_asset(vec![0.0; 3], vec![1.0 - 1e-12, 0.0, 0.0])
                .resolve(&three())
                .is_ok()
        );
    }

    #[test]
    fn test_invalid_per_asset_bounds_are_rejected() {
        let invalid = |field, column, value| CutupError::InvalidAssetValue {
            field,
            column,
            asset: three()[column].clone(),
            value,
        };
        let resolve = |lower: Vec<f64>, upper: Vec<f64>| {
            WeightBounds::per_asset(lower, upper).resolve(&three())
        };
        assert_eq!(
            resolve(vec![0.0, -0.1, 0.0], vec![1.0; 3]).unwrap_err(),
            invalid("lower_bound", 1, -0.1)
        );
        assert_eq!(
            resolve(vec![0.0, 0.0, f64::INFINITY], vec![f64::INFINITY; 3]).unwrap_err(),
            invalid("lower_bound", 2, f64::INFINITY)
        );
        assert_eq!(
            resolve(vec![0.0, 0.3, 0.0], vec![1.0, 0.2, 1.0]).unwrap_err(),
            invalid("upper_bound", 1, 0.2)
        );
    }

    #[test]
    fn test_interior_start_spreads_the_room_above_the_floors() {
        let bounds = qp::Bounds {
            lower: dvector![0.1, 0.0, 0.0],
            upper: dvector![0.5, 1.0, 1.0],
        };
        // Room 2.4 for the 0.9 left above the floors: 0.375 of each asset's room.
        assert_weights_close(
            &Weights::from(interior_start(&bounds).as_slice().to_vec()),
            &Weights::from(vec![0.25, 0.375, 0.375]),
            1e-15,
        );
        // With no room, the floors.
        let fixed = qp::Bounds {
            lower: dvector![0.5, 0.5],
            upper: dvector![0.5, 0.5],
        };
        assert_eq!(interior_start(&fixed), dvector![0.5, 0.5]);
    }

    #[test]
    fn test_reachable_snaps_targets_within_rounding_of_an_end() {
        let unit = qp::Bounds {
            lower: dvector![0.0, 0.0],
            upper: dvector![1.0, 1.0],
        };
        // From 0 to 1. The top's portfolio earns 1, so its rounding is 1e-12: a target exactly
        // that far inside snaps to it. The bottom's earns exactly 0, with no rounding at all.
        let range = ReturnRange::new(&dvector![0.0, 1.0], &unit);
        assert_eq!(range.reachable(1.0 - 1e-12), Ok(1.0));
        assert_eq!(range.reachable(1.0 - 2e-12), Ok(1.0 - 2e-12));
        assert_eq!(range.reachable(1.0 + 1e-12), Ok(1.0));
        assert!(range.reachable(1.0 + 2e-12).is_err());
        assert_eq!(range.reachable(1e-300), Ok(1e-300));
        assert_eq!(range.reachable(0.0), Ok(0.0));
        assert!(range.reachable(-1e-300).is_err());
        // In a range within rounding, a target goes to the nearer end, and halfway to the top.
        let narrow = ReturnRange::new(&dvector![1.0, 1.0 + 4.0 * f64::EPSILON], &unit);
        assert_eq!(narrow.reachable(1.0 + f64::EPSILON), Ok(1.0));
        assert_eq!(
            narrow.reachable(1.0 + 3.0 * f64::EPSILON),
            Ok(1.0 + 4.0 * f64::EPSILON)
        );
        assert_eq!(
            narrow.reachable(1.0 + 2.0 * f64::EPSILON),
            Ok(1.0 + 4.0 * f64::EPSILON)
        );
    }

    #[test]
    fn test_each_end_rounds_by_its_own_portfolio() {
        let unit = qp::Bounds {
            lower: dvector![0.0, 0.0, 0.0],
            upper: dvector![1.0, 1.0, 1.0],
        };
        // The top holds only the second asset, so the first's −1e6 doesn't widen its rounding,
        // 2e-14. The bottom holds the first, so its rounding is 1e-6.
        let mu = dvector![-1e6, 0.02, 0.02 - 5e-7];
        let range = ReturnRange::new(&mu, &unit);
        assert_eq!(range.reachable(0.02 - 1e-7), Ok(0.02 - 1e-7));
        assert_eq!(range.reachable(0.02 - 1e-14), Ok(0.02));
        assert_eq!(range.reachable(-1e6 + 5e-7), Ok(-1e6));
        assert_eq!(range.reachable(-1e6 + 2e-6), Ok(-1e6 + 2e-6));
        assert!(range.reachable(0.02 + 1e-13).is_err());
        assert!(range.reachable(-1e6 - 2e-6).is_err());
    }

    /// The start [`ReturnRange::target_problem`] gives for `target`, checked to be within the
    /// bounds, sum to 1 and reach the target.
    fn start(mu: &DVector<f64>, bounds: &qp::Bounds, target: f64) -> DVector<f64> {
        let problem = ReturnRange::target_problem(mu, bounds.clone(), target).unwrap();
        let w = problem.start;
        assert!((w.sum() - 1.0).abs() <= 1e-15, "{w:?}");
        assert!((0..w.len()).all(|i| bounds.lower[i] <= w[i] && w[i] <= bounds.upper[i]));
        let scale = mu.amax();
        assert!(
            (mu.dot(&w) - target).abs() <= 1e-15 * scale,
            "{} vs {target}",
            mu.dot(&w)
        );
        w
    }

    #[test]
    fn test_the_start_holds_only_assets_near_the_target() {
        let unit = qp::Bounds {
            lower: dvector![0.0, 0.0, 0.0, 0.0],
            upper: dvector![1.0, 1.0, 1.0, 1.0],
        };
        let mu = dvector![-1e6, 0.02, 0.0199995, 0.01];
        // Near the top, only the second and third assets; lower, the third and fourth; far
        // lower, the fourth and first. The first is held only where the target needs it.
        let w = start(&mu, &unit, 0.02 - 1e-7);
        assert_eq!((w[0], w[3]), (0.0, 0.0));
        let w = start(&mu, &unit, 0.0199995 - 1e-4);
        assert_eq!((w[0], w[1]), (0.0, 0.0));
        let w = start(&mu, &unit, -5e5);
        assert_eq!((w[1], w[2]), (0.0, 0.0));
        assert!((w[0] - 0.5).abs() < 1e-6);
        // With floors and caps, and still reaching the target.
        let bounds = qp::Bounds {
            lower: dvector![0.0, 0.1, 0.0, 0.2],
            upper: dvector![0.5, 0.6, 0.15, 1.0],
        };
        assert_eq!(start(&mu, &bounds, 0.016)[0], 0.0);
        start(&mu, &bounds, -1e5);
        // Means near f64::MAX don't overflow it.
        let huge = dvector![-f64::MAX, f64::MAX / 2.0, f64::MAX];
        let w = start(
            &huge,
            &qp::Bounds {
                lower: dvector![0.0, 0.0, 0.0],
                upper: dvector![1.0, 1.0, 1.0],
            },
            f64::MAX * 0.75,
        );
        assert_eq!(w[0], 0.0);
    }

    #[test]
    fn test_the_start_slides_the_budget_along_the_returns() {
        let cap = qp::Bounds {
            lower: DVector::zeros(4),
            upper: DVector::from_element(4, 0.5),
        };
        let start = |mu: &DVector<f64>, bounds: &qp::Bounds, target: f64| {
            start(mu, bounds, target)
                .iter()
                .map(|w| (w * 1e12).round() / 1e12)
                .collect::<Vec<_>>()
        };
        // From 1500 to 3500: the window of 1 slides from [0, 0, 0.5, 0.5] to [0.5, 0.5, 0, 0],
        // through [0, 0.5, 0.5, 0] at 2500.
        let mu = dvector![1000.0, 2000.0, 3000.0, 4000.0];
        assert_eq!(start(&mu, &cap, 3400.0), [0.0, 0.05, 0.5, 0.45]);
        assert_eq!(start(&mu, &cap, 2500.0), [0.0, 0.5, 0.5, 0.0]);
        assert_eq!(start(&mu, &cap, 1600.0), [0.45, 0.5, 0.05, 0.0]);
        // Tied expected returns are laid in column order.
        let tied = dvector![0.0, 1.0, 1.0, 2.0];
        let unit = qp::Bounds {
            lower: DVector::zeros(4),
            upper: DVector::from_element(4, 1.0),
        };
        assert_eq!(start(&tied, &unit, 1.2), [0.0, 0.8, 0.0, 0.2]);
        assert_eq!(start(&tied, &unit, 0.5), [0.5, 0.0, 0.5, 0.0]);
        assert_eq!(start(&tied, &cap, 1.2), [0.0, 0.5, 0.3, 0.2]);
        // Uncapped, each asset's stretch is the budget left after the floors, here 0.6.
        let floors = qp::Bounds {
            lower: dvector![0.1, 0.1, 0.1, 0.1],
            upper: DVector::from_element(4, f64::INFINITY),
        };
        assert_eq!(start(&mu, &floors, 2500.0), [0.1, 0.4, 0.4, 0.1]);
    }

    #[test]
    fn test_a_target_across_several_bounded_assets_is_reached() {
        // Close to a case from the property tests, where a start that moved weight between the
        // closest pair one bound at a time fell short of the target, and the solve failed.
        let returns = DMatrix::from_vec(
            14,
            4,
            vec![
                0.0,
                0.043021496,
                0.0,
                0.039817187,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.031959524,
                0.0,
                -0.003818411,
                -0.020443345,
                -0.017913865,
                -0.034875112,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.037736578,
                0.043941819,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.021989019,
                -0.028166691,
                0.0,
                -0.030074956,
                -0.005915370,
                -0.008912545,
                -0.014654237,
                0.0,
                0.0,
                0.041957695,
                0.047556898,
                -0.018680320,
                0.0,
                0.025386184,
                0.020801107,
                -0.038463244,
                -0.049684629,
                0.006717366,
                0.036041461,
                -0.033560917,
                -0.017856673,
                -0.048719500,
                0.043808973,
                0.019405156,
                0.045918587,
                -0.045944856,
                0.027652960,
                0.009290532,
            ],
        );
        let mu = returns.row_mean().transpose();
        let data = Returns::returns_matrix(returns).unwrap();
        let bounds = WeightBounds::per_asset(
            vec![0.0, 0.068065541, 0.0, 0.185450640],
            vec![0.688408852, f64::INFINITY, 0.730141329, 0.276657586],
        );
        let target = 0.0011;
        let weights = data
            .allocate(&MeanVariance::target_return(target).bounds(bounds))
            .unwrap();
        let earned = mu.dot(&DVector::from_column_slice(weights.as_slice()));
        assert!((earned - target).abs() <= 1e-12 * mu.amax(), "{earned}");
    }

    #[test]
    fn test_targets_near_the_top_beside_a_far_mean_are_reached() {
        // The first asset is a hundred times as volatile, so the minimum variance portfolio
        // holds none of it either.
        let mut returns = crate::test_support::factor_returns(3, 40, 7);
        returns.column_mut(0).scale_mut(100.0);
        let data = Returns::returns_matrix(returns).unwrap();
        let mu = [-1e6, 0.02, 0.02 - 5e-7];
        // Between the two highest means, which the first asset isn't needed to reach.
        for target in [0.02 - 4e-7, 0.02 - 1e-7, 0.02 - 1e-9, 0.02 - 1e-11] {
            let method = MeanVariance::target_return(target).expected_returns(mu);
            let weights = data.allocate(&method).unwrap();
            let earned: f64 = weights.iter().zip(mu).map(|(w, m)| w * m).sum();
            assert!(
                (earned - target).abs() <= 1e-12 * target,
                "{earned} vs {target}"
            );
        }
        // So the frontier stays evenly spaced on them.
        let frontier = data
            .frontier(&MeanVariance::risk_aversion(1.0).expected_returns(mu), 5)
            .unwrap();
        assert_eq!(frontier.len(), 5);
        let (lowest, highest) = (frontier[0].expected_return(), frontier[4].expected_return());
        for (k, point) in frontier.iter().enumerate() {
            let want = lowest + (highest - lowest) * k as f64 / 4.0;
            assert!(
                (point.expected_return() - want).abs() <= 1e-12 * want.abs(),
                "{k}: {}",
                point.expected_return()
            );
        }
    }

    #[test]
    fn test_a_target_at_an_end_needs_only_the_budget() {
        let unit = qp::Bounds {
            lower: dvector![0.0, 0.0, 0.0],
            upper: dvector![1.0, 1.0, 1.0],
        };
        let mu = dvector![0.1, 0.2, 0.3];
        for (target, columns) in [(0.1, 1), (0.3, 1), (0.2, 2)] {
            let problem = ReturnRange::target_problem(&mu, unit.clone(), target).unwrap();
            assert_eq!(problem.e.ncols(), columns, "{target}");
        }
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
                    .bounds(bounds),
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
