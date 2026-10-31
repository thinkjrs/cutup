//! [`Hrp`]: Hierarchical Risk Parity.

use super::cluster_variance;
use crate::{Allocation, CovarianceConfig, CutupError, Linkage, Returns, Weights, cluster};

/// Hierarchical Risk Parity (López de Prado, 2016).
///
/// 1. **Cluster.** Assets are clustered on the correlation distance `sqrt((1 − ρ) / 2)` with
///    the [linkage](Hrp::linkage), and ordered so that correlated assets sit next to each
///    other (see [`Hrp::cluster_order`]).
/// 2. **Recursive bisection.** The ordered list is split in half, repeatedly. At each split,
///    the two halves share their parent's weight in inverse proportion to their risk: each
///    half's variance is that of its own inverse-variance portfolio, and the first half gets
///    `α = 1 − V₁ / (V₁ + V₂)`.
///
/// The clustering follows SciPy and PyPortfolioOpt's `HRPOpt` step for step, including the
/// tie-breaking rule, so results match `HRPOpt(returns).optimize(linkage_method)` up to
/// floating-point rounding. (The correlations are computed here rather than by pandas, so two
/// distances that tie in one may differ by a rounding error in the other and break the tie the
/// other way.)
///
/// HRP never inverts the covariance matrix, so it works when Σ is singular. Weights are never
/// negative, and sum to 1. A half with zero variance, such as a perfectly hedged pair, takes
/// its parent's whole weight and leaves the other half at zero, as in `HRPOpt`.
///
/// [Shrinkage and regularization](Hrp::covariance) adjust the variances that split the weight, but
/// not the correlations the hierarchy is built on, so [`Hrp::cluster_order`] doesn't change. That's
/// what riskfolio-lib's
/// `HCPortfolio.optimization(model="HRP", rm="MV", method_cov="shrunk", leaf_order=False)` does,
/// and results match it up to floating-point rounding. Full shrinkage leaves every variance equal
/// and every covariance zero, which gives equal weights.
///
/// Like the original algorithm and `HRPOpt`, the result can depend on the order of the input
/// columns. The hierarchy puts the lower-indexed subtree on the left, and the bisection splits
/// the ordered list at its midpoint rather than at the tree's branches, so reordering the
/// columns can move a split. Correlated assets still end up next to each other; only where the
/// halves are cut changes. [`Herc`](crate::Herc) splits at the branches instead.
///
/// # Errors
///
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance, even with
///   shrinkage or regularization: a constant asset has no correlation to cluster on.
/// * [`CutupError::InvalidConfig`] if the shrinkage is outside `[0, 1]` or the regularization
///   is negative or not finite.
/// * [`CutupError::NonFiniteCovariance`] if a variance is below about 1e-292,
///   `f64::MIN_POSITIVE / f64::EPSILON`, too small to compute correlations from accurately.
///
/// # Example
///
/// ```
/// use cutup::{Hrp, Linkage, Returns};
///
/// let returns = [
///      [ 0.010,  0.020,  0.011,  0.019],
///      [-0.010,  0.010, -0.012,  0.012],
///      [ 0.020, -0.020,  0.018, -0.021],
///      [-0.015, -0.010, -0.014, -0.009],
/// ];
/// let data = Returns::new(&returns)?;
///
/// let weights = data.allocate(&Hrp::default().linkage(Linkage::Ward))?;
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(weights.iter().all(|&w| w >= 0.0));
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct Hrp {
    linkage: Linkage,
    covariance: CovarianceConfig,
}

impl Hrp {
    /// Sets how the distance between clusters is measured when building the hierarchy.
    /// Defaults to [`Linkage::Single`].
    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    covariance_setters!();

    /// Returns the order HRP puts the assets in: the leaves of the clustering hierarchy, left
    /// to right, so that correlated assets sit next to each other.
    ///
    /// Assets are clustered on the correlation distance `sqrt((1 − ρ) / 2)` with the
    /// [linkage](Hrp::linkage). The clustering reproduces SciPy's `linkage` and
    /// `to_tree().pre_order()` step for step, including the tie-breaking rule, so the order
    /// matches SciPy and PyPortfolioOpt's `HRPOpt` up to floating-point rounding.
    ///
    /// Every asset's column index appears exactly once.
    ///
    /// # Errors
    ///
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
    /// * [`CutupError::NonFiniteCovariance`] if a variance is below about 1e-292,
    ///   `f64::MIN_POSITIVE / f64::EPSILON`, too small to compute correlations from accurately.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{Hrp, Returns};
    ///
    /// // Assets 0 and 2 move together, as do 1 and 3.
    /// let returns = [
    ///      [ 0.010,  0.020,  0.011,  0.019],
    ///      [-0.010,  0.010, -0.012,  0.012],
    ///      [ 0.020, -0.020,  0.018, -0.021],
    ///      [-0.015, -0.010, -0.014, -0.009],
    /// ];
    ///
    /// let data = Returns::new(&returns)?;
    /// let order = Hrp::default().cluster_order(&data)?;
    ///
    /// let position = |asset| order.iter().position(|&a| a == asset).unwrap();
    /// assert_eq!(position(0).abs_diff(position(2)), 1);
    /// assert_eq!(position(1).abs_diff(position(3)), 1);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn cluster_order(&self, data: &Returns) -> Result<Vec<usize>, CutupError> {
        data.cluster_order(self.linkage)
    }
}

impl Allocation for Hrp {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        data.hrp(self.linkage, &self.covariance)
    }
}

impl Returns {
    /// The order HRP puts the assets in, for [`Hrp::cluster_order`].
    fn cluster_order(&self, linkage: Linkage) -> Result<Vec<usize>, CutupError> {
        if let Some(asset) = self.find_zero_variance() {
            return Err(self.zero_variance(asset));
        }
        let distance = self.correlation_distance()?;
        Ok(cluster::hierarchy(&distance, linkage).leaf_order())
    }

    /// The HRP weights, for [`Hrp`]. The hierarchy is built on the sample correlations, and
    /// `covariance` adjusts only the variances that split the weight.
    fn hrp(&self, linkage: Linkage, covariance: &CovarianceConfig) -> Result<Weights, CutupError> {
        // The settings are checked before the data, as every other method checks them.
        covariance.validate()?;
        let order = self.cluster_order(linkage)?;
        let cov = self.adjusted_covariance(covariance)?;
        let mut weights = vec![1.0; order.len()];
        // Clusters are ranges of positions in `order`, bisected one level at a time.
        let mut ranges = vec![(0, order.len())];

        while !ranges.is_empty() {
            ranges = ranges
                .into_iter()
                .filter(|(start, end)| end - start > 1)
                .flat_map(|(start, end)| {
                    let mid = start + (end - start) / 2;
                    [(start, mid), (mid, end)]
                })
                .collect();

            for pair in ranges.chunks(2) {
                let left = &order[pair[0].0..pair[0].1];
                let right = &order[pair[1].0..pair[1].1];
                // On singular data rounding can leave a variance slightly below zero, which
                // would push α outside [0, 1] and short the other half.
                let v_left = cluster_variance(&cov, left).max(0.0);
                let v_right = cluster_variance(&cov, right).max(0.0);
                // Two halves with zero variance each (perfectly hedged pairs) split evenly.
                let alpha = match v_left + v_right {
                    total if total > 0.0 => 1.0 - v_left / total,
                    _ => 0.5,
                };
                for &i in left {
                    weights[i] *= alpha;
                }
                for &i in right {
                    weights[i] *= 1.0 - alpha;
                }
            }
        }

        Ok(self.weights(weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_weights_close, full_rank_prices, rank_one_returns, two_block_returns,
    };
    use crate::{EqualWeight, Herc, MinVariance};
    use nalgebra::{DMatrix, dmatrix};

    #[test]
    fn test_hrp() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let hrp_weights = allocator.allocate(&Hrp::default()).unwrap();
        assert_eq!(hrp_weights.len(), 3);
        assert!((hrp_weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_triggers_continue_path() {
        let prices = dmatrix![
            100.0, 200.0, 300.0, 400.0;
            101.0, 201.0, 301.0, 401.0;
            102.0, 202.0, 302.0, 402.0;
            103.0, 203.0, 303.0, 403.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let weights = allocator.allocate(&Hrp::default()).unwrap();

        assert_eq!(weights.len(), 4);
        for &w in &weights {
            assert!(w.is_finite());
        }
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_rejects_zero_variance() {
        let prices = dmatrix![
            100.0, 50.0, 20.0;
            102.0, 50.0, 20.5;
            101.0, 50.0, 20.2;
            104.0, 50.0, 20.9
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        assert_eq!(
            allocator.allocate(&Hrp::default()).err(),
            Some(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
        assert_eq!(allocator.allocate(&EqualWeight).unwrap().len(), 3);
    }

    #[test]
    fn test_hrp_matches_pyportfolioopt() {
        // HRPOpt(returns).optimize() on the returns of full_rank_prices, with PyPortfolioOpt
        // 1.6.0 and SciPy 1.17.1.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = Weights::from(vec![
            0.43205598172341897,
            0.23653580200415913,
            0.3314082162724219,
        ]);
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &expected,
            1e-12,
        );
    }

    #[test]
    fn test_hrp_two_assets_is_inverse_variance() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = Returns::prices_matrix(prices).unwrap();
        let (v1, v2) = (allocator.cov_matrix[(0, 0)], allocator.cov_matrix[(1, 1)]);
        let w1 = (1.0 / v1) / (1.0 / v1 + 1.0 / v2);
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &Weights::from(vec![w1, 1.0 - w1]),
            1e-12,
        );
    }

    #[test]
    fn test_hrp_uncorrelated_assets_is_inverse_variance() {
        // Uncorrelated assets are all equally far apart, so the clustering is decided by tie
        // breaking; whatever the order, the weights reduce to inverse variance.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let inverse = [1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)];
        let total: f64 = inverse.iter().sum();
        let expected = Weights::from(inverse.iter().map(|x| x / total).collect::<Vec<_>>());
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &expected,
            1e-12,
        );
    }

    #[test]
    fn test_hrp_order_groups_correlated_assets() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        let order = Hrp::default().cluster_order(&allocator).unwrap();

        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);

        let position = |asset: usize| order.iter().position(|&a| a == asset).unwrap();
        let mut first_block: Vec<usize> = [0, 2, 4].iter().map(|&a| position(a)).collect();
        first_block.sort_unstable();
        assert_eq!(first_block[2] - first_block[0], 2, "order {order:?}");
    }

    #[test]
    fn test_hrp_every_linkage_groups_correlated_assets() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        for linkage in [
            Linkage::Single,
            Linkage::Complete,
            Linkage::Average,
            Linkage::Ward,
        ] {
            let config = Hrp::default().linkage(linkage);
            let order = config.cluster_order(&allocator).unwrap();

            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4], "{linkage:?}");

            let position = |asset: usize| order.iter().position(|&a| a == asset).unwrap();
            let mut first_block: Vec<usize> = [0, 2, 4].iter().map(|&a| position(a)).collect();
            first_block.sort_unstable();
            assert_eq!(first_block[2] - first_block[0], 2, "{linkage:?}: {order:?}");

            let weights = allocator.allocate(&config).unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(weights.iter().all(|&w| w >= 0.0));
        }
    }

    #[test]
    fn test_hrp_defaults_to_single_linkage() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        assert_eq!(Hrp::default(), Hrp::default().linkage(Linkage::Single));
        assert_eq!(
            allocator.allocate(&Hrp::default()).unwrap(),
            allocator
                .allocate(&Hrp::default().linkage(Linkage::Single))
                .unwrap()
        );
    }

    #[test]
    fn test_hrp_groups_follow_clustering_not_columns() {
        // On this block-structured input the clustering, not the input order, decides how
        // assets are grouped, so this shuffle leaves the weights unchanged. (In general HRP
        // can depend on column order: see `Hrp`.)
        let returns = two_block_returns();
        let permutation = [3, 0, 4, 1, 2];
        let permuted = DMatrix::from_fn(returns.nrows(), 5, |t, j| returns[(t, permutation[j])]);

        let base = Returns::returns_matrix(returns)
            .unwrap()
            .allocate(&Hrp::default())
            .unwrap();
        let shuffled = Returns::returns_matrix(permuted)
            .unwrap()
            .allocate(&Hrp::default())
            .unwrap();

        for (j, &asset) in permutation.iter().enumerate() {
            assert!((shuffled[j] - base[asset]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hrp_single_asset() {
        let allocator = Returns::prices_matrix(dmatrix![100.0; 110.0; 105.0]).unwrap();
        assert_eq!(Hrp::default().cluster_order(&allocator).unwrap(), vec![0]);
        assert_eq!(
            allocator.allocate(&Hrp::default()).unwrap().as_slice(),
            &[1.0]
        );
    }

    #[test]
    fn test_hrp_singular_covariance() {
        // HRP never inverts Σ, so rank-deficient data still gives positive weights.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let weights = allocator.allocate(&Hrp::default()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_hrp_zero_variance_half_takes_all_weight() {
        // Assets 0 and 1 are a perfect hedge, so their cluster has zero variance and takes all
        // of the weight; the correlated pair 2 and 3 gets none. HRPOpt behaves the same way.
        let a = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
        let b = [0.005, 0.01, -0.02, 0.015, -0.01, 0.004];
        let noise = [0.0005, -0.0003, 0.0002, -0.0004, 0.0001, 0.0003];
        let returns = DMatrix::from_fn(6, 4, |t, j| match j {
            0 => a[t],
            1 => -a[t],
            2 => b[t],
            _ => b[t] + noise[t],
        });
        let weights = Returns::returns_matrix(returns)
            .unwrap()
            .allocate(&Hrp::default())
            .unwrap();

        assert!((weights[0] - 0.5).abs() < 1e-12 && (weights[1] - 0.5).abs() < 1e-12);
        assert!(weights[2] < 1e-12 && weights[3] < 1e-12);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_hrp_and_herc_match_unit_scale_on_tiny_variances() {
        // Variances between about 1e-180 and 1e-290, whose products underflow to zero, so
        // dividing by sqrt(σᵢᵢ·σⱼⱼ) would give an infinite or NaN correlation.
        let unit = dmatrix![1.0, 2.0; -1.0, 3.0; 2.0, -1.0];
        let base = Returns::returns_matrix(unit.clone()).unwrap();
        for s in [1e-90, 1e-100, 1e-140, 1e-145] {
            let tiny = Returns::returns_matrix(&unit * s).unwrap();
            for method in [&Hrp::default() as &dyn Allocation, &Herc::default()] {
                assert_weights_close(
                    &tiny.allocate(method).unwrap(),
                    &base.allocate(method).unwrap(),
                    1e-12,
                );
            }
        }
    }

    #[test]
    fn test_hrp_and_herc_reject_variances_too_small_to_correlate() {
        let methods = [&Hrp::default() as &dyn Allocation, &Herc::default()];
        // Variances around 1e-320 pass validation, but they're subnormal.
        let s = 1e-160;
        let correlated = dmatrix![s, 2.0 * s; -s, 3.0 * s; 2.0 * s, -s];
        let orthogonal = dmatrix![s, s; -s, s; s, -s; -s, -s];
        // Twelve assets with variances of 2.5e-308, just above the subnormal range: most
        // covariances are subnormal, and the inverse variances of six of them, half of HRP's
        // first split, sum past f64::MAX.
        let cycle = [
            1.0, 2.0, -1.0, 3.0, -2.0, 0.5, -0.7, 1.4, 0.2, -1.1, 0.8, -0.3, 2.2, -1.6,
        ];
        let unit = DMatrix::from_fn(14, 12, |t, j| cycle[(t + j) % 14]);
        let variance = Returns::returns_matrix(unit.clone()).unwrap().cov_matrix[(0, 0)];
        let near_subnormal = &unit * (2.5e-308 / variance).sqrt();
        for returns in [correlated, orthogonal, near_subnormal] {
            let allocator = Returns::returns_matrix(returns).unwrap();
            for method in methods {
                assert_eq!(
                    allocator.allocate(method).err(),
                    Some(CutupError::NonFiniteCovariance)
                );
            }
        }
    }

    #[test]
    fn test_hrp_anti_correlated_pair_splits_evenly() {
        // A perfectly anti-correlated pair with equal variances: the two assets are as far apart
        // as possible, and each half carries the same risk.
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &Weights::from(vec![0.5, 0.5]),
            1e-12,
        );
    }
}
