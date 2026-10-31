//! [`Herc`]: Hierarchical Equal Risk Contribution.

use crate::{Allocation, CovarianceConfig, CutupError, Linkage, Returns, Weights, cluster};

/// The most clusters [`Herc`]'s gap statistic may choose, unless set.
const HERC_MAX_CLUSTERS: usize = 10;

/// Hierarchical Equal Risk Contribution (Raffinot, 2018), as riskfolio-lib implements it with
/// the variance as the risk measure.
///
/// 1. **Cluster.** Assets are clustered on the correlation distance `sqrt((1 − ρ) / 2)` with
///    the [linkage](Herc::linkage), as for [`Hrp::cluster_order`](crate::Hrp::cluster_order).
/// 2. **Cut.** The hierarchy is cut into `k` clusters: [`Herc::clusters`] if set, otherwise
///    the number chosen by the two-difference gap statistic, as in riskfolio-lib.
/// 3. **Split down the tree.** Each of the `k − 1` merges above the cut splits its weight
///    between its two branches in inverse proportion to their risk. A branch's risk is the
///    sum of the variances of the clusters under it, each cluster's being that of its own
///    inverse-variance portfolio. The left branch gets `α = 1 − R₁ / (R₁ + R₂)`.
/// 4. **Within clusters.** Each cluster's weight is shared by inverse variance.
///
/// The defaults match riskfolio-lib's `HCPortfolio.optimization(model="HERC")`: single
/// linkage, and the number of clusters chosen by the gap statistic from at most 10.
///
/// Unlike [`Hrp`](crate::Hrp), which bisects the ordered list of assets at its midpoint, HERC
/// splits where the tree branches, so similar assets are never separated and the result doesn't
/// depend on the order of the input columns. Results match riskfolio-lib's
/// `HCPortfolio.optimization(model="HERC", rm="MV")` up to floating-point rounding.
///
/// Both hold except when merges tie in distance at the cut, for example when assets are
/// duplicated. The cut is then made after exactly `n − k` merges, taking tied merges in the
/// order the clustering found them, which depends on column order. riskfolio-lib instead takes
/// the `k − 1` highest merges in the order of an unstable sort, and builds the clusters with
/// SciPy's `fcluster`, which can return fewer than `k` clusters at a tie, so its weights can
/// differ there.
///
/// HERC never inverts the covariance matrix, so it works when Σ is singular, and weights are
/// never negative, and sum to 1. Two branches that both have zero risk split their weight
/// evenly.
///
/// [Shrinkage and regularization](Herc::covariance) adjust the cluster risks and the
/// inverse-variance weights within clusters, but not the correlations the hierarchy is built
/// on, so the clusters, and the number the gap statistic chooses, don't change. That's what
/// riskfolio-lib does with `method_cov="shrunk"`, and results match it up to floating-point
/// rounding.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the number of clusters is 0 or more than the number of
///   assets, or, when the number isn't set, the most clusters is 0; or if the shrinkage is
///   outside `[0, 1]` or the regularization is negative or not finite.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance, even with
///   shrinkage or regularization: a constant asset has no correlation to cluster on.
/// * [`CutupError::NonFiniteCovariance`] if correlations can't be computed because the
///   variances are too small.
///
/// # Example
///
/// ```
/// use cutup::{Herc, InverseVariance, Returns};
///
/// // Assets 0 and 2 move together, as do 1 and 3.
/// let returns = [
///      [ 0.010,  0.020,  0.011,  0.019],
///      [-0.010,  0.010, -0.012,  0.012],
///      [ 0.020, -0.020,  0.018, -0.021],
///      [-0.015, -0.010, -0.014, -0.009],
/// ];
/// let data = Returns::new(&returns)?;
/// let weights = data.allocate(&Herc::default().clusters(2))?;
///
/// // With two clusters, {0, 2} and {1, 3}, each pair is weighted by inverse variance.
/// let inverse_variance = data.allocate(&InverseVariance)?;
/// let ratio = |a: usize, b: usize| weights[a] / weights[b];
/// assert!((ratio(0, 2) - inverse_variance[0] / inverse_variance[2]).abs() < 1e-12);
/// assert!((ratio(1, 3) - inverse_variance[1] / inverse_variance[3]).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Herc {
    linkage: Linkage,
    clusters: Option<usize>,
    max_clusters: usize,
    covariance: CovarianceConfig,
}

impl Default for Herc {
    fn default() -> Self {
        Self {
            linkage: Linkage::default(),
            clusters: None,
            max_clusters: HERC_MAX_CLUSTERS,
            covariance: CovarianceConfig::default(),
        }
    }
}

impl Herc {
    /// Sets how the distance between clusters is measured when building the hierarchy.
    /// Defaults to [`Linkage::Single`].
    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Cuts the hierarchy into `clusters` clusters, from 1 to the number of assets, instead
    /// of choosing the number.
    pub fn clusters(mut self, clusters: usize) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// Sets the most clusters the gap statistic may choose, at least 1. Defaults to 10. It
    /// also never chooses more than `√n` for `n` assets.
    pub fn max_clusters(mut self, max_clusters: usize) -> Self {
        self.max_clusters = max_clusters;
        self
    }

    covariance_setters!();
}

impl Allocation for Herc {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        data.herc(
            self.linkage,
            self.clusters,
            self.max_clusters,
            &self.covariance,
        )
    }
}

impl Returns {
    /// The HERC weights, for [`Herc`]. The hierarchy and the cut are built on the sample
    /// correlations, and `covariance` adjusts only the variances that split the weight.
    fn herc(
        &self,
        linkage: Linkage,
        clusters: Option<usize>,
        max_clusters: usize,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        let n = self.cov_matrix.ncols();
        if let Some(k) = clusters.filter(|k| !(1..=n).contains(k)) {
            return Err(CutupError::InvalidConfig {
                field: "clusters",
                value: k as f64,
            });
        }
        if clusters.is_none() && max_clusters == 0 {
            return Err(CutupError::InvalidConfig {
                field: "max_clusters",
                value: 0.0,
            });
        }
        covariance.validate()?;
        // Shrinkage would give a constant asset a variance, but not a correlation to cluster
        // on, so this comes before adjusting the covariance.
        if let Some(asset) = self.find_zero_variance() {
            return Err(self.zero_variance(asset));
        }
        let cov = self.adjusted_covariance(covariance)?;
        let distance = self.correlation_distance()?;
        let hierarchy = cluster::hierarchy(&distance, linkage);
        let k = clusters.unwrap_or_else(|| hierarchy.cluster_count(&distance, max_clusters));

        // Each cluster below the cut, by the node that formed it.
        let clusters = hierarchy.cut(k);
        let mut risk = vec![0.0; 2 * n - 1];
        for (node, members) in &clusters {
            // On singular data rounding can leave a variance slightly below zero, which would
            // push α outside [0, 1].
            risk[*node] = Returns::cluster_variance(&cov, members).max(0.0);
        }
        // The last k − 1 merges are the ones above the cut. A node's children always have
        // lower ids, so one pass up sums each branch's risk and one pass down splits weight.
        let top = 2 * n - k..2 * n - 1;
        for node in top.clone() {
            let (left, right) = hierarchy.children(node);
            risk[node] = risk[left] + risk[right];
        }
        let mut branch_weight = vec![0.0; 2 * n - 1];
        branch_weight[2 * n - 2] = 1.0;
        for node in top.rev() {
            let (left, right) = hierarchy.children(node);
            let alpha = match risk[left] + risk[right] {
                total if total > 0.0 => 1.0 - risk[left] / total,
                _ => 0.5,
            };
            branch_weight[left] = branch_weight[node] * alpha;
            branch_weight[right] = branch_weight[node] * (1.0 - alpha);
        }

        let mut weights = vec![0.0; n];
        for (node, members) in &clusters {
            let within = Returns::inverse_variance_weights(&cov, members);
            for (&i, w) in members.iter().zip(within) {
                weights[i] = branch_weight[*node] * w;
            }
        }
        Ok(self.weights(weights))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_weights_close, collinear_returns, two_block_returns};
    use crate::{Hrp, InverseVariance};
    use nalgebra::dmatrix;

    #[test]
    fn test_herc_with_one_cluster_is_inverse_variance() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        let config = Herc::default().clusters(1);
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &allocator.allocate(&InverseVariance).unwrap(),
            1e-15,
        );
    }

    #[test]
    fn test_herc_splits_between_blocks_by_risk() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        let weights = allocator.allocate(&Herc::default().clusters(2)).unwrap();
        let (a, b) = ([0, 2, 4], [1, 3]);
        let (risk_a, risk_b) = (
            Returns::cluster_variance(&allocator.cov_matrix, &a),
            Returns::cluster_variance(&allocator.cov_matrix, &b),
        );
        let weight_a: f64 = a.iter().map(|&i| weights[i]).sum();
        assert!((weight_a - risk_b / (risk_a + risk_b)).abs() < 1e-12);
        // Within a block, inverse variance.
        let iv = allocator.allocate(&InverseVariance).unwrap();
        assert!((weights[0] / weights[4] - iv[0] / iv[4]).abs() < 1e-12);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_herc_chooses_the_blocks() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        let config = Herc::default().clusters(2);
        assert_eq!(
            allocator.allocate(&Herc::default()),
            allocator.allocate(&config)
        );
    }

    #[test]
    fn test_herc_two_assets_is_inverse_variance() {
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();
        let iv = allocator.allocate(&InverseVariance).unwrap();
        for clusters in [1, 2] {
            let config = Herc::default().clusters(clusters);
            assert_weights_close(&allocator.allocate(&config).unwrap(), &iv, 1e-15);
        }
        assert_weights_close(&allocator.allocate(&Herc::default()).unwrap(), &iv, 1e-15);
    }

    #[test]
    fn test_herc_hedged_pair_splits_evenly() {
        // Each cluster is a perfectly hedged pair with zero variance.
        let returns = dmatrix![
            0.01, -0.01, 0.02, -0.02;
            -0.02, 0.02, 0.01, -0.01;
            0.015, -0.015, -0.01, 0.01
        ];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = Herc::default().clusters(4);
        let weights = allocator.allocate(&config).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_herc_rejects_invalid_config() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        for clusters in [0, 6] {
            assert_eq!(
                allocator.allocate(&Herc::default().clusters(clusters)),
                Err(CutupError::InvalidConfig {
                    field: "clusters",
                    value: clusters as f64
                })
            );
        }
        // max_clusters only limits the gap statistic, so it's ignored when clusters is set.
        let fixed = Herc::default().clusters(2).max_clusters(0);
        assert!(allocator.allocate(&fixed).is_ok());
        assert_eq!(
            allocator.allocate(&Herc::default().max_clusters(0)),
            Err(CutupError::InvalidConfig {
                field: "max_clusters",
                value: 0.0
            })
        );
        let returns = dmatrix![0.01, 0.0, 0.02; -0.02, 0.0, 0.01; 0.03, 0.0, -0.01];
        assert_eq!(
            Returns::returns_matrix(returns)
                .unwrap()
                .allocate(&Herc::default()),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
    }

    #[test]
    fn test_hierarchical_covariance_errors() {
        // Shrinkage gives a constant asset a variance, but it still has no correlation to
        // cluster on.
        let returns = dmatrix![0.01, 0.0, 0.02; -0.02, 0.0, 0.01; 0.03, 0.0, -0.01];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&Hrp::default().shrinkage(0.5)),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
        assert_eq!(
            allocator.allocate(&Herc::default().regularization(1e-6)),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
        // Even when every asset is constant, which shrinkage alone can't fix.
        let constant = Returns::returns_matrix(dmatrix![0.0, 0.0; 0.0, 0.0]).unwrap();
        assert_eq!(
            constant.allocate(&Hrp::default().shrinkage(0.5)),
            Err(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
        assert_eq!(
            constant.allocate(&Herc::default().shrinkage(0.5)),
            Err(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );

        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        let invalid = Err(CutupError::InvalidConfig {
            field: "shrinkage",
            value: 1.5,
        });
        assert_eq!(allocator.allocate(&Hrp::default().shrinkage(1.5)), invalid);
        assert_eq!(allocator.allocate(&Herc::default().shrinkage(1.5)), invalid);
        let negative = Err(CutupError::InvalidConfig {
            field: "regularization",
            value: -1e-6,
        });
        assert_eq!(
            allocator.allocate(&Hrp::default().regularization(-1e-6)),
            negative
        );
        assert_eq!(
            allocator.allocate(&Herc::default().regularization(-1e-6)),
            negative
        );
        // The settings are checked before the data, so a constant asset doesn't hide them.
        let constant =
            Returns::returns_matrix(dmatrix![0.01, 0.0, 0.02; -0.02, 0.0, 0.01; 0.03, 0.0, -0.01])
                .unwrap();
        assert_eq!(constant.allocate(&Hrp::default().shrinkage(1.5)), invalid);
        assert_eq!(constant.allocate(&Herc::default().shrinkage(1.5)), invalid);
        // The number of clusters is checked first.
        assert_eq!(
            allocator.allocate(&Herc::default().clusters(0).shrinkage(1.5)),
            Err(CutupError::InvalidConfig {
                field: "clusters",
                value: 0.0
            })
        );
    }

    #[test]
    fn test_hierarchical_covariance_changes_only_the_risk_split() {
        let allocator = Returns::returns_matrix(two_block_returns()).unwrap();
        let shrunk = allocator
            .adjusted_covariance(&CovarianceConfig::default().shrinkage(0.4))
            .unwrap();
        let weights = allocator
            .allocate(&Herc::default().clusters(2).shrinkage(0.4))
            .unwrap();
        // The same blocks as without shrinkage, split by their shrunk risk.
        let (a, b) = ([0, 2, 4], [1, 3]);
        let (risk_a, risk_b) = (
            Returns::cluster_variance(&shrunk, &a),
            Returns::cluster_variance(&shrunk, &b),
        );
        let weight_a: f64 = a.iter().map(|&i| weights[i]).sum();
        assert!((weight_a - risk_b / (risk_a + risk_b)).abs() < 1e-12);
        let within = Returns::inverse_variance_weights(&shrunk, &a);
        assert!((weights[0] / weights[4] - within[0] / within[2]).abs() < 1e-12);
        assert_ne!(
            weights,
            allocator.allocate(&Herc::default().clusters(2)).unwrap()
        );
    }
}
