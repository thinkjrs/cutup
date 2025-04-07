use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

/// `MvoConfig` provides optional configuration parameters for
/// the Mean-Variance Optimization (MVO) allocation strategy.
///
/// Both fields are optional. If omitted, standard MVO without
/// regularization or shrinkage is applied.
#[derive(Debug, Clone)]
pub struct MvoConfig {
    /// Optional regularization parameter (ε).
    /// Adds ε * I to the covariance matrix to improve numerical stability.
    /// Recommended small values are in the range of 1e-6 to 1e-3.
    /// If `None`, no regularization is applied.
    ///
    /// https://en.wikipedia.org/wiki/Tikhonov_regularization
    pub regularization: Option<f64>,

    /// Optional shrinkage intensity (λ) toward the identity matrix.
    /// The covariance matrix becomes: λ * I + (1 - λ) * Σ
    /// Helps mitigate estimation error in empirical covariances.
    /// If `None`, no shrinkage is applied.
    ///
    /// https://en.wikipedia.org/wiki/Shrinkage_estimator
    pub shrinkage: Option<f64>,
}

impl Default for MvoConfig {
    fn default() -> Self {
        Self {
            regularization: None,
            shrinkage: None,
        }
    }
}

/// `PortfolioAllocator` handles various portfolio allocation strategies.
/// Supports Mean-Variance Optimization (MVO), Equal Weight (EW), and Hierarchical Risk Parity (HRP).
pub struct PortfolioAllocator {
    price_data: DMatrix<f64>,
    cov_matrix: DMatrix<f64>,
}

impl PortfolioAllocator {
    /// Creates a new `PortfolioAllocator` from a matrix of asset prices.
    ///
    /// # Arguments
    ///
    /// * `price_data` - A `DMatrix<f64>` representing asset price history (rows: time, columns: assets).
    ///
    /// # Returns
    ///
    /// * A new instance of `PortfolioAllocator`.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::PortfolioAllocator;
    /// use nalgebra::dmatrix;
    ///
    /// let prices = dmatrix![
    ///     100.0, 200.0, 300.0;
    ///     110.0, 210.0, 310.0;
    ///     120.0, 220.0, 320.0
    /// ];
    ///
    /// let allocator = PortfolioAllocator::new(prices);
    /// ```
    pub fn new(price_data: DMatrix<f64>) -> Self {
        let cov_matrix = PortfolioAllocator::compute_covariance_matrix(&price_data);
        PortfolioAllocator {
            price_data,
            cov_matrix,
        }
    }

    /// Computes the covariance matrix from asset returns.
    ///
    /// # Arguments
    ///
    /// * `returns` - A `DMatrix<f64>` representing asset returns.
    ///
    /// # Returns
    ///
    /// * A `DMatrix<f64>` representing the covariance matrix.
    fn compute_covariance_matrix(returns: &DMatrix<f64>) -> DMatrix<f64> {
        let mean = returns.row_mean();
        let mean_matrix = DMatrix::from_rows(&vec![mean.clone(); returns.nrows()]);
        let centered = returns - mean_matrix;
        (centered.transpose() * centered) / (returns.nrows() as f64 - 1.0)
    }

    /// Computes Mean-Variance Optimized (MVO) portfolio weights.
    ///
    /// # Returns
    ///
    /// * A `HashMap<usize, f64>` where keys are asset indices and values are allocation weights.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::PortfolioAllocator;
    /// use nalgebra::dmatrix;
    ///
    /// let prices = dmatrix![
    ///     100.0, 200.0, 300.0;
    ///     110.0, 210.0, 310.0;
    ///     120.0, 220.0, 320.0
    /// ];
    ///
    /// let allocator = PortfolioAllocator::new(prices);
    /// let weights = allocator.mvo_allocation();
    ///
    /// let total: f64 = weights.values().sum();
    /// assert!((total - 1.0).abs() < 1e-6);
    /// ```
    pub fn mvo_allocation(&self) -> HashMap<usize, f64> {
        self.mvo_allocation_with_config(&MvoConfig::default())
    }

    /// Computes Mean-Variance Optimized (MVO) portfolio weights using configurable options.
    ///
    /// Accepts an `MvoConfig` struct to apply optional regularization and shrinkage to the
    /// covariance matrix for improved numerical stability and robustness.
    ///
    /// # Arguments
    ///
    /// * `config` - A reference to an `MvoConfig` struct specifying optional regularization and/or shrinkage.
    ///
    /// # Returns
    ///
    /// * A `HashMap<usize, f64>` where keys are asset indices and values are allocation weights.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{PortfolioAllocator, MvoConfig};
    /// use nalgebra::dmatrix;
    ///
    /// let prices = dmatrix![
    ///     100.0, 200.0, 300.0;
    ///     110.0, 210.0, 310.0;
    ///     120.0, 220.0, 320.0
    /// ];
    ///
    /// let allocator = PortfolioAllocator::new(prices);
    /// let config = MvoConfig {
    ///     regularization: Some(1e-6),
    ///     shrinkage: Some(0.05),
    /// };
    ///
    /// let weights = allocator.mvo_allocation_with_config(&config);
    ///
    /// let total: f64 = weights.values().sum();
    /// assert!((total - 1.0).abs() < 1e-6);
    /// ```
    pub fn mvo_allocation_with_config(&self, config: &MvoConfig) -> HashMap<usize, f64> {
        let n = self.cov_matrix.ncols();
        let ones = DVector::from_element(n, 1.0);

        let identity = DMatrix::identity(n, n);

        let shrunk_cov = if let Some(lambda) = config.shrinkage {
            lambda * &identity + (1.0 - lambda) * &self.cov_matrix
        } else {
            self.cov_matrix.clone()
        };

        let regularized_cov = if let Some(eps) = config.regularization {
            &shrunk_cov + eps * &identity
        } else {
            shrunk_cov
        };

        // Note: pseudo_inverse(...).expect(...) is not covered by tests because nalgebra::pseudo_inverse rarely fails.
        // This fallback is defensive and unreachable in practice unless the matrix is non-finite or SVD fails internally.
        let inv_cov = regularized_cov.clone().try_inverse().unwrap_or_else(|| {
            let eps = config.regularization.unwrap_or(1e-8);
            regularized_cov
                .pseudo_inverse(eps)
                .expect("Pseudo-inverse failed")
        });

        let denom = (ones.transpose() * &inv_cov * &ones)[(0, 0)];
        let weights = &inv_cov * &ones / denom;

        (0..n).map(|i| (i, weights[i])).collect()
    }

    /// Computes Equal Weight (EW) portfolio allocation.
    ///
    /// # Returns
    ///
    /// * A `HashMap<usize, f64>` where each asset has equal weight.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::PortfolioAllocator;
    /// use nalgebra::dmatrix;
    ///
    /// let prices = dmatrix![
    ///     100.0, 200.0, 300.0;
    ///     110.0, 210.0, 310.0;
    ///     120.0, 220.0, 320.0
    /// ];
    ///
    /// let allocator = PortfolioAllocator::new(prices);
    /// let weights = allocator.ew_allocation();
    ///
    /// assert_eq!(weights.len(), 3);
    /// ```
    pub fn ew_allocation(&self) -> HashMap<usize, f64> {
        let n = self.price_data.ncols();
        (0..n).map(|i| (i, 1.0 / n as f64)).collect()
    }

    /// Computes Hierarchical Risk Parity (HRP) portfolio allocation.
    ///
    /// # Returns
    ///
    /// * A `HashMap<usize, f64>` representing HRP-optimized portfolio weights.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::PortfolioAllocator;
    /// use nalgebra::dmatrix;
    ///
    /// let prices = dmatrix![
    ///     100.0, 200.0, 300.0;
    ///     110.0, 210.0, 310.0;
    ///     120.0, 220.0, 320.0
    /// ];
    ///
    /// let allocator = PortfolioAllocator::new(prices);
    /// let weights = allocator.hrp_allocation();
    ///
    /// let sum: f64 = weights.values().sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn hrp_allocation(&self) -> HashMap<usize, f64> {
        let n = self.cov_matrix.ncols();
        let mut weights = vec![1.0; n];
        let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];

        while let Some(cluster) = clusters.pop() {
            if cluster.len() != 1 {
                let mid = cluster.len() / 2;
                let left = &cluster[..mid];
                let right = &cluster[mid..];

                let vol_left: f64 = left.iter().map(|&i| self.cov_matrix[(i, i)]).sum();
                let vol_right: f64 = right.iter().map(|&i| self.cov_matrix[(i, i)]).sum();
                let total_vol = vol_left + vol_right;

                for &idx in left {
                    weights[idx] *= vol_right / total_vol;
                }
                for &idx in right {
                    weights[idx] *= vol_left / total_vol;
                }

                clusters.push(left.to_vec());
                clusters.push(right.to_vec());
            }
        }

        (0..n).map(|i| (i, weights[i])).collect()
    }
}

/// Runs portfolio allocation using Mean-Variance Optimization (MVO).
///
/// # Arguments
///
/// * `prices` - A `DMatrix<f64>` of asset price history.
///
/// # Returns
///
/// * A `HashMap<usize, f64>` representing optimized portfolio weights.
/// # Example
///
/// ```
/// use cutup::run_portfolio_allocation;
/// use nalgebra::dmatrix;
///
/// let prices = dmatrix![
///     125.0, 1500.0, 210.0, 600.0;
///     123.0, 1520.0, 215.0, 620.0;
///     130.0, 1510.0, 220.0, 610.0;
///     128.0, 1530.0, 225.0, 630.0
/// ];
///
/// let weights = run_portfolio_allocation(prices);
/// assert_eq!(weights.len(), 4);
/// ```
pub fn run_portfolio_allocation(prices: DMatrix<f64>) -> HashMap<usize, f64> {
    let allocator = PortfolioAllocator::new(prices);
    allocator.mvo_allocation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn test_ew_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let ew_weights = allocator.ew_allocation();
        assert_eq!(ew_weights.len(), 3);
        assert!((ew_weights.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mvo_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let mvo_weights = allocator.mvo_allocation();
        assert_eq!(mvo_weights.len(), 3);
        assert!((mvo_weights.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let hrp_weights = allocator.hrp_allocation();
        assert_eq!(hrp_weights.len(), 3);
        assert!((hrp_weights.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_allocation_triggers_continue_path() {
        let prices = dmatrix![
            100.0, 200.0, 300.0, 400.0;
            101.0, 201.0, 301.0, 401.0;
            102.0, 202.0, 302.0, 402.0;
            103.0, 203.0, 303.0, 403.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let weights = allocator.hrp_allocation();

        assert_eq!(weights.len(), 4);
        for &w in weights.values() {
            assert!(w.is_finite());
        }
        let sum: f64 = weights.values().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_asset() {
        let prices = dmatrix![
            100.0;
            110.0;
            120.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let ew_weights = allocator.ew_allocation();
        assert_eq!(ew_weights.len(), 1);
        assert_eq!(ew_weights.get(&0), Some(&1.0));
    }
    #[test]
    fn test_run_portfolio_allocation() {
        let prices = DMatrix::from_row_slice(
            4,
            4,
            &[
                125.0, 1500.0, 210.0, 600.0, 123.0, 1520.0, 215.0, 620.0, 130.0, 1510.0, 220.0,
                610.0, 128.0, 1530.0, 225.0, 630.0,
            ],
        );

        let weights = run_portfolio_allocation(prices);

        let total_weight: f64 = weights.values().sum();
        assert!(
            (total_weight - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0"
        );

        assert_eq!(weights.len(), 4, "Should return weights for 4 assets");
    }

    #[test]
    fn test_mvo_allocation_with_config_variants() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);

        // 1. No regularization or shrinkage
        let config_none = MvoConfig::default();
        let weights_none = allocator.mvo_allocation_with_config(&config_none);
        assert_eq!(weights_none.len(), 3);
        assert!((weights_none.values().sum::<f64>() - 1.0).abs() < 1e-6);

        // 2. Regularization only
        let config_reg = MvoConfig {
            regularization: Some(1e-6),
            shrinkage: None,
        };
        let weights_reg = allocator.mvo_allocation_with_config(&config_reg);
        assert_eq!(weights_reg.len(), 3);
        assert!((weights_reg.values().sum::<f64>() - 1.0).abs() < 1e-6);

        // 3. Shrinkage only
        let config_shrink = MvoConfig {
            regularization: None,
            shrinkage: Some(0.1),
        };
        let weights_shrink = allocator.mvo_allocation_with_config(&config_shrink);
        assert_eq!(weights_shrink.len(), 3);
        assert!((weights_shrink.values().sum::<f64>() - 1.0).abs() < 1e-6);

        // 4. Both regularization and shrinkage
        let config_both = MvoConfig {
            regularization: Some(1e-6),
            shrinkage: Some(0.2),
        };
        let weights_both = allocator.mvo_allocation_with_config(&config_both);
        assert_eq!(weights_both.len(), 3);
        assert!((weights_both.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mvo_allocation_pseudo_inverse_succeeds() {
        // Create an allocator with any prices
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            101.0, 201.0, 301.0;
            102.0, 202.0, 302.0
        ];
        let mut allocator = PortfolioAllocator::new(prices);

        // Force a corrupted covariance matrix with NaN
        allocator.cov_matrix = dmatrix![
            f64::NAN, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ];

        let config = MvoConfig {
            regularization: None,
            shrinkage: None,
        };

        // This should panic on .expect("Pseudo-inverse failed")
        let _ = allocator.mvo_allocation_with_config(&config);
    }

    #[test]
    fn test_mvo_allocation_pseudo_inverse_on_nan_matrix() {
        // Any shape, all NaNs
        let bad_matrix = DMatrix::<f64>::from_element(3, 3, f64::NAN);

        let mut allocator = PortfolioAllocator::new(dmatrix![
            100.0, 200.0, 300.0;
            101.0, 201.0, 301.0;
            102.0, 202.0, 302.0
        ]);

        allocator.cov_matrix = bad_matrix;

        let config = MvoConfig {
            regularization: None,
            shrinkage: None,
        };

        // Will fail at .pseudo_inverse(...).expect(...)
        let _ = allocator.mvo_allocation_with_config(&config);
    }
}
