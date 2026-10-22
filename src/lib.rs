#![doc = include_str!("../README.md")]

use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::fmt;
use std::ops::Index;

/// Errors returned when building a `PortfolioAllocator` or computing an allocation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CutupError {
    /// The input matrix has no columns, so there are no assets to allocate to.
    NoAssets,
    /// The input matrix has too few rows (observations) for a sample covariance.
    InsufficientData {
        /// Minimum number of rows needed.
        required: usize,
        /// Number of rows supplied.
        actual: usize,
    },
    /// A value in the input matrix is NaN or infinite.
    NonFiniteValue {
        /// Row of the offending value.
        row: usize,
        /// Column (asset) of the offending value.
        col: usize,
    },
    /// A price is zero or negative, so a return can't be computed from it.
    NonPositivePrice {
        /// Row of the offending price.
        row: usize,
        /// Column (asset) of the offending price.
        col: usize,
    },
    /// An asset has zero variance, which the method can't divide by.
    ZeroVariance {
        /// Column (asset) with zero variance.
        asset: usize,
    },
    /// The covariance matrix is singular and no finite weights could be computed from it.
    SingularCovariance,
}

impl fmt::Display for CutupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CutupError::NoAssets => write!(f, "input has no assets (zero columns)"),
            CutupError::InsufficientData { required, actual } => write!(
                f,
                "input has {actual} rows but at least {required} are required"
            ),
            CutupError::NonFiniteValue { row, col } => {
                write!(f, "value at row {row}, column {col} is NaN or infinite")
            }
            CutupError::NonPositivePrice { row, col } => {
                write!(f, "price at row {row}, column {col} is zero or negative")
            }
            CutupError::ZeroVariance { asset } => write!(f, "asset {asset} has zero variance"),
            CutupError::SingularCovariance => write!(
                f,
                "covariance matrix is singular; no finite weights could be computed"
            ),
        }
    }
}

impl std::error::Error for CutupError {}

/// Portfolio weights, one per asset, in the same order as the columns of the input matrix.
///
/// # Example
///
/// ```
/// use cutup::PortfolioAllocator;
/// use nalgebra::dmatrix;
///
/// let prices = dmatrix![
///     100.0, 200.0;
///     110.0, 190.0;
///     121.0, 209.0
/// ];
///
/// let weights = PortfolioAllocator::from_prices(prices)?.ew_allocation();
///
/// assert_eq!(weights.len(), 2);
/// assert_eq!(weights[0], 0.5);
/// assert_eq!(weights.get(1), Some(0.5));
/// assert_eq!(weights.iter().sum::<f64>(), 1.0);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Weights(Vec<f64>);

impl Weights {
    /// Returns the weight of the asset in column `asset`, or `None` if out of range.
    pub fn get(&self, asset: usize) -> Option<f64> {
        self.0.get(asset).copied()
    }

    /// Returns the number of assets.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if there are no weights.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Iterates over the weights in column order.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.0.iter()
    }

    /// Returns the weights as a slice, in column order.
    pub fn as_slice(&self) -> &[f64] {
        &self.0
    }

    /// Consumes the weights and returns them as a `Vec`, in column order.
    pub fn into_vec(self) -> Vec<f64> {
        self.0
    }

    /// Returns the weights keyed by column index, the shape returned before 0.2.0.
    pub fn to_hashmap(&self) -> HashMap<usize, f64> {
        self.0.iter().copied().enumerate().collect()
    }
}

impl Index<usize> for Weights {
    type Output = f64;

    fn index(&self, asset: usize) -> &f64 {
        &self.0[asset]
    }
}

impl<'a> IntoIterator for &'a Weights {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl From<Weights> for Vec<f64> {
    fn from(weights: Weights) -> Self {
        weights.0
    }
}

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
///
/// All covariance-based methods work on the covariance of asset returns, never of price levels.
/// Build an allocator from prices with [`PortfolioAllocator::from_prices`], or from returns you
/// have already computed with [`PortfolioAllocator::from_returns`].
pub struct PortfolioAllocator {
    returns: DMatrix<f64>,
    cov_matrix: DMatrix<f64>,
}

impl PortfolioAllocator {
    /// Creates a new `PortfolioAllocator` from a matrix of asset prices.
    ///
    /// Prices are converted to simple returns, `r_t = p_t / p_{t-1} - 1`, before the covariance
    /// matrix is computed. This matches `pandas.DataFrame.pct_change`. A price matrix with `T`
    /// rows produces `T - 1` rows of returns, so at least three rows of prices are needed for a
    /// sample covariance.
    ///
    /// # Arguments
    ///
    /// * `prices` - A `DMatrix<f64>` representing asset price history (rows: time, columns: assets).
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if `prices` has no columns.
    /// * [`CutupError::InsufficientData`] if `prices` has fewer than three rows.
    /// * [`CutupError::NonFiniteValue`] if any price is NaN or infinite.
    /// * [`CutupError::NonPositivePrice`] if any price is zero or negative.
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
    /// let allocator = PortfolioAllocator::from_prices(prices)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn from_prices(prices: DMatrix<f64>) -> Result<Self, CutupError> {
        PortfolioAllocator::validate(&prices, 3)?;
        if let Some((row, col)) = PortfolioAllocator::find(&prices, |p| p <= 0.0) {
            return Err(CutupError::NonPositivePrice { row, col });
        }
        PortfolioAllocator::from_returns(PortfolioAllocator::compute_returns(&prices))
    }

    /// Creates a new `PortfolioAllocator` from a matrix of asset returns.
    ///
    /// Use this when you already have returns (for example log returns, or returns net of a
    /// benchmark). They are used as-is; at least two rows are needed for a sample covariance.
    ///
    /// # Arguments
    ///
    /// * `returns` - A `DMatrix<f64>` of periodic asset returns (rows: time, columns: assets).
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if `returns` has no columns.
    /// * [`CutupError::InsufficientData`] if `returns` has fewer than two rows.
    /// * [`CutupError::NonFiniteValue`] if any return is NaN or infinite.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::PortfolioAllocator;
    /// use nalgebra::dmatrix;
    ///
    /// let returns = dmatrix![
    ///      0.010, -0.020,  0.005;
    ///     -0.004,  0.015,  0.002;
    ///      0.007,  0.003, -0.006
    /// ];
    ///
    /// let allocator = PortfolioAllocator::from_returns(returns)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn from_returns(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        PortfolioAllocator::validate(&returns, 2)?;
        let cov_matrix = PortfolioAllocator::compute_covariance_matrix(&returns);
        Ok(PortfolioAllocator {
            returns,
            cov_matrix,
        })
    }

    /// Creates a new `PortfolioAllocator` from a matrix of asset prices.
    ///
    /// Identical to [`PortfolioAllocator::from_prices`], but panics instead of returning an
    /// error. Before 0.2.0 this computed the covariance of price levels rather than of returns.
    ///
    /// # Panics
    ///
    /// Panics if `price_data` fails any of the checks listed under
    /// [`PortfolioAllocator::from_prices`].
    #[deprecated(
        since = "0.2.0",
        note = "use `PortfolioAllocator::from_prices` instead"
    )]
    pub fn new(price_data: DMatrix<f64>) -> Self {
        PortfolioAllocator::from_prices(price_data).expect("invalid price data")
    }

    /// Checks that `data` has at least one column, at least `min_rows` rows, and only finite
    /// values.
    fn validate(data: &DMatrix<f64>, min_rows: usize) -> Result<(), CutupError> {
        if data.ncols() == 0 {
            return Err(CutupError::NoAssets);
        }
        if data.nrows() < min_rows {
            return Err(CutupError::InsufficientData {
                required: min_rows,
                actual: data.nrows(),
            });
        }
        if let Some((row, col)) = PortfolioAllocator::find(data, |x| !x.is_finite()) {
            return Err(CutupError::NonFiniteValue { row, col });
        }
        Ok(())
    }

    /// Returns the `(row, col)` of the first value matching `predicate`, scanning row by row.
    fn find(data: &DMatrix<f64>, predicate: impl Fn(f64) -> bool) -> Option<(usize, usize)> {
        (0..data.nrows())
            .flat_map(|row| (0..data.ncols()).map(move |col| (row, col)))
            .find(|&(row, col)| predicate(data[(row, col)]))
    }

    /// Converts a price matrix into simple returns, `r_t = p_t / p_{t-1} - 1`.
    ///
    /// # Arguments
    ///
    /// * `prices` - A `DMatrix<f64>` of prices (rows: time, columns: assets).
    ///
    /// # Returns
    ///
    /// * A `DMatrix<f64>` with one fewer row than `prices`.
    fn compute_returns(prices: &DMatrix<f64>) -> DMatrix<f64> {
        let rows = prices.nrows().saturating_sub(1);
        DMatrix::from_fn(rows, prices.ncols(), |t, j| {
            prices[(t + 1, j)] / prices[(t, j)] - 1.0
        })
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
    /// * [`Weights`], one per asset, summing to 1. Weights can be negative (short positions).
    ///
    /// # Errors
    ///
    /// * [`CutupError::SingularCovariance`] if no finite weights can be computed.
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
    /// let allocator = PortfolioAllocator::from_prices(prices)?;
    /// let weights = allocator.mvo_allocation()?;
    ///
    /// let total: f64 = weights.iter().sum();
    /// assert!((total - 1.0).abs() < 1e-6);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn mvo_allocation(&self) -> Result<Weights, CutupError> {
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
    /// * [`Weights`], one per asset, summing to 1. Weights can be negative (short positions).
    ///
    /// # Errors
    ///
    /// * [`CutupError::SingularCovariance`] if no finite weights can be computed.
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
    /// let allocator = PortfolioAllocator::from_prices(prices)?;
    /// let config = MvoConfig {
    ///     regularization: Some(1e-6),
    ///     shrinkage: Some(0.05),
    /// };
    ///
    /// let weights = allocator.mvo_allocation_with_config(&config)?;
    ///
    /// let total: f64 = weights.iter().sum();
    /// assert!((total - 1.0).abs() < 1e-6);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn mvo_allocation_with_config(&self, config: &MvoConfig) -> Result<Weights, CutupError> {
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

        // Fall back to the pseudo-inverse when Σ is singular, which happens whenever there are
        // fewer observations than assets.
        let inv_cov = match regularized_cov.clone().try_inverse() {
            Some(inv) => inv,
            None => {
                let eps = config.regularization.unwrap_or(1e-8);
                regularized_cov
                    .pseudo_inverse(eps)
                    .map_err(|_| CutupError::SingularCovariance)?
            }
        };

        let denom = (ones.transpose() * &inv_cov * &ones)[(0, 0)];
        let weights = &inv_cov * &ones / denom;

        if weights.iter().any(|w| !w.is_finite()) {
            return Err(CutupError::SingularCovariance);
        }
        Ok(Weights(weights.iter().copied().collect()))
    }

    /// Computes Equal Weight (EW) portfolio allocation.
    ///
    /// # Returns
    ///
    /// * [`Weights`] where each asset has weight `1 / n`.
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
    /// let allocator = PortfolioAllocator::from_prices(prices)?;
    /// let weights = allocator.ew_allocation();
    ///
    /// assert_eq!(weights.len(), 3);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn ew_allocation(&self) -> Weights {
        let n = self.returns.ncols();
        Weights(vec![1.0 / n as f64; n])
    }

    /// Computes Hierarchical Risk Parity (HRP) portfolio allocation.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, summing to 1.
    ///
    /// # Errors
    ///
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
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
    /// let allocator = PortfolioAllocator::from_prices(prices)?;
    /// let weights = allocator.hrp_allocation()?;
    ///
    /// let sum: f64 = weights.iter().sum();
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn hrp_allocation(&self) -> Result<Weights, CutupError> {
        let n = self.cov_matrix.ncols();
        if let Some(asset) = (0..n).find(|&i| self.cov_matrix[(i, i)] <= 0.0) {
            return Err(CutupError::ZeroVariance { asset });
        }

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

        Ok(Weights(weights))
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
/// * [`Weights`] from [`PortfolioAllocator::mvo_allocation`].
///
/// # Errors
///
/// * Any error from [`PortfolioAllocator::from_prices`] or [`PortfolioAllocator::mvo_allocation`].
///
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
/// let weights = run_portfolio_allocation(prices)?;
/// assert_eq!(weights.len(), 4);
/// # Ok::<(), cutup::CutupError>(())
/// ```
pub fn run_portfolio_allocation(prices: DMatrix<f64>) -> Result<Weights, CutupError> {
    PortfolioAllocator::from_prices(prices)?.mvo_allocation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    /// Six periods of prices for three assets whose returns covariance is full rank.
    fn full_rank_prices() -> DMatrix<f64> {
        dmatrix![
            100.0, 50.0, 20.0;
            102.0, 49.0, 20.5;
            101.0, 51.5, 20.2;
            104.0, 51.0, 20.9;
            103.0, 52.5, 21.3;
            106.0, 52.0, 21.0
        ]
    }

    fn assert_weights_close(a: &Weights, b: &Weights, tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (wa, wb)) in a.iter().zip(b).enumerate() {
            assert!((wa - wb).abs() < tol, "asset {i}: {wa} vs {wb}");
        }
    }

    #[test]
    fn test_compute_returns() {
        let prices = dmatrix![
            100.0, 200.0;
            110.0, 190.0;
            121.0, 209.0
        ];
        let returns = PortfolioAllocator::compute_returns(&prices);
        let expected = dmatrix![
            0.1, -0.05;
            0.1, 0.1
        ];
        assert_eq!(returns.shape(), (2, 2));
        assert!((returns - expected).abs().max() < 1e-12);
    }

    #[test]
    fn test_from_prices_matches_from_returns() {
        let prices = full_rank_prices();
        let returns = PortfolioAllocator::compute_returns(&prices);
        let from_prices = PortfolioAllocator::from_prices(prices).unwrap();
        let from_returns = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(from_prices.cov_matrix, from_returns.cov_matrix);
    }

    #[test]
    #[allow(deprecated)]
    fn test_deprecated_new_matches_from_prices() {
        let prices = full_rank_prices();
        let new = PortfolioAllocator::new(prices.clone());
        let from_prices = PortfolioAllocator::from_prices(prices).unwrap();
        assert_eq!(new.cov_matrix, from_prices.cov_matrix);
    }

    #[test]
    #[allow(deprecated)]
    #[should_panic(expected = "invalid price data")]
    fn test_deprecated_new_panics_on_invalid_input() {
        PortfolioAllocator::new(dmatrix![100.0; 110.0]);
    }

    #[test]
    fn test_from_prices_rejects_no_assets() {
        let result = PortfolioAllocator::from_prices(DMatrix::zeros(3, 0));
        assert_eq!(result.err(), Some(CutupError::NoAssets));
    }

    #[test]
    fn test_from_prices_rejects_too_few_rows() {
        let result = PortfolioAllocator::from_prices(dmatrix![100.0, 200.0; 110.0, 210.0]);
        assert_eq!(
            result.err(),
            Some(CutupError::InsufficientData {
                required: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn test_from_returns_rejects_too_few_rows() {
        let result = PortfolioAllocator::from_returns(dmatrix![0.01, -0.02]);
        assert_eq!(
            result.err(),
            Some(CutupError::InsufficientData {
                required: 2,
                actual: 1
            })
        );
    }

    #[test]
    fn test_from_prices_rejects_non_finite_values() {
        let mut prices = full_rank_prices();
        prices[(2, 1)] = f64::NAN;
        prices[(4, 0)] = f64::INFINITY;
        let result = PortfolioAllocator::from_prices(prices);
        assert_eq!(
            result.err(),
            Some(CutupError::NonFiniteValue { row: 2, col: 1 })
        );
    }

    #[test]
    fn test_from_returns_rejects_non_finite_values() {
        let result = PortfolioAllocator::from_returns(dmatrix![0.01, 0.02; f64::NEG_INFINITY, 0.0]);
        assert_eq!(
            result.err(),
            Some(CutupError::NonFiniteValue { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_from_prices_rejects_non_positive_prices() {
        let mut prices = full_rank_prices();
        prices[(3, 2)] = 0.0;
        let result = PortfolioAllocator::from_prices(prices.clone());
        assert_eq!(
            result.err(),
            Some(CutupError::NonPositivePrice { row: 3, col: 2 })
        );

        prices[(1, 0)] = -5.0;
        let result = PortfolioAllocator::from_prices(prices);
        assert_eq!(
            result.err(),
            Some(CutupError::NonPositivePrice { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_from_returns_accepts_negative_returns() {
        let returns = dmatrix![-0.5, 0.01; 0.2, -0.03; -0.1, 0.02];
        assert!(PortfolioAllocator::from_returns(returns).is_ok());
    }

    #[test]
    fn test_error_display() {
        let err = CutupError::InsufficientData {
            required: 3,
            actual: 1,
        };
        assert_eq!(
            err.to_string(),
            "input has 1 rows but at least 3 are required"
        );
        assert_eq!(
            CutupError::ZeroVariance { asset: 2 }.to_string(),
            "asset 2 has zero variance"
        );
    }

    #[test]
    fn test_weights_accessors() {
        let weights = Weights(vec![0.2, 0.3, 0.5]);
        assert_eq!(weights.len(), 3);
        assert!(!weights.is_empty());
        assert_eq!(weights[1], 0.3);
        assert_eq!(weights.get(2), Some(0.5));
        assert_eq!(weights.get(3), None);
        assert_eq!(weights.as_slice(), &[0.2, 0.3, 0.5]);
        assert_eq!((&weights).into_iter().count(), 3);

        let map = weights.to_hashmap();
        assert_eq!(map.len(), 3);
        assert_eq!(map[&0], 0.2);

        assert_eq!(weights.clone().into_vec(), vec![0.2, 0.3, 0.5]);
        assert_eq!(Vec::from(weights), vec![0.2, 0.3, 0.5]);
    }

    #[test]
    fn test_weights_ignore_price_scale() {
        // Rescaling one asset's prices (a stock split, or quoting in cents) leaves its returns
        // unchanged, so no weight should move. On price-level covariance this fails.
        let prices = full_rank_prices();
        let mut scaled = prices.clone();
        scaled.column_mut(1).scale_mut(1000.0);

        let base = PortfolioAllocator::from_prices(prices).unwrap();
        let rescaled = PortfolioAllocator::from_prices(scaled).unwrap();
        assert!(base.cov_matrix.clone().try_inverse().is_some());

        assert_weights_close(
            &base.mvo_allocation().unwrap(),
            &rescaled.mvo_allocation().unwrap(),
            1e-9,
        );
        assert_weights_close(
            &base.hrp_allocation().unwrap(),
            &rescaled.hrp_allocation().unwrap(),
            1e-9,
        );
    }

    #[test]
    fn test_ew_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let ew_weights = allocator.ew_allocation();
        assert_eq!(ew_weights.as_slice(), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_mvo_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let mvo_weights = allocator.mvo_allocation().unwrap();
        assert_eq!(mvo_weights.len(), 3);
        assert!((mvo_weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mvo_allocation_errors_on_non_finite_covariance() {
        let mut allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();

        // Unreachable through the constructors, which reject non-finite input; this guards the
        // check on the computed weights.
        allocator.cov_matrix = dmatrix![
            f64::NAN, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ];
        assert_eq!(
            allocator.mvo_allocation().err(),
            Some(CutupError::SingularCovariance)
        );

        allocator.cov_matrix = DMatrix::from_element(3, 3, f64::NAN);
        assert_eq!(
            allocator.mvo_allocation().err(),
            Some(CutupError::SingularCovariance)
        );
    }

    #[test]
    fn test_hrp_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let hrp_weights = allocator.hrp_allocation().unwrap();
        assert_eq!(hrp_weights.len(), 3);
        assert!((hrp_weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_allocation_triggers_continue_path() {
        let prices = dmatrix![
            100.0, 200.0, 300.0, 400.0;
            101.0, 201.0, 301.0, 401.0;
            102.0, 202.0, 302.0, 402.0;
            103.0, 203.0, 303.0, 403.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let weights = allocator.hrp_allocation().unwrap();

        assert_eq!(weights.len(), 4);
        for &w in &weights {
            assert!(w.is_finite());
        }
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_allocation_rejects_zero_variance() {
        let prices = dmatrix![
            100.0, 50.0, 20.0;
            102.0, 50.0, 20.5;
            101.0, 50.0, 20.2;
            104.0, 50.0, 20.9
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        assert_eq!(
            allocator.hrp_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 1 })
        );
    }

    #[test]
    fn test_single_asset() {
        let prices = dmatrix![
            100.0;
            110.0;
            120.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let ew_weights = allocator.ew_allocation();
        assert_eq!(ew_weights.as_slice(), &[1.0]);
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

        let weights = run_portfolio_allocation(prices).unwrap();

        let total_weight: f64 = weights.iter().sum();
        assert!(
            (total_weight - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0"
        );

        assert_eq!(weights.len(), 4, "Should return weights for 4 assets");
    }

    #[test]
    fn test_run_portfolio_allocation_propagates_errors() {
        assert_eq!(
            run_portfolio_allocation(dmatrix![100.0; 0.0; 110.0]).err(),
            Some(CutupError::NonPositivePrice { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_mvo_allocation_with_config_variants() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();

        let configs = [
            // 1. No regularization or shrinkage
            MvoConfig::default(),
            // 2. Regularization only
            MvoConfig {
                regularization: Some(1e-6),
                shrinkage: None,
            },
            // 3. Shrinkage only
            MvoConfig {
                regularization: None,
                shrinkage: Some(0.1),
            },
            // 4. Both regularization and shrinkage
            MvoConfig {
                regularization: Some(1e-6),
                shrinkage: Some(0.2),
            },
        ];

        for config in &configs {
            let weights = allocator.mvo_allocation_with_config(config).unwrap();
            assert_eq!(weights.len(), 3, "{config:?}");
            assert!(
                (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
                "{config:?}"
            );
        }
    }
}
