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
    /// The input is finite but too extreme for a finite covariance matrix (for example, a
    /// return large enough to overflow when squared).
    NonFiniteCovariance,
    /// The covariance matrix is singular and no finite weights could be computed from it.
    SingularCovariance,
    /// A configuration value is out of range.
    InvalidConfig {
        /// Name of the offending field.
        field: &'static str,
        /// The value supplied.
        value: f64,
    },
    /// A configuration vector doesn't have one entry per asset.
    DimensionMismatch {
        /// Number of assets.
        expected: usize,
        /// Length supplied.
        actual: usize,
    },
    /// Every asset has the same expected return, so no portfolio can target a different one.
    DegenerateExpectedReturns,
    /// The mean-variance objective has no maximum: the covariance matrix is singular, and some
    /// fully invested portfolio gains expected return without adding variance. Add shrinkage or
    /// regularization, or use more observations.
    Unbounded,
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
            CutupError::NonFiniteCovariance => {
                write!(
                    f,
                    "input is too extreme to compute a finite covariance matrix"
                )
            }
            CutupError::SingularCovariance => write!(
                f,
                "covariance matrix is singular; no finite weights could be computed"
            ),
            CutupError::InvalidConfig { field, value } => {
                write!(f, "config value `{field}` is out of range: {value}")
            }
            CutupError::DimensionMismatch { expected, actual } => write!(
                f,
                "expected one value per asset ({expected}) but got {actual}"
            ),
            CutupError::DegenerateExpectedReturns => write!(
                f,
                "every asset has the same expected return, so no other target return can be reached"
            ),
            CutupError::Unbounded => write!(
                f,
                "objective is unbounded: expected return can grow without adding variance"
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

/// Adjustments applied to the sample covariance matrix before it is used by
/// [`PortfolioAllocator::min_variance_allocation_with_config`] and
/// [`PortfolioAllocator::mean_variance_allocation`].
///
/// The default applies no adjustment. Build one with the `with_*` methods:
///
/// ```
/// use cutup::CovarianceConfig;
///
/// let config = CovarianceConfig::default()
///     .with_shrinkage(0.1)
///     .with_regularization(1e-8);
///
/// assert_eq!(config.shrinkage, Some(0.1));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct CovarianceConfig {
    /// Optional regularization parameter (ε).
    /// Adds ε * I to the covariance matrix to improve numerical stability.
    /// Must be finite and non-negative. Unlike shrinkage this is absolute, so choose ε relative
    /// to the scale of your returns' variances.
    /// If `None`, no regularization is applied.
    ///
    /// <https://en.wikipedia.org/wiki/Tikhonov_regularization>
    pub regularization: Option<f64>,

    /// Optional shrinkage intensity (λ) toward a scaled identity matrix.
    /// The covariance matrix becomes: λ * μ * I + (1 - λ) * Σ, where μ = trace(Σ) / n is the
    /// average variance, so the target is on the same scale as Σ.
    /// Must be in `[0, 1]`. Helps mitigate estimation error in empirical covariances.
    /// If `None`, no shrinkage is applied.
    ///
    /// <https://en.wikipedia.org/wiki/Shrinkage_estimator>
    pub shrinkage: Option<f64>,
}

impl CovarianceConfig {
    /// Sets the regularization parameter ε.
    pub fn with_regularization(mut self, epsilon: f64) -> Self {
        self.regularization = Some(epsilon);
        self
    }

    /// Sets the shrinkage intensity λ.
    pub fn with_shrinkage(mut self, lambda: f64) -> Self {
        self.shrinkage = Some(lambda);
        self
    }
}

/// What a mean-variance optimization trades off. See
/// [`PortfolioAllocator::mean_variance_allocation`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MeanVarianceObjective {
    /// Maximize `μᵀw − (γ/2) wᵀΣw` for a risk aversion `γ > 0`. Larger values favor lower
    /// variance; as `γ → ∞` the result approaches the minimum variance portfolio.
    RiskAversion(f64),
    /// Minimize `wᵀΣw` subject to an expected portfolio return `μᵀw` equal to this value.
    TargetReturn(f64),
}

/// Configuration for [`PortfolioAllocator::mean_variance_allocation`].
///
/// Start from [`MeanVarianceConfig::risk_aversion`] or [`MeanVarianceConfig::target_return`]:
///
/// ```
/// use cutup::{CovarianceConfig, MeanVarianceConfig};
/// use nalgebra::dvector;
///
/// let config = MeanVarianceConfig::target_return(0.01)
///     .with_expected_returns(dvector![0.008, 0.012, 0.010])
///     .with_covariance(CovarianceConfig::default().with_shrinkage(0.1));
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MeanVarianceConfig {
    /// The objective to optimize.
    pub objective: MeanVarianceObjective,
    /// Expected return per asset, in the same units and period as the returns. When `None`, the
    /// sample mean of the returns is used.
    pub expected_returns: Option<DVector<f64>>,
    /// Adjustments applied to the covariance matrix.
    pub covariance: CovarianceConfig,
}

impl MeanVarianceConfig {
    /// Maximizes `μᵀw − (γ/2) wᵀΣw` with risk aversion `gamma`.
    pub fn risk_aversion(gamma: f64) -> Self {
        Self::new(MeanVarianceObjective::RiskAversion(gamma))
    }

    /// Minimizes variance subject to an expected portfolio return of `target`.
    pub fn target_return(target: f64) -> Self {
        Self::new(MeanVarianceObjective::TargetReturn(target))
    }

    fn new(objective: MeanVarianceObjective) -> Self {
        Self {
            objective,
            expected_returns: None,
            covariance: CovarianceConfig::default(),
        }
    }

    /// Uses `expected_returns` instead of the sample mean of the returns.
    pub fn with_expected_returns(mut self, expected_returns: DVector<f64>) -> Self {
        self.expected_returns = Some(expected_returns);
        self
    }

    /// Applies `covariance` adjustments to the covariance matrix.
    pub fn with_covariance(mut self, covariance: CovarianceConfig) -> Self {
        self.covariance = covariance;
        self
    }
}

/// `PortfolioAllocator` handles various portfolio allocation strategies.
/// Supports minimum variance, mean-variance optimization, Equal Weight (EW), inverse
/// volatility, inverse variance, and Hierarchical Risk Parity (HRP).
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
    /// * [`CutupError::NonFiniteCovariance`] if the returns are too extreme for a finite covariance.
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
        PortfolioAllocator::build(PortfolioAllocator::compute_returns(&prices))
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
    /// * [`CutupError::NonFiniteCovariance`] if the returns are too extreme for a finite covariance.
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
        PortfolioAllocator::build(returns)
    }

    /// Builds an allocator from returns that have already been validated, rejecting a
    /// covariance matrix that overflowed.
    fn build(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        let cov_matrix = PortfolioAllocator::compute_covariance_matrix(&returns);
        if cov_matrix.iter().any(|x| !x.is_finite()) {
            return Err(CutupError::NonFiniteCovariance);
        }
        Ok(PortfolioAllocator {
            returns,
            cov_matrix,
        })
    }

    /// Returns the first asset whose returns are constant, allowing for floating-point rounding:
    /// a standard deviation below `1e-10` of the asset's largest absolute return counts as zero.
    fn find_zero_variance(&self) -> Option<usize> {
        (0..self.cov_matrix.ncols()).find(|&i| {
            let scale = self.returns.column(i).amax();
            self.cov_matrix[(i, i)].sqrt() <= 1e-10 * scale
        })
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

    /// Applies `config` to the sample covariance matrix, after checking its values.
    ///
    /// Returns [`CutupError::ZeroVariance`] for a constant asset unless shrinkage or
    /// regularization gives it a positive variance.
    fn adjusted_covariance(&self, config: &CovarianceConfig) -> Result<DMatrix<f64>, CutupError> {
        if let Some(value) = config
            .regularization
            .filter(|eps| !(eps.is_finite() && *eps >= 0.0))
        {
            return Err(CutupError::InvalidConfig {
                field: "regularization",
                value,
            });
        }
        if let Some(value) = config
            .shrinkage
            .filter(|lambda| !(0.0..=1.0).contains(lambda))
        {
            return Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value,
            });
        }
        let adjusted =
            config.shrinkage.unwrap_or(0.0) > 0.0 || config.regularization.unwrap_or(0.0) > 0.0;
        if !adjusted {
            if let Some(asset) = self.find_zero_variance() {
                return Err(CutupError::ZeroVariance { asset });
            }
        }

        let n = self.cov_matrix.ncols();
        let identity = DMatrix::identity(n, n);

        let shrunk_cov = if let Some(lambda) = config.shrinkage {
            let mean_variance = self.cov_matrix.trace() / n as f64;
            (lambda * mean_variance) * &identity + (1.0 - lambda) * &self.cov_matrix
        } else {
            self.cov_matrix.clone()
        };

        Ok(if let Some(eps) = config.regularization {
            shrunk_cov + eps * identity
        } else {
            shrunk_cov
        })
    }

    /// Solves `minimize ½ wᵀKw − cᵀw subject to Eᵀw = b` through its KKT system
    /// `[K E; Eᵀ 0] [w; λ] = [c; b]`.
    ///
    /// The system is solved with the pseudo-inverse (singular values below `size · ε · σ_max`
    /// count as zero, matching `numpy.linalg.pinv`), so a singular `K` still gives an exact
    /// optimum: the one with the smallest norm. The solution is then checked against the
    /// system. If it doesn't satisfy it, the problem has no optimum, either because the
    /// objective is unbounded or because the constraints are infeasible, and `Ok(None)` is
    /// returned.
    fn solve_kkt(
        k: &DMatrix<f64>,
        c: &DVector<f64>,
        e: &DMatrix<f64>,
        b: &DVector<f64>,
    ) -> Result<Option<DVector<f64>>, CutupError> {
        // nalgebra's SVD never converges on NaN, so check first and cap the iterations below.
        if k.iter().chain(c.iter()).any(|x| !x.is_finite()) {
            return Err(CutupError::NonFiniteCovariance);
        }
        let (n, m) = (k.nrows(), e.ncols());

        // Put every block on a comparable scale so the pseudo-inverse cutoff treats them alike.
        // Scaling K and c together, or a constraint column of E with its entry of b, leaves the
        // solution unchanged.
        let k_scale = k.trace() / n as f64;
        if !(k_scale.is_finite() && k_scale > 0.0) {
            return Err(CutupError::SingularCovariance);
        }
        let mut kkt = DMatrix::zeros(n + m, n + m);
        let mut rhs = DVector::zeros(n + m);
        kkt.view_mut((0, 0), (n, n)).copy_from(&(k / k_scale));
        rhs.rows_mut(0, n).copy_from(&(c / k_scale));
        for j in 0..m {
            let column_scale = match e.column(j).amax() {
                s if s > 0.0 => s,
                _ => 1.0,
            };
            let column = e.column(j) / column_scale;
            kkt.view_mut((0, n + j), (n, 1)).copy_from(&column);
            kkt.view_mut((n + j, 0), (1, n))
                .copy_from(&column.transpose());
            rhs[n + j] = b[j] / column_scale;
        }

        let size = n + m;
        let svd = kkt
            .clone()
            .try_svd(true, true, f64::EPSILON, 10_000)
            .ok_or(CutupError::SingularCovariance)?;
        let tolerance = svd.singular_values.max() * size as f64 * f64::EPSILON;
        let solution = svd
            .pseudo_inverse(tolerance)
            .map_err(|_| CutupError::SingularCovariance)?
            * &rhs;

        let residual = (&kkt * &solution - &rhs).norm();
        if residual > 1e-8 * (kkt.norm() * solution.norm() + rhs.norm()) {
            return Ok(None);
        }
        let weights = solution.rows(0, n).into_owned();
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(CutupError::SingularCovariance);
        }
        Ok(Some(weights))
    }

    /// Computes the global minimum variance (GMV) portfolio: minimize `wᵀΣw` subject to
    /// `1ᵀw = 1`. When Σ is invertible this is `w = Σ⁻¹1 / 1ᵀΣ⁻¹1`.
    ///
    /// This uses only the covariance matrix, not expected returns, and has no long-only
    /// constraint. For a portfolio that also accounts for expected returns, use
    /// [`PortfolioAllocator::mean_variance_allocation`].
    ///
    /// When Σ is singular (no more rows of returns than assets, or perfectly collinear assets),
    /// many portfolios share the minimum variance and the one with the smallest norm is
    /// returned. That minimum is often zero, an artifact of fitting too little data. Add
    /// shrinkage with [`PortfolioAllocator::min_variance_allocation_with_config`] for a unique,
    /// better-conditioned answer.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, summing to 1. Weights can be negative (short positions).
    ///
    /// # Errors
    ///
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
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
    /// let weights = allocator.min_variance_allocation()?;
    ///
    /// let total: f64 = weights.iter().sum();
    /// assert!((total - 1.0).abs() < 1e-6);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn min_variance_allocation(&self) -> Result<Weights, CutupError> {
        self.min_variance_allocation_with_config(&CovarianceConfig::default())
    }

    /// Computes the global minimum variance portfolio after applying shrinkage and/or
    /// regularization to the covariance matrix. See
    /// [`PortfolioAllocator::min_variance_allocation`].
    ///
    /// # Arguments
    ///
    /// * `config` - Adjustments applied to the covariance matrix.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, summing to 1. Weights can be negative (short positions).
    ///
    /// # Errors
    ///
    /// * [`CutupError::InvalidConfig`] if `regularization` is negative or not finite, or
    ///   `shrinkage` is outside `[0, 1]`.
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
    ///   shrinkage nor regularization is applied.
    /// * [`CutupError::SingularCovariance`] if no finite weights can be computed.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CovarianceConfig, PortfolioAllocator};
    /// use nalgebra::dmatrix;
    ///
    /// let prices = dmatrix![
    ///     100.0, 200.0, 300.0;
    ///     110.0, 210.0, 310.0;
    ///     120.0, 220.0, 320.0
    /// ];
    ///
    /// let allocator = PortfolioAllocator::from_prices(prices)?;
    ///
    /// // Full shrinkage replaces Σ with its average variance times I, giving equal weights.
    /// let config = CovarianceConfig::default().with_shrinkage(1.0);
    /// let weights = allocator.min_variance_allocation_with_config(&config)?;
    ///
    /// for w in &weights {
    ///     assert!((w - 1.0 / 3.0).abs() < 1e-9);
    /// }
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn min_variance_allocation_with_config(
        &self,
        config: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        let cov = self.adjusted_covariance(config)?;
        let n = cov.ncols();
        let ones = DMatrix::from_element(n, 1, 1.0);

        // Variance is bounded below and full investment is always feasible, so an optimum
        // exists; failing to find one is numerical.
        let weights = PortfolioAllocator::solve_kkt(
            &cov,
            &DVector::zeros(n),
            &ones,
            &DVector::from_element(1, 1.0),
        )?
        .ok_or(CutupError::SingularCovariance)?;
        Ok(Weights(weights.iter().copied().collect()))
    }

    /// Computes a Markowitz mean-variance portfolio with the budget constraint `1ᵀw = 1`.
    ///
    /// * [`MeanVarianceObjective::RiskAversion`] `γ`: maximize `μᵀw − (γ/2) wᵀΣw`.
    /// * [`MeanVarianceObjective::TargetReturn`] `ρ`: minimize `wᵀΣw` subject to `μᵀw = ρ`.
    ///
    /// When Σ is invertible, with `S = Σ⁻¹`, `A = 1ᵀS1`, `B = 1ᵀSμ`, `C = μᵀSμ` and
    /// `D = AC − B²`, the solutions are `w = (1/γ) S(μ − η1)` with `η = (B − γ) / A`, and
    /// `w = [(C − Bρ) S1 + (Aρ − B) Sμ] / D` (Merton, 1972). Both are computed through the
    /// KKT system instead, so a singular Σ gives an exact optimum (as in
    /// [`PortfolioAllocator::min_variance_allocation`]) or an error when there is none.
    ///
    /// There is no long-only constraint, so weights can be negative (short positions).
    ///
    /// # Arguments
    ///
    /// * `config` - The objective, optional expected returns (default: the sample mean of the
    ///   returns), and covariance adjustments.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, summing to 1.
    ///
    /// # Errors
    ///
    /// * [`CutupError::InvalidConfig`] if the risk aversion is not finite and positive, the
    ///   target return or an expected return is not finite, or the covariance config is invalid.
    /// * [`CutupError::DimensionMismatch`] if `expected_returns` doesn't have one entry per asset.
    /// * [`CutupError::DegenerateExpectedReturns`] for a target return when every asset has the
    ///   same expected return and the target differs from it.
    /// * [`CutupError::Unbounded`] for a risk aversion when Σ is singular and some fully invested
    ///   portfolio gains expected return without adding variance.
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
    ///   shrinkage nor regularization is applied.
    /// * [`CutupError::SingularCovariance`] if no finite weights can be computed.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MeanVarianceConfig, PortfolioAllocator};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let returns = dmatrix![
    ///      0.010, -0.020,  0.005;
    ///     -0.004,  0.015,  0.002;
    ///      0.007,  0.003, -0.006;
    ///      0.002, -0.001,  0.004;
    ///     -0.003,  0.006,  0.001
    /// ];
    /// let expected = dvector![0.004, 0.006, 0.002];
    ///
    /// let allocator = PortfolioAllocator::from_returns(returns)?;
    /// let config = MeanVarianceConfig::target_return(0.005).with_expected_returns(expected.clone());
    /// let weights = allocator.mean_variance_allocation(&config)?;
    ///
    /// let portfolio_return: f64 = weights.iter().zip(expected.iter()).map(|(w, m)| w * m).sum();
    /// assert!((portfolio_return - 0.005).abs() < 1e-12);
    /// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn mean_variance_allocation(
        &self,
        config: &MeanVarianceConfig,
    ) -> Result<Weights, CutupError> {
        match config.objective {
            MeanVarianceObjective::RiskAversion(gamma) if !(gamma.is_finite() && gamma > 0.0) => {
                return Err(CutupError::InvalidConfig {
                    field: "risk_aversion",
                    value: gamma,
                });
            }
            MeanVarianceObjective::TargetReturn(target) if !target.is_finite() => {
                return Err(CutupError::InvalidConfig {
                    field: "target_return",
                    value: target,
                });
            }
            _ => {}
        }

        let n = self.cov_matrix.ncols();
        let mu = match &config.expected_returns {
            Some(mu) if mu.len() != n => {
                return Err(CutupError::DimensionMismatch {
                    expected: n,
                    actual: mu.len(),
                });
            }
            Some(mu) => {
                if let Some(&value) = mu.iter().find(|m| !m.is_finite()) {
                    return Err(CutupError::InvalidConfig {
                        field: "expected_returns",
                        value,
                    });
                }
                mu.clone()
            }
            None => self.returns.row_mean().transpose(),
        };

        let cov = self.adjusted_covariance(&config.covariance)?;
        let ones = DVector::from_element(n, 1.0);

        let weights = match config.objective {
            MeanVarianceObjective::RiskAversion(gamma) => PortfolioAllocator::solve_kkt(
                &(cov * gamma),
                &mu,
                &DMatrix::from_column_slice(n, 1, ones.as_slice()),
                &DVector::from_element(1, 1.0),
            )?
            .ok_or(CutupError::Unbounded)?,
            MeanVarianceObjective::TargetReturn(target) => PortfolioAllocator::solve_kkt(
                &cov,
                &DVector::zeros(n),
                &DMatrix::from_columns(&[ones, mu]),
                &DVector::from_vec(vec![1.0, target]),
            )?
            .ok_or(CutupError::DegenerateExpectedReturns)?,
        };
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

    /// Weights each asset in proportion to `f(σᵢ²)`, normalized to sum to 1.
    fn weight_by_variance(&self, f: impl Fn(f64) -> f64) -> Result<Weights, CutupError> {
        if let Some(asset) = self.find_zero_variance() {
            return Err(CutupError::ZeroVariance { asset });
        }
        let raw: Vec<f64> = self.cov_matrix.diagonal().iter().map(|&v| f(v)).collect();
        let total: f64 = raw.iter().sum();
        Ok(Weights(raw.iter().map(|w| w / total).collect()))
    }

    /// Computes inverse volatility weights: `wᵢ ∝ 1 / σᵢ`.
    ///
    /// Each asset's weight is inversely proportional to the standard deviation of its returns,
    /// so riskier assets get less. Correlations are ignored. Compared with
    /// [`PortfolioAllocator::inverse_variance_allocation`], the tilt away from risky assets is
    /// milder. When assets are uncorrelated, this gives each asset the same risk contribution.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, positive and summing to 1.
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
    /// // Volatilities in the ratio 1 : 2.
    /// let returns = dmatrix![
    ///      0.01,  0.02;
    ///     -0.01, -0.02;
    ///      0.01, -0.02;
    ///     -0.01,  0.02
    /// ];
    ///
    /// let weights = PortfolioAllocator::from_returns(returns)?.inverse_volatility_allocation()?;
    ///
    /// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
    /// assert!((weights[1] - 1.0 / 3.0).abs() < 1e-12);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn inverse_volatility_allocation(&self) -> Result<Weights, CutupError> {
        self.weight_by_variance(|variance| 1.0 / variance.sqrt())
    }

    /// Computes inverse variance weights: `wᵢ ∝ 1 / σᵢ²`.
    ///
    /// Each asset's weight is inversely proportional to the variance of its returns.
    /// Correlations are ignored. When assets are uncorrelated this is the minimum variance
    /// portfolio, and it is what [`PortfolioAllocator::hrp_allocation`] uses within each
    /// cluster.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, positive and summing to 1.
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
    /// // Volatilities in the ratio 1 : 2, so variances in the ratio 1 : 4.
    /// let returns = dmatrix![
    ///      0.01,  0.02;
    ///     -0.01, -0.02;
    ///      0.01, -0.02;
    ///     -0.01,  0.02
    /// ];
    ///
    /// let weights = PortfolioAllocator::from_returns(returns)?.inverse_variance_allocation()?;
    ///
    /// assert!((weights[0] - 0.8).abs() < 1e-12);
    /// assert!((weights[1] - 0.2).abs() < 1e-12);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn inverse_variance_allocation(&self) -> Result<Weights, CutupError> {
        self.weight_by_variance(|variance| 1.0 / variance)
    }

    /// Orders assets so that correlated assets sit next to each other: the leaf order of a
    /// single-linkage clustering on the correlation distance `dᵢⱼ = sqrt((1 − ρᵢⱼ) / 2)`.
    ///
    /// Follows SciPy's `linkage(method="single")` and `to_tree().pre_order()` step for step,
    /// including the tie-breaking rule, so the order matches PyPortfolioOpt's `HRPOpt`:
    ///
    /// 1. Prim's algorithm builds the minimum spanning tree, starting from asset 0 and taking
    ///    the lowest-index asset among equally distant ones.
    /// 2. The tree's edges are stably sorted by distance and merged in that order. Merge `k`
    ///    creates cluster `n + k`, whose left child is whichever of the two merged clusters has
    ///    the smaller id.
    /// 3. A pre-order traversal from the root lists the assets left to right.
    ///
    /// Returns [`CutupError::NonFiniteCovariance`] if a correlation can't be computed, for
    /// example when variances are so small that their product underflows to zero.
    fn quasi_diagonal_order(&self) -> Result<Vec<usize>, CutupError> {
        let cov = &self.cov_matrix;
        let n = cov.ncols();
        let distance = DMatrix::from_fn(n, n, |i, j| {
            let rho = cov[(i, j)] / (cov[(i, i)] * cov[(j, j)]).sqrt();
            // Mark a non-finite correlation as NaN: clamping would hide an infinite one.
            if rho.is_finite() {
                ((1.0 - rho) / 2.0).clamp(0.0, 1.0).sqrt()
            } else {
                f64::NAN
            }
        });
        // A NaN distance would be skipped by the minimum below and corrupt the tree.
        if distance.iter().any(|d| d.is_nan()) {
            return Err(CutupError::NonFiniteCovariance);
        }

        let mut edges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1);
        let mut distance_to_tree = vec![f64::INFINITY; n];
        let mut in_tree = vec![false; n];
        let mut x = 0;
        for _ in 1..n {
            in_tree[x] = true;
            let (mut y, mut nearest) = (0, f64::INFINITY);
            for (i, to_tree) in distance_to_tree.iter_mut().enumerate() {
                if in_tree[i] {
                    continue;
                }
                *to_tree = to_tree.min(distance[(x, i)]);
                if *to_tree < nearest {
                    (y, nearest) = (i, *to_tree);
                }
            }
            edges.push((x, y, nearest));
            x = y;
        }
        edges.sort_by(|a, b| a.2.total_cmp(&b.2));

        let mut parent: Vec<usize> = (0..2 * n - 1).collect();
        let find = |parent: &mut Vec<usize>, mut node: usize| {
            while parent[node] != node {
                parent[node] = parent[parent[node]];
                node = parent[node];
            }
            node
        };
        let mut children = Vec::with_capacity(n - 1);
        for (k, &(a, b, _)) in edges.iter().enumerate() {
            let (root_a, root_b) = (find(&mut parent, a), find(&mut parent, b));
            parent[root_a] = n + k;
            parent[root_b] = n + k;
            children.push((root_a.min(root_b), root_a.max(root_b)));
        }

        let mut order = Vec::with_capacity(n);
        let mut stack = vec![2 * n - 2];
        while let Some(node) = stack.pop() {
            if node < n {
                order.push(node);
            } else {
                let (left, right) = children[node - n];
                stack.push(right);
                stack.push(left);
            }
        }
        Ok(order)
    }

    /// Returns the variance of the inverse-variance portfolio over `cluster`.
    fn cluster_variance(&self, cluster: &[usize]) -> f64 {
        let inverse: Vec<f64> = cluster
            .iter()
            .map(|&i| 1.0 / self.cov_matrix[(i, i)])
            .collect();
        let total: f64 = inverse.iter().sum();
        let mut variance = 0.0;
        for (a, &i) in cluster.iter().enumerate() {
            for (b, &j) in cluster.iter().enumerate() {
                variance += inverse[a] * inverse[b] * self.cov_matrix[(i, j)];
            }
        }
        variance / (total * total)
    }

    /// Computes Hierarchical Risk Parity (HRP) portfolio allocation (López de Prado, 2016).
    ///
    /// 1. **Cluster.** Assets are clustered by single linkage on the correlation distance
    ///    `sqrt((1 − ρ) / 2)`, and ordered so that correlated assets sit next to each other.
    /// 2. **Recursive bisection.** The ordered list is split in half, repeatedly. At each
    ///    split, the two halves share their parent's weight in inverse proportion to their
    ///    risk: each half's variance is that of its own inverse-variance portfolio, and the
    ///    first half gets `α = 1 − V₁ / (V₁ + V₂)`.
    ///
    /// The clustering follows SciPy and PyPortfolioOpt's `HRPOpt` step for step, including the
    /// tie-breaking rule, so results match `HRPOpt(returns).optimize()` up to floating-point
    /// rounding. (The correlations are computed here rather than by pandas, so two distances
    /// that tie in one may differ by a rounding error in the other and break the tie the other
    /// way.)
    ///
    /// HRP never inverts the covariance matrix, so it works when Σ is singular. Weights are
    /// never negative. A half with zero variance, such as a perfectly hedged pair, takes its
    /// parent's whole weight and leaves the other half at zero, as in `HRPOpt`.
    ///
    /// # Returns
    ///
    /// * [`Weights`], one per asset, summing to 1.
    ///
    /// # Errors
    ///
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
    /// * [`CutupError::NonFiniteCovariance`] if correlations can't be computed because the
    ///   variances are too small.
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
    /// assert!(weights.iter().all(|&w| w >= 0.0));
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn hrp_allocation(&self) -> Result<Weights, CutupError> {
        if let Some(asset) = self.find_zero_variance() {
            return Err(CutupError::ZeroVariance { asset });
        }

        let order = self.quasi_diagonal_order()?;
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
                let v_left = self.cluster_variance(left).max(0.0);
                let v_right = self.cluster_variance(right).max(0.0);
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

        Ok(Weights(weights))
    }
}

/// Runs a global minimum variance allocation on a matrix of prices.
///
/// # Arguments
///
/// * `prices` - A `DMatrix<f64>` of asset price history.
///
/// # Returns
///
/// * [`Weights`] from [`PortfolioAllocator::min_variance_allocation`].
///
/// # Errors
///
/// * Any error from [`PortfolioAllocator::from_prices`] or
///   [`PortfolioAllocator::min_variance_allocation`].
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
    PortfolioAllocator::from_prices(prices)?.min_variance_allocation()
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
        assert_eq!(
            CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            }
            .to_string(),
            "config value `shrinkage` is out of range: 1.5"
        );
        assert_eq!(
            CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            }
            .to_string(),
            "expected one value per asset (3) but got 2"
        );
        assert_eq!(
            CutupError::Unbounded.to_string(),
            "objective is unbounded: expected return can grow without adding variance"
        );
        assert_eq!(
            CutupError::DegenerateExpectedReturns.to_string(),
            "every asset has the same expected return, so no other target return can be reached"
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
            &base.min_variance_allocation().unwrap(),
            &rescaled.min_variance_allocation().unwrap(),
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
    fn test_min_variance_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let weights = allocator.min_variance_allocation().unwrap();
        assert_eq!(weights.len(), 3);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_variance_allocation_errors_on_non_finite_covariance() {
        let mut allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();

        // Unreachable through the constructors, which reject non-finite input; this guards the
        // pseudo-inverse, whose SVD would otherwise never converge.
        allocator.cov_matrix = dmatrix![
            f64::NAN, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ];
        assert_eq!(
            allocator.min_variance_allocation().err(),
            Some(CutupError::NonFiniteCovariance)
        );

        allocator.cov_matrix = DMatrix::from_element(3, 3, f64::NAN);
        assert_eq!(
            allocator.min_variance_allocation().err(),
            Some(CutupError::NonFiniteCovariance)
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
        assert_eq!(
            allocator.min_variance_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 1 })
        );
        assert_eq!(allocator.ew_allocation().len(), 3);
    }

    #[test]
    fn test_constructors_reject_overflowing_covariance() {
        let prices = dmatrix![100.0, 50.0; 1e300, 51.0; 100.0, 50.5];
        assert_eq!(
            PortfolioAllocator::from_prices(prices).err(),
            Some(CutupError::NonFiniteCovariance)
        );

        let returns = dmatrix![1e200, 0.01; -1e200, 0.02];
        assert_eq!(
            PortfolioAllocator::from_returns(returns).err(),
            Some(CutupError::NonFiniteCovariance)
        );
    }

    #[test]
    fn test_rounding_noise_counts_as_zero_variance() {
        // Constant returns whose computed variance is rounding noise rather than exactly zero.
        let allocator =
            PortfolioAllocator::from_returns(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            allocator.hrp_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
        assert_eq!(
            allocator.min_variance_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );

        // Prices growing at a constant 10% per period.
        let prices = dmatrix![100.0, 50.0; 110.0, 49.0; 121.0, 51.5; 133.1, 51.0];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        assert_eq!(
            allocator.hrp_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
    }

    #[test]
    fn test_min_variance_allocation_rejects_invalid_config() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let cases = [
            (Some(-1e-3), None, "regularization"),
            (Some(f64::NAN), None, "regularization"),
            (Some(f64::INFINITY), None, "regularization"),
            (None, Some(-0.1), "shrinkage"),
            (None, Some(1.5), "shrinkage"),
            (None, Some(f64::NAN), "shrinkage"),
        ];
        for (regularization, shrinkage, field) in cases {
            let config = CovarianceConfig {
                regularization,
                shrinkage,
            };
            match allocator.min_variance_allocation_with_config(&config) {
                Err(CutupError::InvalidConfig { field: f, .. }) => assert_eq!(f, field),
                other => panic!("{config:?} gave {other:?}"),
            }
        }

        let edges = CovarianceConfig {
            regularization: Some(0.0),
            shrinkage: Some(1.0),
        };
        assert!(allocator
            .min_variance_allocation_with_config(&edges)
            .is_ok());
    }

    fn portfolio_variance(allocator: &PortfolioAllocator, weights: &Weights) -> f64 {
        let w = DVector::from_column_slice(weights.as_slice());
        (w.transpose() * &allocator.cov_matrix * &w)[(0, 0)]
    }

    fn sample_mean(allocator: &PortfolioAllocator) -> DVector<f64> {
        allocator.returns.row_mean().transpose()
    }

    #[test]
    fn test_covariance_config_builder() {
        assert_eq!(
            CovarianceConfig::default(),
            CovarianceConfig {
                regularization: None,
                shrinkage: None
            }
        );
        let config = CovarianceConfig::default()
            .with_regularization(1e-8)
            .with_shrinkage(0.25);
        assert_eq!(config.regularization, Some(1e-8));
        assert_eq!(config.shrinkage, Some(0.25));
    }

    #[test]
    fn test_full_shrinkage_gives_equal_weights() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().with_shrinkage(1.0);
        let weights = allocator
            .min_variance_allocation_with_config(&config)
            .unwrap();
        assert_weights_close(&weights, &allocator.ew_allocation(), 1e-12);
    }

    #[test]
    fn test_shrinkage_is_scale_invariant() {
        // The shrinkage target is the average variance times I, so rescaling every return by
        // the same factor must leave the weights unchanged. A target of plain I would not.
        let returns = PortfolioAllocator::compute_returns(&full_rank_prices());
        let config = CovarianceConfig::default().with_shrinkage(0.5);

        let base = PortfolioAllocator::from_returns(returns.clone()).unwrap();
        let scaled = PortfolioAllocator::from_returns(returns * 100.0).unwrap();

        assert_weights_close(
            &base.min_variance_allocation_with_config(&config).unwrap(),
            &scaled.min_variance_allocation_with_config(&config).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_min_variance_diagonal_closed_form() {
        // Mutually orthogonal, zero-mean columns give a diagonal covariance, so the minimum
        // variance weights are proportional to 1 / σᵢ².
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let weights = allocator.min_variance_allocation().unwrap();

        let inverse = [1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)];
        let total: f64 = inverse.iter().sum();
        let expected = Weights(inverse.iter().map(|x| x / total).collect());
        assert_weights_close(&weights, &expected, 1e-12);
    }

    #[test]
    fn test_min_variance_two_asset_closed_form() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let weights = allocator.min_variance_allocation().unwrap();

        let cov = &allocator.cov_matrix;
        let (s11, s22, s12) = (cov[(0, 0)], cov[(1, 1)], cov[(0, 1)]);
        let w1 = (s22 - s12) / (s11 + s22 - 2.0 * s12);
        assert_weights_close(&weights, &Weights(vec![w1, 1.0 - w1]), 1e-12);
    }

    #[test]
    fn test_min_variance_beats_equal_weight() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let min_variance = allocator.min_variance_allocation().unwrap();
        let equal = allocator.ew_allocation();
        assert!(
            portfolio_variance(&allocator, &min_variance) <= portfolio_variance(&allocator, &equal)
        );
    }

    #[test]
    fn test_min_variance_rank_deficient_covariance() {
        // Three rows of returns for four assets: Σ has rank at most 2. The pseudo-inverse must
        // still give finite weights that sum to 1 and do no worse than equal weight.
        let prices = dmatrix![
            125.0, 1500.0, 210.0, 600.0;
            123.0, 1520.0, 215.0, 620.0;
            130.0, 1510.0, 220.0, 610.0;
            128.0, 1530.0, 225.0, 630.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let weights = allocator.min_variance_allocation().unwrap();

        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(weights.iter().all(|w| w.is_finite() && w.abs() < 10.0));
        assert!(
            portfolio_variance(&allocator, &weights)
                <= portfolio_variance(&allocator, &allocator.ew_allocation()) + 1e-15
        );
    }

    #[test]
    fn test_mean_variance_target_return() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let mu = sample_mean(&allocator);
        let target = mu.mean() * 1.5;
        let weights = allocator
            .mean_variance_allocation(&MeanVarianceConfig::target_return(target))
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!((mu.dot(&w) - target).abs() < 1e-12);
    }

    #[test]
    fn test_mean_variance_target_at_min_variance_return_is_min_variance() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let min_variance = allocator.min_variance_allocation().unwrap();
        let mu = sample_mean(&allocator);
        let target = mu.dot(&DVector::from_column_slice(min_variance.as_slice()));

        let weights = allocator
            .mean_variance_allocation(&MeanVarianceConfig::target_return(target))
            .unwrap();
        assert_weights_close(&weights, &min_variance, 1e-9);
    }

    #[test]
    fn test_mean_variance_risk_aversion_first_order_condition() {
        // At the optimum of μᵀw − (γ/2) wᵀΣw subject to 1ᵀw = 1, μ − γΣw is a multiple of 1.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let gamma = 3.0;
        let weights = allocator
            .mean_variance_allocation(&MeanVarianceConfig::risk_aversion(gamma))
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        let gradient = sample_mean(&allocator) - &allocator.cov_matrix * &w * gamma;
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!(gradient.max() - gradient.min() < 1e-12);
    }

    #[test]
    fn test_mean_variance_high_risk_aversion_approaches_min_variance() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let weights = allocator
            .mean_variance_allocation(&MeanVarianceConfig::risk_aversion(1e9))
            .unwrap();
        assert_weights_close(
            &weights,
            &allocator.min_variance_allocation().unwrap(),
            1e-6,
        );
    }

    #[test]
    fn test_mean_variance_equal_expected_returns() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let flat = DVector::from_element(3, 0.01);

        // With nothing to choose between on return, risk aversion reduces to minimum variance.
        let config = MeanVarianceConfig::risk_aversion(2.0).with_expected_returns(flat.clone());
        assert_weights_close(
            &allocator.mean_variance_allocation(&config).unwrap(),
            &allocator.min_variance_allocation().unwrap(),
            1e-9,
        );

        // And no portfolio can target any other return.
        let config = MeanVarianceConfig::target_return(0.02).with_expected_returns(flat);
        assert_eq!(
            allocator.mean_variance_allocation(&config).err(),
            Some(CutupError::DegenerateExpectedReturns)
        );
    }

    #[test]
    fn test_mean_variance_defaults_to_sample_mean() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let default = MeanVarianceConfig::risk_aversion(5.0);
        let explicit = default
            .clone()
            .with_expected_returns(sample_mean(&allocator));
        assert_eq!(
            allocator.mean_variance_allocation(&default).unwrap(),
            allocator.mean_variance_allocation(&explicit).unwrap()
        );
    }

    #[test]
    fn test_mean_variance_applies_covariance_config() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let plain = MeanVarianceConfig::risk_aversion(5.0);
        let shrunk = plain
            .clone()
            .with_covariance(CovarianceConfig::default().with_shrinkage(0.5));
        assert_ne!(
            allocator.mean_variance_allocation(&plain).unwrap(),
            allocator.mean_variance_allocation(&shrunk).unwrap()
        );

        let invalid = plain.with_covariance(CovarianceConfig::default().with_shrinkage(2.0));
        assert_eq!(
            allocator.mean_variance_allocation(&invalid).err(),
            Some(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 2.0
            })
        );
    }

    #[test]
    fn test_mean_variance_rejects_invalid_config() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        for gamma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match allocator.mean_variance_allocation(&MeanVarianceConfig::risk_aversion(gamma)) {
                Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "risk_aversion"),
                other => panic!("gamma {gamma} gave {other:?}"),
            }
        }
        match allocator.mean_variance_allocation(&MeanVarianceConfig::target_return(f64::NAN)) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "target_return"),
            other => panic!("NaN target gave {other:?}"),
        }

        let config = MeanVarianceConfig::target_return(0.01)
            .with_expected_returns(DVector::from_vec(vec![0.01, f64::NAN, 0.02]));
        match allocator.mean_variance_allocation(&config) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "expected_returns"),
            other => panic!("NaN expected return gave {other:?}"),
        }

        let config = MeanVarianceConfig::target_return(0.01)
            .with_expected_returns(DVector::from_vec(vec![0.01, 0.02]));
        assert_eq!(
            allocator.mean_variance_allocation(&config).err(),
            Some(CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn test_mean_variance_rejects_zero_variance() {
        let allocator =
            PortfolioAllocator::from_returns(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            allocator
                .mean_variance_allocation(&MeanVarianceConfig::risk_aversion(1.0))
                .err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
    }

    /// Two rows of returns for three assets: Σ has rank 1, and the fully invested portfolio
    /// `[4/7, 0, 3/7]` has zero variance.
    fn rank_one_returns() -> DMatrix<f64> {
        dmatrix![
            0.01, 0.02, -0.01;
            -0.02, 0.01, 0.03
        ]
    }

    #[test]
    fn test_min_variance_finds_zero_variance_portfolio_on_singular_covariance() {
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let weights = allocator.min_variance_allocation().unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_min_variance_anti_correlated_pair() {
        // 1 lies in Σ's null space, so 1ᵀΣ⁺1 is rounding noise; the hedged pair has zero variance.
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let weights = allocator.min_variance_allocation().unwrap();
        assert_weights_close(&weights, &Weights(vec![0.5, 0.5]), 1e-12);
    }

    #[test]
    fn test_mean_variance_unbounded_on_singular_covariance() {
        // With Σ singular, d = [5, -7, 2] has Σd = 0 and 1ᵀd = 0, but μᵀd ≠ 0 for the sample
        // mean, so utility grows without limit along d.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let config = MeanVarianceConfig::risk_aversion(2.0);
        assert_eq!(
            allocator.mean_variance_allocation(&config).err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = config.with_covariance(CovarianceConfig::default().with_shrinkage(0.1));
        assert!(allocator.mean_variance_allocation(&shrunk).is_ok());
    }

    #[test]
    fn test_mean_variance_target_on_singular_covariance() {
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let target = 0.004;
        let weights = allocator
            .mean_variance_allocation(&MeanVarianceConfig::target_return(target))
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!((sample_mean(&allocator).dot(&w) - target).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_mean_variance_target_equal_to_common_return() {
        // Every fully invested portfolio earns the common return, so the target is met by the
        // minimum variance portfolio.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = MeanVarianceConfig::target_return(0.01)
            .with_expected_returns(DVector::from_element(3, 0.01));
        assert_weights_close(
            &allocator.mean_variance_allocation(&config).unwrap(),
            &allocator.min_variance_allocation().unwrap(),
            1e-9,
        );
    }

    #[test]
    fn test_covariance_adjustment_allows_constant_asset() {
        let returns = dmatrix![0.1, 0.01, 0.02; 0.1, -0.02, 0.01; 0.1, 0.03, -0.01];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(
            allocator.min_variance_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );

        let shrunk = CovarianceConfig::default().with_shrinkage(1.0);
        assert_weights_close(
            &allocator
                .min_variance_allocation_with_config(&shrunk)
                .unwrap(),
            &allocator.ew_allocation(),
            1e-12,
        );

        let regularized = CovarianceConfig::default().with_regularization(1e-6);
        let weights = allocator
            .min_variance_allocation_with_config(&regularized)
            .unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let config = MeanVarianceConfig::risk_aversion(2.0).with_covariance(shrunk);
        assert!(allocator.mean_variance_allocation(&config).is_ok());
    }

    /// Five assets driven by two factors, interleaved: assets 0, 2 and 4 follow the first
    /// factor and assets 1 and 3 the second.
    fn two_block_returns() -> DMatrix<f64> {
        let f1 = [0.010, -0.010, 0.020, -0.020, 0.005, -0.005, 0.012, -0.008];
        let f2 = [0.010, 0.010, -0.010, -0.010, 0.003, -0.003, -0.006, 0.009];
        let noise = [0.0011, -0.0007, 0.0004, -0.0013, 0.0009];
        DMatrix::from_fn(8, 5, |t, j| {
            let factor = if j % 2 == 0 { f1[t] } else { f2[t] };
            factor + noise[j] * ((t * (j + 2)) % 5) as f64
        })
    }

    #[test]
    fn test_hrp_matches_pyportfolioopt() {
        // HRPOpt(returns).optimize() on the returns of full_rank_prices, with PyPortfolioOpt
        // 1.6.0 and SciPy 1.17.1.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let expected = Weights(vec![
            0.43205598172341897,
            0.23653580200415913,
            0.3314082162724219,
        ]);
        assert_weights_close(&allocator.hrp_allocation().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn test_hrp_two_assets_is_inverse_variance() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let (v1, v2) = (allocator.cov_matrix[(0, 0)], allocator.cov_matrix[(1, 1)]);
        let w1 = (1.0 / v1) / (1.0 / v1 + 1.0 / v2);
        assert_weights_close(
            &allocator.hrp_allocation().unwrap(),
            &Weights(vec![w1, 1.0 - w1]),
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
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let inverse = [1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)];
        let total: f64 = inverse.iter().sum();
        let expected = Weights(inverse.iter().map(|x| x / total).collect());
        assert_weights_close(&allocator.hrp_allocation().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn test_hrp_order_groups_correlated_assets() {
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
        let order = allocator.quasi_diagonal_order().unwrap();

        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);

        let position = |asset: usize| order.iter().position(|&a| a == asset).unwrap();
        let mut first_block: Vec<usize> = [0, 2, 4].iter().map(|&a| position(a)).collect();
        first_block.sort_unstable();
        assert_eq!(first_block[2] - first_block[0], 2, "order {order:?}");
    }

    #[test]
    fn test_hrp_ignores_column_order() {
        // Index-order bisection fails this: the clustering, not the input order, decides how
        // assets are grouped.
        let returns = two_block_returns();
        let permutation = [3, 0, 4, 1, 2];
        let permuted = DMatrix::from_fn(returns.nrows(), 5, |t, j| returns[(t, permutation[j])]);

        let base = PortfolioAllocator::from_returns(returns)
            .unwrap()
            .hrp_allocation()
            .unwrap();
        let shuffled = PortfolioAllocator::from_returns(permuted)
            .unwrap()
            .hrp_allocation()
            .unwrap();

        for (j, &asset) in permutation.iter().enumerate() {
            assert!((shuffled[j] - base[asset]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hrp_single_asset() {
        let allocator = PortfolioAllocator::from_prices(dmatrix![100.0; 110.0; 105.0]).unwrap();
        assert_eq!(allocator.quasi_diagonal_order().unwrap(), vec![0]);
        assert_eq!(allocator.hrp_allocation().unwrap().as_slice(), &[1.0]);
    }

    #[test]
    fn test_hrp_singular_covariance() {
        // HRP never inverts Σ, so rank-deficient data still gives positive weights.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let weights = allocator.hrp_allocation().unwrap();
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
        let weights = PortfolioAllocator::from_returns(returns)
            .unwrap()
            .hrp_allocation()
            .unwrap();

        assert!((weights[0] - 0.5).abs() < 1e-12 && (weights[1] - 0.5).abs() < 1e-12);
        assert!(weights[2] < 1e-12 && weights[3] < 1e-12);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_hrp_rejects_underflowing_correlation() {
        // Variances around 1e-320 pass validation, but their product underflows to zero, so ρ
        // is infinite (nonzero covariance) or NaN (zero covariance, orthogonal columns).
        let s = 1e-160;
        let infinite = dmatrix![s, 2.0 * s; -s, 3.0 * s; 2.0 * s, -s];
        let nan = dmatrix![s, s; -s, s; s, -s; -s, -s];
        for returns in [infinite, nan] {
            let allocator = PortfolioAllocator::from_returns(returns).unwrap();
            assert_eq!(
                allocator.hrp_allocation().err(),
                Some(CutupError::NonFiniteCovariance)
            );
        }
    }

    #[test]
    fn test_hrp_anti_correlated_pair_splits_evenly() {
        // A perfectly anti-correlated pair with equal variances: the two assets are as far apart
        // as possible, and each half carries the same risk.
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_weights_close(
            &allocator.hrp_allocation().unwrap(),
            &Weights(vec![0.5, 0.5]),
            1e-12,
        );
    }

    #[test]
    fn test_inverse_volatility_and_variance_closed_form() {
        // Orthogonal columns with volatilities s1, s2, s3.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();

        let normalize = |raw: [f64; 3]| {
            let total: f64 = raw.iter().sum();
            Weights(raw.iter().map(|x| x / total).collect())
        };
        assert_weights_close(
            &allocator.inverse_volatility_allocation().unwrap(),
            &normalize([1.0 / s1, 1.0 / s2, 1.0 / s3]),
            1e-12,
        );
        assert_weights_close(
            &allocator.inverse_variance_allocation().unwrap(),
            &normalize([1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)]),
            1e-12,
        );
    }

    #[test]
    fn test_inverse_variance_matches_min_variance_and_hrp_when_uncorrelated() {
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let inverse_variance = allocator.inverse_variance_allocation().unwrap();
        assert_weights_close(
            &inverse_variance,
            &allocator.min_variance_allocation().unwrap(),
            1e-12,
        );
        assert_weights_close(
            &inverse_variance,
            &allocator.hrp_allocation().unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_inverse_volatility_equalizes_risk_when_uncorrelated() {
        // With a diagonal Σ, asset i contributes wᵢ²σᵢ² of the variance, so wᵢ ∝ 1/σᵢ makes
        // every contribution equal.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let weights = allocator.inverse_volatility_allocation().unwrap();
        let contributions: Vec<f64> = (0..3)
            .map(|i| weights[i] * weights[i] * allocator.cov_matrix[(i, i)])
            .collect();
        for c in &contributions {
            assert!((c - contributions[0]).abs() < 1e-18, "{contributions:?}");
        }
    }

    #[test]
    fn test_inverse_weights_ignore_return_scale_and_stay_positive() {
        let returns = PortfolioAllocator::compute_returns(&full_rank_prices());
        let base = PortfolioAllocator::from_returns(returns.clone()).unwrap();
        let scaled = PortfolioAllocator::from_returns(returns * 50.0).unwrap();

        for (a, b) in [
            (
                base.inverse_volatility_allocation().unwrap(),
                scaled.inverse_volatility_allocation().unwrap(),
            ),
            (
                base.inverse_variance_allocation().unwrap(),
                scaled.inverse_variance_allocation().unwrap(),
            ),
        ] {
            assert_weights_close(&a, &b, 1e-12);
            assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(a.iter().all(|&w| w > 0.0));
        }
    }

    #[test]
    fn test_inverse_weights_reject_zero_variance() {
        let allocator =
            PortfolioAllocator::from_returns(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            allocator.inverse_volatility_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
        assert_eq!(
            allocator.inverse_variance_allocation().err(),
            Some(CutupError::ZeroVariance { asset: 0 })
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
    fn test_min_variance_allocation_with_config_variants() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();

        let configs = [
            // 1. No regularization or shrinkage
            CovarianceConfig::default(),
            // 2. Regularization only
            CovarianceConfig {
                regularization: Some(1e-6),
                shrinkage: None,
            },
            // 3. Shrinkage only
            CovarianceConfig {
                regularization: None,
                shrinkage: Some(0.1),
            },
            // 4. Both regularization and shrinkage
            CovarianceConfig {
                regularization: Some(1e-6),
                shrinkage: Some(0.2),
            },
        ];

        for config in &configs {
            let weights = allocator
                .min_variance_allocation_with_config(config)
                .unwrap();
            assert_eq!(weights.len(), 3, "{config:?}");
            assert!(
                (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
                "{config:?}"
            );
        }
    }
}
