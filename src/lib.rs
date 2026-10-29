#![doc = include_str!("../README.md")]
#![warn(missing_docs)]

/// The `covariance`, `shrinkage` and `regularization` setters, for a method with a
/// `covariance: CovarianceConfig` field.
macro_rules! covariance_setters {
    () => {
        /// Applies `covariance` adjustments to the covariance matrix.
        pub fn covariance(mut self, covariance: CovarianceConfig) -> Self {
            self.covariance = covariance;
            self
        }

        /// Shrinks the covariance toward a scaled identity with intensity `lambda`. See
        /// [`CovarianceConfig::shrinkage`].
        pub fn shrinkage(mut self, lambda: f64) -> Self {
            self.covariance = self.covariance.with_shrinkage(lambda);
            self
        }

        /// Adds `epsilon · I` to the covariance. See [`CovarianceConfig::regularization`].
        pub fn regularization(mut self, epsilon: f64) -> Self {
            self.covariance = self.covariance.with_regularization(epsilon);
            self
        }
    };
}

/// The `current_weights` and `max_turnover` setters, for a method with a `trading: Trading`
/// field.
macro_rules! turnover_setters {
    () => {
        /// Sets your current weights, one per asset in column order: what turnover and
        /// trading costs are measured against. They must sum to 1, within `1e-9`; to hold cash,
        /// include it as an asset. On their own they change nothing.
        pub fn current_weights(mut self, current: impl Into<Vec<f64>>) -> Self {
            self.trading.current = Some(current.into());
            self
        }

        /// Caps the turnover `Σᵢ|wᵢ − w₀ᵢ|`, the total weight bought and sold, at `cap`.
        /// Needs [current weights](Self::current_weights). A cap of 0 keeps the current
        /// weights, if they're within the bounds.
        pub fn max_turnover(mut self, cap: f64) -> Self {
            self.trading.max_turnover = Some(cap);
            self
        }
    };
}

mod allocation;
mod black_litterman;
mod cluster;
mod cvar;
mod eigen;
mod lp;
mod qp;
mod turnover;

pub use allocation::{
    Allocation, EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility, MaxDiversification,
    MaxSharpe, MeanVariance, MinVariance, RiskParity,
};
pub use black_litterman::{BlackLitterman, View};
pub use cluster::Linkage;
pub use cvar::MinCvar;

use eigen::SymmetricEigen;

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
    /// The risk-free rate is at or above (or within rounding of) the minimum variance
    /// portfolio's expected return, so no fully invested portfolio attains the maximum Sharpe
    /// ratio: it is only approached as positions grow without limit.
    RiskFreeRateTooHigh {
        /// The risk-free rate supplied.
        risk_free_rate: f64,
        /// The minimum variance portfolio's expected return, which the rate must be below.
        min_variance_return: f64,
    },
    /// An iterative solver stopped before reaching a solution, although one exists. This
    /// happens only with extreme inputs, such as risk budgets more than about 10¹⁴ apart.
    NotConverged,
    /// No long-only portfolio has the target return: it must be between the lowest and the
    /// highest expected return of any asset, or of any portfolio within the weight bounds.
    TargetReturnOutOfRange {
        /// The target return supplied.
        target_return: f64,
        /// The lowest expected return of any long-only portfolio within the bounds.
        lowest: f64,
        /// The highest expected return of any long-only portfolio within the bounds.
        highest: f64,
    },
    /// No asset's expected return is above the risk-free rate by more than rounding, so no
    /// long-only portfolio has a positive Sharpe ratio.
    NoPositiveExcessReturn {
        /// The risk-free rate supplied.
        risk_free_rate: f64,
        /// The highest expected return of any asset, which the rate must be below.
        highest_return: f64,
    },
    /// No portfolio summing to 1 satisfies the weight bounds: the lower bounds sum to more
    /// than 1, or the upper bounds to less.
    InfeasibleBounds {
        /// The sum of the lower bounds.
        lower_sum: f64,
        /// The sum of the upper bounds.
        upper_sum: f64,
    },
    /// An [`Allocation`] returned a weight that is NaN or infinite.
    NonFiniteWeight {
        /// The asset's column index.
        asset: usize,
    },
    /// The views given to [`BlackLitterman`] are degenerate: certain views that repeat or
    /// contradict each other, or a view whose variance follows its portfolio's on a portfolio
    /// with no variance, which would make it certain. Give such a view an explicit
    /// [`View::variance`].
    DegenerateViews,
    /// A turnover cap, turnover penalty or transaction costs were set without current weights
    /// to measure them against.
    MissingCurrentWeights,
    /// No portfolio within the weight bounds can be reached from the current weights within
    /// the turnover cap: moving the current weights into the bounds alone turns over more.
    InfeasibleTurnover {
        /// The least turnover that reaches the bounds.
        minimum: f64,
        /// The turnover cap.
        cap: f64,
    },
    /// A setting isn't supported in combination with another yet.
    NotSupported {
        /// The setting.
        setting: &'static str,
        /// What it can't be combined with.
        with: &'static str,
    },
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
            CutupError::RiskFreeRateTooHigh {
                risk_free_rate,
                min_variance_return,
            } => write!(
                f,
                "risk-free rate {risk_free_rate} must be below the minimum variance portfolio's \
                 expected return {min_variance_return}"
            ),
            CutupError::NotConverged => {
                write!(f, "the solver stopped before reaching a solution")
            }
            CutupError::TargetReturnOutOfRange {
                target_return,
                lowest,
                highest,
            } => write!(
                f,
                "target return {target_return} is outside the reachable expected returns, \
                 from {lowest} to {highest}, so no long-only portfolio reaches it"
            ),
            CutupError::NoPositiveExcessReturn {
                risk_free_rate,
                highest_return,
            } => write!(
                f,
                "risk-free rate {risk_free_rate} must be below the highest expected return \
                 {highest_return} for a long-only portfolio to have a positive Sharpe ratio"
            ),
            CutupError::InfeasibleBounds {
                lower_sum,
                upper_sum,
            } => write!(
                f,
                "weight bounds can't sum to 1: lower bounds sum to {lower_sum} and upper \
                 bounds to {upper_sum}"
            ),
            CutupError::NonFiniteWeight { asset } => {
                write!(f, "weight for asset {asset} is NaN or infinite")
            }
            CutupError::MissingCurrentWeights => write!(
                f,
                "turnover and transaction cost settings need current weights to measure from"
            ),
            CutupError::InfeasibleTurnover { minimum, cap } => write!(
                f,
                "turnover cap {cap} is below {minimum}, the least turnover that reaches the \
                 weight bounds from the current weights"
            ),
            CutupError::NotSupported { setting, with } => {
                write!(f, "`{setting}` isn't supported with {with} yet")
            }
            CutupError::DegenerateViews => write!(
                f,
                "views repeat or contradict each other, or one with a default or confidence \
                 variance is on a portfolio with no variance; give it an explicit variance"
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
/// use cutup::{EqualWeight, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// let prices = dmatrix![
///     100.0, 200.0;
///     110.0, 190.0;
///     121.0, 209.0
/// ];
///
/// let weights = PortfolioAllocator::from_prices(prices)?.allocate(&EqualWeight)?;
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

/// Weights in column order, for an [`Allocation`] of your own. [`PortfolioAllocator::allocate`]
/// checks there's one finite weight per asset, but not the sum, so make them sum to 1, as every
/// method in this crate does.
impl From<Vec<f64>> for Weights {
    fn from(weights: Vec<f64>) -> Self {
        Weights(weights)
    }
}

/// Adjustments applied to the sample covariance matrix before a method uses it, set with its
/// `covariance` setter, such as [`MinVariance::covariance`]. Each method also has `shrinkage`
/// and `regularization` setters for each adjustment on its own.
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

/// What a mean-variance optimization trades off, for [`MeanVariance`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum Objective {
    /// Maximize `μᵀw − (γ/2) wᵀΣw` for a risk aversion `γ > 0`.
    RiskAversion(f64),
    /// Minimize `wᵀΣw` subject to an expected portfolio return `μᵀw` equal to this value.
    TargetReturn(f64),
}

/// Lower and upper limits on each asset's weight, for [`MinVariance::bounds`],
/// [`MeanVariance::bounds`] and [`MinCvar::bounds`].
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
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WeightBounds {
    lower: Limit,
    upper: Limit,
}

/// One side of [`WeightBounds`].
#[derive(Debug, Clone, PartialEq)]
enum Limit {
    Uniform(f64),
    PerAsset(Vec<f64>),
}

impl Limit {
    fn resolve(&self, n: usize) -> Result<DVector<f64>, CutupError> {
        match self {
            Limit::Uniform(value) => Ok(DVector::from_element(n, *value)),
            Limit::PerAsset(values) if values.len() != n => Err(CutupError::DimensionMismatch {
                expected: n,
                actual: values.len(),
            }),
            Limit::PerAsset(values) => Ok(DVector::from_column_slice(values)),
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
        }
    }

    /// Bounds `lower[i] ≤ wᵢ ≤ upper[i]` for each asset `i`, in column order.
    pub fn per_asset(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        Self {
            lower: Limit::PerAsset(lower),
            upper: Limit::PerAsset(upper),
        }
    }

    /// Checks the bounds for `n` assets, and returns them with each upper bound lowered to
    /// what the other assets' lower bounds leave room for, `1 − Σⱼ≠ᵢ lⱼ`. That doesn't change
    /// which portfolios are feasible, and makes every upper bound finite.
    fn resolve(&self, n: usize) -> Result<qp::Bounds, CutupError> {
        let lower = self.lower.resolve(n)?;
        let upper = self.upper.resolve(n)?;
        if let Some(&value) = lower.iter().find(|&&l| !(l.is_finite() && l >= 0.0)) {
            return Err(CutupError::InvalidConfig {
                field: "lower_bound",
                value,
            });
        }
        if let Some((_, &value)) = lower
            .iter()
            .zip(upper.iter())
            .find(|(l, u)| u.is_nan() || u < l)
        {
            return Err(CutupError::InvalidConfig {
                field: "upper_bound",
                value,
            });
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

/// `PortfolioAllocator` handles various portfolio allocation strategies.
/// Supports minimum variance, mean-variance optimization, maximum Sharpe ratio, Equal Weight
/// (EW), inverse volatility, inverse variance, risk parity (equal risk contribution),
/// Hierarchical Risk Parity (HRP), and Hierarchical Equal Risk Contribution (HERC).
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

    /// Computes weights with `method`, any [`Allocation`]: one of this crate's methods, such as
    /// [`MinVariance`] or [`Hrp`], or one of your own.
    ///
    /// # Errors
    ///
    /// * Whatever `method` returns. Each method's type lists its errors.
    /// * [`CutupError::DimensionMismatch`] if `method` doesn't return one weight per asset.
    /// * [`CutupError::NonFiniteWeight`] if `method` returns a weight that is NaN or infinite.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MinVariance, PortfolioAllocator};
    /// use nalgebra::dmatrix;
    ///
    /// let returns = dmatrix![
    ///      0.010, -0.020,  0.005;
    ///     -0.004,  0.015,  0.002;
    ///      0.007,  0.003, -0.006;
    ///      0.002, -0.001,  0.004
    /// ];
    /// let allocator = PortfolioAllocator::from_returns(returns)?;
    ///
    /// let long_only = allocator.allocate(&MinVariance::default())?;
    /// let long_short = allocator.allocate(&MinVariance::default().allow_short())?;
    ///
    /// assert!(long_only.iter().all(|&w| w >= 0.0));
    /// assert!((long_short.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn allocate<A: Allocation + ?Sized>(&self, method: &A) -> Result<Weights, CutupError> {
        let weights = method.allocate(self)?;
        let n = self.returns.ncols();
        if weights.len() != n {
            return Err(CutupError::DimensionMismatch {
                expected: n,
                actual: weights.len(),
            });
        }
        if let Some(asset) = weights.iter().position(|w| !w.is_finite()) {
            return Err(CutupError::NonFiniteWeight { asset });
        }
        Ok(weights)
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
    /// regularization gives it a positive variance, [`CutupError::NonFiniteCovariance`] if the
    /// result isn't finite, and [`CutupError::SingularCovariance`] if it is zero.
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

        let adjusted = if let Some(eps) = config.regularization {
            shrunk_cov + eps * identity
        } else {
            shrunk_cov
        };
        // A solver given NaN or infinity would fail in its own, less telling way.
        if adjusted.iter().any(|x| !x.is_finite()) {
            return Err(CutupError::NonFiniteCovariance);
        }
        // Shrinking or regularizing by zero leaves a covariance of constant assets at zero,
        // which gives nothing to allocate by.
        if adjusted.trace() == 0.0 {
            return Err(CutupError::SingularCovariance);
        }
        Ok(adjusted)
    }

    /// The minimum variance portfolio with short positions allowed, for [`MinVariance`].
    fn min_variance(&self, covariance: &CovarianceConfig) -> Result<Weights, CutupError> {
        let cov = self.adjusted_covariance(covariance)?;
        let n = cov.ncols();
        let ones = DMatrix::from_element(n, 1, 1.0);

        // Variance is bounded below and full investment is always feasible, so an optimum
        // exists; failing to find one is numerical.
        let weights = qp::solve_kkt(
            &cov,
            &DVector::zeros(n),
            &ones,
            &DVector::from_element(1, 1.0),
        )?
        .ok_or(CutupError::SingularCovariance)?
        .0;
        Ok(Weights(weights.iter().copied().collect()))
    }

    /// The minimum variance portfolio within `bounds`, for [`MinVariance`].
    fn bounded_min_variance(
        &self,
        covariance: &CovarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        let bounds = bounds.resolve(self.cov_matrix.ncols())?;
        let cov = self.adjusted_covariance(covariance)?;
        let n = cov.ncols();
        // From a point strictly inside the bounds, the solver's first step is to the
        // unconstrained minimum variance portfolio. With the default bounds it's equal weights.
        let start = interior_start(&bounds);
        long_only_weights(
            &cov,
            &DVector::zeros(n),
            &DMatrix::from_element(n, 1, 1.0),
            &DVector::from_element(1, 1.0),
            &bounds,
            start,
        )
    }

    /// The mean-variance portfolio with short positions allowed, for [`MeanVariance`].
    fn mean_variance(
        &self,
        objective: Objective,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        check_objective(objective)?;
        let n = self.cov_matrix.ncols();
        let mu = self.expected_returns(expected_returns)?;

        let cov = self.adjusted_covariance(covariance)?;
        let ones = DVector::from_element(n, 1.0);

        let weights = match objective {
            Objective::RiskAversion(gamma) => {
                qp::solve_kkt(
                    &(cov * gamma),
                    &mu,
                    &DMatrix::from_column_slice(n, 1, ones.as_slice()),
                    &DVector::from_element(1, 1.0),
                )?
                .ok_or(CutupError::Unbounded)?
                .0
            }
            Objective::TargetReturn(target) => {
                qp::solve_kkt(
                    &cov,
                    &DVector::zeros(n),
                    &DMatrix::from_columns(&[ones, mu]),
                    &DVector::from_vec(vec![1.0, target]),
                )?
                .ok_or(CutupError::DegenerateExpectedReturns)?
                .0
            }
        };
        Ok(Weights(weights.iter().copied().collect()))
    }

    /// The mean-variance portfolio within `bounds`, for [`MeanVariance`].
    fn bounded_mean_variance(
        &self,
        objective: Objective,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        check_objective(objective)?;
        let mu = self.expected_returns(expected_returns)?;
        let bounds = bounds.resolve(mu.len())?;
        let budget = |n: usize| {
            (
                DMatrix::from_element(n, 1, 1.0),
                DVector::from_element(1, 1.0),
            )
        };
        let target = match objective {
            Objective::RiskAversion(gamma) => {
                let cov = self.adjusted_covariance(covariance)?;
                let (e, b) = budget(cov.ncols());
                return long_only_weights(
                    &(cov * gamma),
                    &mu,
                    &e,
                    &b,
                    &bounds,
                    interior_start(&bounds),
                );
            }
            Objective::TargetReturn(target) => target,
        };
        // An unreachable target is rejected before the covariance is built.
        let problem = ReturnRange::target_problem(&mu, bounds, target)?;
        let cov = self.adjusted_covariance(covariance)?;
        long_only_weights(
            &cov,
            &DVector::zeros(cov.ncols()),
            &problem.e,
            &problem.b,
            &problem.bounds,
            problem.start,
        )
    }

    /// Returns `given` after checking it, or the sample mean of the returns when it's `None`.
    fn expected_returns(&self, given: Option<&DVector<f64>>) -> Result<DVector<f64>, CutupError> {
        let n = self.cov_matrix.ncols();
        match given {
            Some(mu) if mu.len() != n => Err(CutupError::DimensionMismatch {
                expected: n,
                actual: mu.len(),
            }),
            Some(mu) => match mu.iter().find(|m| !m.is_finite()) {
                Some(&value) => Err(CutupError::InvalidConfig {
                    field: "expected_returns",
                    value,
                }),
                None => Ok(mu.clone()),
            },
            None => Ok(self.returns.row_mean().transpose()),
        }
    }

    /// The maximum Sharpe ratio portfolio with short positions allowed, for [`MaxSharpe`].
    fn max_sharpe(
        &self,
        risk_free_rate: f64,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        if !risk_free_rate.is_finite() {
            return Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: risk_free_rate,
            });
        }
        let mu = self.expected_returns(expected_returns)?;
        let cov = self.adjusted_covariance(covariance)?;
        let n = cov.ncols();
        let ones = DVector::from_element(n, 1.0);
        let excess = mu.add_scalar(-risk_free_rate);
        let too_high = |min_variance_return: f64| CutupError::RiskFreeRateTooHigh {
            risk_free_rate,
            min_variance_return,
        };

        // Decompose the correlation matrix C into its range and its null space, with
        // numpy.linalg.pinv's cutoff. In units u = Dw, with D = diag(σ), the variance is uᵀCu,
        // the budget (D⁻¹1)ᵀu and the excess return (D⁻¹(μ − r_f))ᵀu.
        let (correlation, volatilities) = correlation(&cov)?;
        // C is positive semidefinite, so its eigenvalues are its singular values.
        let eigen = SymmetricEigen::new(&correlation).ok_or(CutupError::SingularCovariance)?;
        let (range, null) = eigen.split(eigen.values.amax() * n as f64 * f64::EPSILON);
        let budget = ones.component_div(&volatilities);
        let earned = excess.component_div(&volatilities);

        if !null.is_empty() {
            // Coordinates of the budget and the excess returns in the null space.
            let a = DVector::from_iterator(null.len(), null.iter().map(|v| v.dot(&budget)));
            let b = DVector::from_iterator(null.len(), null.iter().map(|v| v.dot(&earned)));
            let noise = 1e-9 * earned.norm().max(f64::MIN_POSITIVE);
            if a.norm() > 1e-9 * budget.norm() {
                // A fully invested zero-variance portfolio exists, w0 ∝ D⁻¹ Σ aₖvₖ, earning
                // excess return b·a / a·a. Any part of b not along a is a budget-neutral,
                // variance-free change that adds return.
                let along = b.dot(&a) / a.dot(&a);
                if (&b - &a * along).norm() > noise || along * a.norm() > noise {
                    return Err(CutupError::Unbounded);
                }
                return Err(too_high(risk_free_rate + along));
            }
            if b.norm() > noise {
                return Err(CutupError::Unbounded);
            }
        }

        // Σ⁺x = D⁻¹C⁺D⁻¹x restricted to the range, where every remaining solution lives.
        let pinv = |x: &DVector<f64>| {
            let x = x.component_div(&volatilities);
            range
                .iter()
                .fold(DVector::zeros(n), |acc, (sigma, v)| {
                    acc + v * (v.dot(&x) / sigma)
                })
                .component_div(&volatilities)
        };
        let inv_ones = pinv(&ones);
        let min_variance_return = inv_ones.dot(&mu) / inv_ones.sum();
        // The rounding in r_mvp comes from its terms wᵢμᵢ, which don't change when an asset is
        // rescaled.
        let terms = inv_ones.component_mul(&mu).abs().sum() / inv_ones.sum().abs();
        let tolerance = 1e-10 * terms;
        if risk_free_rate >= min_variance_return - tolerance {
            return Err(too_high(min_variance_return));
        }

        let z = pinv(&excess);
        let weights = &z / z.sum();
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(CutupError::SingularCovariance);
        }
        Ok(Weights(weights.iter().copied().collect()))
    }

    /// The long-only maximum Sharpe ratio portfolio, for [`MaxSharpe`].
    fn long_only_max_sharpe(
        &self,
        risk_free_rate: f64,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        if !risk_free_rate.is_finite() {
            return Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: risk_free_rate,
            });
        }
        let mu = self.expected_returns(expected_returns)?;
        // Excess returns relative to the largest magnitude, so that neither they nor the
        // tolerance can overflow.
        let (best, highest_return) = mu.argmax();
        let scale = risk_free_rate.abs().max(mu.amax());
        let excess = mu.map(|m| m / scale - risk_free_rate / scale);
        // An excess return counts only beyond the rounding of its own terms, so that rescaling
        // one asset's returns doesn't change whether it has one. A zero scale means every
        // return and the rate are zero.
        let positive = |i: usize| excess[i] > 1e-10 * risk_free_rate.abs().max(mu[i].abs()) / scale;
        if scale == 0.0 || !(0..mu.len()).any(positive) {
            return Err(CutupError::NoPositiveExcessReturn {
                risk_free_rate,
                highest_return,
            });
        }
        let cov = self.adjusted_covariance(covariance)?;
        // Scaled so that the best asset's excess return is 1.
        long_only_max_ratio(&cov, &(&excess / excess[best]), best)?.ok_or(CutupError::Unbounded)
    }

    /// The maximum diversification portfolio, for [`MaxDiversification`].
    fn max_diversification(&self, covariance: &CovarianceConfig) -> Result<Weights, CutupError> {
        let cov = self.adjusted_covariance(covariance)?;
        let volatilities = cov.diagonal().map(f64::sqrt);
        let (best, highest) = volatilities.argmax();
        long_only_max_ratio(&cov, &(volatilities / highest), best)?
            .ok_or(CutupError::SingularCovariance)
    }

    /// Equal weights, for [`EqualWeight`].
    fn equal_weight(&self) -> Weights {
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

    /// The risk budgeting portfolio, for [`RiskParity`].
    fn risk_parity(
        &self,
        risk_budgets: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        let n = self.cov_matrix.ncols();
        let budgets = match risk_budgets {
            Some(b) if b.len() != n => {
                return Err(CutupError::DimensionMismatch {
                    expected: n,
                    actual: b.len(),
                })
            }
            Some(b) => match b.iter().find(|b| !(b.is_finite() && **b > 0.0)) {
                Some(&value) => {
                    return Err(CutupError::InvalidConfig {
                        field: "risk_budgets",
                        value,
                    })
                }
                None => b.clone(),
            },
            None => DVector::from_element(n, 1.0),
        };
        let cov = self.adjusted_covariance(covariance)?;
        PortfolioAllocator::risk_budget_weights(&cov, &budgets)
            .map(|w| Weights(w.iter().copied().collect()))
    }

    /// Minimizes `f(y) = ½ yᵀΣy − bᵀ ln y` over `y > 0` and returns `y / 1ᵀy`.
    ///
    /// At the minimum, `yᵢ (Σy)ᵢ = bᵢ`, so the normalized `y` has the budgeted risk
    /// contributions. `f` is strictly convex, so Newton's method with a backtracking line
    /// search converges to it from any start. `f` is unbounded below exactly when a
    /// long-only `d` has `dᵀΣd = 0`: then `y` grows along `d`, and the normalized `y`
    /// approaches that zero-variance portfolio until the iteration breaks down. A failed
    /// iteration is classified by the variance of its last normalized `y`.
    ///
    /// The iteration runs on the correlation matrix `C`, in units `u = diag(σ)y`, where the
    /// risk contributions `uᵢ(Cu)ᵢ = yᵢ(Σy)ᵢ` are the same. The weights are then `u / σ`,
    /// normalized. That makes the tolerances independent of each asset's units.
    fn risk_budget_weights(
        cov: &DMatrix<f64>,
        budgets: &DVector<f64>,
    ) -> Result<DVector<f64>, CutupError> {
        let n = cov.ncols();
        let (sigma, volatilities) = correlation(cov)?;
        // Scaling b to a sum of 1 makes the tolerances below independent of its scale, and
        // doesn't change the normalized solution. Dividing b by its largest entry first keeps
        // the sum from overflowing.
        let b = budgets / budgets.max();
        let b = &b / b.sum();
        let smallest_budget = b.min();
        let objective = |y: &DVector<f64>| {
            0.5 * y.dot(&(&sigma * y))
                - b.iter().zip(y.iter()).map(|(b, y)| b * y.ln()).sum::<f64>()
        };
        let failed = |y: &DVector<f64>| {
            let w = y / y.sum();
            if w.dot(&(&sigma * &w)) <= 1e-12 {
                CutupError::SingularCovariance
            } else {
                CutupError::NotConverged
            }
        };
        // When a long-only portfolio has zero variance, rounding can stall the iteration on
        // that direction, or let it converge on the rounding left in a singular C. So y is
        // checked with max diversification's null-space cutoff before it's returned.
        let finish = |u: &DVector<f64>| {
            if in_null_space(&sigma, u)? {
                return Err(CutupError::SingularCovariance);
            }
            let w = u.component_div(&volatilities);
            Ok(&w / w.sum())
        };

        // Start from inverse volatility, scaled so that yᵀΣy = 1 as at the solution.
        let mut y = sigma.diagonal().map(|v| 1.0 / v.sqrt());
        let start_variance = y.dot(&(&sigma * &y));
        if start_variance.is_nan() || start_variance <= 1e-12 * y.norm_squared() {
            return Err(CutupError::SingularCovariance);
        }
        y /= start_variance.sqrt();

        let mut previous = f64::INFINITY;
        for _ in 0..500 {
            let gradient = &sigma * &y - b.component_div(&y);
            let mut hessian = sigma.clone();
            for i in 0..n {
                hessian[(i, i)] += b[i] / (y[i] * y[i]);
            }
            let Some(cholesky) = hessian.cholesky() else {
                return Err(failed(&y));
            };
            let step = cholesky.solve(&gradient);
            // The squared Newton decrement, which is never negative unless rounding has broken
            // the Hessian. Divided by the smallest budget, it bounds (Δᵢ / yᵢ)² for every i.
            let decrement = gradient.dot(&step);
            if decrement.is_nan() || decrement < 0.0 {
                return Err(failed(&y));
            }
            let scaled = decrement / smallest_budget;

            if scaled < 1e-2 {
                // Close enough for full steps, which stay positive and converge quadratically.
                // A line search would only see rounding in f from here on.
                if scaled >= previous {
                    // Rounding noise: y is as close as it gets.
                    return finish(&y);
                }
                y -= step;
                if scaled < 1e-20 {
                    return finish(&y);
                }
                previous = scaled;
                continue;
            }
            previous = f64::INFINITY;

            let current = objective(&y);
            let mut t = 1.0;
            loop {
                let candidate = &y - t * &step;
                if candidate.iter().all(|v| v.is_finite() && *v > 0.0)
                    && objective(&candidate) <= current - 0.25 * t * decrement
                {
                    y = candidate;
                    break;
                }
                t *= 0.5;
                if t < 1e-10 {
                    return Err(failed(&y));
                }
            }
        }
        Err(failed(&y))
    }

    /// Returns the correlation distance `dᵢⱼ = sqrt((1 − ρᵢⱼ) / 2)` between every pair of
    /// assets.
    ///
    /// Returns [`CutupError::NonFiniteCovariance`] if a correlation can't be computed, for
    /// example when variances are so small that their product underflows to zero.
    fn correlation_distance(&self) -> Result<DMatrix<f64>, CutupError> {
        let cov = &self.cov_matrix;
        let n = cov.ncols();
        let mut distance = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i + 1..n {
                let rho = cov[(i, j)] / (cov[(i, i)] * cov[(j, j)]).sqrt();
                // Clamping would hide an infinite correlation, so reject it first.
                if !rho.is_finite() {
                    return Err(CutupError::NonFiniteCovariance);
                }
                let d = ((1.0 - rho) / 2.0).clamp(0.0, 1.0).sqrt();
                distance[(i, j)] = d;
                distance[(j, i)] = d;
            }
        }
        Ok(distance)
    }

    /// The order HRP puts the assets in, for [`Hrp::cluster_order`].
    fn cluster_order(&self, linkage: Linkage) -> Result<Vec<usize>, CutupError> {
        if let Some(asset) = self.find_zero_variance() {
            return Err(CutupError::ZeroVariance { asset });
        }
        let distance = self.correlation_distance()?;
        Ok(cluster::hierarchy(&distance, linkage).leaf_order())
    }

    /// Returns the variance of the inverse-variance portfolio over `cluster`.
    fn cluster_variance(&self, cluster: &[usize]) -> f64 {
        let weights = self.inverse_variance_weights(cluster);
        let mut variance = 0.0;
        for (a, &i) in cluster.iter().enumerate() {
            for (b, &j) in cluster.iter().enumerate() {
                variance += weights[a] * weights[b] * self.cov_matrix[(i, j)];
            }
        }
        variance
    }

    /// Returns the inverse-variance weights of the assets in `cluster`, summing to 1.
    fn inverse_variance_weights(&self, cluster: &[usize]) -> Vec<f64> {
        let inverse: Vec<f64> = cluster
            .iter()
            .map(|&i| 1.0 / self.cov_matrix[(i, i)])
            .collect();
        let total: f64 = inverse.iter().sum();
        inverse.iter().map(|w| w / total).collect()
    }

    /// The HRP weights, for [`Hrp`].
    fn hrp(&self, linkage: Linkage) -> Result<Weights, CutupError> {
        let order = self.cluster_order(linkage)?;
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

    /// The HERC weights, for [`Herc`].
    fn herc(
        &self,
        linkage: Linkage,
        clusters: Option<usize>,
        max_clusters: usize,
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
        if let Some(asset) = self.find_zero_variance() {
            return Err(CutupError::ZeroVariance { asset });
        }
        let distance = self.correlation_distance()?;
        let hierarchy = cluster::hierarchy(&distance, linkage);
        let k = clusters.unwrap_or_else(|| hierarchy.cluster_count(&distance, max_clusters));

        // Each cluster below the cut, by the node that formed it.
        let clusters = hierarchy.cut(k);
        let mut risk = vec![0.0; 2 * n - 1];
        for (node, members) in &clusters {
            // On singular data rounding can leave a variance slightly below zero, which would
            // push α outside [0, 1].
            risk[*node] = self.cluster_variance(members).max(0.0);
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
            for (&i, w) in members.iter().zip(self.inverse_variance_weights(members)) {
                weights[i] = branch_weight[*node] * w;
            }
        }
        Ok(Weights(weights))
    }
}

/// Splits a covariance into the correlation matrix `C` and the volatilities `σ`, with
/// `Σ = diag(σ) C diag(σ)`. Working on `C` in units `u = diag(σ)w` keeps assets whose
/// volatilities are many orders of magnitude apart accurate, and makes cutoffs independent of
/// each asset's units.
///
/// Returns [`CutupError::SingularCovariance`] if some variance isn't positive or a
/// correlation isn't finite.
fn correlation(cov: &DMatrix<f64>) -> Result<(DMatrix<f64>, DVector<f64>), CutupError> {
    let volatilities = cov.diagonal().map(f64::sqrt);
    // Dividing by each volatility in turn, so that tiny volatilities' product can't underflow.
    let correlation = DMatrix::from_fn(cov.nrows(), cov.ncols(), |i, j| {
        cov[(i, j)] / volatilities[i] / volatilities[j]
    });
    if volatilities.iter().any(|&s| s.is_nan() || s <= 0.0)
        || correlation.iter().any(|c| !c.is_finite())
    {
        return Err(CutupError::SingularCovariance);
    }
    Ok((correlation, volatilities))
}

/// Solves `minimize ½ wᵀQw − cᵀw subject to Eᵀw = b and l ≤ w ≤ u` from the feasible
/// `start`, and rescales the solution to sum to 1. With a budget constraint `1ᵀw = 1`, that
/// only removes the rounding left by clamping to the bounds; for max Sharpe it maps `y` back
/// to `w`.
fn long_only_weights(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &qp::Bounds,
    start: DVector<f64>,
) -> Result<Weights, CutupError> {
    Ok(normalized(qp::solve_bounded(q, c, e, b, bounds, start)?))
}

/// `x` rescaled to sum to 1. With a budget constraint `1ᵀx = 1`, that only removes the
/// rounding left by clamping to the bounds; for max Sharpe it maps `y` back to `w`.
fn normalized(x: DVector<f64>) -> Weights {
    let total = x.sum();
    Weights(x.iter().map(|w| w / total).collect())
}

/// Maximizes `aᵀw / sqrt(wᵀΣw)` over long-only portfolios summing to 1, for `a` with its
/// largest entry `a[best] = 1`. With `y = w / aᵀw` this is minimizing `yᵀΣy` subject to
/// `aᵀy = 1` and `y ≥ 0`, from one unit of `best`, then `w = y / 1ᵀy`.
///
/// Returns `None` if the optimum, in correlation units `u = diag(σ)w`, lies in the
/// correlation matrix's null space (see [`in_null_space`]), where the ratio has no upper
/// bound, so each caller can name its own error.
fn long_only_max_ratio(
    cov: &DMatrix<f64>,
    a: &DVector<f64>,
    best: usize,
) -> Result<Option<Weights>, CutupError> {
    let n = cov.ncols();
    // Before the solve, so that correlations that can't be computed are reported as such.
    let (correlation, volatilities) = correlation(cov)?;
    let mut start = DVector::zeros(n);
    start[best] = 1.0;
    let weights = long_only_weights(
        cov,
        &DVector::zeros(n),
        &DMatrix::from_column_slice(n, 1, a.as_slice()),
        &DVector::from_element(1, 1.0),
        &qp::Bounds::nonnegative(n),
        start,
    )?;

    let u = DVector::from_column_slice(weights.as_slice()).component_mul(&volatilities);
    if in_null_space(&correlation, &u)? {
        return Ok(None);
    }
    Ok(Some(weights))
}

/// Whether `u` lies in the null space of the correlation matrix `c`, with the cutoff
/// long-short max Sharpe uses: a Rayleigh quotient `uᵀCu / uᵀu` within n·ε of the largest
/// eigenvalue. The trace, n, bounds that eigenvalue, so the decomposition is only needed near
/// the cutoff.
fn in_null_space(c: &DMatrix<f64>, u: &DVector<f64>) -> Result<bool, CutupError> {
    let n = c.ncols() as f64;
    let quotient = u.dot(&(c * u)) / u.norm_squared();
    let cutoff = |largest: f64| quotient <= n * f64::EPSILON * largest;
    if !cutoff(n) {
        return Ok(false);
    }
    let eigen = SymmetricEigen::new(c).ok_or(CutupError::SingularCovariance)?;
    Ok(cutoff(eigen.values.amax()))
}

/// A point strictly inside resolved `bounds` with weights summing to 1, where they allow one:
/// `l + t(u − l)` for the `t` that sums to 1. With the default bounds that's equal weights.
fn interior_start(bounds: &qp::Bounds) -> DVector<f64> {
    let room = &bounds.upper - &bounds.lower;
    let t = match room.sum() {
        total if total > 0.0 => ((1.0 - bounds.lower.sum()) / total).clamp(0.0, 1.0),
        _ => 0.0,
    };
    &bounds.lower + room * t
}

/// A target-return problem from [`ReturnRange::target_problem`].
struct TargetProblem {
    e: DMatrix<f64>,
    b: DVector<f64>,
    bounds: qp::Bounds,
    start: DVector<f64>,
}

/// The lowest and highest expected return of any portfolio within resolved bounds, and the
/// portfolios that reach them: start every asset at its lower bound, then raise assets to
/// their upper bounds in order of expected return until the weights sum to 1.
struct ReturnRange {
    lowest: f64,
    highest: f64,
    lowest_weights: DVector<f64>,
    highest_weights: DVector<f64>,
    /// The expected return of the last asset raised above its lower bound, for the lowest
    /// and the highest, or `None` if the lower bounds already sum to 1.
    margins: (Option<f64>, Option<f64>),
}

impl ReturnRange {
    fn new(mu: &DVector<f64>, bounds: &qp::Bounds) -> Self {
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
    fn reachable(&self, target: f64) -> Result<f64, CutupError> {
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
    fn target_problem(
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

/// Checks a mean-variance objective's parameter.
fn check_objective(objective: Objective) -> Result<(), CutupError> {
    match objective {
        Objective::RiskAversion(gamma) if !(gamma.is_finite() && gamma > 0.0) => {
            Err(CutupError::InvalidConfig {
                field: "risk_aversion",
                value: gamma,
            })
        }
        Objective::TargetReturn(target) if !target.is_finite() => Err(CutupError::InvalidConfig {
            field: "target_return",
            value: target,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

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
            CutupError::RiskFreeRateTooHigh {
                risk_free_rate: 0.02,
                min_variance_return: 0.01
            }
            .to_string(),
            "risk-free rate 0.02 must be below the minimum variance portfolio's expected return 0.01"
        );
        assert_eq!(
            CutupError::Unbounded.to_string(),
            "objective is unbounded: expected return can grow without adding variance"
        );
        assert_eq!(
            CutupError::DegenerateExpectedReturns.to_string(),
            "every asset has the same expected return, so no other target return can be reached"
        );
        assert_eq!(
            CutupError::NotConverged.to_string(),
            "the solver stopped before reaching a solution"
        );
        assert_eq!(
            CutupError::TargetReturnOutOfRange {
                target_return: 0.5,
                lowest: 0.1,
                highest: 0.2
            }
            .to_string(),
            "target return 0.5 is outside the reachable expected returns, from 0.1 to 0.2, so \
             no long-only portfolio reaches it"
        );
        assert_eq!(
            CutupError::NoPositiveExcessReturn {
                risk_free_rate: 0.03,
                highest_return: 0.02
            }
            .to_string(),
            "risk-free rate 0.03 must be below the highest expected return 0.02 for a long-only \
             portfolio to have a positive Sharpe ratio"
        );
        assert_eq!(
            CutupError::InfeasibleBounds {
                lower_sum: 0.0,
                upper_sum: 0.9
            }
            .to_string(),
            "weight bounds can't sum to 1: lower bounds sum to 0 and upper bounds to 0.9"
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
            &base
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            &rescaled
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );
        assert_weights_close(
            &base.allocate(&Hrp::default()).unwrap(),
            &rescaled.allocate(&Hrp::default()).unwrap(),
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
        let ew_weights = allocator.allocate(&EqualWeight).unwrap();
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
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
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
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::NonFiniteCovariance)
        );

        allocator.cov_matrix = DMatrix::from_element(3, 3, f64::NAN);
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
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
        let hrp_weights = allocator.allocate(&Hrp::default()).unwrap();
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
        let weights = allocator.allocate(&Hrp::default()).unwrap();

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
            allocator.allocate(&Hrp::default()).err(),
            Some(CutupError::ZeroVariance { asset: 1 })
        );
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance { asset: 1 })
        );
        assert_eq!(allocator.allocate(&EqualWeight).unwrap().len(), 3);
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
            allocator.allocate(&Hrp::default()).err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );

        // Prices growing at a constant 10% per period.
        let prices = dmatrix![100.0, 50.0; 110.0, 49.0; 121.0, 51.5; 133.1, 51.0];
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        assert_eq!(
            allocator.allocate(&Hrp::default()).err(),
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
            match allocator.allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(config.clone()),
            ) {
                Err(CutupError::InvalidConfig { field: f, .. }) => assert_eq!(f, field),
                other => panic!("{config:?} gave {other:?}"),
            }
        }

        let edges = CovarianceConfig {
            regularization: Some(0.0),
            shrinkage: Some(1.0),
        };
        assert!(allocator
            .allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(edges.clone())
            )
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
            .allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(config.clone()),
            )
            .unwrap();
        assert_weights_close(&weights, &allocator.allocate(&EqualWeight).unwrap(), 1e-12);
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
            &base
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(config.clone()),
                )
                .unwrap(),
            &scaled
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(config.clone()),
                )
                .unwrap(),
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
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        let inverse = [1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)];
        let total: f64 = inverse.iter().sum();
        let expected = Weights(inverse.iter().map(|x| x / total).collect());
        assert_weights_close(&weights, &expected, 1e-12);
    }

    #[test]
    fn test_min_variance_two_asset_closed_form() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        let cov = &allocator.cov_matrix;
        let (s11, s22, s12) = (cov[(0, 0)], cov[(1, 1)], cov[(0, 1)]);
        let w1 = (s22 - s12) / (s11 + s22 - 2.0 * s12);
        assert_weights_close(&weights, &Weights(vec![w1, 1.0 - w1]), 1e-12);
    }

    #[test]
    fn test_min_variance_beats_equal_weight() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let min_variance = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        let equal = allocator.allocate(&EqualWeight).unwrap();
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
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(weights.iter().all(|w| w.is_finite() && w.abs() < 10.0));
        assert!(
            portfolio_variance(&allocator, &weights)
                <= portfolio_variance(&allocator, &allocator.allocate(&EqualWeight).unwrap())
                    + 1e-15
        );
    }

    #[test]
    fn test_mean_variance_target_return() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let mu = sample_mean(&allocator);
        let target = mu.mean() * 1.5;
        let weights = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!((mu.dot(&w) - target).abs() < 1e-12);
    }

    #[test]
    fn test_mean_variance_target_at_min_variance_return_is_min_variance() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let min_variance = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        let mu = sample_mean(&allocator);
        let target = mu.dot(&DVector::from_column_slice(min_variance.as_slice()));

        let weights = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();
        assert_weights_close(&weights, &min_variance, 1e-9);
    }

    #[test]
    fn test_mean_variance_risk_aversion_first_order_condition() {
        // At the optimum of μᵀw − (γ/2) wᵀΣw subject to 1ᵀw = 1, μ − γΣw is a multiple of 1.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let gamma = 3.0;
        let weights = allocator
            .allocate(&MeanVariance::risk_aversion(gamma).allow_short())
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
            .allocate(&MeanVariance::risk_aversion(1e9).allow_short())
            .unwrap();
        assert_weights_close(
            &weights,
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-6,
        );
    }

    #[test]
    fn test_mean_variance_equal_expected_returns() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let flat = DVector::from_element(3, 0.01);

        // With nothing to choose between on return, risk aversion reduces to minimum variance.
        let config = MeanVariance::risk_aversion(2.0).expected_returns(flat.as_slice());
        assert_weights_close(
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );

        // And no portfolio can target any other return.
        let config = MeanVariance::target_return(0.02).expected_returns(flat.as_slice());
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::DegenerateExpectedReturns)
        );
    }

    #[test]
    fn test_mean_variance_defaults_to_sample_mean() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let default = MeanVariance::risk_aversion(5.0);
        let explicit = default
            .clone()
            .expected_returns(sample_mean(&allocator).as_slice());
        assert_eq!(
            allocator.allocate(&default.clone().allow_short()).unwrap(),
            allocator.allocate(&explicit.clone().allow_short()).unwrap()
        );
    }

    #[test]
    fn test_mean_variance_applies_covariance_config() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let plain = MeanVariance::risk_aversion(5.0);
        let shrunk = plain
            .clone()
            .covariance(CovarianceConfig::default().with_shrinkage(0.5));
        assert_ne!(
            allocator.allocate(&plain.clone().allow_short()).unwrap(),
            allocator.allocate(&shrunk.clone().allow_short()).unwrap()
        );

        let invalid = plain.covariance(CovarianceConfig::default().with_shrinkage(2.0));
        assert_eq!(
            allocator.allocate(&invalid.clone().allow_short()).err(),
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
            match allocator.allocate(&MeanVariance::risk_aversion(gamma).allow_short()) {
                Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "risk_aversion"),
                other => panic!("gamma {gamma} gave {other:?}"),
            }
        }
        match allocator.allocate(&MeanVariance::target_return(f64::NAN).allow_short()) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "target_return"),
            other => panic!("NaN target gave {other:?}"),
        }

        let config = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_vec(vec![0.01, f64::NAN, 0.02]).as_slice());
        match allocator.allocate(&config.clone().allow_short()) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "expected_returns"),
            other => panic!("NaN expected return gave {other:?}"),
        }

        let config = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_vec(vec![0.01, 0.02]).as_slice());
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
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
                .allocate(&MeanVariance::risk_aversion(1.0).allow_short())
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
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_min_variance_anti_correlated_pair() {
        // 1 lies in Σ's null space, so 1ᵀΣ⁺1 is rounding noise; the hedged pair has zero variance.
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert_weights_close(&weights, &Weights(vec![0.5, 0.5]), 1e-12);
    }

    #[test]
    fn test_min_variance_survives_inaccurate_svd() {
        // Heavily shrunk, this covariance is close to a multiple of I. nalgebra's SVD with a
        // convergence tolerance of exactly ε reconstructed its KKT matrix only to about 1e-2,
        // and the solve was rejected as inconsistent.
        let returns = DMatrix::from_vec(
            10,
            5,
            vec![
                0.014883239895260967,
                -0.01493530508682466,
                0.016430855321270307,
                -0.013476652725590775,
                0.048376573241446175,
                -0.01454922384002048,
                0.013863889151550382,
                0.03060490436889639,
                0.005947862449558874,
                0.007242902806749439,
                -0.049791777589274035,
                -0.010276099234288459,
                0.03787642939756265,
                -0.01009095345402214,
                0.039051035567899794,
                0.04379567627495853,
                -0.014218203192808818,
                -0.0009414262734083997,
                0.04410238107402796,
                -0.0023072042163211695,
                -0.036553907215322874,
                -0.0496177694619739,
                0.033432410527783465,
                -0.04522593873754994,
                -0.013996807386067985,
                0.04641859793989761,
                -0.019811327585646278,
                0.014370795975672165,
                -0.044680407876113194,
                -0.03280219001859793,
                -0.03639028312023631,
                0.03263436706874201,
                0.03115846887684503,
                -0.0029601214724074795,
                0.0035616215605822,
                0.02380399736864574,
                -0.02372734365258026,
                0.012992821815841767,
                0.012965514435532502,
                0.042600843138559705,
                -0.024029906325403177,
                0.0,
                0.04784068605333592,
                0.005896517391334338,
                -0.01686648883472212,
                0.048269384986887494,
                0.03695469685973938,
                0.015311227622385607,
                -0.03242117667024552,
                0.034392821320402465,
            ],
        );
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = CovarianceConfig::default().with_shrinkage(0.9832865013278819);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(config.clone()),
            )
            .unwrap();
        let cov = allocator.adjusted_covariance(&config).unwrap();
        let marginal = &cov * DVector::from_column_slice(weights.as_slice());
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert!(marginal.max() - marginal.min() < 1e-14 * marginal.amax());
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().covariance(config.clone()))
                .unwrap(),
            &weights,
            1e-14,
        );
    }

    #[test]
    fn test_zero_covariance_is_singular_after_shrinkage() {
        // Shrinkage lets constant assets past the zero-variance check, but shrinking a zero
        // covariance leaves it zero.
        let returns = dmatrix![0.01, 0.02; 0.01, 0.02; 0.01, 0.02];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let shrunk = CovarianceConfig::default().with_shrinkage(0.5);
        let mean_variance = MeanVariance::risk_aversion(10.0).covariance(shrunk.clone());
        let max_sharpe = MaxSharpe::default().covariance(shrunk.clone());
        let risk_parity = RiskParity::default().covariance(shrunk.clone());
        for result in [
            allocator.allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(shrunk.clone()),
            ),
            allocator.allocate(&MinVariance::default().covariance(shrunk.clone())),
            allocator.allocate(&mean_variance.clone().allow_short()),
            allocator.allocate(&max_sharpe.clone().allow_short()),
            allocator.allocate(&risk_parity),
        ] {
            assert_eq!(result, Err(CutupError::SingularCovariance));
        }
    }

    #[test]
    fn test_min_variance_on_rank_deficient_kkt() {
        // Three rows for four assets give Σ rank 2, and its KKT matrix a zero singular value.
        // nalgebra's SVD left a residual of about 4e-7 here, so the solve was rejected as
        // inconsistent; NumPy's solves it to rounding.
        let returns = DMatrix::from_vec(
            3,
            4,
            vec![
                0.03923072979353582,
                -0.027034431663761255,
                -0.02017662174798146,
                -0.030105448299475653,
                0.02262405031858968,
                0.01653614746213769,
                -0.009248234151376263,
                0.02095233950501841,
                -0.049975234461268725,
                -0.010174824524215684,
                -0.043910804533156975,
                0.04613336884944048,
            ],
        );
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        // numpy.linalg.pinv's minimum-norm solution of the same KKT system, in correlation
        // units u = diag(σ)w.
        let expected = Weights(vec![0.275093649, 0.331571165, 0.219203116, 0.174132070]);
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            &expected,
            1e-9,
        );
        let weights = allocator.allocate(&MinVariance::default()).unwrap();
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_long_only_min_variance_matches_unconstrained_when_long() {
        // A diagonal covariance gives all-positive minimum variance weights, so the
        // constraint is inactive.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&MinVariance::default()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_min_variance_drops_the_hedge() {
        // Unconstrained, (2, −1) hedges the collinear pair to zero variance. Long-only, any mix
        // of the two has more variance than the first asset alone.
        let allocator = PortfolioAllocator::from_returns(collinear_returns()).unwrap();
        assert!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap()[1]
                < 0.0
        );
        assert_eq!(
            allocator.allocate(&MinVariance::default()).unwrap(),
            Weights(vec![1.0, 0.0])
        );
    }

    #[test]
    fn test_long_only_min_variance_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, so the minimum is zero.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let weights = allocator.allocate(&MinVariance::default()).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);

        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&MinVariance::default()).unwrap(),
            &Weights(vec![0.5, 0.5]),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_min_variance_is_optimal_on_short_histories() {
        // Fewer rows than assets make Σ singular, so the free problems have many minima. The
        // solver must still reach a point where no asset's marginal variance (Σw)ᵢ is below
        // the portfolio's variance wᵀΣw.
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 0.1 - 0.05
        };
        for case in 0..300 {
            let assets = 4 + case % 9;
            let rows = 2 + case % (assets - 1);
            let returns = DMatrix::from_fn(rows, assets, |_, _| next());
            let allocator = PortfolioAllocator::from_returns(returns).unwrap();
            let weights = allocator.allocate(&MinVariance::default()).unwrap();
            let w = DVector::from_column_slice(weights.as_slice());
            let marginal = &allocator.cov_matrix * &w;
            let v = w.dot(&marginal);
            let tolerance = 1e-9 * allocator.cov_matrix.amax();
            assert!(w.iter().all(|&x| x >= 0.0), "case {case}: {weights:?}");
            assert!(
                marginal.iter().all(|&m| m >= v - tolerance),
                "case {case}: {marginal:?} vs {v}"
            );
        }
    }

    #[test]
    fn test_long_only_min_variance_rejects_bad_input() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().with_shrinkage(1.5);
        assert_eq!(
            allocator.allocate(&MinVariance::default().covariance(config.clone())),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            })
        );

        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(
            allocator.allocate(&MinVariance::default()),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
    }

    #[test]
    fn test_long_only_mean_variance_matches_unconstrained_when_long() {
        // A diagonal covariance and close expected returns keep every unconstrained weight
        // positive, so the constraint is inactive.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let expected = dvector![0.001, 0.002, 0.003];
        for config in [
            MeanVariance::risk_aversion(100.0),
            MeanVariance::target_return(0.0015),
        ] {
            let config = config.expected_returns(expected.as_slice());
            let unconstrained = allocator.allocate(&config.clone().allow_short()).unwrap();
            assert!(unconstrained.iter().all(|&w| w > 0.0), "{unconstrained:?}");
            assert_weights_close(&allocator.allocate(&config).unwrap(), &unconstrained, 1e-12);
        }
    }

    #[test]
    fn test_long_only_mean_variance_at_the_extremes() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];

        // With almost no aversion to risk, everything goes to the highest expected return.
        let config = MeanVariance::risk_aversion(1e-9).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // Only the asset with the highest expected return reaches it.
        let config = MeanVariance::target_return(0.003).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights(vec![0.0, 1.0, 0.0]),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_mean_variance_hits_the_target() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];
        for target in [0.001, 0.0012, 0.0025, 0.0029] {
            let config = MeanVariance::target_return(target).expected_returns(expected.as_slice());
            let weights = allocator.allocate(&config).unwrap();
            let w = DVector::from_column_slice(weights.as_slice());
            assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
            assert!((w.sum() - 1.0).abs() < 1e-12);
            assert!(
                (w.dot(&expected) - target).abs() < 1e-12 * target,
                "{weights:?}"
            );
        }
    }

    #[test]
    fn test_a_range_narrower_than_rounding_snaps_to_the_nearer_end() {
        // Returns of 100, 100 + 5e-11 and 100: the whole range is within rounding, and a
        // target at its top is only reached by the second asset.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = MeanVariance::target_return(100.0 + 5e-11)
            .expected_returns(dvector![100.0, 100.0 + 5e-11, 100.0].as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights(vec![0.0, 1.0, 0.0]),
            1e-15,
        );
    }

    #[test]
    fn test_long_only_mean_variance_rejects_unreachable_targets() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = MeanVariance::target_return(0.004)
            .expected_returns(dvector![0.001, 0.003, 0.002].as_slice());
        assert_eq!(
            allocator.allocate(&config),
            Err(CutupError::TargetReturnOutOfRange {
                target_return: 0.004,
                lowest: 0.001,
                highest: 0.003
            })
        );

        // With equal expected returns, only that return can be targeted, by any portfolio.
        let flat = dvector![0.002, 0.002, 0.002];
        let config = MeanVariance::target_return(0.001).expected_returns(flat.as_slice());
        assert!(matches!(
            allocator.allocate(&config),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        let config = MeanVariance::target_return(0.002).expected_returns(flat.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &allocator.allocate(&MinVariance::default()).unwrap(),
            1e-12,
        );
    }

    /// Returns of three uncorrelated assets whose variances are in the ratio 1 : 2 : 4, so the
    /// minimum variance weights are inverse variance, 4/7, 2/7 and 1/7.
    fn uncorrelated_returns() -> DMatrix<f64> {
        let s = [0.01, 0.01 * 2f64.sqrt(), 0.02];
        dmatrix![
            s[0], s[1], s[2];
            -s[0], s[1], -s[2];
            s[0], -s[1], -s[2];
            -s[0], -s[1], s[2]
        ]
    }

    #[test]
    fn test_bounded_min_variance_water_fills() {
        let allocator = PortfolioAllocator::from_returns(uncorrelated_returns()).unwrap();
        let config = CovarianceConfig::default();
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().covariance(config.clone()))
                .unwrap(),
            &Weights(vec![4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]),
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
        assert_weights_close(&weights, &Weights(vec![0.5, 1.0 / 3.0, 1.0 / 6.0]), 1e-12);

        // A floor of 25% on the last asset leaves 75% to split 2 : 1.
        let floored = WeightBounds::per_asset(vec![0.0, 0.0, 0.25], vec![0.5, 1.0, 1.0]);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(floored.clone()),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights(vec![0.5, 0.25, 0.25]), 1e-12);

        // Floors and caps of 1/3 leave equal weights as the only portfolio.
        let pinned = WeightBounds::uniform(1.0 / 3.0, 1.0 / 3.0);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .covariance(config.clone())
                    .bounds(pinned.clone()),
            )
            .unwrap();
        assert_weights_close(&weights, &Weights(vec![1.0 / 3.0; 3]), 1e-15);
    }

    #[test]
    fn test_bounds_that_dont_bind_change_nothing() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().with_shrinkage(0.2);
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
        let allocator = PortfolioAllocator::from_returns(uncorrelated_returns()).unwrap();
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
        assert_weights_close(&weights, &Weights(vec![0.6, 0.4, 0.0]), 1e-12);

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
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
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
            assert_weights_close(&weights, &Weights(vec![1.0 / n as f64; n]), 1e-15);
            for &w in &weights {
                assert!(
                    w >= lower * (1.0 - 1e-12) && w <= upper * (1.0 + 1e-12),
                    "{w}"
                );
            }
        }

        // Within rounding of a budget of 1 is feasible: ten caps of 0.1 sum to just under it.
        let ten = PortfolioAllocator::from_returns(DMatrix::from_fn(12, 10, |i, j| {
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
        assert_weights_close(&weights, &Weights(vec![0.1; 10]), 1e-15);
    }

    #[test]
    fn test_long_only_mean_variance_target_edge_cases() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();

        // A target a rounding error above the highest expected return is that return.
        let expected = dvector![0.001, 0.003, 0.002];
        let config = MeanVariance::target_return(0.003 * (1.0 + 4.0 * f64::EPSILON))
            .expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // highest − lowest overflows here, which mustn't spoil the starting point.
        let expected = dvector![-1e308, 1e308, 0.0];
        let config = MeanVariance::target_return(5e307).expected_returns(expected.as_slice());
        let weights = allocator.allocate(&config).unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
        assert!(
            (w.dot(&expected) - 5e307).abs() < 1e-12 * 5e307,
            "{weights:?}"
        );

        // An unreachable target is reported before the covariance is checked.
        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = MeanVariance::target_return(1.0);
        assert!(matches!(
            allocator.allocate(&config),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
    }

    #[test]
    fn test_long_only_max_sharpe_matches_unconstrained_when_long() {
        // Uncorrelated assets with positive excess returns: the tangency weights,
        // proportional to (μᵢ − r_f) / σᵢ², are all positive.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = MaxSharpe::default()
            .risk_free_rate(0.0005)
            .expected_returns(dvector![0.001, 0.002, 0.003].as_slice());
        let unconstrained = allocator.allocate(&config.clone().allow_short()).unwrap();
        assert!(unconstrained.iter().all(|&w| w > 0.0), "{unconstrained:?}");
        assert_weights_close(&allocator.allocate(&config).unwrap(), &unconstrained, 1e-12);
    }

    #[test]
    fn test_long_only_max_sharpe_risk_free_rate_limits() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];
        let at = |rate: f64| {
            let config = MaxSharpe::default()
                .risk_free_rate(rate)
                .expected_returns(expected.as_slice());
            (
                allocator.allocate(&config.clone().allow_short()),
                allocator.allocate(&config),
            )
        };

        // Just below the highest expected return, the unconstrained maximum doesn't exist,
        // but the long-only one does, and holds only that asset.
        let (unconstrained, long_only) = at(0.0029);
        assert!(matches!(
            unconstrained,
            Err(CutupError::RiskFreeRateTooHigh { .. })
        ));
        assert_weights_close(&long_only.unwrap(), &Weights(vec![0.0, 1.0, 0.0]), 1e-9);

        for rate in [0.003, 0.003 * (1.0 + 1e-12), 0.01] {
            assert!(matches!(
                at(rate).1,
                Err(CutupError::NoPositiveExcessReturn {
                    highest_return: 0.003,
                    ..
                })
            ));
        }
    }

    #[test]
    fn test_long_only_max_sharpe_extreme_returns() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let at = |rate: f64, expected: DVector<f64>| {
            let config = MaxSharpe::default()
                .risk_free_rate(rate)
                .expected_returns(expected.as_slice());
            allocator.allocate(&config)
        };

        // A subnormal excess return mustn't make the starting point infinite. With a zero
        // rate the weights don't depend on the scale of μ.
        assert_weights_close(
            &at(0.0, dvector![1e-310, 0.0, 0.0]).unwrap(),
            &at(0.0, dvector![1e-3, 0.0, 0.0]).unwrap(),
            1e-12,
        );

        // Neither the tolerance nor the excess returns may overflow.
        let weights = at(-1.7e308, dvector![1.7e308, 1.0, 1.0]).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // Every return and the rate zero: no excess return at all.
        assert!(matches!(
            at(0.0, DVector::zeros(3)),
            Err(CutupError::NoPositiveExcessReturn { .. })
        ));
    }

    #[test]
    fn test_max_diversification_matches_closed_form() {
        // When every asset is held, the optimum is v ∝ C⁻¹1 for the correlation matrix C, and
        // w ∝ v / σ. These correlations are unequal, so it isn't inverse volatility.
        let returns = dmatrix![
            0.01, 0.02, -0.01;
            -0.02, 0.01, 0.02;
            0.03, 0.01, 0.01;
            -0.01, -0.03, 0.0;
            0.0, 0.01, -0.02
        ];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let cov = &allocator.cov_matrix;
        let sigma = cov.diagonal().map(f64::sqrt);
        let correlation = DMatrix::from_fn(3, 3, |i, j| cov[(i, j)] / (sigma[i] * sigma[j]));
        let v = correlation
            .lu()
            .solve(&DVector::from_element(3, 1.0))
            .unwrap();
        assert!(v.iter().all(|&v| v > 0.0), "{v}");
        let w = v.component_div(&sigma);
        let expected = Weights((&w / w.sum()).iter().copied().collect());

        let weights = allocator.allocate(&MaxDiversification::default()).unwrap();
        assert_weights_close(&weights, &expected, 1e-12);
        assert_weights_close(
            &weights,
            &Weights(vec![0.29228043, 0.23774194, 0.46997763]),
            1e-8,
        );
    }

    #[test]
    fn test_volatilities_far_apart() {
        // Two uncorrelated assets have closed forms however far apart their volatilities are:
        // inverse volatility for max diversification and risk parity, inverse variance for
        // min variance, and (μᵢ / σᵢ²)-weighting for max Sharpe. Solving on Σ in raw units
        // got these wrong, or returned errors, at a gap of 1e8. The unit-free methods are
        // accurate relative to each weight. Min variance's weights are about 1 and s², so only
        // absolute accuracy means anything for the small one.
        let close = |weights: &Weights, expected: &Weights, s: f64| {
            for (w, e) in weights.iter().zip(expected.iter()) {
                assert!(
                    (w - e).abs() <= 1e-12 * e,
                    "s = {s}: {weights:?} vs {expected:?}"
                );
            }
        };
        let absolute = |weights: &Weights, expected: &Weights, s: f64| {
            for (w, e) in weights.iter().zip(expected.iter()) {
                assert!(
                    (w - e).abs() <= 1e-15,
                    "s = {s}: {weights:?} vs {expected:?}"
                );
            }
        };
        for s in [1e-4, 1e-8, 1e-12] {
            let returns = dmatrix![1.0, s; -1.0, s; 1.0, -s; -1.0, -s];
            let allocator = PortfolioAllocator::from_returns(returns).unwrap();
            let inverse_volatility = allocator.allocate(&InverseVolatility::default()).unwrap();
            let inverse_variance = allocator.allocate(&InverseVariance::default()).unwrap();
            close(
                &allocator.allocate(&MaxDiversification::default()).unwrap(),
                &inverse_volatility,
                s,
            );
            close(
                &allocator.allocate(&RiskParity::default()).unwrap(),
                &inverse_volatility,
                s,
            );
            absolute(
                &allocator
                    .allocate(&MinVariance::default().allow_short())
                    .unwrap(),
                &inverse_variance,
                s,
            );
            absolute(
                &allocator.allocate(&MinVariance::default()).unwrap(),
                &inverse_variance,
                s,
            );

            // Equal Sharpe ratios: μᵢ ∝ σᵢ gives weights ∝ 1 / σᵢ.
            let config = MaxSharpe::default().expected_returns(dvector![0.1, 0.1 * s].as_slice());
            close(
                &allocator.allocate(&config.clone().allow_short()).unwrap(),
                &inverse_volatility,
                s,
            );
            close(
                &allocator.allocate(&config).unwrap(),
                &inverse_volatility,
                s,
            );
        }
    }

    #[test]
    fn test_max_diversification_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, so the ratio has no upper bound.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        assert_eq!(
            allocator.allocate(&MaxDiversification::default()),
            Err(CutupError::SingularCovariance)
        );

        // Shrinkage gives every portfolio some variance.
        let config = CovarianceConfig::default().with_shrinkage(0.3);
        let weights = allocator
            .allocate(&MaxDiversification::default().covariance(config.clone()))
            .unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // The collinear pair's zero-variance hedge shorts one asset, so long-only the ratio
        // is bounded: every mix of perfectly correlated assets has a ratio of 1.
        let allocator = PortfolioAllocator::from_returns(collinear_returns()).unwrap();
        let weights = allocator.allocate(&MaxDiversification::default()).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
    }

    #[test]
    fn test_max_diversification_rejects_bad_input() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().with_shrinkage(1.5);
        assert_eq!(
            allocator.allocate(&MaxDiversification::default().covariance(config.clone())),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            })
        );

        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(
            allocator.allocate(&MaxDiversification::default()),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
    }

    #[test]
    fn test_long_only_max_sharpe_unbounded_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, and with these expected returns its
        // excess return is positive, so the Sharpe ratio grows without limit.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let config = MaxSharpe::default().expected_returns(dvector![0.01, 0.0, 0.01].as_slice());
        assert_eq!(allocator.allocate(&config), Err(CutupError::Unbounded));

        // Shrinkage gives every portfolio some variance.
        let config = config.covariance(CovarianceConfig::default().with_shrinkage(0.3));
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_long_only_max_sharpe_rejects_bad_input() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let config = MaxSharpe::default().risk_free_rate(f64::NAN);
        assert!(matches!(
            allocator.allocate(&config),
            Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                ..
            })
        ));
        let config = MaxSharpe::default().expected_returns(dvector![0.01].as_slice());
        assert_eq!(
            allocator.allocate(&config),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 1
            })
        );
    }

    #[test]
    fn test_long_only_mean_variance_bounded_on_singular_covariance() {
        // Unconstrained, utility grows without limit (see the next test). Long-only, the
        // weights can't grow, so there is always an optimum.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let config = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()),
            Err(CutupError::Unbounded)
        );
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_long_only_mean_variance_rejects_bad_input() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        assert_eq!(
            allocator.allocate(&MeanVariance::risk_aversion(0.0)),
            Err(CutupError::InvalidConfig {
                field: "risk_aversion",
                value: 0.0
            })
        );
        assert_eq!(
            allocator.allocate(&MeanVariance::target_return(f64::INFINITY)),
            Err(CutupError::InvalidConfig {
                field: "target_return",
                value: f64::INFINITY
            })
        );
        let config = MeanVariance::risk_aversion(1.0).expected_returns(dvector![0.01].as_slice());
        assert_eq!(
            allocator.allocate(&config),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 1
            })
        );
    }

    #[test]
    fn test_mean_variance_unbounded_on_singular_covariance() {
        // With Σ singular, d = [5, -7, 2] has Σd = 0 and 1ᵀd = 0, but μᵀd ≠ 0 for the sample
        // mean, so utility grows without limit along d.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let config = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = config.covariance(CovarianceConfig::default().with_shrinkage(0.1));
        assert!(allocator.allocate(&shrunk.clone().allow_short()).is_ok());
    }

    #[test]
    fn test_mean_variance_target_on_singular_covariance() {
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        let target = 0.004;
        let weights = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
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
        let config = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_element(3, 0.01).as_slice());
        assert_weights_close(
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );
    }

    #[test]
    fn test_covariance_adjustment_allows_constant_asset() {
        let returns = dmatrix![0.1, 0.01, 0.02; 0.1, -0.02, 0.01; 0.1, 0.03, -0.01];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );

        let shrunk = CovarianceConfig::default().with_shrinkage(1.0);
        assert_weights_close(
            &allocator
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(shrunk.clone()),
                )
                .unwrap(),
            &allocator.allocate(&EqualWeight).unwrap(),
            1e-12,
        );

        let regularized = CovarianceConfig::default().with_regularization(1e-6);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(regularized.clone()),
            )
            .unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let config = MeanVariance::risk_aversion(2.0).covariance(shrunk);
        assert!(allocator.allocate(&config.clone().allow_short()).is_ok());
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
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &expected,
            1e-12,
        );
    }

    #[test]
    fn test_hrp_two_assets_is_inverse_variance() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = PortfolioAllocator::from_prices(prices).unwrap();
        let (v1, v2) = (allocator.cov_matrix[(0, 0)], allocator.cov_matrix[(1, 1)]);
        let w1 = (1.0 / v1) / (1.0 / v1 + 1.0 / v2);
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
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
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &expected,
            1e-12,
        );
    }

    #[test]
    fn test_hrp_order_groups_correlated_assets() {
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
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
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
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
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
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

        let base = PortfolioAllocator::from_returns(returns)
            .unwrap()
            .allocate(&Hrp::default())
            .unwrap();
        let shuffled = PortfolioAllocator::from_returns(permuted)
            .unwrap()
            .allocate(&Hrp::default())
            .unwrap();

        for (j, &asset) in permutation.iter().enumerate() {
            assert!((shuffled[j] - base[asset]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_hrp_single_asset() {
        let allocator = PortfolioAllocator::from_prices(dmatrix![100.0; 110.0; 105.0]).unwrap();
        assert_eq!(Hrp::default().cluster_order(&allocator).unwrap(), vec![0]);
        assert_eq!(
            allocator.allocate(&Hrp::default()).unwrap().as_slice(),
            &[1.0]
        );
    }

    #[test]
    fn test_hrp_singular_covariance() {
        // HRP never inverts Σ, so rank-deficient data still gives positive weights.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
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
        let weights = PortfolioAllocator::from_returns(returns)
            .unwrap()
            .allocate(&Hrp::default())
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
                allocator.allocate(&Hrp::default()).err(),
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
            &allocator.allocate(&Hrp::default()).unwrap(),
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
            &allocator.allocate(&InverseVolatility::default()).unwrap(),
            &normalize([1.0 / s1, 1.0 / s2, 1.0 / s3]),
            1e-12,
        );
        assert_weights_close(
            &allocator.allocate(&InverseVariance::default()).unwrap(),
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
        let inverse_variance = allocator.allocate(&InverseVariance::default()).unwrap();
        assert_weights_close(
            &inverse_variance,
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-12,
        );
        assert_weights_close(
            &inverse_variance,
            &allocator.allocate(&Hrp::default()).unwrap(),
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
        let weights = allocator.allocate(&InverseVolatility::default()).unwrap();
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
                base.allocate(&InverseVolatility::default()).unwrap(),
                scaled.allocate(&InverseVolatility::default()).unwrap(),
            ),
            (
                base.allocate(&InverseVariance::default()).unwrap(),
                scaled.allocate(&InverseVariance::default()).unwrap(),
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
            allocator.allocate(&InverseVolatility::default()).err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
        assert_eq!(
            allocator.allocate(&InverseVariance::default()).err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
    }

    fn sharpe(allocator: &PortfolioAllocator, weights: &Weights, risk_free_rate: f64) -> f64 {
        let w = DVector::from_column_slice(weights.as_slice());
        (sample_mean(allocator).dot(&w) - risk_free_rate)
            / portfolio_variance(allocator, weights).sqrt()
    }

    fn min_variance_return(allocator: &PortfolioAllocator) -> f64 {
        let w = DVector::from_column_slice(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap()
                .as_slice(),
        );
        sample_mean(allocator).dot(&w)
    }

    #[test]
    fn test_max_sharpe_closed_form() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let weights = allocator.allocate(&config.clone().allow_short()).unwrap();

        let excess = sample_mean(&allocator).add_scalar(-risk_free_rate);
        let z = allocator.cov_matrix.clone().try_inverse().unwrap() * excess;
        let expected = Weights((&z / z.sum()).iter().copied().collect());
        assert_weights_close(&weights, &expected, 1e-10);
    }

    #[test]
    fn test_max_sharpe_beats_every_other_method() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let best = sharpe(
            &allocator,
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            risk_free_rate,
        );

        let others = [
            allocator.allocate(&EqualWeight).unwrap(),
            allocator.allocate(&InverseVolatility::default()).unwrap(),
            allocator.allocate(&InverseVariance::default()).unwrap(),
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            allocator.allocate(&Hrp::default()).unwrap(),
            allocator
                .allocate(&MeanVariance::risk_aversion(50.0).allow_short())
                .unwrap(),
        ];
        for weights in &others {
            assert!(sharpe(&allocator, weights, risk_free_rate) <= best + 1e-12);
        }
    }

    #[test]
    fn test_max_sharpe_lies_on_the_efficient_frontier() {
        // The tangency portfolio is the minimum variance portfolio for its own return.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let tangency = allocator
            .allocate(
                &MaxSharpe::default()
                    .risk_free_rate(risk_free_rate)
                    .allow_short(),
            )
            .unwrap();
        let target = sample_mean(&allocator).dot(&DVector::from_column_slice(tangency.as_slice()));
        let frontier = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();
        assert_weights_close(&tangency, &frontier, 1e-9);
    }

    #[test]
    fn test_max_sharpe_rejects_risk_free_rate_at_or_above_min_variance_return() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let threshold = min_variance_return(&allocator);
        for risk_free_rate in [threshold, threshold + 0.001] {
            let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
            match allocator.allocate(&config.clone().allow_short()) {
                Err(CutupError::RiskFreeRateTooHigh {
                    risk_free_rate: got,
                    min_variance_return,
                }) => {
                    assert_eq!(got, risk_free_rate);
                    assert!((min_variance_return - threshold).abs() < 1e-12);
                }
                other => panic!("risk-free rate {risk_free_rate} gave {other:?}"),
            }
        }
    }

    #[test]
    fn test_max_sharpe_singular_covariance() {
        // The sample mean isn't in Σ's range, so some zero-variance portfolio earns excess
        // return and the Sharpe ratio has no maximum. Shrinkage restores one.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        assert_eq!(
            allocator
                .allocate(&MaxSharpe::default().allow_short())
                .err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = MaxSharpe::default()
            .risk_free_rate(-0.01)
            .covariance(CovarianceConfig::default().with_shrinkage(0.5));
        let weights = allocator.allocate(&shrunk.clone().allow_short()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// Asset 1 is exactly 2 × asset 0 + 0.001, so (2, −1) is a fully invested portfolio with
    /// zero variance that returns −0.001 every period.
    fn collinear_returns() -> DMatrix<f64> {
        let base = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
        DMatrix::from_fn(6, 2, |t, j| {
            if j == 0 {
                base[t]
            } else {
                2.0 * base[t] + 0.001
            }
        })
    }

    #[test]
    fn test_max_sharpe_collinear_assets() {
        let allocator = PortfolioAllocator::from_returns(collinear_returns()).unwrap();

        // The zero-variance portfolio is the minimum variance portfolio.
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            &Weights(vec![2.0, -1.0]),
            1e-9,
        );

        // It returns −0.001, below a risk-free rate of 0: no maximum, and the threshold reported
        // is its return.
        match allocator.allocate(&MaxSharpe::default().allow_short()) {
            Err(CutupError::RiskFreeRateTooHigh {
                min_variance_return,
                ..
            }) => assert!((min_variance_return + 0.001).abs() < 1e-12),
            other => panic!("expected RiskFreeRateTooHigh, got {other:?}"),
        }

        // Below its return, the zero-variance portfolio has excess return and no risk, so the
        // Sharpe ratio is unbounded.
        let config = MaxSharpe::default().risk_free_rate(-0.01);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::Unbounded)
        );
    }

    #[test]
    fn test_max_sharpe_just_below_the_threshold() {
        // A rate clearly below the threshold (by far more than the rounding tolerance) has a
        // maximum, however leveraged.
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let threshold = min_variance_return(&allocator);
        let config = MaxSharpe::default().risk_free_rate(threshold - 1e-6);
        let weights = allocator.allocate(&config.clone().allow_short()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_sharpe_rejects_invalid_config() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        match allocator.allocate(&MaxSharpe::default().risk_free_rate(f64::NAN).allow_short()) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "risk_free_rate"),
            other => panic!("NaN risk-free rate gave {other:?}"),
        }
        let config =
            MaxSharpe::default().expected_returns(DVector::from_vec(vec![0.01]).as_slice());
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::DimensionMismatch {
                expected: 3,
                actual: 1
            })
        );

        let constant =
            PortfolioAllocator::from_returns(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            constant.allocate(&MaxSharpe::default().allow_short()).err(),
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
        let ew_weights = allocator.allocate(&EqualWeight).unwrap();
        assert_eq!(ew_weights.as_slice(), &[1.0]);
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
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(config.clone()),
                )
                .unwrap();
            assert_eq!(weights.len(), 3, "{config:?}");
            assert!(
                (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
                "{config:?}"
            );
        }
    }

    /// Each asset's share of the portfolio variance, `wᵢ (Σw)ᵢ / wᵀΣw`.
    fn risk_shares(allocator: &PortfolioAllocator, weights: &Weights) -> Vec<f64> {
        let w = DVector::from_column_slice(weights.as_slice());
        let contributions = w.component_mul(&(&allocator.cov_matrix * &w));
        let total = contributions.sum();
        contributions.iter().map(|c| c / total).collect()
    }

    #[test]
    fn test_risk_parity_equalizes_risk_contributions() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
        for share in risk_shares(&allocator, &weights) {
            assert!((share - 1.0 / 3.0).abs() < 1e-12, "{share}");
        }
    }

    #[test]
    fn test_risk_parity_meets_risk_budgets() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        // Spanning eight orders of magnitude, and not summing to 1.
        let budgets = [1e-8, 3.0, 1.0];
        let config =
            RiskParity::default().risk_budgets(DVector::from_row_slice(&budgets).as_slice());
        let weights = allocator.allocate(&config).unwrap();
        let total: f64 = budgets.iter().sum();
        for (share, budget) in risk_shares(&allocator, &weights).iter().zip(budgets) {
            assert!(
                (share - budget / total).abs() < 1e-12,
                "{share} vs {}",
                budget / total
            );
        }
    }

    #[test]
    fn test_risk_parity_converges_below_rounding_noise() {
        // Found by the property tests: near the solution the decrease a line search checks
        // for is below the rounding in the objective, so only full Newton steps finish.
        let returns = DMatrix::from_column_slice(
            5,
            2,
            &[
                -0.011054998419255687,
                0.02268592417867769,
                0.028346158812172808,
                -0.04316318369020081,
                0.04868903549835187,
                -0.01337776138331863,
                0.012639084624998789,
                -0.04497829673506405,
                -0.030576298169914445,
                -0.04666742643312797,
            ],
        );
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = RiskParity::default()
            .risk_budgets(
                DVector::from_row_slice(&[0.8571666963481355, 0.45606979884293564]).as_slice(),
            )
            .covariance(CovarianceConfig::default().with_shrinkage(0.29613802865778266));
        assert!(allocator.allocate(&config).is_ok());
    }

    #[test]
    fn test_risk_parity_extreme_budgets() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let budgets =
            |b: &[f64]| RiskParity::default().risk_budgets(DVector::from_row_slice(b).as_slice());
        // Budgets whose sum overflows give the same weights as equal budgets.
        assert_weights_close(
            &allocator
                .allocate(&budgets(&[1e308, 1e308, 1e308]))
                .unwrap(),
            &allocator.allocate(&RiskParity::default()).unwrap(),
            1e-15,
        );
        // Budgets 10¹⁴ apart still converge.
        let weights = allocator.allocate(&budgets(&[1.0, 1e-14, 1e-14])).unwrap();
        let shares = risk_shares(&allocator, &weights);
        // Rounding in (Σw)ᵢ limits a tiny share's relative accuracy, so compare absolutely.
        for (share, budget) in shares.iter().zip([1.0, 1e-14, 1e-14]) {
            assert!((share - budget).abs() < 1e-12, "{share} vs {budget}");
        }
        // Far beyond that the solver can stop early, which isn't a singular covariance.
        assert_eq!(
            allocator.allocate(&budgets(&[1.0, 1e-200, 1.0])),
            Err(CutupError::NotConverged)
        );
    }

    #[test]
    fn test_risk_parity_two_assets_is_inverse_volatility() {
        // With two assets, equal contributions mean w₀σ₀ = w₁σ₁ whatever the correlation.
        let allocator =
            PortfolioAllocator::from_prices(full_rank_prices().columns(0, 2).into_owned()).unwrap();
        assert_weights_close(
            &allocator.allocate(&RiskParity::default()).unwrap(),
            &allocator.allocate(&InverseVolatility::default()).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_risk_parity_ignores_return_and_budget_scale() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let scaled = PortfolioAllocator::from_returns(&allocator.returns * 1e-3).unwrap();
        let budgets = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let config = RiskParity::default().risk_budgets(budgets.as_slice());
        let scaled_config = RiskParity::default().risk_budgets((budgets * 7.0).as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &scaled.allocate(&scaled_config).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_risk_parity_on_singular_covariance() {
        // Positively collinear assets have a singular Σ, but every long-only portfolio is risky.
        let allocator = PortfolioAllocator::from_returns(collinear_returns()).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        assert_weights_close(&weights, &Weights(vec![2.0 / 3.0, 1.0 / 3.0]), 1e-12);

        // Here the long-only portfolio (1, 1, 1) / 3 has zero variance.
        let allocator = PortfolioAllocator::from_returns(rank_one_returns()).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::SingularCovariance)
        );
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().with_shrinkage(0.3));
        let weights = allocator.allocate(&config).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_risk_parity_anti_correlated_pair() {
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.015, -0.015];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::SingularCovariance)
        );
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().with_regularization(1e-6));
        let weights = allocator.allocate(&config).unwrap();
        assert_weights_close(&weights, &Weights(vec![0.5, 0.5]), 1e-12);
    }

    #[test]
    fn test_risk_parity_rejects_invalid_config() {
        let allocator = PortfolioAllocator::from_prices(full_rank_prices()).unwrap();
        let budgets =
            |b: &[f64]| RiskParity::default().risk_budgets(DVector::from_row_slice(b).as_slice());
        assert_eq!(
            allocator.allocate(&budgets(&[1.0, 1.0])),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        for bad in [0.0, -1.0, f64::INFINITY] {
            assert_eq!(
                allocator.allocate(&budgets(&[1.0, bad, 1.0])),
                Err(CutupError::InvalidConfig {
                    field: "risk_budgets",
                    value: bad
                })
            );
        }
        assert!(matches!(
            allocator.allocate(&budgets(&[1.0, 1.0, f64::NAN])),
            Err(CutupError::InvalidConfig {
                field: "risk_budgets",
                ..
            })
        ));
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().with_shrinkage(2.0));
        assert_eq!(
            allocator.allocate(&config),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 2.0
            })
        );
    }

    #[test]
    fn test_risk_parity_rejects_zero_variance() {
        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
        // Regularization gives the constant asset a variance, and nearly all the weight.
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().with_regularization(1e-8));
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights[1] > 0.99);
    }

    #[test]
    fn test_herc_with_one_cluster_is_inverse_variance() {
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
        let config = Herc::default().clusters(1);
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &allocator.allocate(&InverseVariance::default()).unwrap(),
            1e-15,
        );
    }

    #[test]
    fn test_herc_splits_between_blocks_by_risk() {
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
        let weights = allocator.allocate(&Herc::default().clusters(2)).unwrap();
        let (a, b) = ([0, 2, 4], [1, 3]);
        let (risk_a, risk_b) = (
            allocator.cluster_variance(&a),
            allocator.cluster_variance(&b),
        );
        let weight_a: f64 = a.iter().map(|&i| weights[i]).sum();
        assert!((weight_a - risk_b / (risk_a + risk_b)).abs() < 1e-12);
        // Within a block, inverse variance.
        let iv = allocator.allocate(&InverseVariance::default()).unwrap();
        assert!((weights[0] / weights[4] - iv[0] / iv[4]).abs() < 1e-12);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_herc_chooses_the_blocks() {
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
        let config = Herc::default().clusters(2);
        assert_eq!(
            allocator.allocate(&Herc::default()),
            allocator.allocate(&config)
        );
    }

    #[test]
    fn test_herc_two_assets_is_inverse_variance() {
        let allocator = PortfolioAllocator::from_returns(collinear_returns()).unwrap();
        let iv = allocator.allocate(&InverseVariance::default()).unwrap();
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
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = Herc::default().clusters(4);
        let weights = allocator.allocate(&config).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_herc_rejects_invalid_config() {
        let allocator = PortfolioAllocator::from_returns(two_block_returns()).unwrap();
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
            PortfolioAllocator::from_returns(returns)
                .unwrap()
                .allocate(&Herc::default()),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
    }
}
