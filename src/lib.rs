#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

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
            self.covariance = self.covariance.shrinkage(lambda);
            self
        }

        /// Adds `epsilon · I` to the covariance. See [`CovarianceConfig::regularization`].
        pub fn regularization(mut self, epsilon: f64) -> Self {
            self.covariance = self.covariance.regularization(epsilon);
            self
        }
    };
}

/// The `current_weights` and `max_turnover` setters, for a method with a `trading: Trading`
/// field.
macro_rules! turnover_setters {
    () => {
        /// Sets your current weights, [one per asset](PerAsset) in column order or by asset:
        /// what turnover and trading costs are measured against. By asset, those left out are
        /// 0. They must sum to 1, within `1e-9`; to hold cash, include it as an asset. On their
        /// own they change nothing.
        pub fn current_weights(mut self, current: impl Into<PerAsset>) -> Self {
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
mod asset;
mod black_litterman;
mod cluster;
mod cvar;
mod deprecated;
mod eigen;
mod lp;
mod qp;
mod turnover;

pub use allocation::{
    Allocation, EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility, MaxDiversification,
    MaxSharpe, MeanVariance, MinVariance, RiskParity,
};
pub use asset::{Asset, PerAsset};
pub use black_litterman::{BlackLitterman, View};
pub use cluster::Linkage;
pub use cvar::MinCvar;
#[allow(deprecated)]
pub use deprecated::PortfolioAllocator;

use eigen::SymmetricEigen;

use nalgebra::{DMatrix, DVector};
use std::fmt;
use std::ops::Index;
use std::sync::Arc;

/// Errors returned when building a `Returns` or computing an allocation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum CutupError {
    /// The input has no assets to allocate to: no rows, empty rows, or zero assets.
    NoAssets,
    /// The input has too few rows (observations) for a sample covariance.
    InsufficientData {
        /// Minimum number of rows needed.
        required: usize,
        /// Number of rows supplied.
        actual: usize,
    },
    /// A value in the input is NaN or infinite.
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
    /// A row of the input has a different number of values from the first row. For flat
    /// input, the last row is short: the number of values isn't a multiple of the number of
    /// assets.
    RaggedRows {
        /// The row, counting from 0.
        row: usize,
        /// The number of values in the first row, or the number of assets for flat input.
        expected: usize,
        /// The number of values in this row.
        actual: usize,
    },
    /// A configuration vector doesn't have one entry per asset.
    DimensionMismatch {
        /// Number of assets.
        expected: usize,
        /// Length supplied.
        actual: usize,
    },
    /// An [`Asset`] isn't in the data: no asset has its label, or its column is out of range.
    UnknownAsset {
        /// The label, or the column.
        asset: String,
    },
    /// An asset is given more than once: a label repeated in [`Returns::labeled`], or an asset
    /// named twice in the pairs of a [`PerAsset`].
    DuplicateAsset {
        /// The asset's label.
        asset: String,
    },
    /// [`Weights`] are labeled with other assets than the data they're used with, or in another
    /// order.
    LabelMismatch {
        /// The first column whose labels differ.
        column: usize,
        /// The data's label for it.
        expected: String,
        /// The weights' label for it.
        actual: String,
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
            CutupError::NoAssets => write!(f, "input has no assets"),
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
            CutupError::RaggedRows {
                row,
                expected,
                actual,
            } => write!(
                f,
                "row {row} has the wrong number of values: {actual}, not {expected}"
            ),
            CutupError::DimensionMismatch { expected, actual } => write!(
                f,
                "expected one value per asset ({expected}) but got {actual}"
            ),
            CutupError::UnknownAsset { asset } => write!(f, "the data has no asset `{asset}`"),
            CutupError::DuplicateAsset { asset } => {
                write!(f, "asset `{asset}` is given more than once")
            }
            CutupError::LabelMismatch {
                column,
                expected,
                actual,
            } => write!(
                f,
                "the weights label column {column} `{actual}`, but the data labels it `{expected}`"
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

/// Portfolio weights, one per asset, in the order of the columns of the [`Returns`], and labeled
/// with their assets.
///
/// Like a slice, it indexes and iterates by column. [`Weights::weight`] and [`Weights::labeled`]
/// go by label, which is the asset's column unless the data was
/// [labeled](Returns::labeled).
///
/// # Example
///
/// ```
/// use cutup::{EqualWeight, Returns};
///
/// let prices = [
///    [100.0, 200.0],
///    [110.0, 190.0],
///    [121.0, 209.0],
/// ];
///
/// let weights = Returns::from_prices(&prices)?.allocate(&EqualWeight)?;
///
/// assert_eq!(weights.len(), 2);
/// assert_eq!(weights[0], 0.5);
/// assert_eq!(weights.get(1), Some(0.5));
/// assert_eq!(weights.iter().sum::<f64>(), 1.0);
///
/// let labeled = Returns::from_prices(&prices)?.labeled(["SPY", "TLT"])?.allocate(&EqualWeight)?;
/// assert_eq!(labeled.weight("TLT"), Some(0.5));
/// assert_eq!(labeled["SPY"], 0.5);
/// for (label, weight) in labeled.labeled() {
///     println!("{label}: {weight}");
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    values: Vec<f64>,
    labels: Arc<[String]>,
}

impl Weights {
    /// Returns the weight of the asset in column `asset`, or `None` if out of range.
    pub fn get(&self, asset: usize) -> Option<f64> {
        self.values.get(asset).copied()
    }

    /// Returns the weight of the asset labeled `label`, or `None` if no asset has that label.
    pub fn weight(&self, label: &str) -> Option<f64> {
        self.position(label).map(|i| self.values[i])
    }

    /// Returns the number of assets.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if there are no weights.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterates over the weights in column order.
    pub fn iter(&self) -> std::slice::Iter<'_, f64> {
        self.values.iter()
    }

    /// Iterates over each asset's label and weight, in column order.
    pub fn labeled(&self) -> impl DoubleEndedIterator<Item = (&str, f64)> + ExactSizeIterator {
        self.labels
            .iter()
            .map(String::as_str)
            .zip(self.values.iter().copied())
    }

    /// Returns the assets' labels, in column order.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Returns the weights as a slice, in column order.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Consumes the weights and returns them as a `Vec`, in column order.
    pub fn into_vec(self) -> Vec<f64> {
        self.values
    }

    fn position(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| l == label)
    }
}

impl Index<usize> for Weights {
    type Output = f64;

    fn index(&self, asset: usize) -> &f64 {
        &self.values[asset]
    }
}

/// The weight of the asset labeled with the index.
///
/// # Panics
///
/// If no asset has that label. [`Weights::weight`] returns `None` instead.
impl Index<&str> for Weights {
    type Output = f64;

    fn index(&self, label: &str) -> &f64 {
        match self.position(label) {
            Some(i) => &self.values[i],
            None => panic!("no asset is labeled `{label}`"),
        }
    }
}

impl<'a> IntoIterator for &'a Weights {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl From<Weights> for Vec<f64> {
    fn from(weights: Weights) -> Self {
        weights.values
    }
}

/// The weights as a column vector, with the `nalgebra` feature.
#[cfg(feature = "nalgebra")]
#[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
impl From<Weights> for DVector<f64> {
    fn from(weights: Weights) -> Self {
        DVector::from_vec(weights.values)
    }
}

/// Weights in column order, for an [`Allocation`] of your own. [`Returns::allocate`]
/// checks there's one finite weight per asset, but not the sum, so make them sum to 1, as every
/// method in this crate does. It labels them with the data's labels; until then each is labeled
/// with its column.
impl From<Vec<f64>> for Weights {
    fn from(values: Vec<f64>) -> Self {
        let labels = asset::column_labels(values.len());
        Weights { values, labels }
    }
}

/// Adjustments applied to the sample covariance matrix before a method uses it, set with its
/// `covariance` setter, such as [`MinVariance::covariance`]. Each method also has `shrinkage`
/// and `regularization` setters for each adjustment on its own.
///
/// The default applies no adjustment, and an adjustment that isn't set isn't applied. With
/// both, the covariance becomes `λ·μ·I + (1 − λ)·Σ + ε·I`, which is the same whichever is
/// applied first.
///
/// ```
/// use cutup::{CovarianceConfig, MinVariance, Returns};
///
/// let config = CovarianceConfig::default().shrinkage(0.1).regularization(1e-8);
/// assert_eq!((config.lambda(), config.epsilon()), (Some(0.1), Some(1e-8)));
/// assert_eq!(CovarianceConfig::default().lambda(), None);
///
/// // Full shrinkage leaves every asset with the same variance and no correlation, so the
/// // minimum variance portfolio holds them equally.
/// let data = Returns::new(&[
///      [ 0.010,  0.021,  0.004],
///      [-0.008, -0.015, -0.002],
///      [ 0.004,  0.009,  0.006],
/// ])?;
/// let full = CovarianceConfig::default().shrinkage(1.0);
/// let weights = data.allocate(&MinVariance::default().covariance(full))?;
/// assert!(weights.iter().all(|w| (w - 1.0 / 3.0).abs() < 1e-12));
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CovarianceConfig {
    regularization: Option<f64>,
    shrinkage: Option<f64>,
}

impl CovarianceConfig {
    /// Adds `ε · I` to the covariance matrix to improve numerical stability. `ε` must be finite
    /// and nonnegative. Unlike shrinkage it's absolute, so choose it relative to the scale of
    /// your returns' variances.
    ///
    /// <https://en.wikipedia.org/wiki/Tikhonov_regularization>
    pub fn regularization(mut self, epsilon: f64) -> Self {
        self.regularization = Some(epsilon);
        self
    }

    /// Shrinks the covariance matrix toward a scaled identity with intensity `λ` in `[0, 1]`:
    /// it becomes `λ·μ·I + (1 − λ)·Σ`, where `μ = trace(Σ) / n` is the average variance, so
    /// the target is on the same scale as `Σ`. Shrinkage reduces the estimation error in a
    /// sample covariance.
    ///
    /// <https://en.wikipedia.org/wiki/Shrinkage_estimator>
    pub fn shrinkage(mut self, lambda: f64) -> Self {
        self.shrinkage = Some(lambda);
        self
    }

    /// The shrinkage intensity `λ` set with [`CovarianceConfig::shrinkage`], or `None`.
    pub fn lambda(&self) -> Option<f64> {
        self.shrinkage
    }

    /// The regularization `ε` set with [`CovarianceConfig::regularization`], or `None`.
    pub fn epsilon(&self) -> Option<f64> {
        self.regularization
    }

    /// Returns [`CutupError::InvalidConfig`] if the regularization is negative or not finite,
    /// or the shrinkage is outside `[0, 1]`.
    fn validate(&self) -> Result<(), CutupError> {
        if let Some(value) = self
            .regularization
            .filter(|eps| !(eps.is_finite() && *eps >= 0.0))
        {
            return Err(CutupError::InvalidConfig {
                field: "regularization",
                value,
            });
        }
        if let Some(value) = self
            .shrinkage
            .filter(|lambda| !(0.0..=1.0).contains(lambda))
        {
            return Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value,
            });
        }
        Ok(())
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
    fn resolve(&self, labels: &[String]) -> Result<DVector<f64>, CutupError> {
        let n = labels.len();
        match self {
            Limit::Uniform(value) => Ok(DVector::from_element(n, *value)),
            Limit::PerAsset(values) => {
                let values = values.resolve(labels, None)?;
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
    fn resolve(&self, labels: &[String]) -> Result<qp::Bounds, CutupError> {
        let n = labels.len();
        let mut lower = self.lower.resolve(labels)?;
        let mut upper = self.upper.resolve(labels)?;
        for (asset, l, u) in &self.assets {
            let i = asset.column(labels)?;
            lower[i] = *l;
            upper[i] = *u;
        }
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

/// Asset returns, with their sample covariance: the data every allocation method works on.
/// Pass an [`Allocation`], such as [`MinVariance`] or [`Hrp`], to [`Returns::allocate`] to
/// get [`Weights`].
///
/// All covariance-based methods work on the covariance of asset returns, never of price levels.
/// Build one from prices with [`Returns::from_prices`], which converts them to returns, or from
/// returns you have already computed with [`Returns::new`].
///
/// Before 0.3.0 this was `PortfolioAllocator`, which stays as a deprecated alias until 1.0.0.
pub struct Returns {
    returns: DMatrix<f64>,
    cov_matrix: DMatrix<f64>,
    labels: Arc<[String]>,
}

impl Returns {
    /// Creates a new `Returns` from asset prices, one row per period and one value
    /// per asset in each row.
    ///
    /// Prices are converted to simple returns, `r_t = p_t / p_{t-1} - 1`, before the covariance
    /// matrix is computed. This matches `pandas.DataFrame.pct_change`. `T` rows of prices
    /// produce `T - 1` rows of returns, so at least three rows of prices are needed for a
    /// sample covariance.
    ///
    /// Rows can be arrays, `Vec`s or slices: anything that is `AsRef<[f64]>`. For one flat,
    /// row-major slice, use [`Returns::from_prices_flat`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if there are no rows, or the rows are empty.
    /// * [`CutupError::RaggedRows`] if a row has a different length from the first.
    /// * [`CutupError::InsufficientData`] if there are fewer than three rows.
    /// * [`CutupError::NonFiniteValue`] if any price is NaN or infinite.
    /// * [`CutupError::NonPositivePrice`] if any price is zero or negative.
    /// * [`CutupError::NonFiniteCovariance`] if the returns are too extreme for a finite covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let prices = [
    ///    [100.0, 200.0, 300.0],
    ///    [110.0, 210.0, 310.0],
    ///    [120.0, 220.0, 320.0],
    /// ];
    ///
    /// let data = Returns::from_prices(&prices)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn from_prices<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, CutupError> {
        Returns::prices_matrix(Returns::rows_matrix(rows)?)
    }

    /// Creates a new `Returns` from asset prices in one flat slice, row-major: the
    /// first `assets` values are the first period's prices, the next `assets` the second's,
    /// and so on. This is the layout of a JavaScript `Float64Array` or a CSV read row by row.
    /// Otherwise it's the same as [`Returns::from_prices`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if `assets` is 0.
    /// * [`CutupError::RaggedRows`] if the number of values isn't a multiple of `assets`.
    /// * Otherwise as for [`Returns::from_prices`].
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let prices = [
    ///     100.0, 200.0, 300.0,
    ///     110.0, 210.0, 310.0,
    ///     120.0, 220.0, 320.0,
    /// ];
    ///
    /// let data = Returns::from_prices_flat(&prices, 3)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn from_prices_flat(values: &[f64], assets: usize) -> Result<Self, CutupError> {
        Returns::prices_matrix(Returns::flat_matrix(values, assets)?)
    }

    /// Creates a new `Returns` from a matrix of asset prices (rows: time, columns:
    /// assets), with the `nalgebra` feature. Otherwise it's the same as
    /// [`Returns::from_prices`].
    ///
    /// The feature tracks the nalgebra version cutup depends on, so a nalgebra release that
    /// isn't compatible can change which `DMatrix` this takes.
    ///
    /// # Errors
    ///
    /// As for [`Returns::from_prices`], except [`CutupError::RaggedRows`], which a
    /// matrix can't have.
    #[cfg(feature = "nalgebra")]
    #[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
    pub fn from_prices_matrix(prices: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::prices_matrix(prices)
    }

    /// Creates a new `Returns` from asset returns, one row per period and one value
    /// per asset in each row.
    ///
    /// Use this when you already have returns (for example log returns, or returns net of a
    /// benchmark). They are used as-is; at least two rows are needed for a sample covariance.
    /// For prices, use [`Returns::from_prices`]: in 0.1, `PortfolioAllocator::new` took prices,
    /// but this takes returns, and prices passed here are treated as returns.
    ///
    /// Rows can be arrays, `Vec`s or slices: anything that is `AsRef<[f64]>`. For one flat,
    /// row-major slice, use [`Returns::new_flat`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if there are no rows, or the rows are empty.
    /// * [`CutupError::RaggedRows`] if a row has a different length from the first.
    /// * [`CutupError::InsufficientData`] if there are fewer than two rows.
    /// * [`CutupError::NonFiniteValue`] if any return is NaN or infinite.
    /// * [`CutupError::NonFiniteCovariance`] if the returns are too extreme for a finite covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let returns = [
    ///    [0.010, -0.020, 0.005],
    ///    [-0.004, 0.015, 0.002],
    ///    [0.007, 0.003, -0.006],
    /// ];
    ///
    /// let data = Returns::new(&returns)?;
    ///
    /// // Rows built at runtime work the same way.
    /// let rows: Vec<Vec<f64>> = returns.iter().map(|row| row.to_vec()).collect();
    /// let from_vecs = Returns::new(&rows)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn new<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, CutupError> {
        Returns::returns_matrix(Returns::rows_matrix(rows)?)
    }

    /// Creates a new `Returns` from asset returns in one flat slice, row-major: the
    /// first `assets` values are the first period's returns, the next `assets` the second's,
    /// and so on. This is the layout of a JavaScript `Float64Array` or a CSV read row by row.
    /// Otherwise it's the same as [`Returns::new`].
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if `assets` is 0.
    /// * [`CutupError::RaggedRows`] if the number of values isn't a multiple of `assets`.
    /// * Otherwise as for [`Returns::new`].
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::Returns;
    ///
    /// let returns = [
    ///     0.010, -0.020, 0.005,
    ///     -0.004, 0.015, 0.002,
    ///     0.007, 0.003, -0.006,
    /// ];
    ///
    /// let data = Returns::new_flat(&returns, 3)?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn new_flat(values: &[f64], assets: usize) -> Result<Self, CutupError> {
        Returns::returns_matrix(Returns::flat_matrix(values, assets)?)
    }

    /// Creates a new `Returns` from a matrix of asset returns (rows: time, columns:
    /// assets), with the `nalgebra` feature. Otherwise it's the same as
    /// [`Returns::new`].
    ///
    /// The feature tracks the nalgebra version cutup depends on, so a nalgebra release that
    /// isn't compatible can change which `DMatrix` this takes.
    ///
    /// # Errors
    ///
    /// As for [`Returns::new`], except [`CutupError::RaggedRows`], which a
    /// matrix can't have.
    #[cfg(feature = "nalgebra")]
    #[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
    pub fn from_matrix(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::returns_matrix(returns)
    }

    /// Builds the matrix for rows, checking that they all have the first row's length.
    fn rows_matrix<R: AsRef<[f64]>>(rows: &[R]) -> Result<DMatrix<f64>, CutupError> {
        let assets = rows.first().map_or(0, |row| row.as_ref().len());
        if let Some((row, values)) = rows
            .iter()
            .map(AsRef::as_ref)
            .enumerate()
            .find(|(_, values)| values.len() != assets)
        {
            return Err(CutupError::RaggedRows {
                row,
                expected: assets,
                actual: values.len(),
            });
        }
        Ok(DMatrix::from_row_iterator(
            rows.len(),
            assets,
            rows.iter().flat_map(|row| row.as_ref().iter().copied()),
        ))
    }

    /// Builds the matrix for flat, row-major values, checking that they fill whole rows.
    fn flat_matrix(values: &[f64], assets: usize) -> Result<DMatrix<f64>, CutupError> {
        if assets == 0 {
            return Err(CutupError::NoAssets);
        }
        if values.len() % assets != 0 {
            return Err(CutupError::RaggedRows {
                row: values.len() / assets,
                expected: assets,
                actual: values.len() % assets,
            });
        }
        Ok(DMatrix::from_row_slice(
            values.len() / assets,
            assets,
            values,
        ))
    }

    /// The `Returns` for a matrix of prices.
    fn prices_matrix(prices: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::validate(&prices, 3)?;
        if let Some((row, col)) = Returns::find(&prices, |p| p <= 0.0) {
            return Err(CutupError::NonPositivePrice { row, col });
        }
        Returns::build(Returns::compute_returns(&prices))
    }

    /// The `Returns` for a matrix of returns.
    fn returns_matrix(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        Returns::validate(&returns, 2)?;
        Returns::build(returns)
    }

    /// Labels the assets, in column order, for example with their tickers. The [`Weights`]
    /// carry the labels, and every setting that takes an [`Asset`] or [`PerAsset`] values can
    /// name assets by them. Without labels, each asset's label is its column: `"0"`, `"1"`, and
    /// so on.
    ///
    /// # Errors
    ///
    /// * [`CutupError::DimensionMismatch`] if there isn't one label per asset.
    /// * [`CutupError::DuplicateAsset`] if a label is repeated.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{InverseVariance, Returns};
    ///
    /// let returns = [
    ///    [0.010, -0.020, 0.005],
    ///    [-0.004, 0.015, 0.002],
    ///    [0.007, 0.003, -0.006],
    /// ];
    ///
    /// let data = Returns::new(&returns)?.labeled(["SPY", "TLT", "GLD"])?;
    /// assert_eq!(data.labels(), ["SPY", "TLT", "GLD"]);
    ///
    /// let weights = data.allocate(&InverseVariance)?;
    /// assert_eq!(weights.weight("TLT"), Some(weights[1]));
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn labeled<L: Into<String>>(
        mut self,
        labels: impl IntoIterator<Item = L>,
    ) -> Result<Self, CutupError> {
        let labels: Vec<String> = labels.into_iter().map(Into::into).collect();
        let n = self.returns.ncols();
        if labels.len() != n {
            return Err(CutupError::DimensionMismatch {
                expected: n,
                actual: labels.len(),
            });
        }
        let mut seen = std::collections::HashSet::with_capacity(n);
        if let Some(label) = labels.iter().find(|label| !seen.insert(label.as_str())) {
            return Err(CutupError::DuplicateAsset {
                asset: label.clone(),
            });
        }
        self.labels = labels.into();
        Ok(self)
    }

    /// Returns the assets' labels, in column order: those set with [`Returns::labeled`], or
    /// each asset's column.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Computes weights with `method`, any [`Allocation`]: one of this crate's methods, such as
    /// [`MinVariance`] or [`Hrp`], or one of your own. The weights carry the data's
    /// [labels](Returns::labels).
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
    /// use cutup::{MinVariance, Returns};
    ///
    /// let returns = [
    ///      [ 0.010, -0.020,  0.005],
    ///      [-0.004,  0.015,  0.002],
    ///      [ 0.007,  0.003, -0.006],
    ///      [ 0.002, -0.001,  0.004],
    /// ];
    /// let data = Returns::new(&returns)?;
    ///
    /// let long_only = data.allocate(&MinVariance::default())?;
    /// let long_short = data.allocate(&MinVariance::default().allow_short())?;
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
        Ok(self.weights(weights.values))
    }

    /// `values`, given for every asset, in column order as a vector.
    fn resolve(&self, values: Option<&PerAsset>) -> Result<Option<DVector<f64>>, CutupError> {
        values
            .map(|values| Ok(DVector::from_vec(values.resolve(&self.labels, None)?)))
            .transpose()
    }

    /// `values`, one per asset in column order, as weights labeled with the data's labels.
    fn weights(&self, values: Vec<f64>) -> Weights {
        Weights {
            values,
            labels: Arc::clone(&self.labels),
        }
    }

    /// Builds a `Returns` from returns that have already been validated, rejecting a
    /// covariance matrix that overflowed.
    fn build(returns: DMatrix<f64>) -> Result<Self, CutupError> {
        let cov_matrix = Returns::compute_covariance_matrix(&returns);
        if cov_matrix.iter().any(|x| !x.is_finite()) {
            return Err(CutupError::NonFiniteCovariance);
        }
        let labels = asset::column_labels(returns.ncols());
        Ok(Returns {
            returns,
            cov_matrix,
            labels,
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
        if let Some((row, col)) = Returns::find(data, |x| !x.is_finite()) {
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
        config.validate()?;
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
        Ok(self.weights(weights.iter().copied().collect()))
    }

    /// The minimum variance portfolio within `bounds`, for [`MinVariance`].
    fn bounded_min_variance(
        &self,
        covariance: &CovarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        let bounds = bounds.resolve(&self.labels)?;
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
        .map(|w| self.weights(w))
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
        Ok(self.weights(weights.iter().copied().collect()))
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
        let bounds = bounds.resolve(&self.labels)?;
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
                )
                .map(|w| self.weights(w));
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
        .map(|w| self.weights(w))
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
        Ok(self.weights(weights.iter().copied().collect()))
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
        long_only_max_ratio(&cov, &(&excess / excess[best]), best)?
            .map(|w| self.weights(w))
            .ok_or(CutupError::Unbounded)
    }

    /// The maximum diversification portfolio, for [`MaxDiversification`].
    fn max_diversification(&self, covariance: &CovarianceConfig) -> Result<Weights, CutupError> {
        let cov = self.adjusted_covariance(covariance)?;
        let volatilities = cov.diagonal().map(f64::sqrt);
        let (best, highest) = volatilities.argmax();
        long_only_max_ratio(&cov, &(volatilities / highest), best)?
            .map(|w| self.weights(w))
            .ok_or(CutupError::SingularCovariance)
    }

    /// Weights each asset in proportion to `f(σᵢ²)`, normalized to sum to 1.
    fn weight_by_variance(&self, f: impl Fn(f64) -> f64) -> Result<Weights, CutupError> {
        if let Some(asset) = self.find_zero_variance() {
            return Err(CutupError::ZeroVariance { asset });
        }
        let raw: Vec<f64> = self.cov_matrix.diagonal().iter().map(|&v| f(v)).collect();
        let total: f64 = raw.iter().sum();
        Ok(self.weights(raw.iter().map(|w| w / total).collect()))
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
                });
            }
            Some(b) => match b.iter().find(|b| !(b.is_finite() && **b > 0.0)) {
                Some(&value) => {
                    return Err(CutupError::InvalidConfig {
                        field: "risk_budgets",
                        value,
                    });
                }
                None => b.clone(),
            },
            None => DVector::from_element(n, 1.0),
        };
        let cov = self.adjusted_covariance(covariance)?;
        Returns::risk_budget_weights(&cov, &budgets)
            .map(|w| self.weights(w.iter().copied().collect()))
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

    /// Returns the variance under `cov` of the inverse-variance portfolio over `cluster`.
    fn cluster_variance(cov: &DMatrix<f64>, cluster: &[usize]) -> f64 {
        let weights = Returns::inverse_variance_weights(cov, cluster);
        let mut variance = 0.0;
        for (a, &i) in cluster.iter().enumerate() {
            for (b, &j) in cluster.iter().enumerate() {
                variance += weights[a] * weights[b] * cov[(i, j)];
            }
        }
        variance
    }

    /// Returns the inverse-variance weights under `cov` of the assets in `cluster`, summing
    /// to 1.
    fn inverse_variance_weights(cov: &DMatrix<f64>, cluster: &[usize]) -> Vec<f64> {
        let inverse: Vec<f64> = cluster.iter().map(|&i| 1.0 / cov[(i, i)]).collect();
        let total: f64 = inverse.iter().sum();
        inverse.iter().map(|w| w / total).collect()
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
                let v_left = Returns::cluster_variance(&cov, left).max(0.0);
                let v_right = Returns::cluster_variance(&cov, right).max(0.0);
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
            return Err(CutupError::ZeroVariance { asset });
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
) -> Result<Vec<f64>, CutupError> {
    Ok(normalized(qp::solve_bounded(q, c, e, b, bounds, start)?))
}

/// `x` rescaled to sum to 1. With a budget constraint `1ᵀx = 1`, that only removes the
/// rounding left by clamping to the bounds; for max Sharpe it maps `y` back to `w`.
fn normalized(x: DVector<f64>) -> Vec<f64> {
    let total = x.sum();
    x.iter().map(|w| w / total).collect()
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
) -> Result<Option<Vec<f64>>, CutupError> {
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
        let returns = Returns::compute_returns(&prices);
        let expected = dmatrix![
            0.1, -0.05;
            0.1, 0.1
        ];
        assert_eq!(returns.shape(), (2, 2));
        assert!((returns - expected).abs().max() < 1e-12);
    }

    #[test]
    fn test_from_prices_matches_new() {
        let prices = full_rank_prices();
        let returns = Returns::compute_returns(&prices);
        let from_prices = Returns::prices_matrix(prices).unwrap();
        let new = Returns::returns_matrix(returns).unwrap();
        assert_eq!(from_prices.cov_matrix, new.cov_matrix);
    }

    #[test]
    fn test_plain_inputs_match_the_matrix() {
        let rows = [
            [0.01, -0.02, 0.005],
            [-0.004, 0.015, 0.002],
            [0.007, 0.003, -0.006],
        ];
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let matrix = DMatrix::from_fn(3, 3, |t, j| rows[t][j]);
        let expected = Returns::returns_matrix(matrix.clone()).unwrap();
        let vecs: Vec<Vec<f64>> = rows.iter().map(|row| row.to_vec()).collect();
        let slices: Vec<&[f64]> = rows.iter().map(|row| row.as_slice()).collect();
        for allocator in [
            Returns::new(&rows).unwrap(),
            Returns::new(&vecs).unwrap(),
            Returns::new(&slices).unwrap(),
            Returns::new_flat(&flat, 3).unwrap(),
        ] {
            assert_eq!(allocator.returns, expected.returns);
            assert_eq!(allocator.cov_matrix, expected.cov_matrix);
        }

        let prices = [[100.0, 50.0], [102.0, 49.0], [101.0, 51.5]];
        let flat: Vec<f64> = prices.iter().flatten().copied().collect();
        let expected = Returns::prices_matrix(DMatrix::from_fn(3, 2, |t, j| prices[t][j])).unwrap();
        for allocator in [
            Returns::from_prices(&prices).unwrap(),
            Returns::from_prices_flat(&flat, 2).unwrap(),
        ] {
            assert_eq!(allocator.returns, expected.returns);
            assert_eq!(allocator.cov_matrix, expected.cov_matrix);
        }
    }

    #[test]
    #[allow(deprecated)]
    fn test_old_names_are_the_new_ones() {
        let rows = [[0.01, -0.02], [-0.004, 0.015], [0.007, 0.003]];
        let old: PortfolioAllocator = PortfolioAllocator::from_returns(&rows).unwrap();
        let new = Returns::new(&rows).unwrap();
        assert_eq!(old.returns, new.returns);
        assert_eq!(old.cov_matrix, new.cov_matrix);
    }

    #[test]
    fn test_plain_inputs_reject_bad_shapes() {
        let ragged = vec![vec![0.01, 0.02], vec![0.03], vec![0.04, 0.05]];
        let error = Some(CutupError::RaggedRows {
            row: 1,
            expected: 2,
            actual: 1,
        });
        assert_eq!(Returns::new(&ragged).err(), error);
        assert_eq!(Returns::from_prices(&ragged).err(), error);
        // Seven values don't fill rows of three: the third row has one.
        let flat = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        let error = Some(CutupError::RaggedRows {
            row: 2,
            expected: 3,
            actual: 1,
        });
        assert_eq!(Returns::new_flat(&flat, 3).err(), error);
        assert_eq!(Returns::from_prices_flat(&flat, 3).err(), error);

        let none: [[f64; 2]; 0] = [];
        let empty: [[f64; 0]; 3] = [[], [], []];
        for result in [
            Returns::new(&none),
            Returns::new(&empty),
            Returns::from_prices(&none),
            Returns::new_flat(&flat, 0),
            Returns::new_flat(&[], 0),
            Returns::from_prices_flat(&flat, 0),
        ] {
            assert_eq!(result.err(), Some(CutupError::NoAssets));
        }
        // No values for a known number of assets is too few rows, not too few assets.
        assert_eq!(
            Returns::new_flat(&[], 3).err(),
            Some(CutupError::InsufficientData {
                required: 2,
                actual: 0
            })
        );
        assert_eq!(
            CutupError::RaggedRows {
                row: 1,
                expected: 2,
                actual: 1
            }
            .to_string(),
            "row 1 has the wrong number of values: 1, not 2"
        );
    }

    #[cfg(feature = "nalgebra")]
    #[test]
    fn test_matrix_inputs_and_vector_weights() {
        let prices = dmatrix![100.0, 50.0; 102.0, 49.0; 101.0, 51.5];
        let allocator = Returns::from_prices_matrix(prices.clone()).unwrap();
        assert_eq!(
            allocator.cov_matrix,
            Returns::prices_matrix(prices).unwrap().cov_matrix
        );
        let returns = allocator.returns.clone();
        let weights = Returns::from_matrix(returns)
            .unwrap()
            .allocate(&EqualWeight)
            .unwrap();
        assert_eq!(DVector::from(weights), DVector::from_element(2, 0.5));
    }

    #[test]
    fn test_from_prices_rejects_no_assets() {
        let result = Returns::prices_matrix(DMatrix::zeros(3, 0));
        assert_eq!(result.err(), Some(CutupError::NoAssets));
    }

    #[test]
    fn test_from_prices_rejects_too_few_rows() {
        let result = Returns::prices_matrix(dmatrix![100.0, 200.0; 110.0, 210.0]);
        assert_eq!(
            result.err(),
            Some(CutupError::InsufficientData {
                required: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn test_new_rejects_too_few_rows() {
        let result = Returns::returns_matrix(dmatrix![0.01, -0.02]);
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
        let result = Returns::prices_matrix(prices);
        assert_eq!(
            result.err(),
            Some(CutupError::NonFiniteValue { row: 2, col: 1 })
        );
    }

    #[test]
    fn test_new_rejects_non_finite_values() {
        let result = Returns::returns_matrix(dmatrix![0.01, 0.02; f64::NEG_INFINITY, 0.0]);
        assert_eq!(
            result.err(),
            Some(CutupError::NonFiniteValue { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_from_prices_rejects_non_positive_prices() {
        let mut prices = full_rank_prices();
        prices[(3, 2)] = 0.0;
        let result = Returns::prices_matrix(prices.clone());
        assert_eq!(
            result.err(),
            Some(CutupError::NonPositivePrice { row: 3, col: 2 })
        );

        prices[(1, 0)] = -5.0;
        let result = Returns::prices_matrix(prices);
        assert_eq!(
            result.err(),
            Some(CutupError::NonPositivePrice { row: 1, col: 0 })
        );
    }

    #[test]
    fn test_new_accepts_negative_returns() {
        let returns = dmatrix![-0.5, 0.01; 0.2, -0.03; -0.1, 0.02];
        assert!(Returns::returns_matrix(returns).is_ok());
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
            CutupError::UnknownAsset {
                asset: "QQQ".into()
            }
            .to_string(),
            "the data has no asset `QQQ`"
        );
        assert_eq!(
            CutupError::DuplicateAsset {
                asset: "SPY".into()
            }
            .to_string(),
            "asset `SPY` is given more than once"
        );
        assert_eq!(
            CutupError::LabelMismatch {
                column: 1,
                expected: "TLT".into(),
                actual: "GLD".into()
            }
            .to_string(),
            "the weights label column 1 `GLD`, but the data labels it `TLT`"
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
        let weights = Weights::from(vec![0.2, 0.3, 0.5]);
        assert_eq!(weights.len(), 3);
        assert!(!weights.is_empty());
        assert_eq!(weights[1], 0.3);
        assert_eq!(weights.get(2), Some(0.5));
        assert_eq!(weights.get(3), None);
        assert_eq!(weights.as_slice(), &[0.2, 0.3, 0.5]);
        assert_eq!((&weights).into_iter().count(), 3);

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

        let base = Returns::prices_matrix(prices).unwrap();
        let rescaled = Returns::prices_matrix(scaled).unwrap();
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
    fn test_equal_weight() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let ew_weights = allocator.allocate(&EqualWeight).unwrap();
        assert_eq!(ew_weights.as_slice(), &[1.0 / 3.0; 3]);
    }

    #[test]
    fn test_long_short_min_variance() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert_eq!(weights.len(), 3);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_variance_errors_on_non_finite_covariance() {
        let mut allocator = Returns::prices_matrix(full_rank_prices()).unwrap();

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
            Returns::prices_matrix(prices).err(),
            Some(CutupError::NonFiniteCovariance)
        );

        let returns = dmatrix![1e200, 0.01; -1e200, 0.02];
        assert_eq!(
            Returns::returns_matrix(returns).err(),
            Some(CutupError::NonFiniteCovariance)
        );
    }

    #[test]
    fn test_rounding_noise_counts_as_zero_variance() {
        // Constant returns whose computed variance is rounding noise rather than exactly zero.
        let allocator =
            Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
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
        let allocator = Returns::prices_matrix(prices).unwrap();
        assert_eq!(
            allocator.allocate(&Hrp::default()).err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
    }

    #[test]
    fn test_min_variance_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        assert!(
            allocator
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(edges.clone())
                )
                .is_ok()
        );
    }

    fn portfolio_variance(allocator: &Returns, weights: &Weights) -> f64 {
        let w = DVector::from_column_slice(weights.as_slice());
        (w.transpose() * &allocator.cov_matrix * &w)[(0, 0)]
    }

    fn sample_mean(allocator: &Returns) -> DVector<f64> {
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
            .regularization(1e-8)
            .shrinkage(0.25);
        assert_eq!(config.epsilon(), Some(1e-8));
        assert_eq!(config.lambda(), Some(0.25));
    }

    #[test]
    fn test_full_shrinkage_gives_equal_weights() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().shrinkage(1.0);
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
        let returns = Returns::compute_returns(&full_rank_prices());
        let config = CovarianceConfig::default().shrinkage(0.5);

        let base = Returns::returns_matrix(returns.clone()).unwrap();
        let scaled = Returns::returns_matrix(returns * 100.0).unwrap();

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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        let inverse = [1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)];
        let total: f64 = inverse.iter().sum();
        let expected = Weights::from(inverse.iter().map(|x| x / total).collect::<Vec<_>>());
        assert_weights_close(&weights, &expected, 1e-12);
    }

    #[test]
    fn test_min_variance_two_asset_closed_form() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = Returns::prices_matrix(prices).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        let cov = &allocator.cov_matrix;
        let (s11, s22, s12) = (cov[(0, 0)], cov[(1, 1)], cov[(0, 1)]);
        let w1 = (s22 - s12) / (s11 + s22 - 2.0 * s12);
        assert_weights_close(&weights, &Weights::from(vec![w1, 1.0 - w1]), 1e-12);
    }

    #[test]
    fn test_min_variance_beats_equal_weight() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(prices).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let plain = MeanVariance::risk_aversion(5.0);
        let shrunk = plain
            .clone()
            .covariance(CovarianceConfig::default().shrinkage(0.5));
        assert_ne!(
            allocator.allocate(&plain.clone().allow_short()).unwrap(),
            allocator.allocate(&shrunk.clone().allow_short()).unwrap()
        );

        let invalid = plain.covariance(CovarianceConfig::default().shrinkage(2.0));
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
            Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
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
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.5]), 1e-12);
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = CovarianceConfig::default().shrinkage(0.9832865013278819);
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let shrunk = CovarianceConfig::default().shrinkage(0.5);
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        // numpy.linalg.pinv's minimum-norm solution of the same KKT system, in correlation
        // units u = diag(σ)w.
        let expected = Weights::from(vec![0.275093649, 0.331571165, 0.219203116, 0.174132070]);
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
        let allocator = Returns::returns_matrix(returns).unwrap();
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
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();
        assert!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap()[1]
                < 0.0
        );
        assert_eq!(
            allocator.allocate(&MinVariance::default()).unwrap(),
            Weights::from(vec![1.0, 0.0])
        );
    }

    #[test]
    fn test_long_only_min_variance_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, so the minimum is zero.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let weights = allocator.allocate(&MinVariance::default()).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);

        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&MinVariance::default()).unwrap(),
            &Weights::from(vec![0.5, 0.5]),
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
            let allocator = Returns::returns_matrix(returns).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().shrinkage(1.5);
        assert_eq!(
            allocator.allocate(&MinVariance::default().covariance(config.clone())),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            })
        );

        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = Returns::returns_matrix(returns).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];

        // With almost no aversion to risk, everything goes to the highest expected return.
        let config = MeanVariance::risk_aversion(1e-9).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // Only the asset with the highest expected return reaches it.
        let config = MeanVariance::target_return(0.003).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_mean_variance_hits_the_target() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
    fn test_long_only_mean_variance_rejects_unreachable_targets() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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

    #[test]
    fn test_long_only_mean_variance_target_edge_cases() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();

        // A target a rounding error above the highest expected return is that return.
        let expected = dvector![0.001, 0.003, 0.002];
        let config = MeanVariance::target_return(0.003 * (1.0 + 4.0 * f64::EPSILON))
            .expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
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
        let allocator = Returns::returns_matrix(returns).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = MaxSharpe::default()
            .risk_free_rate(0.0005)
            .expected_returns(dvector![0.001, 0.002, 0.003].as_slice());
        let unconstrained = allocator.allocate(&config.clone().allow_short()).unwrap();
        assert!(unconstrained.iter().all(|&w| w > 0.0), "{unconstrained:?}");
        assert_weights_close(&allocator.allocate(&config).unwrap(), &unconstrained, 1e-12);
    }

    #[test]
    fn test_long_only_max_sharpe_risk_free_rate_limits() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        assert_weights_close(
            &long_only.unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-9,
        );

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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let cov = &allocator.cov_matrix;
        let sigma = cov.diagonal().map(f64::sqrt);
        let correlation = DMatrix::from_fn(3, 3, |i, j| cov[(i, j)] / (sigma[i] * sigma[j]));
        let v = correlation
            .lu()
            .solve(&DVector::from_element(3, 1.0))
            .unwrap();
        assert!(v.iter().all(|&v| v > 0.0), "{v}");
        let w = v.component_div(&sigma);
        let expected = Weights::from((&w / w.sum()).iter().copied().collect::<Vec<_>>());

        let weights = allocator.allocate(&MaxDiversification::default()).unwrap();
        assert_weights_close(&weights, &expected, 1e-12);
        assert_weights_close(
            &weights,
            &Weights::from(vec![0.29228043, 0.23774194, 0.46997763]),
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
            let allocator = Returns::returns_matrix(returns).unwrap();
            let inverse_volatility = allocator.allocate(&InverseVolatility).unwrap();
            let inverse_variance = allocator.allocate(&InverseVariance).unwrap();
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
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert_eq!(
            allocator.allocate(&MaxDiversification::default()),
            Err(CutupError::SingularCovariance)
        );

        // Shrinkage gives every portfolio some variance.
        let config = CovarianceConfig::default().shrinkage(0.3);
        let weights = allocator
            .allocate(&MaxDiversification::default().covariance(config.clone()))
            .unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // The collinear pair's zero-variance hedge shorts one asset, so long-only the ratio
        // is bounded: every mix of perfectly correlated assets has a ratio of 1.
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();
        let weights = allocator.allocate(&MaxDiversification::default()).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
    }

    #[test]
    fn test_max_diversification_rejects_bad_input() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().shrinkage(1.5);
        assert_eq!(
            allocator.allocate(&MaxDiversification::default().covariance(config.clone())),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            })
        );

        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&MaxDiversification::default()),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
    }

    #[test]
    fn test_long_only_max_sharpe_unbounded_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, and with these expected returns its
        // excess return is positive, so the Sharpe ratio grows without limit.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let config = MaxSharpe::default().expected_returns(dvector![0.01, 0.0, 0.01].as_slice());
        assert_eq!(allocator.allocate(&config), Err(CutupError::Unbounded));

        // Shrinkage gives every portfolio some variance.
        let config = config.covariance(CovarianceConfig::default().shrinkage(0.3));
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_long_only_max_sharpe_rejects_bad_input() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let config = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = config.covariance(CovarianceConfig::default().shrinkage(0.1));
        assert!(allocator.allocate(&shrunk.clone().allow_short()).is_ok());
    }

    #[test]
    fn test_mean_variance_target_on_singular_covariance() {
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );

        let shrunk = CovarianceConfig::default().shrinkage(1.0);
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

        let regularized = CovarianceConfig::default().regularization(1e-6);
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
    fn test_hrp_rejects_underflowing_correlation() {
        // Variances around 1e-320 pass validation, but their product underflows to zero, so ρ
        // is infinite (nonzero covariance) or NaN (zero covariance, orthogonal columns).
        let s = 1e-160;
        let infinite = dmatrix![s, 2.0 * s; -s, 3.0 * s; 2.0 * s, -s];
        let nan = dmatrix![s, s; -s, s; s, -s; -s, -s];
        for returns in [infinite, nan] {
            let allocator = Returns::returns_matrix(returns).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&Hrp::default()).unwrap(),
            &Weights::from(vec![0.5, 0.5]),
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
        let allocator = Returns::returns_matrix(returns).unwrap();

        let normalize = |raw: [f64; 3]| {
            let total: f64 = raw.iter().sum();
            Weights::from(raw.iter().map(|x| x / total).collect::<Vec<_>>())
        };
        assert_weights_close(
            &allocator.allocate(&InverseVolatility).unwrap(),
            &normalize([1.0 / s1, 1.0 / s2, 1.0 / s3]),
            1e-12,
        );
        assert_weights_close(
            &allocator.allocate(&InverseVariance).unwrap(),
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let inverse_variance = allocator.allocate(&InverseVariance).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let weights = allocator.allocate(&InverseVolatility).unwrap();
        let contributions: Vec<f64> = (0..3)
            .map(|i| weights[i] * weights[i] * allocator.cov_matrix[(i, i)])
            .collect();
        for c in &contributions {
            assert!((c - contributions[0]).abs() < 1e-18, "{contributions:?}");
        }
    }

    #[test]
    fn test_inverse_weights_ignore_return_scale_and_stay_positive() {
        let returns = Returns::compute_returns(&full_rank_prices());
        let base = Returns::returns_matrix(returns.clone()).unwrap();
        let scaled = Returns::returns_matrix(returns * 50.0).unwrap();

        for (a, b) in [
            (
                base.allocate(&InverseVolatility).unwrap(),
                scaled.allocate(&InverseVolatility).unwrap(),
            ),
            (
                base.allocate(&InverseVariance).unwrap(),
                scaled.allocate(&InverseVariance).unwrap(),
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
            Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            allocator.allocate(&InverseVolatility).err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
        assert_eq!(
            allocator.allocate(&InverseVariance).err(),
            Some(CutupError::ZeroVariance { asset: 0 })
        );
    }

    fn sharpe(allocator: &Returns, weights: &Weights, risk_free_rate: f64) -> f64 {
        let w = DVector::from_column_slice(weights.as_slice());
        (sample_mean(allocator).dot(&w) - risk_free_rate)
            / portfolio_variance(allocator, weights).sqrt()
    }

    fn min_variance_return(allocator: &Returns) -> f64 {
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let weights = allocator.allocate(&config.clone().allow_short()).unwrap();

        let excess = sample_mean(&allocator).add_scalar(-risk_free_rate);
        let z = allocator.cov_matrix.clone().try_inverse().unwrap() * excess;
        let expected = Weights::from((&z / z.sum()).iter().copied().collect::<Vec<_>>());
        assert_weights_close(&weights, &expected, 1e-10);
    }

    #[test]
    fn test_max_sharpe_beats_every_other_method() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let best = sharpe(
            &allocator,
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            risk_free_rate,
        );

        let others = [
            allocator.allocate(&EqualWeight).unwrap(),
            allocator.allocate(&InverseVolatility).unwrap(),
            allocator.allocate(&InverseVariance).unwrap(),
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert_eq!(
            allocator
                .allocate(&MaxSharpe::default().allow_short())
                .err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = MaxSharpe::default()
            .risk_free_rate(-0.01)
            .covariance(CovarianceConfig::default().shrinkage(0.5));
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
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();

        // The zero-variance portfolio is the minimum variance portfolio.
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            &Weights::from(vec![2.0, -1.0]),
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
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let threshold = min_variance_return(&allocator);
        let config = MaxSharpe::default().risk_free_rate(threshold - 1e-6);
        let weights = allocator.allocate(&config.clone().allow_short()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_sharpe_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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

        let constant = Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
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
        let allocator = Returns::prices_matrix(prices).unwrap();
        let ew_weights = allocator.allocate(&EqualWeight).unwrap();
        assert_eq!(ew_weights.as_slice(), &[1.0]);
    }

    #[test]
    fn test_min_variance_with_covariance_variants() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();

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
    fn risk_shares(allocator: &Returns, weights: &Weights) -> Vec<f64> {
        let w = DVector::from_column_slice(weights.as_slice());
        let contributions = w.component_mul(&(&allocator.cov_matrix * &w));
        let total = contributions.sum();
        contributions.iter().map(|c| c / total).collect()
    }

    #[test]
    fn test_risk_parity_equalizes_risk_contributions() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
        for share in risk_shares(&allocator, &weights) {
            assert!((share - 1.0 / 3.0).abs() < 1e-12, "{share}");
        }
    }

    #[test]
    fn test_risk_parity_meets_risk_budgets() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = RiskParity::default()
            .risk_budgets(
                DVector::from_row_slice(&[0.8571666963481355, 0.45606979884293564]).as_slice(),
            )
            .covariance(CovarianceConfig::default().shrinkage(0.29613802865778266));
        assert!(allocator.allocate(&config).is_ok());
    }

    #[test]
    fn test_risk_parity_extreme_budgets() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
            Returns::prices_matrix(full_rank_prices().columns(0, 2).into_owned()).unwrap();
        assert_weights_close(
            &allocator.allocate(&RiskParity::default()).unwrap(),
            &allocator.allocate(&InverseVolatility).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_risk_parity_ignores_return_and_budget_scale() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let scaled = Returns::returns_matrix(&allocator.returns * 1e-3).unwrap();
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
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        assert_weights_close(&weights, &Weights::from(vec![2.0 / 3.0, 1.0 / 3.0]), 1e-12);

        // Here the long-only portfolio (1, 1, 1) / 3 has zero variance.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::SingularCovariance)
        );
        let config = RiskParity::default().covariance(CovarianceConfig::default().shrinkage(0.3));
        let weights = allocator.allocate(&config).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_risk_parity_anti_correlated_pair() {
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.015, -0.015];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::SingularCovariance)
        );
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().regularization(1e-6));
        let weights = allocator.allocate(&config).unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.5]), 1e-12);
    }

    #[test]
    fn test_risk_parity_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
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
        let config = RiskParity::default().covariance(CovarianceConfig::default().shrinkage(2.0));
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
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
        // Regularization gives the constant asset a variance, and nearly all the weight.
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().regularization(1e-8));
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights[1] > 0.99);
    }

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
            Err(CutupError::ZeroVariance { asset: 1 })
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
            Err(CutupError::ZeroVariance { asset: 1 })
        );
        assert_eq!(
            allocator.allocate(&Herc::default().regularization(1e-6)),
            Err(CutupError::ZeroVariance { asset: 1 })
        );
        // Even when every asset is constant, which shrinkage alone can't fix.
        let constant = Returns::returns_matrix(dmatrix![0.0, 0.0; 0.0, 0.0]).unwrap();
        assert_eq!(
            constant.allocate(&Hrp::default().shrinkage(0.5)),
            Err(CutupError::ZeroVariance { asset: 0 })
        );
        assert_eq!(
            constant.allocate(&Herc::default().shrinkage(0.5)),
            Err(CutupError::ZeroVariance { asset: 0 })
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

    fn labeled() -> Returns {
        Returns::prices_matrix(full_rank_prices())
            .unwrap()
            .labeled(["SPY", "TLT", "GLD"])
            .unwrap()
    }

    #[test]
    fn test_labels_default_to_columns() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        assert_eq!(allocator.labels(), ["0", "1", "2"]);
        let weights = allocator.allocate(&InverseVariance).unwrap();
        assert_eq!(weights.labels(), ["0", "1", "2"]);
        assert_eq!(weights.weight("1"), Some(weights[1]));
        assert_eq!(weights, Weights::from(weights.as_slice().to_vec()));
    }

    #[test]
    fn test_labeled_rejects_a_wrong_count_and_repeats() {
        let allocator = || Returns::prices_matrix(full_rank_prices()).unwrap();
        assert_eq!(
            allocator().labeled(["SPY", "TLT"]).err(),
            Some(CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            allocator().labeled(["SPY", "TLT", "SPY"]).err(),
            Some(CutupError::DuplicateAsset {
                asset: "SPY".into()
            })
        );
        let labels: Vec<String> = ["a", "b", "c"].map(String::from).to_vec();
        assert_eq!(
            allocator().labeled(labels).unwrap().labels(),
            ["a", "b", "c"]
        );
    }

    #[test]
    fn test_every_method_labels_its_weights() {
        let allocator = labeled();
        let current = [0.4, 0.3, 0.3];
        let methods: Vec<Box<dyn Allocation>> = vec![
            Box::new(EqualWeight),
            Box::new(InverseVolatility),
            Box::new(InverseVariance),
            Box::new(MinVariance::default()),
            Box::new(MinVariance::default().allow_short()),
            Box::new(
                MinVariance::default()
                    .current_weights(current)
                    .max_turnover(0.1),
            ),
            Box::new(MeanVariance::risk_aversion(2.0)),
            Box::new(MeanVariance::risk_aversion(2.0).allow_short()),
            Box::new(
                MeanVariance::risk_aversion(2.0)
                    .current_weights(current)
                    .transaction_costs(0.001),
            ),
            Box::new(MaxSharpe::default()),
            Box::new(MaxDiversification::default()),
            Box::new(RiskParity::default()),
            Box::new(Hrp::default()),
            Box::new(Herc::default()),
            Box::new(MinCvar::default()),
        ];
        for (i, method) in methods.iter().enumerate() {
            let weights = allocator.allocate(method).unwrap();
            assert_eq!(weights.labels(), ["SPY", "TLT", "GLD"], "method {i}");
            // Called directly, not through `Returns::allocate`, too.
            assert_eq!(method.allocate(&allocator).unwrap(), weights);
        }
    }

    #[test]
    fn test_own_allocations_get_the_labels() {
        struct Fixed;
        impl Allocation for Fixed {
            fn allocate(&self, _data: &Returns) -> Result<Weights, CutupError> {
                Ok(vec![0.2, 0.3, 0.5].into())
            }
        }
        let weights = labeled().allocate(&Fixed).unwrap();
        assert_eq!(weights.labels(), ["SPY", "TLT", "GLD"]);
        assert_eq!(weights.weight("GLD"), Some(0.5));
    }

    #[test]
    fn test_weights_by_label() {
        let weights = labeled().allocate(&InverseVariance).unwrap();
        assert_eq!(weights.weight("TLT"), Some(weights[1]));
        assert_eq!(weights.weight("QQQ"), None);
        assert_eq!(weights["GLD"], weights[2]);
        let pairs: Vec<(&str, f64)> = weights.labeled().collect();
        assert_eq!(
            pairs,
            [
                ("SPY", weights[0]),
                ("TLT", weights[1]),
                ("GLD", weights[2])
            ]
        );
        assert_eq!(weights.labeled().len(), 3);
        assert_eq!(weights.labeled().next_back(), Some(("GLD", weights[2])));
    }

    #[test]
    #[should_panic(expected = "no asset is labeled `QQQ`")]
    fn test_weights_index_by_unknown_label_panics() {
        let _ = labeled().allocate(&EqualWeight).unwrap()["QQQ"];
    }

    #[test]
    fn test_assets_by_label_match_by_column() {
        let allocator = labeled();
        let same = |a: &dyn Allocation, b: &dyn Allocation| {
            assert_eq!(
                allocator.allocate(a).unwrap(),
                allocator.allocate(b).unwrap()
            );
        };

        let in_order = WeightBounds::per_asset(vec![0.1, 0.0, 0.0], vec![0.5, 0.6, 0.4]);
        let by_label = WeightBounds::per_asset(
            [("GLD", 0.0), ("SPY", 0.1), ("TLT", 0.0)],
            [("TLT", 0.6), ("SPY", 0.5), ("GLD", 0.4)],
        );
        // Each asset's bounds replace the uniform ones, and the last for an asset wins.
        let overridden = WeightBounds::uniform(0.0, 0.6)
            .asset("SPY", 0.0, 0.2)
            .asset("SPY", 0.1, 0.5)
            .asset(2, 0.0, 0.4);
        for bounds in [by_label, overridden] {
            same(
                &MinVariance::default().bounds(in_order.clone()),
                &MinVariance::default().bounds(bounds.clone()),
            );
            same(
                &MinCvar::default().bounds(in_order.clone()),
                &MinCvar::default().bounds(bounds.clone()),
            );
            same(
                &MeanVariance::risk_aversion(2.0)
                    .bounds(in_order.clone())
                    .current_weights([0.4, 0.3, 0.3])
                    .max_turnover(0.3),
                &MeanVariance::risk_aversion(2.0)
                    .bounds(bounds)
                    .current_weights([0.4, 0.3, 0.3])
                    .max_turnover(0.3),
            );
        }

        // Current weights left out are 0.
        let current = [("TLT", 0.4), ("SPY", 0.6)];
        same(
            &MinVariance::default()
                .current_weights([0.6, 0.4, 0.0])
                .max_turnover(0.2),
            &MinVariance::default()
                .current_weights(current)
                .max_turnover(0.2),
        );
        same(
            &MeanVariance::target_return(0.01)
                .current_weights([0.6, 0.4, 0.0])
                .max_turnover(0.5),
            &MeanVariance::target_return(0.01)
                .current_weights(current)
                .max_turnover(0.5),
        );

        let mu = [0.001, 0.002, 0.0015];
        let mu_by_label = [("GLD", 0.0015), ("TLT", 0.002), ("SPY", 0.001)];
        same(
            &MeanVariance::risk_aversion(2.0).expected_returns(mu),
            &MeanVariance::risk_aversion(2.0).expected_returns(mu_by_label),
        );
        same(
            &MaxSharpe::default().expected_returns(mu),
            &MaxSharpe::default().expected_returns(mu_by_label),
        );
        same(
            &MeanVariance::risk_aversion(2.0)
                .current_weights([0.4, 0.3, 0.3])
                .transaction_costs_per_asset([0.001, 0.002, 0.003]),
            &MeanVariance::risk_aversion(2.0)
                .current_weights([("GLD", 0.3), ("SPY", 0.4), ("TLT", 0.3)])
                .transaction_costs_per_asset([(2, 0.003), (0, 0.001), (1, 0.002)]),
        );
        same(
            &RiskParity::default().risk_budgets([1.0, 2.0, 3.0]),
            &RiskParity::default().risk_budgets([("TLT", 2.0), ("GLD", 3.0), ("SPY", 1.0)]),
        );
    }

    #[test]
    fn test_views_by_label_match_by_column() {
        let allocator = labeled();
        let in_order = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
            .view(View::absolute(0, 0.001))
            .view(View::relative(1, 2, 0.0005))
            .view(View::portfolio([0.5, 0.0, -0.5], 0.0002));
        let by_label = BlackLitterman::new([("TLT", 0.3), ("GLD", 0.2), ("SPY", 0.5)], 2.5)
            .view(View::absolute("SPY", 0.001))
            .view(View::relative("TLT", "GLD", 0.0005))
            .view(View::portfolio([("GLD", -0.5), ("SPY", 0.5)], 0.0002));
        assert_eq!(
            in_order.expected_returns(&allocator).unwrap(),
            by_label.expected_returns(&allocator).unwrap()
        );
        assert_eq!(
            in_order.prior_returns(&allocator).unwrap(),
            by_label.prior_returns(&allocator).unwrap()
        );
    }

    #[test]
    fn test_assets_not_in_the_data_or_given_twice() {
        let allocator = labeled();
        let model = || BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let unknown = |asset: &str| CutupError::UnknownAsset {
            asset: asset.into(),
        };
        assert_eq!(
            allocator.allocate(
                &MinVariance::default()
                    .bounds(WeightBounds::uniform(0.0, 1.0).asset("QQQ", 0.0, 0.5))
            ),
            Err(unknown("QQQ"))
        );
        assert_eq!(
            allocator.allocate(
                &MinVariance::default().bounds(WeightBounds::uniform(0.0, 1.0).asset(3, 0.0, 0.5))
            ),
            Err(unknown("3"))
        );
        assert_eq!(
            model()
                .view(View::absolute("QQQ", 0.001))
                .expected_returns(&allocator),
            Err(unknown("QQQ"))
        );
        // The same asset by label and by column.
        assert_eq!(
            model()
                .view(View::relative("SPY", 0, 0.001))
                .expected_returns(&allocator),
            Err(CutupError::InvalidConfig {
                field: "view_asset",
                value: 0.0
            })
        );
        assert_eq!(
            allocator.allocate(
                &MinVariance::default()
                    .current_weights([("SPY", 0.5), ("SPY", 0.5)])
                    .max_turnover(0.1)
            ),
            Err(CutupError::DuplicateAsset {
                asset: "SPY".into()
            })
        );
        // Every asset needs an expected return.
        assert_eq!(
            allocator.allocate(
                &MeanVariance::risk_aversion(2.0)
                    .expected_returns([("SPY", 0.001), ("TLT", 0.002)])
            ),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
    }
}
