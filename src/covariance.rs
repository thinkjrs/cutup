//! The covariance a method uses, set with [`CovarianceConfig`]: the sample covariance or one
//! you give, and adjustments to it.

use crate::eigen::SymmetricEigen;
use crate::stats::{binary_exponent, pow2_scaled};
use crate::{CutupError, Returns};
use nalgebra::DMatrix;
use std::sync::Arc;

/// The covariance matrix a method uses, set with its `covariance` setter, such as
/// [`MinVariance::covariance`](crate::MinVariance::covariance): the sample covariance of the
/// returns, by default, or [one you give](CovarianceConfig::matrix), with adjustments applied
/// to it. Each method also has `shrinkage`, `ledoit_wolf` and `regularization` setters for each
/// adjustment on its own.
///
/// The default applies no adjustment, and an adjustment that isn't set isn't applied. With
/// both, the covariance `Σ` becomes `λ·μ·I + (1 − λ)·Σ + ε·I`, which is the same whichever is
/// applied first. [Ledoit-Wolf](CovarianceConfig::ledoit_wolf) shrinkage estimates `λ` from the
/// data in place of a fixed one, and whichever of the two is set last applies.
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
    estimate: Estimate,
    regularization: Option<f64>,
    shrinkage: Option<Shrinkage>,
}

/// What [Ledoit-Wolf shrinkage](CovarianceConfig::ledoit_wolf) shrinks the sample covariance
/// toward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShrinkageTarget {
    /// The average variance times the identity, `μ·I` with `μ = trace(Σ) / n`: every asset
    /// with the average variance and no correlation, as for
    /// [`CovarianceConfig::shrinkage`]. From Ledoit and Wolf, "A well-conditioned estimator for
    /// large-dimensional covariance matrices" (2004), as in scikit-learn's `LedoitWolf`.
    ConstantVariance,
    /// Each asset's own variance, and the average correlation `r̄` between every pair of
    /// assets: `σᵢ²` on the diagonal and `r̄·σᵢ·σⱼ` off it. From Ledoit and Wolf, "Honey, I
    /// shrunk the sample covariance matrix" (2004). With two assets it's the sample
    /// covariance itself, so it changes nothing.
    ConstantCorrelation,
}

/// The shrinkage intensity: fixed, or estimated from the returns.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Shrinkage {
    Fixed(f64),
    LedoitWolf(ShrinkageTarget),
}

/// The covariance the adjustments apply to.
#[derive(Debug, Clone, Default, PartialEq)]
enum Estimate {
    /// The sample covariance of the returns.
    #[default]
    Sample,
    /// A covariance given in column order, checked when it was given.
    Given(Arc<DMatrix<f64>>),
    /// The exponentially weighted covariance, with the decay as it was given.
    ExponentiallyWeighted(Decay),
    /// The semicovariance below each asset's mean, for `None`, or below a fixed threshold.
    Semi(Option<f64>),
}

impl Estimate {
    /// What [`CutupError::NotSupported`] calls the estimate, if it isn't the sample one.
    fn name(&self) -> Option<&'static str> {
        match self {
            Estimate::Sample => None,
            Estimate::Given(_) => Some("a given covariance"),
            Estimate::ExponentiallyWeighted(_) => Some("an exponentially weighted covariance"),
            Estimate::Semi(_) => Some("a semicovariance"),
        }
    }
}

/// How fast the weights of an exponentially weighted covariance decay.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Decay {
    HalfLife(f64),
    Lambda(f64),
}

impl Decay {
    /// The smallest `ln λ`: `e^−708`, about 3.3e-308, is above `f64::MIN_POSITIVE`, so `λ` is
    /// a normal number and the weights `λ^k` it gives can only underflow to 0 or stay exact
    /// enough.
    const MIN_LOG: f64 = -708.0;

    /// `ln λ`, after checking it's in `[−708, 0)`. From a half-life it's `−ln 2 / h`, without
    /// rounding `λ` first.
    fn log(self) -> Result<f64, CutupError> {
        let (field, value, log) = match self {
            Decay::HalfLife(h) => ("half_life", h, -std::f64::consts::LN_2 / h),
            Decay::Lambda(lambda) => ("decay", lambda, lambda.ln()),
        };
        // Outside it are NaN, a half-life of 0 or below, whose log is −∞ or positive, or of
        // infinity, whose log is −0, a λ of 0 or below, or 1 or above, and a half-life or λ so
        // small that λ would be subnormal.
        if (Decay::MIN_LOG..0.0).contains(&log) {
            Ok(log)
        } else {
            Err(CutupError::InvalidConfig { field, value })
        }
    }
}

impl CovarianceConfig {
    /// Uses `rows`, a covariance matrix with one row and one column per asset in column
    /// order, instead of the sample covariance of the returns: an estimate of your own, such as
    /// one from a factor model or another library. Shrinkage and regularization still apply on
    /// top of it.
    ///
    /// It's checked here, and its size against the data when a method uses it. Rounding can
    /// leave a computed covariance a little asymmetric, or its smallest eigenvalues a little
    /// negative, so entries that differ from their transposes by up to `1e-12` of the largest
    /// are averaged, and eigenvalues down to `−10nε` of the largest are allowed.
    ///
    /// HRP and HERC still cluster on the correlations of the returns: a given covariance, like
    /// shrinkage, changes only how they split the weight.
    ///
    /// # Errors
    ///
    /// * [`CutupError::NoAssets`] if there are no rows.
    /// * [`CutupError::RaggedRows`] if a row has a different number of values from the first.
    /// * [`CutupError::DimensionMismatch`] if the rows don't have one value per row.
    /// * [`CutupError::NonFiniteValue`] if a value is NaN or infinite.
    /// * [`CutupError::AsymmetricCovariance`] if an entry differs from its transpose by more
    ///   than rounding.
    /// * [`CutupError::IndefiniteCovariance`] if it has a negative eigenvalue beyond rounding,
    ///   so some portfolio would have a negative variance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CovarianceConfig, MinVariance, Returns};
    ///
    /// let data = Returns::new(&[
    ///      [ 0.010,  0.021],
    ///      [-0.008, -0.015],
    ///      [ 0.004,  0.009],
    /// ])?;
    /// // Two uncorrelated assets, the second with four times the variance.
    /// let covariance = CovarianceConfig::matrix(&[[1e-4, 0.0], [0.0, 4e-4]])?;
    /// let weights = data.allocate(&MinVariance::default().covariance(covariance))?;
    /// assert!((weights[0] - 0.8).abs() < 1e-12 && (weights[1] - 0.2).abs() < 1e-12);
    ///
    /// assert!(CovarianceConfig::matrix(&[[1e-4, 2e-4], [2e-4, 1e-4]]).is_err());
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn matrix<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self, CutupError> {
        CovarianceConfig::given(Returns::rows_matrix(rows)?)
    }

    /// Uses a covariance matrix given as `assets × assets` values in one flat, row-major slice,
    /// such as a JavaScript `Float64Array`. Otherwise it's the same as
    /// [`CovarianceConfig::matrix`].
    ///
    /// # Errors
    ///
    /// As for [`CovarianceConfig::matrix`], with [`CutupError::NoAssets`] for 0 assets and
    /// [`CutupError::RaggedRows`] if the number of values isn't a multiple of `assets`.
    pub fn matrix_flat(values: &[f64], assets: usize) -> Result<Self, CutupError> {
        CovarianceConfig::given(Returns::flat_matrix(values, assets)?)
    }

    /// Uses a covariance matrix given as a nalgebra `DMatrix`, with the `nalgebra` feature.
    /// Otherwise it's the same as [`CovarianceConfig::matrix`].
    ///
    /// # Errors
    ///
    /// As for [`CovarianceConfig::matrix`], except [`CutupError::RaggedRows`], which a matrix
    /// can't have.
    #[cfg(feature = "nalgebra")]
    #[cfg_attr(docsrs, doc(cfg(feature = "nalgebra")))]
    pub fn from_matrix(matrix: DMatrix<f64>) -> Result<Self, CutupError> {
        CovarianceConfig::given(matrix)
    }

    /// Uses the exponentially weighted covariance of the returns in place of the sample
    /// covariance, so recent periods count for more: the weight of each period halves every
    /// `half_life` periods back from the last row. It's the same estimate as
    /// [`CovarianceConfig::exponentially_weighted_decay`] with `λ = 2^(−1/half_life)`; RiskMetrics'
    /// daily `λ = 0.94` is a half-life of about 11.2 periods.
    ///
    /// The weights `wₜ = λ^(T−1−t)` give a weighted mean `m = Σwₜrₜ / Σwₜ`, and the covariance is
    /// `Σwₜ(rₜ − m)(rₜ − m)ᵀ / (Σwₜ − Σwₜ² / Σwₜ)`, the unbiased estimate for these weights. It's
    /// pandas' `ewm(alpha=1 − λ).cov()`, which riskfolio-lib's `"ewma1"` and skfolio's
    /// `EWCovariance` use. With equal weights the denominator is `T − 1`, so as the half-life
    /// grows the estimate becomes the sample covariance.
    ///
    /// Shrinkage and regularization apply on top of it. HRP and HERC still cluster on the
    /// correlations of the returns: like a [given covariance](CovarianceConfig::matrix), it
    /// changes only how they split the weight.
    ///
    /// # Errors
    ///
    /// A method using it returns:
    ///
    /// * [`CutupError::InvalidConfig`] if `half_life` isn't finite and positive, or is so short,
    ///   below `ln 2 / 708`, about 0.001 periods, that `λ` would be below about 3.3e-308.
    /// * [`CutupError::ZeroVariance`] for an asset whose weighted variance is 0, to rounding,
    ///   unless an adjustment gives it a positive variance.
    /// * [`CutupError::NotSupported`] with [Ledoit-Wolf shrinkage](CovarianceConfig::ledoit_wolf),
    ///   whose intensity is for the sample covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CovarianceConfig, MinVariance, Returns};
    ///
    /// // The second asset was calm, then turned volatile.
    /// let data = Returns::new(&[
    ///      [ 0.010,  0.001],
    ///      [-0.012, -0.001],
    ///      [ 0.011,  0.001],
    ///      [-0.010,  0.030],
    ///      [ 0.012, -0.028],
    /// ])?;
    /// let sample = data.allocate(&MinVariance::default())?;
    /// let recent = CovarianceConfig::exponentially_weighted(1.0);
    /// let weighted = data.allocate(&MinVariance::default().covariance(recent))?;
    /// // Weighting recent periods more holds less of it.
    /// assert!(weighted[1] < sample[1]);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    ///
    /// <https://en.wikipedia.org/wiki/Exponential_smoothing>
    pub fn exponentially_weighted(half_life: f64) -> Self {
        CovarianceConfig {
            estimate: Estimate::ExponentiallyWeighted(Decay::HalfLife(half_life)),
            ..CovarianceConfig::default()
        }
    }

    /// Uses the exponentially weighted covariance of the returns with decay `λ`, in `(0, 1)`:
    /// each period back from the last row weighs `λ` times the one after it, as riskfolio-lib's
    /// `d`. Otherwise it's the same as [`CovarianceConfig::exponentially_weighted`], with a
    /// half-life of `−ln 2 / ln λ`.
    ///
    /// # Errors
    ///
    /// As for [`CovarianceConfig::exponentially_weighted`], with
    /// [`CutupError::InvalidConfig`] if `λ` isn't in `(0, 1)`, or is below `e^−708`, about
    /// 3.3e-308.
    pub fn exponentially_weighted_decay(lambda: f64) -> Self {
        CovarianceConfig {
            estimate: Estimate::ExponentiallyWeighted(Decay::Lambda(lambda)),
            ..CovarianceConfig::default()
        }
    }

    /// Uses the semicovariance of the returns below each asset's mean in place of the sample
    /// covariance, for downside risk: each return above its asset's mean counts as the mean.
    /// With `dₜᵢ = min(rₜᵢ − r̄ᵢ, 0)` it's `Σₜ dₜᵢ dₜⱼ / (T − 1)`, riskfolio-lib's `"semi"`.
    ///
    /// Shrinkage and regularization apply on top of it. HRP and HERC still cluster on the
    /// correlations of the returns: like a [given covariance](CovarianceConfig::matrix), it
    /// changes only how they split the weight.
    ///
    /// # Errors
    ///
    /// A method using it returns:
    ///
    /// * [`CutupError::ZeroVariance`] for a constant asset, unless an adjustment gives it a
    ///   positive variance.
    /// * [`CutupError::NotSupported`] with [Ledoit-Wolf shrinkage](CovarianceConfig::ledoit_wolf),
    ///   whose intensity is for the sample covariance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CovarianceConfig, MinVariance, Returns};
    ///
    /// // The same spread, but the first asset's large moves are gains and the second's losses.
    /// let data = Returns::new(&[
    ///      [ 0.030, -0.010],
    ///      [-0.010,  0.010],
    ///      [-0.010, -0.030],
    ///      [-0.010,  0.010],
    /// ])?;
    /// let downside = CovarianceConfig::semicovariance();
    /// let weights = data.allocate(&MinVariance::default().covariance(downside))?;
    /// assert!(weights[0] > weights[1]);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    ///
    /// <https://en.wikipedia.org/wiki/Downside_risk>
    pub fn semicovariance() -> Self {
        CovarianceConfig {
            estimate: Estimate::Semi(None),
            ..CovarianceConfig::default()
        }
    }

    /// Uses the semicovariance of the returns below `threshold`, such as 0 or a risk-free rate
    /// per period, in place of the sample covariance: each return above it counts as the
    /// threshold. With `dₜᵢ = min(rₜᵢ − threshold, 0)` it's `Σₜ dₜᵢ dₜⱼ / T`, PyPortfolioOpt's
    /// `semicovariance` with `benchmark=threshold`. Nothing is estimated, so it divides by `T`,
    /// unlike [`CovarianceConfig::semicovariance`], whose mean takes a degree of freedom.
    ///
    /// A constant asset below the threshold has a semivariance: each period loses against it.
    ///
    /// # Errors
    ///
    /// As for [`CovarianceConfig::semicovariance`], with [`CutupError::InvalidConfig`] if
    /// `threshold` isn't finite, and [`CutupError::ZeroVariance`] for an asset that's never
    /// below it, to rounding, which has no downside to measure.
    pub fn semicovariance_below(threshold: f64) -> Self {
        CovarianceConfig {
            estimate: Estimate::Semi(Some(threshold)),
            ..CovarianceConfig::default()
        }
    }

    /// Checks `matrix` is a covariance, made exactly symmetric, and uses it.
    fn given(matrix: DMatrix<f64>) -> Result<Self, CutupError> {
        let n = matrix.nrows();
        if n == 0 || matrix.ncols() == 0 {
            return Err(CutupError::NoAssets);
        }
        if matrix.ncols() != n {
            return Err(CutupError::DimensionMismatch {
                field: "covariance",
                expected: n,
                actual: matrix.ncols(),
            });
        }
        if let Some((row, col)) = Returns::find(&matrix, |x| !x.is_finite()) {
            return Err(CutupError::NonFiniteValue { row, col });
        }
        let tolerance = 1e-12 * matrix.amax();
        let difference = &matrix - matrix.transpose();
        if let Some((row, col)) = Returns::find(&difference, |d| d.abs() > tolerance) {
            return Err(CutupError::AsymmetricCovariance { row, col });
        }
        let symmetric = (&matrix + matrix.transpose()) / 2.0;
        // Divided by its largest entry first, so neither the eigenvalues nor the cutoff can
        // overflow or underflow.
        let scale = symmetric.amax();
        if scale > 0.0 {
            let eigen =
                SymmetricEigen::new(&(&symmetric / scale)).ok_or(CutupError::NotConverged)?;
            let smallest = eigen.values.min();
            if smallest < -10.0 * n as f64 * f64::EPSILON * eigen.values.amax() {
                return Err(CutupError::IndefiniteCovariance {
                    eigenvalue: smallest * scale,
                });
            }
        }
        Ok(CovarianceConfig {
            estimate: Estimate::Given(Arc::new(symmetric)),
            ..CovarianceConfig::default()
        })
    }

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
    /// sample covariance. It replaces [Ledoit-Wolf shrinkage](CovarianceConfig::ledoit_wolf),
    /// if that was set.
    ///
    /// <https://en.wikipedia.org/wiki/Shrinkage_estimator>
    pub fn shrinkage(mut self, lambda: f64) -> Self {
        self.shrinkage = Some(Shrinkage::Fixed(lambda));
        self
    }

    /// Shrinks the sample covariance `Σ` toward `target` with the intensity `δ` Ledoit and Wolf
    /// estimate from the returns, the one that minimizes the expected squared error: it
    /// becomes `δ·F + (1 − δ)·Σ`, for the target `F` built from `Σ`. It replaces a
    /// [fixed intensity](CovarianceConfig::shrinkage), if one was set, and regularization
    /// still applies on top.
    ///
    /// `δ` comes from the papers' formulas, on the `1/T` moments of the centered returns, and
    /// is in `[0, 1]`. It's `0` for one asset, or when `Σ` already equals the target. With
    /// [`ShrinkageTarget::ConstantVariance`] it's scikit-learn's `ledoit_wolf_shrinkage`, and
    /// the result is the same as [`CovarianceConfig::shrinkage`] with `λ = δ`. The papers shrink
    /// the `1/T` sample covariance, and cutup's has `T − 1`, so the result is `T / (T − 1)`
    /// times theirs: the weights are the same for any method that doesn't depend on the
    /// covariance's scale.
    ///
    /// # Errors
    ///
    /// A method using it returns:
    ///
    /// * [`CutupError::NotSupported`] with a [given covariance](CovarianceConfig::matrix),
    ///   since the intensity is estimated from the returns that covariance would come from.
    /// * [`CutupError::ZeroVariance`] for a constant asset with
    ///   [`ShrinkageTarget::ConstantCorrelation`], which has no correlations to average, or
    ///   with [`ShrinkageTarget::ConstantVariance`] if `δ` is `0`, unless regularization gives
    ///   it a positive variance.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{CovarianceConfig, MinVariance, Returns, ShrinkageTarget};
    ///
    /// let data = Returns::new(&[
    ///      [ 0.010,  0.021,  0.004],
    ///      [-0.008, -0.015, -0.002],
    ///      [ 0.004,  0.009,  0.006],
    ///      [ 0.002, -0.004,  0.003],
    /// ])?;
    /// let shrunk = CovarianceConfig::default().ledoit_wolf(ShrinkageTarget::ConstantCorrelation);
    /// let weights = data.allocate(&MinVariance::default().covariance(shrunk))?;
    /// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    ///
    /// // The method's own setter does the same.
    /// let method = MinVariance::default().ledoit_wolf(ShrinkageTarget::ConstantCorrelation);
    /// assert_eq!(data.allocate(&method)?, weights);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    ///
    /// <https://en.wikipedia.org/wiki/Estimation_of_covariance_matrices#Shrinkage_estimation>
    pub fn ledoit_wolf(mut self, target: ShrinkageTarget) -> Self {
        self.shrinkage = Some(Shrinkage::LedoitWolf(target));
        self
    }

    /// The fixed shrinkage intensity `λ` set with [`CovarianceConfig::shrinkage`], or `None`,
    /// including under [Ledoit-Wolf shrinkage](CovarianceConfig::ledoit_wolf), whose intensity
    /// depends on the returns.
    pub fn lambda(&self) -> Option<f64> {
        match self.shrinkage {
            Some(Shrinkage::Fixed(lambda)) => Some(lambda),
            _ => None,
        }
    }

    /// The regularization `ε` set with [`CovarianceConfig::regularization`], or `None`.
    pub fn epsilon(&self) -> Option<f64> {
        self.regularization
    }

    /// Returns [`CutupError::InvalidConfig`] if the regularization is negative or not finite,
    /// the shrinkage is outside `[0, 1]`, or the decay or threshold of the estimate is invalid,
    /// and [`CutupError::NotSupported`] for Ledoit-Wolf shrinkage of anything but the sample
    /// covariance.
    pub(crate) fn validate(&self) -> Result<(), CutupError> {
        if let Some(value) = self
            .regularization
            .filter(|eps| !(eps.is_finite() && *eps >= 0.0))
        {
            return Err(CutupError::InvalidConfig {
                field: "regularization",
                value,
            });
        }
        if let Some(value) = self.lambda().filter(|lambda| !(0.0..=1.0).contains(lambda)) {
            return Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value,
            });
        }
        match self.estimate {
            Estimate::ExponentiallyWeighted(decay) => {
                decay.log()?;
            }
            Estimate::Semi(Some(value)) if !value.is_finite() => {
                return Err(CutupError::InvalidConfig {
                    field: "threshold",
                    value,
                });
            }
            _ => {}
        }
        if let (Some(Shrinkage::LedoitWolf(_)), Some(with)) = (self.shrinkage, self.estimate.name())
        {
            return Err(CutupError::NotSupported {
                setting: "ledoit_wolf",
                with,
            });
        }
        Ok(())
    }
}

impl Returns {
    /// The covariance `config` sets, adjusted, after checking its values.
    ///
    /// Returns [`CutupError::DimensionMismatch`] for a given covariance of another size,
    /// [`CutupError::ZeroVariance`] for an asset whose estimated variance is 0, to rounding, or
    /// one given a variance of 0, unless shrinkage toward a constant variance or regularization
    /// gives it a positive variance,
    /// [`CutupError::NonFiniteCovariance`] if the result isn't finite, and
    /// [`CutupError::SingularCovariance`] if it is zero.
    pub(crate) fn adjusted_covariance(
        &self,
        config: &CovarianceConfig,
    ) -> Result<DMatrix<f64>, CutupError> {
        config.validate()?;
        let shrinkage = match config.shrinkage {
            None => None,
            Some(Shrinkage::Fixed(lambda)) => Some((lambda, ShrinkageTarget::ConstantVariance)),
            Some(Shrinkage::LedoitWolf(target)) => Some((self.ledoit_wolf(target)?, target)),
        };
        let adjusted = shrinkage.is_some_and(|(lambda, _)| lambda > 0.0)
            || config.regularization.unwrap_or(0.0) > 0.0;
        let n = self.cov_matrix.ncols();
        let estimated;
        let base = match &config.estimate {
            Estimate::Sample => &self.cov_matrix,
            Estimate::Given(matrix) => {
                if matrix.nrows() != n {
                    return Err(CutupError::DimensionMismatch {
                        field: "covariance",
                        expected: n,
                        actual: matrix.nrows(),
                    });
                }
                matrix.as_ref()
            }
            Estimate::ExponentiallyWeighted(decay) => {
                estimated = self.exponentially_weighted(decay.log()?);
                &estimated
            }
            Estimate::Semi(threshold) => {
                estimated = self.semicovariance(*threshold);
                &estimated
            }
        };
        if !adjusted {
            // A given variance is exact; one estimated from the returns is 0 to rounding.
            let zero = match &config.estimate {
                Estimate::Given(_) => (0..n).find(|&i| base[(i, i)] == 0.0),
                _ => self.find_zero_variance_in(base),
            };
            if let Some(asset) = zero {
                return Err(self.zero_variance(asset));
            }
        }

        let identity = DMatrix::identity(n, n);

        let shrunk_cov = match shrinkage {
            Some((lambda, ShrinkageTarget::ConstantVariance)) => {
                let mean_variance = base.trace() / n as f64;
                (lambda * mean_variance) * &identity + (1.0 - lambda) * base
            }
            Some((lambda, ShrinkageTarget::ConstantCorrelation)) => {
                lambda * constant_correlation(base) + (1.0 - lambda) * base
            }
            None => base.clone(),
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

    /// The exponentially weighted covariance for the decay `λ = e^log_decay`, the last row
    /// weighted most.
    fn exponentially_weighted(&self, log_decay: f64) -> DMatrix<f64> {
        let (periods, n) = self.returns.shape();
        let weights: Vec<f64> = (0..periods)
            .map(|t| ((periods - 1 - t) as f64 * log_decay).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mean: Vec<f64> = (0..n)
            .map(|i| {
                let column = self.returns.column(i);
                weights
                    .iter()
                    .zip(column.iter())
                    .map(|(w, r)| w * r)
                    .sum::<f64>()
                    / total
            })
            .collect();
        // Σw − Σw²/Σw, from its closed form 2λ(1 − λ^(T−1)) / (1 − λ²), with each 1 − λ^k as
        // −expm1(k·ln λ), so neither cancels for λ near 0 or 1.
        let lambda = log_decay.exp();
        let denominator = 2.0 * lambda * -((periods - 1) as f64 * log_decay).exp_m1()
            / -(2.0 * log_decay).exp_m1();
        // Each weight over the denominator: for λ near 0 both are about λ, and dividing only
        // the product would take the deviations times λ through subnormal numbers first.
        let scales: Vec<f64> = weights.iter().map(|w| (w / denominator).sqrt()).collect();
        // Each centered return times the square root of its scaled weight, so the covariance
        // is a product XᵀX: symmetric and positive semidefinite.
        let x = DMatrix::from_fn(periods, n, |t, i| {
            (self.returns[(t, i)] - mean[i]) * scales[t]
        });
        x.tr_mul(&x)
    }

    /// The semicovariance below each asset's mean, divided by `T − 1`, for `None`, or below a
    /// fixed threshold, divided by `T`.
    fn semicovariance(&self, threshold: Option<f64>) -> DMatrix<f64> {
        let (periods, n) = self.returns.shape();
        let floor = match threshold {
            Some(threshold) => vec![threshold; n],
            None => self.returns.row_mean().iter().copied().collect(),
        };
        let below = DMatrix::from_fn(periods, n, |t, i| {
            (self.returns[(t, i)] - floor[i]).min(0.0)
        });
        let degrees = periods - usize::from(threshold.is_none());
        below.tr_mul(&below) / degrees as f64
    }

    /// The Ledoit-Wolf shrinkage intensity toward `target`, in `[0, 1]`, from the `1/T`
    /// moments of the centered returns.
    ///
    /// Both intensities are ratios of fourth moments, so the centered returns are first
    /// divided by a power of two near the largest, which is exact and keeps their fourth
    /// powers from overflowing or underflowing.
    ///
    /// Returns [`CutupError::ZeroVariance`] for a constant asset with
    /// [`ShrinkageTarget::ConstantCorrelation`].
    pub(crate) fn ledoit_wolf(&self, target: ShrinkageTarget) -> Result<f64, CutupError> {
        if target == ShrinkageTarget::ConstantCorrelation {
            if let Some(asset) = self.find_zero_variance() {
                return Err(self.zero_variance(asset));
            }
        }
        let n = self.returns.ncols();
        let mean = self.returns.row_mean();
        let mut x = DMatrix::from_fn(self.returns.nrows(), n, |t, i| {
            self.returns[(t, i)] - mean[i]
        });
        let exponent = binary_exponent(x.amax());
        x.apply(|v| *v = pow2_scaled(*v, -exponent));
        let t = x.nrows() as f64;
        let sample = x.transpose() * &x / t;

        // π_ij, the variance of the sample covariance's entry ij, times T.
        let mut pi = DMatrix::zeros(n, n);
        for row in x.row_iter() {
            for i in 0..n {
                for j in 0..n {
                    let d = row[i] * row[j] - sample[(i, j)];
                    pi[(i, j)] += d * d;
                }
            }
        }
        let pi = pi / t;
        let (error, distance) = match target {
            ShrinkageTarget::ConstantVariance => {
                let mean_variance = sample.trace() / n as f64;
                let identity = DMatrix::<f64>::identity(n, n);
                (
                    pi.sum(),
                    (&sample - mean_variance * identity).norm_squared(),
                )
            }
            ShrinkageTarget::ConstantCorrelation => {
                let r = average_correlation(&sample);
                let sd = sample.diagonal().map(f64::sqrt);
                // ρ, the target's covariance with the sample covariance: the variances' own
                // π_ii, and off the diagonal θ_ij = Σ_t (x_ti² − s_ii)·x_ti·x_tj / T, the
                // covariance of x_ti² with x_ti·x_tj. The squares are centered, so the products
                // needn't be.
                let mut rho = pi.trace();
                for i in 0..n {
                    for j in (0..n).filter(|&j| j != i) {
                        let theta = x
                            .row_iter()
                            .map(|row| (row[i] * row[i] - sample[(i, i)]) * row[i] * row[j])
                            .sum::<f64>()
                            / t;
                        rho += r * sd[j] / sd[i] * theta;
                    }
                }
                let distance = (&sample - constant_correlation(&sample)).norm_squared();
                (pi.sum() - rho, distance)
            }
        };
        // δ = (estimation error / T) / (distance to the target), in [0, 1], and nothing to
        // shrink if the sample is the target already.
        Ok(if distance > 0.0 {
            (error / t / distance).clamp(0.0, 1.0)
        } else {
            0.0
        })
    }
}

/// The average correlation between distinct assets in `cov`, which has at least two assets,
/// none with a variance of 0.
fn average_correlation(cov: &DMatrix<f64>) -> f64 {
    let n = cov.ncols();
    let sd = cov.diagonal().map(f64::sqrt);
    let mut total = 0.0;
    for i in 0..n {
        for j in (0..n).filter(|&j| j != i) {
            total += cov[(i, j)] / sd[i] / sd[j];
        }
    }
    total / (n * (n - 1)) as f64
}

/// The constant correlation target for `cov`: its variances, and the average correlation
/// times the two standard deviations off the diagonal.
fn constant_correlation(cov: &DMatrix<f64>) -> DMatrix<f64> {
    let n = cov.ncols();
    if n == 1 {
        return cov.clone();
    }
    let r = average_correlation(cov);
    let sd = cov.diagonal().map(f64::sqrt);
    DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            cov[(i, i)]
        } else {
            r * sd[i] * sd[j]
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_weights_close, factor_returns, full_rank_prices, rank_one_returns,
        two_block_returns, uncorrelated_returns,
    };
    use crate::{
        Allocation, BlackLitterman, EqualWeight, Herc, Hrp, MaxDiversification, MaxSharpe,
        MeanVariance, MinVariance, RiskParity, Weights,
    };
    use nalgebra::dmatrix;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_covariance_config_builder() {
        assert_eq!(
            CovarianceConfig::default(),
            CovarianceConfig {
                estimate: Estimate::Sample,
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
            .allocate(&MinVariance::default().allow_short().covariance(config))
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
                .allocate(&MinVariance::default().allow_short().covariance(config))
                .unwrap(),
            1e-12,
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
            allocator.allocate(&MinVariance::default().covariance(shrunk)),
            allocator.allocate(&mean_variance.allow_short()),
            allocator.allocate(&max_sharpe.allow_short()),
            allocator.allocate(&risk_parity),
        ] {
            assert_eq!(result, Err(CutupError::SingularCovariance));
        }
    }

    #[test]
    fn test_covariance_adjustment_allows_constant_asset() {
        let returns = dmatrix![0.1, 0.01, 0.02; 0.1, -0.02, 0.01; 0.1, 0.03, -0.01];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
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
            .allocate(&MinVariance::default().allow_short().covariance(regularized))
            .unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let mean_variance = MeanVariance::risk_aversion(2.0).covariance(shrunk);
        assert!(allocator.allocate(&mean_variance.allow_short()).is_ok());
    }

    fn rows(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
        m.row_iter().map(|r| r.iter().copied().collect()).collect()
    }

    /// Every method that takes a covariance config, with `config`.
    fn methods(config: &CovarianceConfig) -> Vec<Box<dyn Allocation>> {
        vec![
            Box::new(MinVariance::default().covariance(config.clone())),
            Box::new(
                MinVariance::default()
                    .allow_short()
                    .covariance(config.clone()),
            ),
            Box::new(MeanVariance::risk_aversion(20.0).covariance(config.clone())),
            Box::new(
                MaxSharpe::default()
                    .allow_short()
                    .covariance(config.clone()),
            ),
            Box::new(MaxDiversification::default().covariance(config.clone())),
            Box::new(RiskParity::default().covariance(config.clone())),
            Box::new(Hrp::default().covariance(config.clone())),
            Box::new(Herc::default().covariance(config.clone())),
        ]
    }

    #[test]
    fn a_given_sample_covariance_changes_nothing() {
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        let given = CovarianceConfig::matrix(&rows(&data.cov_matrix)).unwrap();
        for (adjust, sample) in [
            (given.clone(), CovarianceConfig::default()),
            (
                given.clone().shrinkage(0.3),
                CovarianceConfig::default().shrinkage(0.3),
            ),
            (
                given.regularization(1e-6),
                CovarianceConfig::default().regularization(1e-6),
            ),
        ] {
            for (a, b) in methods(&adjust).iter().zip(methods(&sample)) {
                assert_eq!(data.allocate(a.as_ref()), data.allocate(b.as_ref()));
            }
            let model = |c: CovarianceConfig| BlackLitterman::new(vec![0.2; 5], 2.5).covariance(c);
            assert_eq!(
                model(adjust.clone()).prior_returns(&data),
                model(sample).prior_returns(&data)
            );
        }
    }

    #[test]
    fn a_given_covariance_is_the_one_used() {
        let data = Returns::returns_matrix(factor_returns(3, 20, 4)).unwrap();
        // Uncorrelated, with variances 1 : 4 : 16: minimum variance is inverse variance.
        let given =
            CovarianceConfig::matrix(&[[1e-4, 0.0, 0.0], [0.0, 4e-4, 0.0], [0.0, 0.0, 16e-4]])
                .unwrap();
        let weights = data
            .allocate(&MinVariance::default().covariance(given.clone()))
            .unwrap();
        assert_weights_close(
            &weights,
            &Weights::from(vec![16.0 / 21.0, 4.0 / 21.0, 1.0 / 21.0]),
            1e-12,
        );
        // Full shrinkage of it is equal weight, at its own average variance.
        let full = data
            .allocate(&MinVariance::default().covariance(given.shrinkage(1.0)))
            .unwrap();
        assert_weights_close(&full, &data.allocate(&EqualWeight).unwrap(), 1e-12);
    }

    #[test]
    fn flat_and_matrix_covariances_match_rows() {
        let rows = [[2.0, 0.5], [0.5, 1.0]];
        let given = CovarianceConfig::matrix(&rows).unwrap();
        assert_eq!(
            CovarianceConfig::matrix_flat(&[2.0, 0.5, 0.5, 1.0], 2),
            Ok(given.clone())
        );
        #[cfg(feature = "nalgebra")]
        assert_eq!(
            CovarianceConfig::from_matrix(dmatrix![2.0, 0.5; 0.5, 1.0]),
            Ok(given)
        );
    }

    #[test]
    fn given_covariances_must_be_square_and_finite() {
        let none: [[f64; 0]; 0] = [];
        assert_eq!(CovarianceConfig::matrix(&none), Err(CutupError::NoAssets));
        assert_eq!(
            CovarianceConfig::matrix(&[[]; 2]),
            Err(CutupError::NoAssets)
        );
        assert_eq!(
            CovarianceConfig::matrix_flat(&[1.0], 0),
            Err(CutupError::NoAssets)
        );
        assert_eq!(
            CovarianceConfig::matrix(&[vec![1.0, 0.0], vec![0.0]]),
            Err(CutupError::RaggedRows {
                row: 1,
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CovarianceConfig::matrix(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            Err(CutupError::DimensionMismatch {
                field: "covariance",
                expected: 2,
                actual: 3
            })
        );
        assert_eq!(
            CovarianceConfig::matrix_flat(&[1.0, 0.0, 0.0], 2),
            Err(CutupError::RaggedRows {
                row: 1,
                expected: 2,
                actual: 1
            })
        );
        assert_eq!(
            CovarianceConfig::matrix(&[[1.0, 0.0], [0.0, f64::NAN]]),
            Err(CutupError::NonFiniteValue { row: 1, col: 1 })
        );
    }

    #[test]
    fn given_covariances_must_be_symmetric_to_rounding() {
        assert_eq!(
            CovarianceConfig::matrix(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.3], [0.0, 0.2, 1.0]]),
            Err(CutupError::AsymmetricCovariance { row: 1, col: 2 })
        );
        // Within 1e-12 of the largest entry, the two are averaged.
        let off = 1e-12 * 4.0;
        let given = CovarianceConfig::matrix(&[[4.0, 1.0 + off], [1.0, 2.0]]).unwrap();
        assert_eq!(
            given,
            CovarianceConfig::matrix(&[[4.0, 1.0 + off / 2.0], [1.0 + off / 2.0, 2.0]]).unwrap()
        );
        assert!(CovarianceConfig::matrix(&[[4.0, 1.0 + 1.01 * off], [1.0, 2.0]]).is_err());
    }

    #[test]
    fn given_covariances_must_be_positive_semidefinite_to_rounding() {
        // Correlation 2: the portfolio (1, −1) has variance −2.
        assert_eq!(
            CovarianceConfig::matrix(&[[1.0, 2.0], [2.0, 1.0]]),
            Err(CutupError::IndefiniteCovariance { eigenvalue: -1.0 })
        );
        assert!(matches!(
            CovarianceConfig::matrix(&[[1e-300, 0.0], [0.0, -1e-300]]),
            Err(CutupError::IndefiniteCovariance { eigenvalue }) if eigenvalue == -1e-300
        ));
        // Down to −10nε of the largest eigenvalue is rounding: 4.4e-15 for n = 2, exactly
        // 20ε.
        assert!(CovarianceConfig::matrix(&[[1.0, 0.0], [0.0, -20.0 * f64::EPSILON]]).is_ok());
        assert!(CovarianceConfig::matrix(&[[1.0, 0.0], [0.0, -4.5e-15]]).is_err());
        // Relative to the largest eigenvalue, here 2, not the largest entry, 1.
        let nearly = |d: f64| CovarianceConfig::matrix(&[[1.0, 1.0], [1.0, 1.0 - d]]);
        assert!(nearly(1e-14).is_ok());
        assert!(nearly(2e-14).is_err());
        // A sample covariance with fewer periods than assets is singular, but semidefinite.
        let data = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert!(CovarianceConfig::matrix(&rows(&data.cov_matrix)).is_ok());
        // So are the zero matrix and covariances of any scale.
        assert!(CovarianceConfig::matrix(&[[0.0, 0.0], [0.0, 0.0]]).is_ok());
        for scale in [1e-300, 1e300] {
            assert!(CovarianceConfig::matrix(&[[2.0 * scale, scale], [scale, scale]]).is_ok());
        }
    }

    #[test]
    fn a_given_covariance_must_fit_the_data() {
        let data = Returns::returns_matrix(factor_returns(3, 12, 2)).unwrap();
        let two = CovarianceConfig::matrix(&[[1.0, 0.0], [0.0, 1.0]]).unwrap();
        let mismatch = Err(CutupError::DimensionMismatch {
            field: "covariance",
            expected: 3,
            actual: 2,
        });
        for method in methods(&two) {
            assert_eq!(data.allocate(method.as_ref()), mismatch);
        }
        // An asset given a variance of 0 has zero variance, unless an adjustment gives it one.
        let zero =
            CovarianceConfig::matrix(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
        assert_eq!(
            data.allocate(&MinVariance::default().covariance(zero.clone())),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
        assert!(
            data.allocate(&MinVariance::default().covariance(zero.regularization(1e-6)))
                .is_ok()
        );
        let none = CovarianceConfig::matrix(&[[0.0; 3]; 3])
            .unwrap()
            .shrinkage(0.5);
        assert_eq!(
            data.allocate(&MinVariance::default().covariance(none)),
            Err(CutupError::SingularCovariance)
        );
    }

    const TARGETS: [ShrinkageTarget; 2] = [
        ShrinkageTarget::ConstantVariance,
        ShrinkageTarget::ConstantCorrelation,
    ];

    fn ledoit_wolf(target: ShrinkageTarget) -> CovarianceConfig {
        CovarianceConfig::default().ledoit_wolf(target)
    }

    /// The datasets in `tests/fixtures/datasets.json`, by name.
    fn datasets() -> Vec<(String, Returns)> {
        let datasets: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/datasets.json")).unwrap();
        let datasets = datasets.as_object().unwrap();
        datasets
            .iter()
            .map(|(name, dataset)| {
                let rows: Vec<Vec<f64>> = dataset["data"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| {
                        row.as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap())
                            .collect()
                    })
                    .collect();
                let data = match dataset["kind"].as_str().unwrap() {
                    "prices" => Returns::from_prices(&rows),
                    _ => Returns::new(&rows),
                };
                (name.clone(), data.unwrap())
            })
            .collect()
    }

    /// Each case in `tests/fixtures/covariance.json`, from `reference/covariance.py`, with its
    /// dataset's name and returns.
    fn reference_cases() -> Vec<(serde_json::Value, String, Returns)> {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/covariance.json")).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        let datasets = datasets();
        assert_eq!(cases.len(), datasets.len());
        cases
            .iter()
            .map(|case| {
                let (name, data) = datasets
                    .iter()
                    .find(|(name, _)| case["dataset"] == name.as_str())
                    .unwrap();
                (case.clone(), name.clone(), data.clone())
            })
            .collect()
    }

    /// Asserts `config`'s covariance on `data` is the fixture's `expected["covariance"]`, to
    /// 1e-12 of its largest entry.
    fn assert_reference(
        data: &Returns,
        config: &CovarianceConfig,
        expected: &serde_json::Value,
        what: &str,
    ) {
        let got = data.adjusted_covariance(config).unwrap();
        let want = DMatrix::from_row_iterator(
            got.nrows(),
            got.ncols(),
            expected["covariance"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap())),
        );
        assert!(
            (&got - &want).amax() <= 1e-12 * want.amax(),
            "{what}: {got} vs {want}"
        );
    }

    /// Against the fixture: scikit-learn's intensity toward a constant variance, and the
    /// paper's toward a constant correlation, cross-checked against PyPortfolioOpt's.
    #[test]
    fn ledoit_wolf_matches_the_python_reference() {
        for (case, name, data) in reference_cases() {
            for (target, key) in TARGETS
                .into_iter()
                .zip(["constant_variance", "constant_correlation"])
            {
                let expected = &case[key];
                // Null where the target is the sample covariance but for rounding, which makes
                // any intensity give the same covariance.
                if let Some(want) = expected["intensity"].as_f64() {
                    let got = data.ledoit_wolf(target).unwrap();
                    assert!(
                        (got - want).abs() <= 1e-12 * want,
                        "{name} {key}: {got} vs {want}"
                    );
                }
                assert_reference(
                    &data,
                    &ledoit_wolf(target),
                    expected,
                    &format!("{name} {key}"),
                );
            }
        }
    }

    /// Against the fixture: the exponentially weighted covariance, cross-checked against
    /// pandas and riskfolio-lib, and the semicovariances, against riskfolio-lib and
    /// PyPortfolioOpt.
    #[test]
    fn estimates_match_the_python_reference() {
        for (case, name, data) in reference_cases() {
            for expected in case["exponentially_weighted"].as_array().unwrap() {
                let half_life = expected["half_life"].as_f64().unwrap();
                let what = format!("{name} half-life {half_life}");
                let config = CovarianceConfig::exponentially_weighted(half_life);
                assert_reference(&data, &config, expected, &what);
                let decay =
                    CovarianceConfig::exponentially_weighted_decay(2f64.powf(-1.0 / half_life));
                assert_reference(&data, &decay, expected, &what);
            }
            let expected = &case["semicovariance"];
            assert_reference(&data, &CovarianceConfig::semicovariance(), expected, &name);
            let expected = &case["semicovariance_below"];
            let threshold = expected["threshold"].as_f64().unwrap();
            let config = CovarianceConfig::semicovariance_below(threshold);
            assert_reference(&data, &config, expected, &name);
        }
    }

    #[test]
    fn constant_variance_ledoit_wolf_is_shrinkage_by_its_intensity() {
        for (name, data) in datasets() {
            let intensity = data.ledoit_wolf(ShrinkageTarget::ConstantVariance).unwrap();
            assert!(intensity > 0.0 && intensity < 1.0, "{name}: {intensity}");
            for (estimated, fixed) in [
                (
                    ledoit_wolf(ShrinkageTarget::ConstantVariance),
                    CovarianceConfig::default().shrinkage(intensity),
                ),
                (
                    ledoit_wolf(ShrinkageTarget::ConstantVariance).regularization(1e-6),
                    CovarianceConfig::default()
                        .shrinkage(intensity)
                        .regularization(1e-6),
                ),
            ] {
                assert_eq!(
                    data.adjusted_covariance(&estimated),
                    data.adjusted_covariance(&fixed),
                    "{name}"
                );
                for (a, b) in methods(&estimated).iter().zip(methods(&fixed)) {
                    assert_eq!(
                        data.allocate(a.as_ref()),
                        data.allocate(b.as_ref()),
                        "{name}"
                    );
                }
                let model = |c: CovarianceConfig| {
                    BlackLitterman::new(vec![0.1; data.labels().len()], 2.5).covariance(c)
                };
                assert_eq!(
                    model(estimated).prior_returns(&data),
                    model(fixed).prior_returns(&data),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn the_last_shrinkage_setter_applies() {
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        for target in TARGETS {
            let estimated = ledoit_wolf(target);
            let fixed = CovarianceConfig::default().shrinkage(0.3);
            assert_eq!(fixed.clone().ledoit_wolf(target), estimated);
            assert_eq!(estimated.clone().shrinkage(0.3), fixed);
            assert_eq!(estimated.lambda(), None);
            assert_eq!(
                data.allocate(&MinVariance::default().shrinkage(0.3).ledoit_wolf(target)),
                data.allocate(&MinVariance::default().covariance(estimated.clone()))
            );
            assert_eq!(
                data.allocate(&MinVariance::default().ledoit_wolf(target).shrinkage(0.3)),
                data.allocate(&MinVariance::default().covariance(fixed))
            );
        }
        // The two targets differ, and the intensity isn't the fixed one.
        let [variance, correlation] =
            TARGETS.map(|target| data.adjusted_covariance(&ledoit_wolf(target)).unwrap());
        assert_ne!(variance, correlation);
        let intensity = data
            .ledoit_wolf(ShrinkageTarget::ConstantCorrelation)
            .unwrap();
        assert!(intensity > 0.0 && intensity < 1.0 && intensity != 0.3);
    }

    #[test]
    fn ledoit_wolf_is_only_for_the_sample_covariance() {
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        let given = CovarianceConfig::matrix(&rows(&data.cov_matrix)).unwrap();
        for (base, with) in [
            (given, "a given covariance"),
            (
                CovarianceConfig::exponentially_weighted(10.0),
                "an exponentially weighted covariance",
            ),
            (CovarianceConfig::semicovariance(), "a semicovariance"),
            (
                CovarianceConfig::semicovariance_below(0.0),
                "a semicovariance",
            ),
        ] {
            for target in TARGETS {
                for config in [
                    base.clone().ledoit_wolf(target),
                    base.clone().shrinkage(0.2).ledoit_wolf(target),
                ] {
                    assert_eq!(
                        data.allocate(&MinVariance::default().covariance(config)),
                        Err(CutupError::NotSupported {
                            setting: "ledoit_wolf",
                            with
                        })
                    );
                }
                // A fixed intensity after it is supported.
                let fixed = base.clone().ledoit_wolf(target).shrinkage(0.2);
                assert_eq!(
                    data.allocate(&MinVariance::default().covariance(fixed)),
                    data.allocate(&MinVariance::default().covariance(base.clone().shrinkage(0.2)))
                );
            }
        }
    }

    #[test]
    fn ledoit_wolf_with_a_constant_asset() {
        let zero = Err(CutupError::ZeroVariance {
            column: 0,
            asset: "0".into(),
        });
        let returns = dmatrix![0.1, 0.01, 0.02; 0.1, -0.02, 0.01; 0.1, 0.03, -0.01];
        let data = Returns::returns_matrix(returns).unwrap();
        // The constant asset has no correlations to average, even with regularization.
        let correlation = ledoit_wolf(ShrinkageTarget::ConstantCorrelation);
        assert_eq!(
            data.ledoit_wolf(ShrinkageTarget::ConstantCorrelation),
            zero.clone()
        );
        for config in [correlation.clone(), correlation.regularization(1e-6)] {
            assert_eq!(
                data.allocate(&MinVariance::default().covariance(config))
                    .err(),
                zero.clone().err()
            );
        }
        // Shrinking toward a constant variance gives it one.
        let variance = ledoit_wolf(ShrinkageTarget::ConstantVariance);
        assert!(data.ledoit_wolf(ShrinkageTarget::ConstantVariance).unwrap() > 0.0);
        assert!(
            data.allocate(&MinVariance::default().covariance(variance.clone()))
                .is_ok()
        );

        // Over two periods each centered return is the other's negative, so every product of
        // two is the same in both and the estimation error is 0: nothing is shrunk, and the
        // constant asset keeps its variance of 0, unless regularization gives it one. Powers
        // of two keep the centering exact.
        let data = Returns::returns_matrix(dmatrix![0.125, 0.25, 0.5; 0.125, -0.25, 0.75]).unwrap();
        assert_eq!(data.ledoit_wolf(ShrinkageTarget::ConstantVariance), Ok(0.0));
        assert_eq!(
            data.allocate(&MinVariance::default().covariance(variance.clone()))
                .err(),
            zero.err()
        );
        assert!(
            data.allocate(&MinVariance::default().covariance(variance.regularization(1e-6)))
                .is_ok()
        );
    }

    #[test]
    fn ledoit_wolf_leaves_one_asset_alone() {
        let data = Returns::returns_matrix(dmatrix![0.01; -0.02; 0.03]).unwrap();
        for target in TARGETS {
            assert_eq!(data.ledoit_wolf(target), Ok(0.0));
            assert_eq!(
                data.adjusted_covariance(&ledoit_wolf(target)),
                Ok(data.cov_matrix.clone())
            );
        }
    }

    #[test]
    fn ledoit_wolf_leaves_a_covariance_equal_to_its_target_alone() {
        // Uncorrelated, so the average correlation is exactly 0 and the target is the sample
        // covariance: there's an estimation error, but nothing to shrink toward.
        let data = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let off_diagonal = DMatrix::from_fn(3, 3, |i, j| if i == j { 0.0 } else { 1.0 });
        assert_eq!(data.cov_matrix.component_mul(&off_diagonal).amax(), 0.0);
        assert_eq!(
            data.ledoit_wolf(ShrinkageTarget::ConstantCorrelation),
            Ok(0.0)
        );
        assert!(data.ledoit_wolf(ShrinkageTarget::ConstantVariance).unwrap() > 0.0);

        // The same variances too: the constant variance target is the sample covariance.
        let s = 0.01;
        let equal = dmatrix![s, s; -s, s; s, -s; -s, -s];
        let data = Returns::returns_matrix(equal).unwrap();
        assert_eq!(data.ledoit_wolf(ShrinkageTarget::ConstantVariance), Ok(0.0));
    }

    #[test]
    fn ledoit_wolf_is_exactly_scale_invariant() {
        // A power of two scales every centered return exactly, so the intensities are the
        // same bit for bit, here at scales whose fourth powers would overflow or underflow
        // unless the returns were scaled first.
        let returns = factor_returns(4, 30, 7);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        for k in [-400, -40, 40, 400] {
            let factor = 2f64.powi(k);
            let scaled = Returns::returns_matrix(&returns * factor).unwrap();
            for target in TARGETS {
                assert_eq!(
                    scaled.ledoit_wolf(target),
                    data.ledoit_wolf(target),
                    "2^{k}"
                );
                assert_eq!(
                    scaled.adjusted_covariance(&ledoit_wolf(target)).unwrap(),
                    data.adjusted_covariance(&ledoit_wolf(target)).unwrap() * (factor * factor),
                    "2^{k}"
                );
            }
        }
        // Other scales, to rounding.
        for factor in [1e-120, 1e120] {
            let scaled = Returns::returns_matrix(&returns * factor).unwrap();
            for target in TARGETS {
                let (a, b) = (
                    scaled.ledoit_wolf(target).unwrap(),
                    data.ledoit_wolf(target).unwrap(),
                );
                assert!((a - b).abs() <= 1e-13 * b, "{factor}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn ledoit_wolf_of_returns_far_from_zero() {
        // Centering comes before the fourth moments: shifting every return changes nothing
        // but rounding.
        let returns = factor_returns(4, 30, 3);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let shifted = Returns::returns_matrix(returns.add_scalar(1000.0)).unwrap();
        for target in TARGETS {
            let (a, b) = (
                shifted.ledoit_wolf(target).unwrap(),
                data.ledoit_wolf(target).unwrap(),
            );
            assert!((a - b).abs() <= 1e-9 * b, "{target:?}: {a} vs {b}");
        }
    }

    /// The computed estimates, by name.
    fn estimates() -> Vec<(&'static str, CovarianceConfig)> {
        vec![
            ("half-life 3", CovarianceConfig::exponentially_weighted(3.0)),
            (
                "decay 0.9",
                CovarianceConfig::exponentially_weighted_decay(0.9),
            ),
            ("semicovariance", CovarianceConfig::semicovariance()),
            ("below 0", CovarianceConfig::semicovariance_below(0.0)),
        ]
    }

    #[test]
    fn computed_estimates_are_used_like_given_ones() {
        // Given back as a matrix, each estimate gives the same weights for every method, with
        // and without adjustments: HRP and HERC included, which cluster on the returns'
        // correlations either way.
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        for (name, config) in estimates() {
            let matrix = data.adjusted_covariance(&config).unwrap();
            let given = CovarianceConfig::matrix(&rows(&matrix)).unwrap();
            assert_ne!(matrix, data.cov_matrix, "{name}");
            for (estimated, given) in [
                (config.clone(), given.clone()),
                (config.clone().shrinkage(0.3), given.clone().shrinkage(0.3)),
                (config.regularization(1e-6), given.regularization(1e-6)),
            ] {
                let estimated_methods = methods(&estimated);
                assert!(
                    data.allocate(estimated_methods[0].as_ref()).is_ok(),
                    "{name}"
                );
                for (a, b) in estimated_methods.iter().zip(methods(&given)) {
                    assert_eq!(
                        data.allocate(a.as_ref()),
                        data.allocate(b.as_ref()),
                        "{name}"
                    );
                }
                let model =
                    |c: CovarianceConfig| BlackLitterman::new(vec![0.2; 5], 2.5).covariance(c);
                assert_eq!(
                    model(estimated).prior_returns(&data),
                    model(given).prior_returns(&data),
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn exponentially_weighted_by_hand() {
        // λ = 1/2 over three periods: weights 1/4, 1/2 and 1, the last period most.
        let returns = dmatrix![0.01, 0.03; -0.02, 0.01; 0.04, -0.02];
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let w = [0.25, 0.5, 1.0];
        let total: f64 = w.iter().sum();
        let squares: f64 = w.iter().map(|w| w * w).sum();
        let mean = |i: usize| (0..3).map(|t| w[t] * returns[(t, i)]).sum::<f64>() / total;
        let want = DMatrix::from_fn(2, 2, |i, j| {
            let products: f64 = (0..3)
                .map(|t| w[t] * (returns[(t, i)] - mean(i)) * (returns[(t, j)] - mean(j)))
                .sum();
            products / (total - squares / total)
        });
        for config in [
            CovarianceConfig::exponentially_weighted(1.0),
            CovarianceConfig::exponentially_weighted_decay(0.5),
        ] {
            let got = data.adjusted_covariance(&config).unwrap();
            assert!(
                (&got - &want).amax() <= 1e-15 * want.amax(),
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn long_half_lives_give_the_sample_covariance() {
        // Equal weights, and a denominator of T − 1, even where λ rounds to 1.
        let data = Returns::returns_matrix(factor_returns(4, 30, 5)).unwrap();
        // At a half-life of 1e12 the weights are 1 − O(T / 1e12), and the estimate moves as
        // much.
        for (half_life, tolerance) in [(1e12, 1e-9), (1e300, 1e-12), (f64::MAX, 1e-12)] {
            let got = data
                .adjusted_covariance(&CovarianceConfig::exponentially_weighted(half_life))
                .unwrap();
            let gap = (&got - &data.cov_matrix).amax();
            assert!(
                gap <= tolerance * data.cov_matrix.amax(),
                "{half_life}: {gap}"
            );
        }
        let near_one = CovarianceConfig::exponentially_weighted_decay(1.0 - f64::EPSILON);
        let got = data.adjusted_covariance(&near_one).unwrap();
        assert!((&got - &data.cov_matrix).amax() <= 1e-12 * data.cov_matrix.amax());
    }

    #[test]
    fn short_half_lives_give_the_last_two_periods() {
        // λ near 0 weighs the last period and, by λ, the one before, whose deviations carry
        // the covariance: the sample covariance of the last two periods, to O(λ).
        let returns = factor_returns(3, 20, 9);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let last_two = Returns::returns_matrix(returns.rows(18, 2).into_owned()).unwrap();
        for config in [
            CovarianceConfig::exponentially_weighted(0.01),
            CovarianceConfig::exponentially_weighted_decay(1e-300),
        ] {
            let got = data.adjusted_covariance(&config).unwrap();
            let want = &last_two.cov_matrix;
            assert!(
                (&got - want).amax() <= 1e-12 * want.amax(),
                "{got} vs {want}"
            );
        }
        // At the shortest half-life λ is about 3.3e-308, so the deviations' squares times λ
        // would be subnormal, or 0, for returns this small.
        let shortest = std::f64::consts::LN_2 / 708.0 * (1.0 + 1e-15);
        let small = &returns * 1e-9;
        let data = Returns::returns_matrix(small.clone()).unwrap();
        let last_two = Returns::returns_matrix(small.rows(18, 2).into_owned()).unwrap();
        let got = data
            .adjusted_covariance(&CovarianceConfig::exponentially_weighted(shortest))
            .unwrap();
        let want = &last_two.cov_matrix;
        assert!(
            (&got - want).amax() <= 1e-12 * want.amax(),
            "{got} vs {want}"
        );
    }

    #[test]
    fn decays_must_be_in_range() {
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        let invalid = |field, value| Err(CutupError::InvalidConfig { field, value });
        for h in [0.0, -0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, 1e-4] {
            let config = CovarianceConfig::exponentially_weighted(h);
            assert_eq!(config.validate(), invalid("half_life", h), "{h}");
            assert_eq!(
                data.allocate(&MinVariance::default().covariance(config))
                    .err(),
                invalid("half_life", h).err()
            );
        }
        for lambda in [0.0, -0.5, 1.0, 1.5, f64::INFINITY, 1e-308, 5e-324] {
            let config = CovarianceConfig::exponentially_weighted_decay(lambda);
            assert_eq!(config.validate(), invalid("decay", lambda), "{lambda}");
        }
        for (field, config) in [
            (
                "half_life",
                CovarianceConfig::exponentially_weighted(f64::NAN),
            ),
            (
                "decay",
                CovarianceConfig::exponentially_weighted_decay(f64::NAN),
            ),
            (
                "threshold",
                CovarianceConfig::semicovariance_below(f64::NAN),
            ),
        ] {
            assert!(
                matches!(config.validate(), Err(CutupError::InvalidConfig { field: f, value }) if f == field && value.is_nan()),
                "{field}"
            );
        }
        for t in [f64::INFINITY, f64::NEG_INFINITY] {
            let config = CovarianceConfig::semicovariance_below(t);
            assert_eq!(config.validate(), invalid("threshold", t));
        }
        // The range of ln λ is [−708, 0), so its edges are exact.
        assert_eq!(Decay::Lambda((-708f64).exp()).log(), Ok(-708.0));
        assert!(Decay::Lambda((-708.001f64).exp()).log().is_err());
        assert!(Decay::Lambda(1.0 - f64::EPSILON / 2.0).log().unwrap() < 0.0);
        let shortest = std::f64::consts::LN_2 / 708.0;
        for h in [shortest * (1.0 + 1e-15), 1e-3] {
            let config = CovarianceConfig::exponentially_weighted(h);
            let weights = data
                .allocate(&MinVariance::default().covariance(config))
                .unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{h}");
        }
    }

    #[test]
    fn estimates_scale_exactly_by_powers_of_two() {
        let returns = factor_returns(4, 30, 11);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        for k in [-200, 200] {
            let factor = 2f64.powi(k);
            let scaled = Returns::returns_matrix(&returns * factor).unwrap();
            for (name, config) in estimates() {
                let config = match config.estimate {
                    Estimate::Semi(Some(t)) => CovarianceConfig::semicovariance_below(t * factor),
                    _ => config,
                };
                assert_eq!(
                    scaled.adjusted_covariance(&config).unwrap(),
                    data.adjusted_covariance(&config).unwrap() * (factor * factor),
                    "{name} 2^{k}"
                );
            }
        }
    }

    #[test]
    fn shifting_the_returns_changes_only_a_fixed_threshold() {
        let returns = factor_returns(4, 30, 13);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let shifted = Returns::returns_matrix(returns.add_scalar(1.0)).unwrap();
        for (name, config) in estimates() {
            let a = data.adjusted_covariance(&config).unwrap();
            let b = if let Estimate::Semi(Some(_)) = config.estimate {
                // Every shifted return is above 0, so none falls below the same threshold, but
                // they do below one shifted with them.
                assert!(matches!(
                    shifted.adjusted_covariance(&config),
                    Err(CutupError::ZeroVariance { .. })
                ));
                shifted
                    .adjusted_covariance(&CovarianceConfig::semicovariance_below(1.0))
                    .unwrap()
            } else {
                shifted.adjusted_covariance(&config).unwrap()
            };
            let gap = (&a - &b).amax();
            assert!(gap <= 1e-12 * a.amax(), "{name}: {gap}");
        }
    }

    #[test]
    fn returns_above_the_threshold_are_ignored() {
        // Raising the gains changes nothing below the threshold, or the mean if it's fixed.
        let returns = factor_returns(3, 25, 17);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let raised =
            Returns::returns_matrix(returns.map(|r| if r > 0.0 { 3.0 * r } else { r })).unwrap();
        let below = CovarianceConfig::semicovariance_below(0.0);
        assert_eq!(
            raised.adjusted_covariance(&below),
            data.adjusted_covariance(&below)
        );
        // Below the mean, by hand: only the returns below each mean count.
        let semi = data
            .adjusted_covariance(&CovarianceConfig::semicovariance())
            .unwrap();
        let mean = returns.row_mean();
        let d = returns.map_with_location(|_, i, r| (r - mean[i]).min(0.0));
        let want = d.transpose() * &d / 24.0;
        assert!((&semi - &want).amax() <= 1e-15 * want.amax());
    }

    #[test]
    fn semicovariances_of_assets_with_no_downside() {
        let zero = |column: usize| CutupError::ZeroVariance {
            column,
            asset: column.to_string(),
        };
        let returns = dmatrix![0.01, 0.02, 0.03; 0.02, -0.01, 0.01; 0.03, 0.02, -0.02];
        let data = Returns::returns_matrix(returns).unwrap();
        // Asset 0 is never below 0, so it has no downside.
        let below = CovarianceConfig::semicovariance_below(0.0);
        assert_eq!(
            data.allocate(&MinVariance::default().covariance(below.clone()))
                .err(),
            Some(zero(0))
        );
        assert!(
            data.allocate(&MinVariance::default().covariance(below.regularization(1e-6)))
                .is_ok()
        );
        // Every asset varies, so each has returns below its mean.
        assert!(
            data.allocate(&MinVariance::default().covariance(CovarianceConfig::semicovariance()))
                .is_ok()
        );

        // A constant asset has none below its mean, but each period loses against a threshold
        // above it.
        let constant = dmatrix![0.01, 0.02; 0.01, -0.01; 0.01, 0.03];
        let data = Returns::returns_matrix(constant).unwrap();
        assert_eq!(
            data.allocate(&MinVariance::default().covariance(CovarianceConfig::semicovariance()))
                .err(),
            Some(zero(0))
        );
        let above = data
            .adjusted_covariance(&CovarianceConfig::semicovariance_below(0.02))
            .unwrap();
        // Each period is 0.01 below it.
        assert!((above[(0, 0)] - 1e-4).abs() <= 1e-18, "{above}");
        // A threshold so far away that the shortfalls overflow.
        assert_eq!(
            data.adjusted_covariance(&CovarianceConfig::semicovariance_below(1e300)),
            Err(CutupError::NonFiniteCovariance)
        );
    }

    #[test]
    fn an_exponentially_weighted_constant_asset_has_zero_variance() {
        let returns = dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03; 0.1, 0.01];
        let data = Returns::returns_matrix(returns).unwrap();
        let config = CovarianceConfig::exponentially_weighted(2.0);
        assert_eq!(
            data.allocate(&MinVariance::default().covariance(config.clone()))
                .err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
        assert!(
            data.allocate(&MinVariance::default().covariance(config.shrinkage(0.5)))
                .is_ok()
        );
    }

    #[test]
    fn a_given_variance_counts_however_small() {
        // Only an estimate from the returns is 0 to rounding. A given variance of 1e-30 next
        // to returns near 0.01 would count as 0 by that rule, but it's what was given.
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        let mut diagonal = vec![vec![0.0; 5]; 5];
        for (i, row) in diagonal.iter_mut().enumerate() {
            row[i] = if i == 1 { 1e-30 } else { 1e-4 };
        }
        let given = CovarianceConfig::matrix(&diagonal).unwrap();
        let weights = data
            .allocate(&MinVariance::default().covariance(given))
            .unwrap();
        assert!(weights[1] > 0.999, "{weights:?}");
    }
}
