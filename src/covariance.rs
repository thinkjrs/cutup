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
    /// or the shrinkage is outside `[0, 1]`, and [`CutupError::NotSupported`] for Ledoit-Wolf
    /// shrinkage of a given covariance.
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
        if let (Some(Shrinkage::LedoitWolf(_)), Estimate::Given(_)) =
            (self.shrinkage, &self.estimate)
        {
            return Err(CutupError::NotSupported {
                setting: "ledoit_wolf",
                with: "a given covariance",
            });
        }
        Ok(())
    }
}

impl Returns {
    /// The covariance `config` sets, adjusted, after checking its values.
    ///
    /// Returns [`CutupError::DimensionMismatch`] for a given covariance of another size,
    /// [`CutupError::ZeroVariance`] for a constant asset, or one given a variance of 0, unless
    /// shrinkage toward a constant variance or regularization gives it a positive variance,
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
        let base = match &config.estimate {
            Estimate::Sample => {
                if !adjusted {
                    if let Some(asset) = self.find_zero_variance() {
                        return Err(self.zero_variance(asset));
                    }
                }
                &self.cov_matrix
            }
            Estimate::Given(matrix) => {
                if matrix.nrows() != n {
                    return Err(CutupError::DimensionMismatch {
                        field: "covariance",
                        expected: n,
                        actual: matrix.nrows(),
                    });
                }
                if !adjusted {
                    if let Some(asset) = (0..n).find(|&i| matrix[(i, i)] == 0.0) {
                        return Err(self.zero_variance(asset));
                    }
                }
                matrix.as_ref()
            }
        };

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

    /// Against `tests/fixtures/covariance.json`, from `reference/covariance.py`: scikit-learn's
    /// intensity toward a constant variance, and the paper's toward a constant correlation,
    /// cross-checked against PyPortfolioOpt's.
    #[test]
    fn ledoit_wolf_matches_the_python_reference() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/covariance.json")).unwrap();
        let cases = fixture["cases"].as_array().unwrap();
        let datasets = datasets();
        assert_eq!(cases.len(), datasets.len());
        for case in cases {
            let (name, data) = datasets
                .iter()
                .find(|(name, _)| case["dataset"] == name.as_str())
                .unwrap();
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
                let got = data.adjusted_covariance(&ledoit_wolf(target)).unwrap();
                let want = DMatrix::from_row_iterator(
                    got.nrows(),
                    got.ncols(),
                    expected["covariance"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| {
                            row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap())
                        }),
                );
                assert!(
                    (&got - &want).amax() <= 1e-12 * want.amax(),
                    "{name} {key}: {got} vs {want}"
                );
            }
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
    fn ledoit_wolf_of_a_given_covariance_is_not_supported() {
        let data = Returns::returns_matrix(two_block_returns()).unwrap();
        let given = CovarianceConfig::matrix(&rows(&data.cov_matrix)).unwrap();
        for target in TARGETS {
            for config in [
                given.clone().ledoit_wolf(target),
                ledoit_wolf(target).shrinkage(0.2).ledoit_wolf(target),
            ] {
                let config = match config.estimate {
                    Estimate::Given(_) => config,
                    Estimate::Sample => CovarianceConfig {
                        estimate: given.estimate.clone(),
                        ..config
                    },
                };
                assert_eq!(
                    data.allocate(&MinVariance::default().covariance(config)),
                    Err(CutupError::NotSupported {
                        setting: "ledoit_wolf",
                        with: "a given covariance"
                    })
                );
            }
            // A fixed intensity after it is supported.
            let fixed = given.clone().ledoit_wolf(target).shrinkage(0.2);
            assert_eq!(
                data.allocate(&MinVariance::default().covariance(fixed)),
                data.allocate(&MinVariance::default().shrinkage(0.2))
            );
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
}
