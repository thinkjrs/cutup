//! The covariance a method uses, set with [`CovarianceConfig`]: the sample covariance or one
//! you give, and adjustments to it.

use crate::eigen::SymmetricEigen;
use crate::{CutupError, Returns};
use nalgebra::DMatrix;
use std::sync::Arc;

/// The covariance matrix a method uses, set with its `covariance` setter, such as
/// [`MinVariance::covariance`](crate::MinVariance::covariance): the sample covariance of the
/// returns, by default, or [one you give](CovarianceConfig::matrix), with adjustments applied
/// to it. Each method also has `shrinkage` and `regularization` setters for each adjustment on
/// its own.
///
/// The default applies no adjustment, and an adjustment that isn't set isn't applied. With
/// both, the covariance `Σ` becomes `λ·μ·I + (1 − λ)·Σ + ε·I`, which is the same whichever is
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
    estimate: Estimate,
    regularization: Option<f64>,
    shrinkage: Option<f64>,
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

impl Returns {
    /// The covariance `config` sets, adjusted, after checking its values.
    ///
    /// Returns [`CutupError::DimensionMismatch`] for a given covariance of another size,
    /// [`CutupError::ZeroVariance`] for a constant asset, or one given a variance of 0, unless
    /// shrinkage or regularization gives it a positive variance,
    /// [`CutupError::NonFiniteCovariance`] if the result isn't finite, and
    /// [`CutupError::SingularCovariance`] if it is zero.
    pub(crate) fn adjusted_covariance(
        &self,
        config: &CovarianceConfig,
    ) -> Result<DMatrix<f64>, CutupError> {
        config.validate()?;
        let adjusted =
            config.shrinkage.unwrap_or(0.0) > 0.0 || config.regularization.unwrap_or(0.0) > 0.0;
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

        let shrunk_cov = if let Some(lambda) = config.shrinkage {
            let mean_variance = base.trace() / n as f64;
            (lambda * mean_variance) * &identity + (1.0 - lambda) * base
        } else {
            base.clone()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_weights_close, factor_returns, full_rank_prices, rank_one_returns, two_block_returns,
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
}
