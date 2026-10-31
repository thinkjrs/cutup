//! Adjustments to the sample covariance, set with [`CovarianceConfig`].

use crate::{CutupError, Returns};
use nalgebra::DMatrix;

/// Adjustments applied to the sample covariance matrix before a method uses it, set with its
/// `covariance` setter, such as [`MinVariance::covariance`](crate::MinVariance::covariance). Each
/// method also has `shrinkage` and `regularization` setters for each adjustment on its own.
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
    /// Applies `config` to the sample covariance matrix, after checking its values.
    ///
    /// Returns [`CutupError::ZeroVariance`] for a constant asset unless shrinkage or
    /// regularization gives it a positive variance, [`CutupError::NonFiniteCovariance`] if the
    /// result isn't finite, and [`CutupError::SingularCovariance`] if it is zero.
    pub(crate) fn adjusted_covariance(
        &self,
        config: &CovarianceConfig,
    ) -> Result<DMatrix<f64>, CutupError> {
        config.validate()?;
        let adjusted =
            config.shrinkage.unwrap_or(0.0) > 0.0 || config.regularization.unwrap_or(0.0) > 0.0;
        if !adjusted {
            if let Some(asset) = self.find_zero_variance() {
                return Err(self.zero_variance(asset));
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_weights_close, full_rank_prices};
    use crate::{EqualWeight, MaxSharpe, MeanVariance, MinVariance, RiskParity};
    use nalgebra::dmatrix;

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
}
