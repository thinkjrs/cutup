//! [`MaxDiversification`]: the maximum diversification portfolio.

use crate::{Allocation, CovarianceConfig, CutupError, Returns, Weights, long_only_max_ratio};

/// The maximum diversification portfolio (Choueifaty and Coignard, 2008): maximize the
/// diversification ratio `σᵀw / sqrt(wᵀΣw)` subject to `1ᵀw = 1` and `w ≥ 0`, where `σ` holds
/// the assets' volatilities.
///
/// The ratio is the weighted average volatility over the portfolio's volatility, which is at least
/// 1, and is 1 only when the held assets are perfectly correlated. It is long-only by definition.
/// This is the long-only [`MaxSharpe`](crate::MaxSharpe) problem with `σ` in place of the excess
/// returns, solved the same way: with `y = w / σᵀw`, minimize `yᵀΣy` subject to `σᵀy = 1` and
/// `y ≥ 0`, then `w = y / 1ᵀy`. The solver works in correlation units `diag(σ)·y`, where that is
/// minimizing `vᵀCv` for the correlation matrix `C` subject to `1ᵀv = 1`, so assets whose
/// volatilities differ by many orders of magnitude stay accurate. For uncorrelated assets it is
/// inverse volatility weighting. With shrinkage or regularization, the volatilities are those of
/// the adjusted covariance.
///
/// When Σ is singular, several portfolios can share the maximum, and which one is returned
/// depends on the solver's path. Add shrinkage for a unique answer.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if `regularization` is negative or not finite, or
///   `shrinkage` is outside `[0, 1]`.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::NonFiniteCovariance`] if the adjusted covariance isn't finite.
/// * [`CutupError::SingularCovariance`] if some long-only portfolio has no variance (in the
///   correlation matrix's null space, with the cutoff of long-short
///   [`MaxSharpe`](crate::MaxSharpe)), so the ratio has no upper bound, or no finite weights can be
///   computed.
/// * [`CutupError::NotConverged`] if the solver stops early, which only extreme inputs should
///   cause.
///
/// # Example
///
/// ```
/// use cutup::{MaxDiversification, Returns};
///
/// // Uncorrelated assets, with volatilities in the ratio 1 : 2 : 4.
/// let returns = [
///      [ 0.01,  0.02,  0.04],
///      [-0.01,  0.02, -0.04],
///      [ 0.01, -0.02, -0.04],
///      [-0.01, -0.02,  0.04],
/// ];
/// let data = Returns::new(&returns)?;
/// let weights = data.allocate(&MaxDiversification::default())?;
///
/// // Inverse volatility: 4 : 2 : 1.
/// for (w, expected) in weights.iter().zip([4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]) {
///     assert!((w - expected).abs() < 1e-12);
/// }
///
/// // Fewer observations than assets: some long-only portfolio has no variance.
/// let data = Returns::new(&[
///      [ 0.01, -0.02,  0.03],
///      [-0.01,  0.02, -0.03],
/// ])?;
/// assert!(data.allocate(&MaxDiversification::default()).is_err());
/// assert!(data.allocate(&MaxDiversification::default().shrinkage(0.3)).is_ok());
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct MaxDiversification {
    covariance: CovarianceConfig,
}

impl MaxDiversification {
    covariance_setters!();
}

impl Allocation for MaxDiversification {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        data.max_diversification(&self.covariance)
    }
}

impl Returns {
    /// The maximum diversification portfolio, for [`MaxDiversification`].
    fn max_diversification(&self, covariance: &CovarianceConfig) -> Result<Weights, CutupError> {
        let cov = self.adjusted_covariance(covariance)?;
        let volatilities = cov.diagonal().map(f64::sqrt);
        let (best, highest) = volatilities.argmax();
        long_only_max_ratio(&cov, &(volatilities / highest), best)?
            .map(|w| self.weights(w))
            .ok_or(CutupError::SingularCovariance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_weights_close, collinear_returns, full_rank_prices, rank_one_returns,
    };
    use crate::{InverseVariance, InverseVolatility, MaxSharpe, MinVariance, RiskParity};
    use nalgebra::{DMatrix, DVector, dmatrix, dvector};

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
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
    }
}
