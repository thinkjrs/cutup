//! [`InverseVolatility`] and [`InverseVariance`]: weights by each asset's own risk.

use crate::{Allocation, CutupError, Returns, Weights};

/// Inverse volatility weights, `wᵢ ∝ 1 / σᵢ`.
///
/// Each asset's weight is inversely proportional to the standard deviation of its returns,
/// so riskier assets get less. Correlations are ignored. Compared with [`InverseVariance`],
/// the tilt away from risky assets is milder. When assets are uncorrelated, this gives each
/// asset the same risk contribution.
///
/// The weights are positive and sum to 1.
///
/// # Errors
///
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
///
/// # Example
///
/// ```
/// use cutup::{InverseVolatility, Returns};
///
/// // Volatilities in the ratio 1 : 2.
/// let returns = [
///      [ 0.01,  0.02],
///      [-0.01, -0.02],
///      [ 0.01, -0.02],
///      [-0.01,  0.02],
/// ];
///
/// let data = Returns::new(&returns)?;
/// let weights = data.allocate(&InverseVolatility)?;
///
/// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
/// assert!((weights[1] - 1.0 / 3.0).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct InverseVolatility;

impl Allocation for InverseVolatility {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        data.weight_by_variance(|variance| 1.0 / variance.sqrt())
    }
}

/// Inverse variance weights, `wᵢ ∝ 1 / σᵢ²`.
///
/// Each asset's weight is inversely proportional to the variance of its returns.
/// Correlations are ignored. When assets are uncorrelated this is the minimum variance
/// portfolio, and it is what [`Hrp`](crate::Hrp) and [`Herc`](crate::Herc) use within each cluster.
///
/// The weights are positive and sum to 1.
///
/// # Errors
///
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
///
/// # Example
///
/// ```
/// use cutup::{InverseVariance, Returns};
///
/// // Volatilities in the ratio 1 : 2, so variances in the ratio 1 : 4.
/// let returns = [
///      [ 0.01,  0.02],
///      [-0.01, -0.02],
///      [ 0.01, -0.02],
///      [-0.01,  0.02],
/// ];
///
/// let weights = Returns::new(&returns)?.allocate(&InverseVariance)?;
///
/// assert!((weights[0] - 0.8).abs() < 1e-12);
/// assert!((weights[1] - 0.2).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct InverseVariance;

impl Allocation for InverseVariance {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        data.weight_by_variance(|variance| 1.0 / variance)
    }
}

impl Returns {
    /// Weights each asset in proportion to `f(σᵢ²)`, normalized to sum to 1.
    fn weight_by_variance(&self, f: impl Fn(f64) -> f64) -> Result<Weights, CutupError> {
        if let Some(asset) = self.find_zero_variance() {
            return Err(self.zero_variance(asset));
        }
        let raw: Vec<f64> = self.cov_matrix.diagonal().iter().map(|&v| f(v)).collect();
        let total: f64 = raw.iter().sum();
        Ok(self.weights(raw.iter().map(|w| w / total).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_weights_close, full_rank_prices};
    use crate::{Hrp, MinVariance};
    use nalgebra::dmatrix;

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
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
        assert_eq!(
            allocator.allocate(&InverseVariance).err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
    }
}
