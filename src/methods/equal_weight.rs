//! [`EqualWeight`]: `1/n` each.

use crate::{Allocation, CutupError, Returns, Weights};

/// Equal weights, `1/n` each. Never fails.
///
/// ```
/// use cutup::{EqualWeight, Returns};
///
/// let prices = [
///    [100.0, 200.0, 300.0],
///    [110.0, 210.0, 310.0],
///    [120.0, 220.0, 320.0],
/// ];
///
/// let weights = Returns::from_prices(&prices)?.allocate(&EqualWeight)?;
/// assert!(weights.iter().all(|&w| w == 1.0 / 3.0));
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EqualWeight;

impl Allocation for EqualWeight {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let n = data.returns.ncols();
        Ok(data.weights(vec![1.0 / n as f64; n]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_weights_close, full_rank_prices};
    use crate::{Hrp, MinVariance};
    use nalgebra::dmatrix;

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
}
