use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;

pub struct PortfolioAllocator {
    price_data: DMatrix<f64>,
    cov_matrix: DMatrix<f64>,
}

impl PortfolioAllocator {
    pub fn new(price_data: DMatrix<f64>) -> Self {
        let cov_matrix = PortfolioAllocator::compute_covariance_matrix(&price_data);
        PortfolioAllocator {
            price_data,
            cov_matrix,
        }
    }

    fn compute_covariance_matrix(returns: &DMatrix<f64>) -> DMatrix<f64> {
        let mean = returns.row_mean();
        let mean_matrix = DMatrix::from_rows(&vec![mean.clone(); returns.nrows()]);
        let centered = returns - mean_matrix;
        (centered.transpose() * centered) / (returns.nrows() as f64 - 1.0)
    }

    pub fn mvo_allocation(&self) -> HashMap<usize, f64> {
        let n = self.cov_matrix.ncols();
        let ones = DVector::from_element(n, 1.0);
        let lambda = 1e-6; // Regularization factor

        // Regularized covariance matrix to ensure invertibility
        let regularized_cov = self.cov_matrix.clone() + DMatrix::identity(n, n) * lambda;

        let inv_cov = regularized_cov
            .try_inverse()
            .expect("Matrix inversion shouldn't fail due to regularization.");

        let denominator = (ones.transpose() * &inv_cov * &ones)[(0, 0)];
        let weights = &inv_cov * &ones / denominator;

        (0..n).map(|i| (i, weights[i])).collect()
    }
    pub fn ew_allocation(&self) -> HashMap<usize, f64> {
        let n = self.price_data.ncols();
        (0..n).map(|i| (i, 1.0 / n as f64)).collect()
    }

    pub fn hrp_allocation(&self) -> HashMap<usize, f64> {
        let n = self.cov_matrix.ncols();
        let mut weights = vec![1.0; n];
        let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];

        while let Some(cluster) = clusters.pop() {
            if cluster.len() == 1 {
                continue;
            }
            let mid = cluster.len() / 2;
            let left = &cluster[..mid];
            let right = &cluster[mid..];

            let vol_left: f64 = left.iter().map(|&i| self.cov_matrix[(i, i)]).sum();
            let vol_right: f64 = right.iter().map(|&i| self.cov_matrix[(i, i)]).sum();
            let total_vol = vol_left + vol_right;

            for &idx in left {
                weights[idx] *= vol_right / total_vol;
            }
            for &idx in right {
                weights[idx] *= vol_left / total_vol;
            }

            clusters.push(left.to_vec());
            clusters.push(right.to_vec());
        }

        (0..n).map(|i| (i, weights[i])).collect()
    }
}

pub fn run_portfolio_allocation(prices: DMatrix<f64>) -> HashMap<usize, f64> {
    let allocator = PortfolioAllocator::new(prices);
    allocator.mvo_allocation()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    #[test]
    fn test_ew_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let ew_weights = allocator.ew_allocation();
        assert_eq!(ew_weights.len(), 3);
        assert!((ew_weights.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mvo_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let mvo_weights = allocator.mvo_allocation();
        assert_eq!(mvo_weights.len(), 3);
        assert!((mvo_weights.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hrp_allocation() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let hrp_weights = allocator.hrp_allocation();
        assert_eq!(hrp_weights.len(), 3);
        assert!((hrp_weights.values().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_asset() {
        let prices = dmatrix![
            100.0;
            110.0;
            120.0
        ];
        let allocator = PortfolioAllocator::new(prices);
        let ew_weights = allocator.ew_allocation();
        assert_eq!(ew_weights.len(), 1);
        assert_eq!(ew_weights.get(&0), Some(&1.0));
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

        let weights = run_portfolio_allocation(prices);

        // Ensure weights sum to 1.0
        let total_weight: f64 = weights.values().sum();
        assert!(
            (total_weight - 1.0).abs() < 1e-6,
            "Weights should sum to 1.0"
        );

        // Ensure the correct number of assets
        assert_eq!(weights.len(), 4, "Should return weights for 4 assets");
    }
}
