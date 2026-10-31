//! Fixtures and checks shared by the unit tests.

use crate::{MinVariance, Returns, Weights};
use nalgebra::{DMatrix, DVector, dmatrix};

/// Six periods of prices for three assets whose returns covariance is full rank.
pub(crate) fn full_rank_prices() -> DMatrix<f64> {
    dmatrix![
        100.0, 50.0, 20.0;
        102.0, 49.0, 20.5;
        101.0, 51.5, 20.2;
        104.0, 51.0, 20.9;
        103.0, 52.5, 21.3;
        106.0, 52.0, 21.0
    ]
}

pub(crate) fn assert_weights_close(a: &Weights, b: &Weights, tol: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (wa, wb)) in a.iter().zip(b).enumerate() {
        assert!((wa - wb).abs() < tol, "asset {i}: {wa} vs {wb}");
    }
}

pub(crate) fn portfolio_variance(allocator: &Returns, weights: &Weights) -> f64 {
    let w = DVector::from_column_slice(weights.as_slice());
    (w.transpose() * &allocator.cov_matrix * &w)[(0, 0)]
}

pub(crate) fn sample_mean(allocator: &Returns) -> DVector<f64> {
    allocator.returns.row_mean().transpose()
}

/// Two rows of returns for three assets: Σ has rank 1, and the fully invested portfolio
/// `[4/7, 0, 3/7]` has zero variance.
pub(crate) fn rank_one_returns() -> DMatrix<f64> {
    dmatrix![
        0.01, 0.02, -0.01;
        -0.02, 0.01, 0.03
    ]
}

/// Returns of three uncorrelated assets whose variances are in the ratio 1 : 2 : 4, so the
/// minimum variance weights are inverse variance, 4/7, 2/7 and 1/7.
pub(crate) fn uncorrelated_returns() -> DMatrix<f64> {
    let s = [0.01, 0.01 * 2f64.sqrt(), 0.02];
    dmatrix![
        s[0], s[1], s[2];
        -s[0], s[1], -s[2];
        s[0], -s[1], -s[2];
        -s[0], -s[1], s[2]
    ]
}

/// Five assets driven by two factors, interleaved: assets 0, 2 and 4 follow the first
/// factor and assets 1 and 3 the second.
pub(crate) fn two_block_returns() -> DMatrix<f64> {
    let f1 = [0.010, -0.010, 0.020, -0.020, 0.005, -0.005, 0.012, -0.008];
    let f2 = [0.010, 0.010, -0.010, -0.010, 0.003, -0.003, -0.006, 0.009];
    let noise = [0.0011, -0.0007, 0.0004, -0.0013, 0.0009];
    DMatrix::from_fn(8, 5, |t, j| {
        let factor = if j % 2 == 0 { f1[t] } else { f2[t] };
        factor + noise[j] * ((t * (j + 2)) % 5) as f64
    })
}

pub(crate) fn sharpe(allocator: &Returns, weights: &Weights, risk_free_rate: f64) -> f64 {
    let w = DVector::from_column_slice(weights.as_slice());
    (sample_mean(allocator).dot(&w) - risk_free_rate)
        / portfolio_variance(allocator, weights).sqrt()
}

pub(crate) fn min_variance_return(allocator: &Returns) -> f64 {
    let w = DVector::from_column_slice(
        allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap()
            .as_slice(),
    );
    sample_mean(allocator).dot(&w)
}

/// Asset 1 is exactly 2 × asset 0 + 0.001, so (2, −1) is a fully invested portfolio with
/// zero variance that returns −0.001 every period.
pub(crate) fn collinear_returns() -> DMatrix<f64> {
    let base = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
    DMatrix::from_fn(6, 2, |t, j| {
        if j == 0 {
            base[t]
        } else {
            2.0 * base[t] + 0.001
        }
    })
}

/// Each asset's share of the portfolio variance, `wᵢ (Σw)ᵢ / wᵀΣw`.
pub(crate) fn risk_shares(allocator: &Returns, weights: &Weights) -> Vec<f64> {
    let w = DVector::from_column_slice(weights.as_slice());
    let contributions = w.component_mul(&(&allocator.cov_matrix * &w));
    let total = contributions.sum();
    contributions.iter().map(|c| c / total).collect()
}

pub(crate) fn labeled() -> Returns {
    Returns::prices_matrix(full_rank_prices())
        .unwrap()
        .labeled(["SPY", "TLT", "GLD"])
        .unwrap()
}
