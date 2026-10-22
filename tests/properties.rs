//! Property tests: invariants every allocation method must satisfy on random returns.

use cutup::{
    CovarianceConfig, HrpConfig, Linkage, MaxSharpeConfig, MeanVarianceConfig, PortfolioAllocator,
    Weights,
};
use nalgebra::{DMatrix, DVector};
use proptest::prelude::*;

const LINKAGES: [Linkage; 4] = [
    Linkage::Single,
    Linkage::Complete,
    Linkage::Average,
    Linkage::Ward,
];

/// Random daily-scale returns with more periods than assets, so the covariance is full rank.
fn returns() -> impl Strategy<Value = DMatrix<f64>> {
    (2usize..=6)
        .prop_flat_map(|assets| (Just(assets), (assets + 3)..=40))
        .prop_flat_map(|(assets, periods)| {
            prop::collection::vec(-0.05f64..0.05, assets * periods)
                .prop_map(move |values| DMatrix::from_vec(periods, assets, values))
        })
}

/// Rejects inputs the invariants don't apply to: a near-constant asset (an error by design)
/// or a covariance so ill-conditioned that rounding swamps a 1e-9 comparison.
fn well_posed(returns: &DMatrix<f64>) -> bool {
    let cov = covariance(returns);
    let singular_values = cov.clone().svd(false, false).singular_values;
    let smallest_volatility = cov
        .diagonal()
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        .sqrt();
    smallest_volatility > 1e-3 && singular_values.max() < 1e8 * singular_values.min()
}

fn covariance(returns: &DMatrix<f64>) -> DMatrix<f64> {
    let mean = returns.row_mean();
    let centered = DMatrix::from_fn(returns.nrows(), returns.ncols(), |t, j| {
        returns[(t, j)] - mean[j]
    });
    centered.transpose() * &centered / (returns.nrows() as f64 - 1.0)
}

fn variance(cov: &DMatrix<f64>, weights: &Weights) -> f64 {
    let w = DVector::from_column_slice(weights.as_slice());
    w.dot(&(cov * &w))
}

/// HRP with every linkage, by name.
fn hrp_methods(allocator: &PortfolioAllocator) -> Vec<(String, Weights)> {
    LINKAGES
        .iter()
        .map(|&linkage| {
            let config = HrpConfig::default().with_linkage(linkage);
            (
                format!("hrp ({linkage:?})"),
                allocator.hrp_allocation_with_config(&config).unwrap(),
            )
        })
        .collect()
}

/// Every method that returns fully invested weights, by name.
fn all_methods(allocator: &PortfolioAllocator, mean: f64) -> Vec<(String, Weights)> {
    let mut out = vec![
        ("equal weight".to_string(), allocator.ew_allocation()),
        (
            "inverse volatility".to_string(),
            allocator.inverse_volatility_allocation().unwrap(),
        ),
        (
            "inverse variance".to_string(),
            allocator.inverse_variance_allocation().unwrap(),
        ),
        (
            "min variance".to_string(),
            allocator.min_variance_allocation().unwrap(),
        ),
        (
            "mean-variance (risk aversion)".to_string(),
            allocator
                .mean_variance_allocation(&MeanVarianceConfig::risk_aversion(50.0))
                .unwrap(),
        ),
        (
            "mean-variance (target return)".to_string(),
            allocator
                .mean_variance_allocation(&MeanVarianceConfig::target_return(mean))
                .unwrap(),
        ),
    ];
    out.extend(hrp_methods(allocator));
    out
}

/// Methods that depend only on Σ, and on neither column order nor return scale. HRP is
/// tested separately: like the original algorithm, it can depend on column order.
fn risk_only_methods(allocator: &PortfolioAllocator) -> Vec<(String, Weights)> {
    let shrunk = CovarianceConfig::default().with_shrinkage(0.3);
    vec![
        (
            "inverse volatility".to_string(),
            allocator.inverse_volatility_allocation().unwrap(),
        ),
        (
            "inverse variance".to_string(),
            allocator.inverse_variance_allocation().unwrap(),
        ),
        (
            "min variance".to_string(),
            allocator.min_variance_allocation().unwrap(),
        ),
        (
            "min variance (shrinkage)".to_string(),
            allocator
                .min_variance_allocation_with_config(&shrunk)
                .unwrap(),
        ),
    ]
}

fn assert_same(
    base: &[(String, Weights)],
    other: &[(String, Weights)],
) -> Result<(), TestCaseError> {
    for ((name, a), (_, b)) in base.iter().zip(other) {
        for (x, y) in a.iter().zip(b) {
            prop_assert!((x - y).abs() < 1e-9, "{name}: {a:?} vs {b:?}");
        }
    }
    Ok(())
}

proptest! {
    #[test]
    fn weights_are_finite_and_sum_to_one(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let mean = returns.mean() * 1.5;
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        for (name, weights) in all_methods(&allocator, mean) {
            prop_assert!(weights.iter().all(|w| w.is_finite()), "{name}: {weights:?}");
            let total: f64 = weights.iter().sum();
            prop_assert!((total - 1.0).abs() < 1e-9, "{name} sums to {total}");
        }
    }

    #[test]
    fn long_only_methods_never_short(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let mut long_only = vec![
            allocator.ew_allocation(),
            allocator.inverse_volatility_allocation().unwrap(),
            allocator.inverse_variance_allocation().unwrap(),
        ];
        long_only.extend(hrp_methods(&allocator).into_iter().map(|(_, weights)| weights));
        for weights in long_only {
            prop_assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        }
    }

    #[test]
    fn min_variance_has_the_lowest_variance(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let mean = returns.mean() * 1.5;
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let minimum = variance(&cov, &allocator.min_variance_allocation().unwrap());
        for (name, weights) in all_methods(&allocator, mean) {
            let other = variance(&cov, &weights);
            prop_assert!(minimum <= other * (1.0 + 1e-9), "{name}: {other} < {minimum}");
        }
    }

    #[test]
    fn column_order_does_not_matter(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let reversed = DMatrix::from_fn(returns.nrows(), n, |t, j| returns[(t, n - 1 - j)]);
        let base = risk_only_methods(&PortfolioAllocator::from_returns(returns).unwrap());
        let flipped = risk_only_methods(&PortfolioAllocator::from_returns(reversed).unwrap());
        for ((name, a), (_, b)) in base.iter().zip(&flipped) {
            for j in 0..n {
                prop_assert!((a[j] - b[n - 1 - j]).abs() < 1e-9, "{name}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn return_scale_does_not_matter(returns in returns(), scale in 0.1f64..10.0) {
        prop_assume!(well_posed(&returns));
        let base = PortfolioAllocator::from_returns(returns.clone()).unwrap();
        let scaled = PortfolioAllocator::from_returns(returns * scale).unwrap();
        assert_same(&risk_only_methods(&base), &risk_only_methods(&scaled))?;
    }

    #[test]
    fn hrp_ignores_return_scale(returns in returns(), exponent in -4i32..=4) {
        // A power of two scales every value exactly, so the correlations are bit-identical
        // and a near-tie between distances can't flip a merge. Any other factor could round
        // a tie differently and legitimately move a split.
        prop_assume!(well_posed(&returns));
        let base = PortfolioAllocator::from_returns(returns.clone()).unwrap();
        let scaled = PortfolioAllocator::from_returns(returns * 2f64.powi(exponent)).unwrap();
        assert_same(&hrp_methods(&base), &hrp_methods(&scaled))?;
    }

    #[test]
    fn mean_variance_meets_its_objective(returns in returns(), gamma in 1.0f64..500.0) {
        prop_assume!(well_posed(&returns));
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let target = mu.mean() * 1.5;
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();

        let targeted = allocator
            .mean_variance_allocation(&MeanVarianceConfig::target_return(target))
            .unwrap();
        let w = DVector::from_column_slice(targeted.as_slice());
        prop_assert!((mu.dot(&w) - target).abs() < 1e-9 * (1.0 + target.abs()));

        // At the optimum of μᵀw − (γ/2)wᵀΣw with 1ᵀw = 1, μ − γΣw is a constant vector.
        let averse = allocator
            .mean_variance_allocation(&MeanVarianceConfig::risk_aversion(gamma))
            .unwrap();
        let w = DVector::from_column_slice(averse.as_slice());
        let gradient = &mu - &cov * &w * gamma;
        let spread = gradient.max() - gradient.min();
        prop_assert!(spread < 1e-9 * (1.0 + mu.amax()), "spread {spread}");
    }

    #[test]
    fn max_sharpe_has_the_highest_sharpe_ratio(returns in returns(), margin in 1e-4f64..0.01) {
        prop_assume!(well_posed(&returns));
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let mean = returns.mean() * 1.5;
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();

        // The maximum exists for any risk-free rate below the minimum variance return.
        let min_variance = allocator.min_variance_allocation().unwrap();
        let risk_free_rate = mu.dot(&DVector::from_column_slice(min_variance.as_slice())) - margin;
        let sharpe = |weights: &Weights| {
            let w = DVector::from_column_slice(weights.as_slice());
            (mu.dot(&w) - risk_free_rate) / variance(&cov, weights).sqrt()
        };

        let config = MaxSharpeConfig::default().with_risk_free_rate(risk_free_rate);
        let best = sharpe(&allocator.max_sharpe_allocation(&config).unwrap());
        for (name, weights) in all_methods(&allocator, mean) {
            let other = sharpe(&weights);
            prop_assert!(other <= best + 1e-9 * (1.0 + best.abs()), "{name}: {other} > {best}");
        }
    }
}
