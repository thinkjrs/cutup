//! Property tests: invariants every allocation method must satisfy on random returns.

use cutup::{
    CovarianceConfig, HrpConfig, Linkage, MeanVarianceConfig, PortfolioAllocator, Weights,
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
    for linkage in LINKAGES {
        let config = HrpConfig::default().with_linkage(linkage);
        out.push((
            format!("hrp ({linkage:?})"),
            allocator.hrp_allocation_with_config(&config).unwrap(),
        ));
    }
    out
}

/// Methods that depend only on Σ and must not depend on return scale. With `with_hrp` false,
/// HRP is left out: like the original algorithm, it can depend on column order.
fn risk_only_methods(allocator: &PortfolioAllocator, with_hrp: bool) -> Vec<(String, Weights)> {
    let shrunk = CovarianceConfig::default().with_shrinkage(0.3);
    let mut out = vec![
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
    ];
    for linkage in LINKAGES.iter().filter(|_| with_hrp) {
        let config = HrpConfig::default().with_linkage(*linkage);
        out.push((
            format!("hrp ({linkage:?})"),
            allocator.hrp_allocation_with_config(&config).unwrap(),
        ));
    }
    out
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
        for linkage in LINKAGES {
            let config = HrpConfig::default().with_linkage(linkage);
            long_only.push(allocator.hrp_allocation_with_config(&config).unwrap());
        }
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
        let base = risk_only_methods(&PortfolioAllocator::from_returns(returns).unwrap(), false);
        let flipped = risk_only_methods(&PortfolioAllocator::from_returns(reversed).unwrap(), false);
        for ((name, a), (_, b)) in base.iter().zip(&flipped) {
            for j in 0..n {
                prop_assert!((a[j] - b[n - 1 - j]).abs() < 1e-9, "{name}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn return_scale_does_not_matter(returns in returns(), scale in 0.1f64..10.0) {
        prop_assume!(well_posed(&returns));
        let base = risk_only_methods(&PortfolioAllocator::from_returns(returns.clone()).unwrap(), true);
        let scaled = risk_only_methods(&PortfolioAllocator::from_returns(returns * scale).unwrap(), true);
        for ((name, a), (_, b)) in base.iter().zip(&scaled) {
            for (x, y) in a.iter().zip(b) {
                prop_assert!((x - y).abs() < 1e-9, "{name}: {a:?} vs {b:?}");
            }
        }
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
}
