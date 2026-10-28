//! Property tests: invariants every allocation method must satisfy on random returns.

use cutup::{
    CovarianceConfig, CutupError, EqualWeight, Herc, HercConfig, Hrp, HrpConfig, InverseVariance,
    InverseVolatility, Linkage, MaxDiversification, MaxSharpe, MaxSharpeConfig, MeanVariance,
    MeanVarianceConfig, MinVariance, PortfolioAllocator, RiskParity, RiskParityConfig,
    WeightBounds, Weights,
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

/// HERC with every linkage, by name.
fn herc_methods(allocator: &PortfolioAllocator) -> Vec<(String, Weights)> {
    LINKAGES
        .iter()
        .map(|&linkage| {
            let config = HercConfig::default().with_linkage(linkage);
            (
                format!("herc ({linkage:?})"),
                allocator.herc_allocation_with_config(&config).unwrap(),
            )
        })
        .collect()
}

/// A risk-free rate below every generated expected return, so long-only max Sharpe exists.
fn low_rate() -> MaxSharpeConfig {
    MaxSharpeConfig::default().with_risk_free_rate(-1.0)
}

/// `aᵀw / sqrt(wᵀΣw)`: the Sharpe ratio for excess returns `a`, or the diversification ratio
/// for volatilities.
fn sharpe(cov: &DMatrix<f64>, excess: &DVector<f64>, weights: &Weights) -> f64 {
    let w = DVector::from_column_slice(weights.as_slice());
    excess.dot(&w) / variance(cov, weights).sqrt()
}

/// Methods whose objective doesn't depend on each asset's units: rescaling one asset's
/// returns only rescales its weight. Long-only max Sharpe gets each asset's mean absolute
/// return as its expected return, which is positive and rescales with the asset.
fn unit_free_methods(returns: &DMatrix<f64>) -> Vec<(String, Weights)> {
    let allocator = PortfolioAllocator::from_returns(returns.clone()).unwrap();
    let mean_absolute = returns.abs().row_mean().transpose();
    let zero_rate = MaxSharpeConfig::default().with_expected_returns(mean_absolute);
    vec![
        (
            "max diversification".into(),
            allocator.max_diversification_allocation().unwrap(),
        ),
        (
            "risk parity".into(),
            allocator.risk_parity_allocation().unwrap(),
        ),
        (
            "long-only max Sharpe".into(),
            allocator
                .long_only_max_sharpe_allocation(&zero_rate)
                .unwrap(),
        ),
    ]
}

/// Random feasible bounds for up to six assets, as fractions: each floor is up to 90% of
/// `1/n`, and each cap is at least `1/n`, or absent. So the floors sum to at most 0.9 and the
/// caps to at least 1.
fn bounds() -> impl Strategy<Value = Vec<(f64, f64, bool)>> {
    prop::collection::vec(
        (
            prop_oneof![Just(0.0), 0.0f64..0.9],
            0.0f64..1.0,
            prop::bool::weighted(0.8),
        ),
        6,
    )
}

/// The bounds for `n` assets from the fractions [`bounds`] draws.
fn weight_bounds(fractions: &[(f64, f64, bool)], n: usize) -> (Vec<f64>, Vec<f64>) {
    let even = 1.0 / n as f64;
    fractions[..n]
        .iter()
        .map(|&(floor, cap, capped)| {
            let upper = if capped {
                even + cap * (1.0 - even)
            } else {
                f64::INFINITY
            };
            (floor * even, upper)
        })
        .unzip()
}

/// Checks that `w` is within the bounds, sums to 1, and minimizes an objective with gradient
/// `gradient` over them: no asset that can grow has a lower marginal cost than any asset that
/// can shrink, since moving weight between them would lower the objective.
fn assert_bounded_optimum(
    w: &DVector<f64>,
    gradient: &DVector<f64>,
    (lower, upper): (&[f64], &[f64]),
    tolerance: f64,
) -> Result<(), TestCaseError> {
    let n = w.len();
    for i in 0..n {
        prop_assert!(
            w[i] >= lower[i] - 1e-15 && w[i] <= upper[i] + 1e-15,
            "asset {i}: {w:?}"
        );
    }
    prop_assert!((w.sum() - 1.0).abs() < 1e-12, "sums to {}", w.sum());
    let can_shrink = (0..n).filter(|&i| w[i] > lower[i] + 1e-9);
    let can_grow: Vec<usize> = (0..n).filter(|&i| w[i] < upper[i] - 1e-9).collect();
    for j in can_shrink {
        for &i in &can_grow {
            prop_assert!(
                gradient[j] <= gradient[i] + tolerance,
                "moving weight from {j} to {i} lowers the objective: {w:?}"
            );
        }
    }
    Ok(())
}

/// Every long-only method's weights, to compare an optimum against.
fn long_only_methods(allocator: &PortfolioAllocator) -> Vec<Weights> {
    let mut methods = vec![
        allocator.ew_allocation(),
        allocator.inverse_volatility_allocation().unwrap(),
        allocator.inverse_variance_allocation().unwrap(),
        allocator.long_only_min_variance_allocation().unwrap(),
        allocator
            .long_only_mean_variance_allocation(&MeanVarianceConfig::risk_aversion(50.0))
            .unwrap(),
        allocator
            .long_only_max_sharpe_allocation(&low_rate())
            .unwrap(),
        allocator.max_diversification_allocation().unwrap(),
        allocator.risk_parity_allocation().unwrap(),
    ];
    methods.extend(
        hrp_methods(allocator)
            .into_iter()
            .map(|(_, weights)| weights),
    );
    methods.extend(
        herc_methods(allocator)
            .into_iter()
            .map(|(_, weights)| weights),
    );
    methods
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
    out.push((
        "long-only min variance".to_string(),
        allocator.long_only_min_variance_allocation().unwrap(),
    ));
    out.push((
        "long-only mean-variance".to_string(),
        allocator
            .long_only_mean_variance_allocation(&MeanVarianceConfig::risk_aversion(50.0))
            .unwrap(),
    ));
    out.push((
        "max diversification".to_string(),
        allocator.max_diversification_allocation().unwrap(),
    ));
    out.push((
        "long-only max Sharpe".to_string(),
        allocator
            .long_only_max_sharpe_allocation(&low_rate())
            .unwrap(),
    ));
    out.push((
        "risk parity".to_string(),
        allocator.risk_parity_allocation().unwrap(),
    ));
    out.extend(hrp_methods(allocator));
    out.extend(herc_methods(allocator));
    out
}

/// Methods that depend only on Σ, and on neither column order nor return scale. HRP and
/// HERC are tested separately: like the original algorithm, HRP can depend on column order,
/// and both cluster on correlations, which only an exact rescaling leaves bit-identical.
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
        (
            "long-only min variance".to_string(),
            allocator.long_only_min_variance_allocation().unwrap(),
        ),
        (
            "long-only min variance (shrinkage)".to_string(),
            allocator
                .long_only_min_variance_allocation_with_config(&shrunk)
                .unwrap(),
        ),
        (
            "max diversification".to_string(),
            allocator.max_diversification_allocation().unwrap(),
        ),
        (
            "max diversification (shrinkage)".to_string(),
            allocator
                .max_diversification_allocation_with_config(&shrunk)
                .unwrap(),
        ),
        (
            "risk parity".to_string(),
            allocator.risk_parity_allocation().unwrap(),
        ),
        (
            "risk parity (shrinkage)".to_string(),
            allocator
                .risk_parity_allocation_with_config(
                    &RiskParityConfig::default().with_covariance(shrunk),
                )
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
            // Long-short weights can be in the thousands when expected returns are nearly
            // equal, so their sum is checked relative to the largest position.
            let total: f64 = weights.iter().sum();
            let tolerance = if name == "min variance" || name.starts_with("mean-variance") {
                1e-9 * weights.iter().fold(1.0f64, |m, w| m.max(w.abs()))
            } else {
                1e-12
            };
            prop_assert!((total - 1.0).abs() < tolerance, "{name} sums to {total}");
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
            allocator.long_only_min_variance_allocation().unwrap(),
            allocator
                .long_only_mean_variance_allocation(&MeanVarianceConfig::risk_aversion(50.0))
                .unwrap(),
            allocator.long_only_max_sharpe_allocation(&low_rate()).unwrap(),
            allocator.max_diversification_allocation().unwrap(),
            allocator.risk_parity_allocation().unwrap(),
        ];
        long_only.extend(hrp_methods(&allocator).into_iter().map(|(_, weights)| weights));
        long_only.extend(herc_methods(&allocator).into_iter().map(|(_, weights)| weights));
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
    fn long_only_min_variance_satisfies_kkt(
        returns in returns(),
        shrinkage in prop::option::of(0.0f64..1.0),
    ) {
        // w minimizes wᵀΣw over long-only portfolios exactly when every asset's marginal
        // variance (Σw)ᵢ is at least the portfolio's variance wᵀΣw, with equality wherever
        // wᵢ > 0: then no shift of weight between assets lowers the variance.
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let mut config = CovarianceConfig::default();
        let mut cov = covariance(&returns);
        if let Some(lambda) = shrinkage {
            config = config.with_shrinkage(lambda);
            let average = cov.trace() / n as f64;
            cov = cov * (1.0 - lambda) + DMatrix::identity(n, n) * (lambda * average);
        }
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let weights = allocator.long_only_min_variance_allocation_with_config(&config).unwrap();
        prop_assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        prop_assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let w = DVector::from_column_slice(weights.as_slice());
        let marginal = &cov * &w;
        let v = w.dot(&marginal);
        let tolerance = 1e-8 * marginal.amax();
        for i in 0..n {
            prop_assert!(marginal[i] >= v - tolerance, "asset {i}: {} < {v}", marginal[i]);
            prop_assert!(w[i] * (marginal[i] - v).abs() <= tolerance, "asset {i}: {weights:?}");
        }
    }

    #[test]
    fn long_only_min_variance_is_lowest_among_long_only(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let unconstrained = allocator.min_variance_allocation().unwrap();
        let long_only = allocator.long_only_min_variance_allocation().unwrap();
        let minimum = variance(&cov, &long_only);
        prop_assert!(minimum >= variance(&cov, &unconstrained) * (1.0 - 1e-9));

        // When the unconstrained portfolio is already long-only, the two are the same.
        if unconstrained.iter().all(|&w| w >= 0.0) {
            for (a, b) in unconstrained.iter().zip(&long_only) {
                prop_assert!((a - b).abs() < 1e-9, "{unconstrained:?} vs {long_only:?}");
            }
        }

        for weights in long_only_methods(&allocator) {
            let other = variance(&cov, &weights);
            prop_assert!(minimum <= other * (1.0 + 1e-9), "{weights:?}: {other} < {minimum}");
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
    fn herc_ignores_column_order(returns in returns()) {
        // HERC splits where the tree branches, not at a position in the ordered list, so
        // unlike HRP it doesn't depend on the order of the columns.
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let reversed = DMatrix::from_fn(returns.nrows(), n, |t, j| returns[(t, n - 1 - j)]);
        let base = herc_methods(&PortfolioAllocator::from_returns(returns).unwrap());
        let flipped = herc_methods(&PortfolioAllocator::from_returns(reversed).unwrap());
        for ((name, a), (_, b)) in base.iter().zip(&flipped) {
            for j in 0..n {
                prop_assert!((a[j] - b[n - 1 - j]).abs() < 1e-9, "{name}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn risk_parity_meets_its_budgets(
        returns in returns(),
        budgets in prop::collection::vec(0.01f64..1.0, 6),
        shrinkage in prop::option::of(0.0f64..1.0),
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let mut covariance_config = CovarianceConfig::default();
        if let Some(lambda) = shrinkage {
            covariance_config = covariance_config.with_shrinkage(lambda);
        }
        let budgets = DVector::from_column_slice(&budgets[..n]);
        let config = RiskParityConfig::default()
            .with_risk_budgets(budgets.clone())
            .with_covariance(covariance_config);
        let allocator = PortfolioAllocator::from_returns(returns.clone()).unwrap();
        let weights = allocator.risk_parity_allocation_with_config(&config).unwrap();
        prop_assert!(weights.iter().all(|&w| w > 0.0), "{weights:?}");

        // The same shrinkage, applied here: Σ' = λ·(tr Σ / n)·I + (1 − λ)·Σ.
        let mut cov = covariance(&returns);
        if let Some(lambda) = shrinkage {
            let average = cov.trace() / n as f64;
            cov = cov * (1.0 - lambda) + DMatrix::identity(n, n) * (lambda * average);
        }
        let w = DVector::from_column_slice(weights.as_slice());
        let contributions = w.component_mul(&(&cov * &w));
        for i in 0..n {
            let share = contributions[i] / contributions.sum();
            let budget = budgets[i] / budgets.sum();
            prop_assert!((share - budget).abs() < 1e-9, "asset {i}: {share} vs {budget}");
        }
    }

    #[test]
    fn hrp_and_herc_ignore_return_scale(returns in returns(), exponent in -4i32..=4) {
        // A power of two scales every value exactly, so the correlations are bit-identical
        // and a near-tie between distances can't flip a merge. Any other factor could round
        // a tie differently and legitimately move a split.
        prop_assume!(well_posed(&returns));
        let base = PortfolioAllocator::from_returns(returns.clone()).unwrap();
        let scaled = PortfolioAllocator::from_returns(returns * 2f64.powi(exponent)).unwrap();
        assert_same(&hrp_methods(&base), &hrp_methods(&scaled))?;
        assert_same(&herc_methods(&base), &herc_methods(&scaled))?;
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
    fn long_only_mean_variance_satisfies_kkt(
        returns in returns(),
        gamma in 0.1f64..500.0,
        fraction in 0.05f64..0.95,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();

        // For a risk aversion, the gradient g = μ − γΣw of the utility is the same on every
        // held asset and no higher on the others: no shift of weight between assets adds
        // utility.
        let averse = allocator
            .long_only_mean_variance_allocation(&MeanVarianceConfig::risk_aversion(gamma))
            .unwrap();
        let w = DVector::from_column_slice(averse.as_slice());
        prop_assert!(w.iter().all(|&x| x >= 0.0), "{averse:?}");
        prop_assert!((w.sum() - 1.0).abs() < 1e-12);
        let risk = &cov * &w * gamma;
        let gradient = &mu - &risk;
        let level = w.dot(&gradient);
        let tolerance = 1e-8 * (mu.amax() + risk.amax());
        for i in 0..n {
            prop_assert!(gradient[i] <= level + tolerance, "asset {i}: {averse:?}");
            prop_assert!(w[i] * (gradient[i] - level).abs() <= tolerance, "asset {i}: {averse:?}");
        }

        // For a target return, the marginal variance Σw is λ₁ + λ₂μᵢ on every held asset and
        // no lower on the others, for some multipliers λ. Fitting λ needs expected returns
        // that differ by more than rounding.
        prop_assume!(mu.max() - mu.min() > 1e-6);
        let target = mu.min() + fraction * (mu.max() - mu.min());
        let targeted = allocator
            .long_only_mean_variance_allocation(&MeanVarianceConfig::target_return(target))
            .unwrap();
        let w = DVector::from_column_slice(targeted.as_slice());
        prop_assert!(w.iter().all(|&x| x >= 0.0), "{targeted:?}");
        prop_assert!((w.sum() - 1.0).abs() < 1e-12);
        prop_assert!((w.dot(&mu) - target).abs() < 1e-12 * mu.amax(), "{targeted:?}");
        let marginal = &cov * &w;
        let held: Vec<usize> = (0..n).filter(|&i| w[i] > 1e-9).collect();
        let basis = DMatrix::from_fn(held.len(), 2, |k, j| if j == 0 { 1.0 } else { mu[held[k]] / mu.amax() });
        let lambda = basis
            .svd(true, true)
            .solve(&DVector::from_fn(held.len(), |k, _| marginal[held[k]]), 1e-12)
            .unwrap();
        let tolerance = 1e-8 * marginal.amax();
        for i in 0..n {
            let residual = marginal[i] - lambda[0] - lambda[1] * mu[i] / mu.amax();
            prop_assert!(residual >= -tolerance, "asset {i}: {targeted:?}");
            prop_assert!(w[i] * residual.abs() <= tolerance, "asset {i}: {targeted:?}");
        }
    }

    #[test]
    fn long_only_mean_variance_matches_unconstrained_when_long(
        returns in returns(),
        gamma in 1.0f64..500.0,
        fraction in 0.05f64..0.95,
    ) {
        prop_assume!(well_posed(&returns));
        let mu = returns.row_mean().transpose();
        let target = mu.min() + fraction * (mu.max() - mu.min());
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        for config in [
            MeanVarianceConfig::risk_aversion(gamma),
            MeanVarianceConfig::target_return(target),
        ] {
            let unconstrained = allocator.mean_variance_allocation(&config).unwrap();
            let long_only = allocator.long_only_mean_variance_allocation(&config).unwrap();
            if unconstrained.iter().all(|&w| w >= 0.0) {
                for (a, b) in unconstrained.iter().zip(&long_only) {
                    prop_assert!((a - b).abs() < 1e-9, "{unconstrained:?} vs {long_only:?}");
                }
            }
        }
    }

    #[test]
    fn long_only_max_sharpe_satisfies_kkt(returns in returns(), fraction in 0.0f64..0.9) {
        // The Sharpe ratio doesn't change when w is scaled, so at a long-only maximum its
        // gradient, proportional to (μ − r_f)·σ² − (μ − r_f)ᵀw·Σw, is zero on every held
        // asset and no higher on the others.
        prop_assume!(well_posed(&returns));
        let mu = returns.row_mean().transpose();
        prop_assume!(mu.max() - mu.min() > 1e-6);
        let rate = mu.min() + fraction * (mu.max() - mu.min());
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = MaxSharpeConfig::default().with_risk_free_rate(rate);
        let weights = allocator.long_only_max_sharpe_allocation(&config).unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        prop_assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
        prop_assert!((w.sum() - 1.0).abs() < 1e-12);

        let excess = mu.add_scalar(-rate);
        let marginal = &cov * &w;
        let reward = &excess * w.dot(&marginal);
        let risk = marginal * excess.dot(&w);
        let gradient = &reward - &risk;
        let tolerance = 1e-8 * (reward.amax() + risk.amax());
        for i in 0..w.len() {
            prop_assert!(gradient[i] <= tolerance, "asset {i}: {weights:?}");
            prop_assert!(w[i] * gradient[i].abs() <= tolerance, "asset {i}: {weights:?}");
        }
    }

    #[test]
    fn long_only_max_sharpe_is_highest_among_long_only(
        returns in returns(),
        fraction in 0.0f64..0.9,
    ) {
        prop_assume!(well_posed(&returns));
        let mu = returns.row_mean().transpose();
        prop_assume!(mu.max() - mu.min() > 1e-6);
        let rate = mu.min() + fraction * (mu.max() - mu.min());
        let excess = mu.add_scalar(-rate);
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = MaxSharpeConfig::default().with_risk_free_rate(rate);
        let best = sharpe(&cov, &excess, &allocator.long_only_max_sharpe_allocation(&config).unwrap());

        for weights in long_only_methods(&allocator) {
            let other = sharpe(&cov, &excess, &weights);
            prop_assert!(other <= best + 1e-9 * best.abs(), "{weights:?}: {other} > {best}");
        }

        // When the tangency portfolio is long-only, it is the long-only maximum too.
        if let Ok(unconstrained) = allocator.max_sharpe_allocation(&config) {
            if unconstrained.iter().all(|&w| w >= 0.0) {
                let long_only = allocator.long_only_max_sharpe_allocation(&config).unwrap();
                for (a, b) in unconstrained.iter().zip(&long_only) {
                    prop_assert!((a - b).abs() < 1e-9, "{unconstrained:?} vs {long_only:?}");
                }
            }
        }
    }

    #[test]
    fn max_diversification_satisfies_kkt(returns in returns()) {
        // As for max Sharpe with σ in place of the excess returns: the ratio's gradient,
        // proportional to σ·wᵀΣw − σᵀw·Σw, is zero on every held asset and no higher on the
        // others.
        prop_assume!(well_posed(&returns));
        let cov = covariance(&returns);
        let volatilities = cov.diagonal().map(f64::sqrt);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let weights = allocator.max_diversification_allocation().unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        prop_assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
        prop_assert!((w.sum() - 1.0).abs() < 1e-12);

        let marginal = &cov * &w;
        let spread = &volatilities * w.dot(&marginal);
        let concentration = marginal * volatilities.dot(&w);
        let gradient = &spread - &concentration;
        let tolerance = 1e-8 * (spread.amax() + concentration.amax());
        for i in 0..w.len() {
            prop_assert!(gradient[i] <= tolerance, "asset {i}: {weights:?}");
            prop_assert!(w[i] * gradient[i].abs() <= tolerance, "asset {i}: {weights:?}");
        }
    }

    #[test]
    fn max_diversification_is_highest_among_long_only(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let cov = covariance(&returns);
        let volatilities = cov.diagonal().map(f64::sqrt);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let ratio = |weights: &Weights| sharpe(&cov, &volatilities, weights);
        let best = ratio(&allocator.max_diversification_allocation().unwrap());
        // A weighted average volatility is at least the portfolio's volatility.
        prop_assert!(best >= 1.0 - 1e-12, "{best}");

        for weights in long_only_methods(&allocator) {
            let other = ratio(&weights);
            prop_assert!(other <= best * (1.0 + 1e-9), "{weights:?}: {other} > {best}");
        }
    }

    #[test]
    fn risk_parity_and_max_diversification_agree_on_singular_covariances(
        returns in (3usize..=6).prop_flat_map(|assets| {
            prop::collection::vec(-0.05f64..0.05, 2 * assets)
                .prop_map(move |values| DMatrix::from_vec(2, assets, values))
        }),
    ) {
        // With two periods, the covariance has rank 1. Both methods have no answer exactly
        // when some long-only portfolio has zero variance.
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let risk_parity = allocator.risk_parity_allocation();
        let max_diversification = allocator.max_diversification_allocation();
        prop_assert_eq!(
            risk_parity.as_ref().err(),
            max_diversification.as_ref().err(),
            "{:?} vs {:?}",
            risk_parity,
            max_diversification
        );
    }

    #[test]
    fn unit_free_methods_ignore_each_asset_scale(
        returns in returns(),
        asset in 0usize..6,
        exponent in -8.0f64..8.0,
    ) {
        // These objectives don't change when one asset's returns are rescaled by k, so
        // neither does the optimum, measured in that asset's new units: its weight is divided
        // by k, and the weights are renormalized.
        prop_assume!(well_posed(&returns));
        let (asset, k) = (asset % returns.ncols(), 10f64.powf(exponent));
        let mut scaled = returns.clone();
        scaled.column_mut(asset).scale_mut(k);
        let base = unit_free_methods(&returns);
        let rescaled = unit_free_methods(&scaled);

        for ((name, base), (_, weights)) in base.iter().zip(&rescaled) {
            let mut expected = DVector::from_column_slice(base.as_slice());
            expected[asset] /= k;
            let expected = &expected / expected.sum();
            for (w, e) in weights.iter().zip(&expected) {
                prop_assert!(
                    (w - e).abs() <= 1e-8 * w.max(*e) + 1e-14,
                    "{name}, asset {asset} scaled by {k}: {weights:?} vs {expected}"
                );
            }
        }
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

    #[test]
    fn max_sharpe_ignores_column_order_and_joint_scale(
        returns in returns(),
        margin in 1e-4f64..0.01,
        exponent in -3i32..=3,
        custom in prop::collection::vec(-0.01f64..0.02, 6),
        shrink in any::<bool>(),
    ) {
        // Reordering the assets, or scaling returns, expected returns and the risk-free rate
        // together, must not change the tangency weights.
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let covariance = CovarianceConfig::default().with_shrinkage(if shrink { 0.3 } else { 0.0 });
        let mu = DVector::from_column_slice(&custom[..n]);
        let allocator = PortfolioAllocator::from_returns(returns.clone()).unwrap();

        // A rate below the minimum variance return (under the same covariance) has a maximum.
        let min_variance = allocator.min_variance_allocation_with_config(&covariance).unwrap();
        let rate = mu.dot(&DVector::from_column_slice(min_variance.as_slice())) - margin;
        let config = |mu: DVector<f64>, rate: f64| {
            MaxSharpeConfig::default()
                .with_risk_free_rate(rate)
                .with_expected_returns(mu)
                .with_covariance(covariance.clone())
        };
        let base = allocator.max_sharpe_allocation(&config(mu.clone(), rate)).unwrap();

        let reversed = DMatrix::from_fn(returns.nrows(), n, |t, j| returns[(t, n - 1 - j)]);
        let mu_reversed = DVector::from_fn(n, |j, _| mu[n - 1 - j]);
        let flipped = PortfolioAllocator::from_returns(reversed)
            .unwrap()
            .max_sharpe_allocation(&config(mu_reversed, rate))
            .unwrap();

        let s = 2f64.powi(exponent);
        let scaled = PortfolioAllocator::from_returns(returns * s)
            .unwrap()
            .max_sharpe_allocation(&config(mu.clone() * s, rate * s))
            .unwrap();

        let tolerance = 1e-9 * (1.0 + base.iter().fold(0.0f64, |m, w| m.max(w.abs())));
        for j in 0..n {
            prop_assert!((base[j] - flipped[n - 1 - j]).abs() < tolerance, "{base:?} vs {flipped:?}");
            prop_assert!((base[j] - scaled[j]).abs() < tolerance, "{base:?} vs {scaled:?}");
        }
    }
    #[test]
    fn bounded_long_only_weights_satisfy_kkt(
        returns in returns(),
        fractions in bounds(),
        gamma in prop::option::of(1.0f64..500.0),
        shrinkage in prop::option::of(0.0f64..1.0),
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let (lower, upper) = weight_bounds(&fractions, n);
        let bounds = WeightBounds::per_asset(lower.clone(), upper.clone());
        let mu = returns.row_mean().transpose();
        let mut config = CovarianceConfig::default();
        let mut cov = covariance(&returns);
        if let Some(lambda) = shrinkage {
            config = config.with_shrinkage(lambda);
            let average = cov.trace() / n as f64;
            cov = cov * (1.0 - lambda) + DMatrix::identity(n, n) * (lambda * average);
        }
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();

        // Min variance has gradient Σw, and risk aversion γΣw − μ.
        let weights = match gamma {
            None => allocator.long_only_min_variance_allocation_with_bounds(&config, &bounds),
            Some(gamma) => allocator.long_only_mean_variance_allocation_with_bounds(
                &MeanVarianceConfig::risk_aversion(gamma).with_covariance(config),
                &bounds,
            ),
        }
        .unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        let (gradient, linear) = match gamma {
            None => (&cov * &w, 0.0),
            Some(gamma) => (&cov * &w * gamma - &mu, mu.amax()),
        };
        let tolerance = 1e-8 * gradient.amax().max(linear);
        assert_bounded_optimum(&w, &gradient, (&lower, &upper), tolerance)?;
    }

    #[test]
    fn bounded_target_returns_satisfy_kkt(
        returns in returns(),
        fractions in bounds(),
        fraction in 0.0f64..=1.0,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let (lower, upper) = weight_bounds(&fractions, n);
        let bounds = WeightBounds::per_asset(lower.clone(), upper.clone());
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();

        // An unreachable target reports the reachable range.
        let Err(CutupError::TargetReturnOutOfRange { lowest, highest, .. }) = allocator
            .long_only_mean_variance_allocation_with_bounds(
                &MeanVarianceConfig::target_return(1.0),
                &bounds,
            )
        else {
            panic!("a target return of 1 is out of range");
        };
        prop_assert!(lowest <= highest);
        let target = if fraction == 1.0 { highest } else { lowest + fraction * (highest - lowest) };
        let weights = allocator
            .long_only_mean_variance_allocation_with_bounds(
                &MeanVarianceConfig::target_return(target),
                &bounds,
            )
            .unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        prop_assert!((w.dot(&mu) - target).abs() < 1e-12 * mu.amax(), "{weights:?}");

        // The marginal variance Σw is λ₁ + λ₂μᵢ on every asset strictly inside its bounds, at
        // least that on assets at their floor, and at most on assets at their cap. Fitting λ
        // needs two such assets with expected returns that differ by more than rounding.
        for i in 0..n {
            prop_assert!(w[i] >= lower[i] - 1e-15 && w[i] <= upper[i] + 1e-15, "asset {i}: {w:?}");
        }
        prop_assert!((w.sum() - 1.0).abs() < 1e-12, "sums to {}", w.sum());
        let marginal = &cov * &w;
        let inside: Vec<usize> = (0..n)
            .filter(|&i| w[i] > lower[i] + 1e-9 && w[i] < upper[i] - 1e-9)
            .collect();
        let spread = inside.iter().map(|&i| mu[i]).fold(f64::NEG_INFINITY, f64::max)
            - inside.iter().map(|&i| mu[i]).fold(f64::INFINITY, f64::min);
        prop_assume!(inside.len() >= 2 && spread > 1e-6);
        let basis = DMatrix::from_fn(inside.len(), 2, |k, j| if j == 0 { 1.0 } else { mu[inside[k]] / mu.amax() });
        let lambda = basis
            .svd(true, true)
            .solve(&DVector::from_fn(inside.len(), |k, _| marginal[inside[k]]), 1e-12)
            .unwrap();
        let tolerance = 1e-8 * marginal.amax();
        for i in 0..n {
            let residual = marginal[i] - lambda[0] - lambda[1] * mu[i] / mu.amax();
            if w[i] > lower[i] + 1e-9 {
                prop_assert!(residual <= tolerance, "asset {i} could shrink: {weights:?}");
            }
            if w[i] < upper[i] - 1e-9 {
                prop_assert!(residual >= -tolerance, "asset {i} could grow: {weights:?}");
            }
        }
    }

    #[test]
    fn tightening_a_cap_never_lowers_variance(
        returns in returns(),
        fractions in bounds(),
        asset in 0usize..6,
        squeeze in 0.0f64..1.0,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let asset = asset % n;
        let (lower, upper) = weight_bounds(&fractions, n);
        let cov = covariance(&returns);
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let config = CovarianceConfig::default();
        let solve = |upper: Vec<f64>| {
            allocator.long_only_min_variance_allocation_with_bounds(
                &config,
                &WeightBounds::per_asset(lower.clone(), upper),
            )
        };
        let loose = solve(upper.clone()).unwrap();

        // Lower the asset's cap below its weight, toward its floor, but no further than the
        // other caps allow, so the caps can still sum to 1. That limit is at most the weight,
        // since the other weights are within their caps.
        let others: f64 = (0..n).filter(|&j| j != asset).map(|j| upper[j]).sum();
        let mut tight = upper.clone();
        tight[asset] = (lower[asset] + squeeze * (loose[asset] - lower[asset]).max(0.0))
            .max(1.0 - others);
        let tight = solve(tight).unwrap();
        let (v_loose, v_tight) = (variance(&cov, &loose), variance(&cov, &tight));
        prop_assert!(v_tight >= v_loose * (1.0 - 1e-12), "{v_tight} < {v_loose}");
    }

    #[test]
    fn bounds_that_never_bind_change_nothing(returns in returns(), gamma in 1.0f64..500.0) {
        prop_assume!(well_posed(&returns));
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let loose = WeightBounds::uniform(0.0, 1.0);
        let config = CovarianceConfig::default();
        prop_assert_eq!(
            allocator.long_only_min_variance_allocation_with_bounds(&config, &loose).unwrap(),
            allocator.long_only_min_variance_allocation_with_config(&config).unwrap()
        );
        let config = MeanVarianceConfig::risk_aversion(gamma);
        prop_assert_eq!(
            allocator.long_only_mean_variance_allocation_with_bounds(&config, &loose).unwrap(),
            allocator.long_only_mean_variance_allocation(&config).unwrap()
        );
    }

    #[test]
    fn method_values_match_the_methods_they_wrap(
        returns in returns(),
        shrinkage in prop::option::of(0.0f64..1.0),
        fractions in bounds(),
        gamma in 1.0f64..500.0,
        target in -0.01f64..0.01,
        risk_free_rate in -0.01f64..0.01,
        budgets in prop::collection::vec(0.1f64..10.0, 6),
        linkage in 0usize..4,
        clusters in prop::option::of(1usize..=6),
    ) {
        // No `well_posed` filter: errors must match too.
        let n = returns.ncols();
        let allocator = PortfolioAllocator::from_returns(returns).unwrap();
        let covariance = match shrinkage {
            Some(lambda) => CovarianceConfig::default().with_shrinkage(lambda),
            None => CovarianceConfig::default(),
        };
        let (lower, upper) = weight_bounds(&fractions, n);
        let bounds = WeightBounds::per_asset(lower, upper);
        let mu: Vec<f64> = (0..n).map(|i| target * (i as f64 - 1.0)).collect();
        let linkage = LINKAGES[linkage];
        let budgets = budgets[..n].to_vec();

        prop_assert_eq!(allocator.allocate(&EqualWeight), Ok(allocator.ew_allocation()));
        prop_assert_eq!(
            allocator.allocate(&InverseVolatility::default()),
            allocator.inverse_volatility_allocation()
        );
        prop_assert_eq!(
            allocator.allocate(&InverseVariance::default()),
            allocator.inverse_variance_allocation()
        );

        let min_variance = MinVariance::default().covariance(covariance.clone());
        prop_assert_eq!(
            allocator.allocate(&min_variance),
            allocator.long_only_min_variance_allocation_with_config(&covariance)
        );
        prop_assert_eq!(
            allocator.allocate(&min_variance.clone().bounds(bounds.clone())),
            allocator.long_only_min_variance_allocation_with_bounds(&covariance, &bounds)
        );
        prop_assert_eq!(
            allocator.allocate(&min_variance.allow_short()),
            allocator.min_variance_allocation_with_config(&covariance)
        );

        let risk_aversion = MeanVariance::default().risk_aversion(gamma).covariance(covariance.clone());
        let config = MeanVarianceConfig::risk_aversion(gamma).with_covariance(covariance.clone());
        prop_assert_eq!(
            allocator.allocate(&risk_aversion),
            allocator.long_only_mean_variance_allocation(&config)
        );
        prop_assert_eq!(
            allocator.allocate(&risk_aversion.clone().bounds(bounds.clone())),
            allocator.long_only_mean_variance_allocation_with_bounds(&config, &bounds)
        );
        prop_assert_eq!(
            allocator.allocate(&risk_aversion.allow_short()),
            allocator.mean_variance_allocation(&config)
        );
        let target_return = MeanVariance::default()
            .target_return(target)
            .expected_returns(mu.clone())
            .covariance(covariance.clone());
        let config = MeanVarianceConfig::target_return(target)
            .with_expected_returns(DVector::from_vec(mu.clone()))
            .with_covariance(covariance.clone());
        prop_assert_eq!(
            allocator.allocate(&target_return.clone().bounds(bounds.clone())),
            allocator.long_only_mean_variance_allocation_with_bounds(&config, &bounds)
        );
        prop_assert_eq!(
            allocator.allocate(&target_return.allow_short()),
            allocator.mean_variance_allocation(&config)
        );
        prop_assert_eq!(
            allocator.allocate(&MeanVariance::default()),
            allocator.long_only_mean_variance_allocation(&MeanVarianceConfig::risk_aversion(1.0))
        );

        let max_sharpe = MaxSharpe::default()
            .risk_free_rate(risk_free_rate)
            .expected_returns(mu.clone())
            .covariance(covariance.clone());
        let config = MaxSharpeConfig::default()
            .with_risk_free_rate(risk_free_rate)
            .with_expected_returns(DVector::from_vec(mu))
            .with_covariance(covariance.clone());
        prop_assert_eq!(
            allocator.allocate(&max_sharpe),
            allocator.long_only_max_sharpe_allocation(&config)
        );
        prop_assert_eq!(
            allocator.allocate(&max_sharpe.allow_short()),
            allocator.max_sharpe_allocation(&config)
        );

        prop_assert_eq!(
            allocator.allocate(&MaxDiversification::default().covariance(covariance.clone())),
            allocator.max_diversification_allocation_with_config(&covariance)
        );
        prop_assert_eq!(
            allocator.allocate(
                &RiskParity::default().risk_budgets(budgets.clone()).covariance(covariance.clone())
            ),
            allocator.risk_parity_allocation_with_config(
                &RiskParityConfig::default()
                    .with_risk_budgets(DVector::from_vec(budgets))
                    .with_covariance(covariance)
            )
        );

        prop_assert_eq!(
            allocator.allocate(&Hrp::default().linkage(linkage)),
            allocator.hrp_allocation_with_config(&HrpConfig::default().with_linkage(linkage))
        );
        let mut herc = Herc::default().linkage(linkage);
        let mut config = HercConfig::default().with_linkage(linkage);
        if let Some(k) = clusters {
            herc = herc.clusters(k);
            config = config.with_clusters(k);
        }
        prop_assert_eq!(
            allocator.allocate(&herc),
            allocator.herc_allocation_with_config(&config)
        );
    }
}
