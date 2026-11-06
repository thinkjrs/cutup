//! Property tests: invariants every allocation method must satisfy on random returns.
//!
//! They run natively only, since proptest needs other features and a getrandom backend to build
//! for wasm.

#![cfg(not(target_arch = "wasm32"))]

use cutup::{
    Allocation, BlackLitterman, CovarianceConfig, CutupError, EqualWeight, Herc, Hrp,
    InverseVariance, InverseVolatility, Linkage, MaxDiversification, MaxSharpe, MeanVariance,
    MinCvar, MinVariance, Returns, RiskParity, ShrinkageTarget, View, WeightBounds, Weights,
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

/// The `Returns` for a matrix of returns, through the flat row-major constructor.
fn returns_via_flat(returns: DMatrix<f64>) -> Result<Returns, CutupError> {
    Returns::new_flat(returns.transpose().as_slice(), returns.ncols())
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

/// This file's Ledoit-Wolf estimate, written out from scikit-learn's `ledoit_wolf_shrinkage`
/// and from Ledoit and Wolf's `covCor.m`, as differences of moments where cutup sums squares,
/// applied to the ddof=1 sample covariance as cutup applies it.
fn ledoit_wolf(returns: &DMatrix<f64>, target: ShrinkageTarget) -> DMatrix<f64> {
    let (t, n) = (returns.nrows() as f64, returns.ncols());
    let mean = returns.row_mean();
    let x = DMatrix::from_fn(returns.nrows(), n, |s, j| returns[(s, j)] - mean[j]);
    let sample = x.transpose() * &x / t;
    let y = x.map(|v| v * v);
    let pi = y.transpose() * &y / t - sample.map(|v| v * v);
    let cov = covariance(returns);
    let identity = DMatrix::<f64>::identity(n, n);
    let constant_correlation = |cov: &DMatrix<f64>| {
        let sd = cov.diagonal().map(f64::sqrt);
        let correlation = DMatrix::from_fn(n, n, |i, j| cov[(i, j)] / (sd[i] * sd[j]));
        let r_bar = (correlation.sum() - n as f64) / (n * (n - 1)) as f64;
        let prior = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                cov[(i, i)]
            } else {
                r_bar * sd[i] * sd[j]
            }
        });
        (prior, r_bar, sd)
    };
    let (intensity, prior) = match target {
        ShrinkageTarget::ConstantVariance => {
            let mu = sample.trace() / n as f64;
            let beta = pi.sum() / (n as f64 * t);
            let delta = (&sample - mu * &identity).norm_squared() / n as f64;
            let intensity = if delta == 0.0 {
                0.0
            } else {
                beta.min(delta) / delta
            };
            (intensity, cov.trace() / n as f64 * identity)
        }
        _ => {
            let (prior, r_bar, sd) = constant_correlation(&sample);
            let cubes = x.map(|v| v * v * v);
            let theta = DMatrix::from_fn(n, n, |i, j| {
                if i == j {
                    0.0
                } else {
                    (cubes.column(i).dot(&x.column(j))) / t - sample[(i, i)] * sample[(i, j)]
                }
            });
            let ratios = DMatrix::from_fn(n, n, |i, j| sd[j] / sd[i]);
            let rho = pi.trace() + r_bar * ratios.component_mul(&theta).sum();
            let gamma = (&sample - prior).norm_squared();
            let kappa = (pi.sum() - rho) / gamma;
            let intensity = if gamma == 0.0 {
                0.0
            } else {
                (kappa / t).clamp(0.0, 1.0)
            };
            (intensity, constant_correlation(&cov).0)
        }
    };
    intensity * prior + (1.0 - intensity) * cov
}

/// This file's exponentially weighted covariance, pandas' `ewm(halflife=h).cov()`: weights
/// normalized to sum to 1 and the correction `1 / (1 − Σwₜ²)`.
fn exponentially_weighted(returns: &DMatrix<f64>, half_life: f64) -> DMatrix<f64> {
    let (t, n) = returns.shape();
    let raw: Vec<f64> = (0..t)
        .map(|s| 0.5f64.powf((t - 1 - s) as f64 / half_life))
        .collect();
    let total: f64 = raw.iter().sum();
    let w: Vec<f64> = raw.iter().map(|x| x / total).collect();
    let mean: Vec<f64> = (0..n)
        .map(|i| (0..t).map(|s| w[s] * returns[(s, i)]).sum())
        .collect();
    let correction = 1.0 - w.iter().map(|x| x * x).sum::<f64>();
    DMatrix::from_fn(n, n, |i, j| {
        (0..t)
            .map(|s| w[s] * (returns[(s, i)] - mean[i]) * (returns[(s, j)] - mean[j]))
            .sum::<f64>()
            / correction
    })
}

/// This file's semicovariance below each mean, over `T − 1`, or below `threshold`, over `T`.
fn semicovariance(returns: &DMatrix<f64>, threshold: Option<f64>) -> DMatrix<f64> {
    let (t, n) = returns.shape();
    let mean = returns.row_mean();
    let shortfall = |s: usize, i: usize| (returns[(s, i)] - threshold.unwrap_or(mean[i])).min(0.0);
    let divisor = if threshold.is_some() { t } else { t - 1 } as f64;
    DMatrix::from_fn(n, n, |i, j| {
        (0..t)
            .map(|s| shortfall(s, i) * shortfall(s, j))
            .sum::<f64>()
            / divisor
    })
}

fn variance(cov: &DMatrix<f64>, weights: &Weights) -> f64 {
    let w = DVector::from_column_slice(weights.as_slice());
    w.dot(&(cov * &w))
}

/// HRP with every linkage, by name.
fn hrp_methods(allocator: &Returns) -> Vec<(String, Weights)> {
    LINKAGES
        .iter()
        .map(|&linkage| {
            let hrp = Hrp::default().linkage(linkage);
            (
                format!("hrp ({linkage:?})"),
                allocator.allocate(&hrp).unwrap(),
            )
        })
        .collect()
}

/// HERC with every linkage, by name.
fn herc_methods(allocator: &Returns) -> Vec<(String, Weights)> {
    LINKAGES
        .iter()
        .map(|&linkage| {
            let herc = Herc::default().linkage(linkage);
            (
                format!("herc ({linkage:?})"),
                allocator.allocate(&herc).unwrap(),
            )
        })
        .collect()
}

/// A risk-free rate below every generated expected return, so long-only max Sharpe exists.
fn low_rate() -> MaxSharpe {
    MaxSharpe::default().risk_free_rate(-1.0)
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
    let allocator = returns_via_flat(returns.clone()).unwrap();
    let mean_absolute = returns.abs().row_mean().transpose();
    let zero_rate = MaxSharpe::default().expected_returns(mean_absolute.as_slice());
    vec![
        (
            "max diversification".into(),
            allocator.allocate(&MaxDiversification::default()).unwrap(),
        ),
        (
            "risk parity".into(),
            allocator.allocate(&RiskParity::default()).unwrap(),
        ),
        (
            "long-only max Sharpe".into(),
            allocator.allocate(&zero_rate).unwrap(),
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
fn long_only_methods(allocator: &Returns) -> Vec<Weights> {
    let mut methods = vec![
        allocator.allocate(&EqualWeight).unwrap(),
        allocator.allocate(&InverseVolatility).unwrap(),
        allocator.allocate(&InverseVariance).unwrap(),
        allocator.allocate(&MinVariance::default()).unwrap(),
        allocator
            .allocate(&MeanVariance::risk_aversion(50.0))
            .unwrap(),
        allocator.allocate(&low_rate()).unwrap(),
        allocator.allocate(&MaxDiversification::default()).unwrap(),
        allocator.allocate(&RiskParity::default()).unwrap(),
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
fn all_methods(allocator: &Returns, mean: f64) -> Vec<(String, Weights)> {
    let mut out = vec![
        (
            "equal weight".to_string(),
            allocator.allocate(&EqualWeight).unwrap(),
        ),
        (
            "inverse volatility".to_string(),
            allocator.allocate(&InverseVolatility).unwrap(),
        ),
        (
            "inverse variance".to_string(),
            allocator.allocate(&InverseVariance).unwrap(),
        ),
        (
            "min variance".to_string(),
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
        ),
        (
            "mean-variance (risk aversion)".to_string(),
            allocator
                .allocate(&MeanVariance::risk_aversion(50.0).allow_short())
                .unwrap(),
        ),
        (
            "mean-variance (target return)".to_string(),
            allocator
                .allocate(&MeanVariance::target_return(mean).allow_short())
                .unwrap(),
        ),
    ];
    out.push((
        "long-only min variance".to_string(),
        allocator.allocate(&MinVariance::default()).unwrap(),
    ));
    out.push((
        "long-only mean-variance".to_string(),
        allocator
            .allocate(&MeanVariance::risk_aversion(50.0))
            .unwrap(),
    ));
    out.push((
        "max diversification".to_string(),
        allocator.allocate(&MaxDiversification::default()).unwrap(),
    ));
    out.push((
        "long-only max Sharpe".to_string(),
        allocator.allocate(&low_rate()).unwrap(),
    ));
    out.push((
        "risk parity".to_string(),
        allocator.allocate(&RiskParity::default()).unwrap(),
    ));
    out.extend(hrp_methods(allocator));
    out.extend(herc_methods(allocator));
    out
}

/// Methods that depend only on Σ, and on neither column order nor return scale. HRP and
/// HERC are tested separately: like the original algorithm, HRP can depend on column order,
/// and both cluster on correlations, which only an exact rescaling leaves bit-identical.
fn risk_only_methods(allocator: &Returns) -> Vec<(String, Weights)> {
    let shrunk = CovarianceConfig::default().shrinkage(0.3);
    vec![
        (
            "inverse volatility".to_string(),
            allocator.allocate(&InverseVolatility).unwrap(),
        ),
        (
            "inverse variance".to_string(),
            allocator.allocate(&InverseVariance).unwrap(),
        ),
        (
            "min variance".to_string(),
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
        ),
        (
            "min variance (shrinkage)".to_string(),
            allocator
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(shrunk.clone()),
                )
                .unwrap(),
        ),
        (
            "long-only min variance".to_string(),
            allocator.allocate(&MinVariance::default()).unwrap(),
        ),
        (
            "long-only min variance (shrinkage)".to_string(),
            allocator
                .allocate(&MinVariance::default().covariance(shrunk.clone()))
                .unwrap(),
        ),
        (
            "max diversification".to_string(),
            allocator.allocate(&MaxDiversification::default()).unwrap(),
        ),
        (
            "max diversification (shrinkage)".to_string(),
            allocator
                .allocate(&MaxDiversification::default().covariance(shrunk.clone()))
                .unwrap(),
        ),
        (
            "risk parity".to_string(),
            allocator.allocate(&RiskParity::default()).unwrap(),
        ),
        (
            "risk parity (shrinkage)".to_string(),
            allocator
                .allocate(&RiskParity::default().covariance(shrunk))
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

/// Random current weights for `n` assets: positive fractions, normalized to sum to 1.
fn current_weights(fractions: &[f64], n: usize) -> Vec<f64> {
    let total: f64 = fractions[..n].iter().sum();
    fractions[..n].iter().map(|f| f / total).collect()
}

fn turnover(weights: &Weights, current: &[f64]) -> f64 {
    weights
        .iter()
        .zip(current)
        .map(|(w, c)| (w - c).abs())
        .sum()
}

/// Checks that `w` minimizes a smooth objective with gradient `gradient` plus `Σᵢ costsᵢ|wᵢ −
/// currentᵢ|` within the bounds: no move of weight from an asset that can shrink to one that
/// can grow lowers it. Going up, asset i's marginal cost is its gradient plus its cost if it's
/// at or above its current weight, minus it if below; going down, the reverse. At its current
/// weight both directions cost.
fn assert_kinked_optimum(
    w: &DVector<f64>,
    gradient: &DVector<f64>,
    (current, costs): (&[f64], &[f64]),
    (lower, upper): (&[f64], &[f64]),
    tolerance: f64,
) -> Result<(), TestCaseError> {
    let n = w.len();
    prop_assert!((w.sum() - 1.0).abs() < 1e-12, "sums to {}", w.sum());
    let at = |i: usize| (w[i] - current[i]).abs() <= 1e-12;
    let up = |i: usize| {
        gradient[i]
            + if at(i) || w[i] > current[i] {
                costs[i]
            } else {
                -costs[i]
            }
    };
    let down = |i: usize| {
        -gradient[i]
            + if at(i) || w[i] < current[i] {
                costs[i]
            } else {
                -costs[i]
            }
    };
    for j in (0..n).filter(|&j| w[j] > lower[j] + 1e-9) {
        for i in (0..n).filter(|&i| i != j && w[i] < upper[i] - 1e-9) {
            prop_assert!(
                up(i) + down(j) >= -tolerance,
                "moving weight from {j} to {i} lowers the objective by {}: {w:?}",
                -(up(i) + down(j))
            );
        }
    }
    Ok(())
}

proptest! {
    #[test]
    fn weights_are_finite_and_sum_to_one(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let mean = returns.mean() * 1.5;
        let allocator = returns_via_flat(returns).unwrap();
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
    fn cleaning_drops_small_weights_and_keeps_the_ratios_of_the_rest(
        returns in returns(),
        fraction in prop_oneof![Just(0.0), 0.0f64..1.0],
    ) {
        prop_assume!(well_posed(&returns));
        let mean = returns.mean() * 1.5;
        let allocator = returns_via_flat(returns).unwrap();
        for (name, weights) in all_methods(&allocator, mean) {
            let largest = weights.iter().fold(0.0f64, |m, w| m.max(w.abs()));
            let cutoff = fraction * largest;
            let kept: f64 = weights.iter().filter(|w| w.abs() >= cutoff).sum();
            let cleaned = match weights.clean(cutoff) {
                Ok(cleaned) => cleaned,
                // Only when the shorts it keeps outweigh the longs.
                Err(CutupError::InvalidConfig { field: "cutoff", .. }) => {
                    prop_assert!(kept <= 0.0, "{name}: {weights:?} keeps {kept}");
                    continue;
                }
                Err(other) => return Err(TestCaseError::fail(format!("{name}: {other:?}"))),
            };
            prop_assert_eq!(cleaned.labels(), weights.labels());
            let total: f64 = cleaned.iter().sum();
            prop_assert!((total - 1.0).abs() <= 1e-12 * largest.max(1.0) / kept, "{name} sums to {total}");
            for (c, w) in cleaned.iter().zip(&weights) {
                if w.abs() < cutoff {
                    prop_assert_eq!(*c, 0.0);
                } else {
                    // Each kept weight is rescaled by the same factor.
                    prop_assert!((c * kept - w).abs() <= 1e-15 * largest, "{name}: {c} vs {w}");
                }
            }
            if weights.iter().all(|&w| w >= 0.0) {
                // Long-only, the kept weights only grow, so cleaning again changes nothing more.
                prop_assert!(cleaned.iter().all(|&c| c == 0.0 || c >= cutoff), "{name}: {cleaned:?}");
                let again = cleaned.clean(cutoff).unwrap();
                for (a, c) in again.iter().zip(&cleaned) {
                    prop_assert!((a - c).abs() <= 1e-15, "{name}: {again:?} vs {cleaned:?}");
                }
            }
        }
    }

    #[test]
    fn stats_match_their_closed_forms(returns in returns(), rate in -0.002f64..0.002) {
        prop_assume!(well_posed(&returns));
        let mean = returns.mean() * 1.5;
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let volatilities = cov.diagonal().map(f64::sqrt);
        let allocator = returns_via_flat(returns).unwrap();
        for (name, weights) in all_methods(&allocator, mean) {
            let stats = allocator.stats(&weights).unwrap();
            let w = DVector::from_column_slice(weights.as_slice());
            // Long-short weights can be in the thousands, so the return is checked against the
            // size of the positions, and everything else relative to itself.
            let gross: f64 = w.iter().map(|w| w.abs()).sum();
            let expected = mu.dot(&w);
            prop_assert!((stats.expected_return() - expected).abs() <= 1e-12 * gross * mu.amax(), "{name}");
            let volatility = variance(&cov, &weights).sqrt();
            prop_assert!((stats.volatility() - volatility).abs() <= 1e-9 * volatility, "{name}: {} vs {volatility}", stats.volatility());
            let ratio = stats.sharpe_ratio(rate).unwrap();
            let exact = (expected - rate) / volatility;
            prop_assert!((ratio - exact).abs() <= 1e-9 * exact.abs() + 1e-12 * gross * mu.amax() / volatility, "{name}: {ratio} vs {exact}");
            let ratio = stats.diversification_ratio().unwrap();
            let exact = volatilities.dot(&w) / volatility;
            prop_assert!((ratio - exact).abs() <= 1e-9 * gross * volatilities.amax() / volatility, "{name}: {ratio} vs {exact}");
            let shares = stats.risk_contributions().unwrap();
            let marginal = &cov * &w;
            for (i, share) in shares.iter().enumerate() {
                let exact = w[i] * marginal[i] / (volatility * volatility);
                prop_assert!((share - exact).abs() <= 1e-9 * gross * gross, "{name}: {shares:?}");
            }
            let total: f64 = shares.iter().sum();
            prop_assert!((total - 1.0).abs() <= 1e-9 * gross * gross, "{name}: {total}");
            for beta in [0.5, 0.9] {
                prop_assert_eq!(stats.cvar(beta), allocator.cvar(&weights, beta));
            }
        }
    }

    #[test]
    fn stats_favor_the_methods_that_optimize_them(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns).unwrap();
        let stats = |method: &dyn Allocation| allocator.stats(&allocator.allocate(method).unwrap()).unwrap();
        let lowest = stats(&MinVariance::default()).volatility();
        let highest = stats(&MaxDiversification::default()).diversification_ratio().unwrap();
        let best = stats(&low_rate()).sharpe_ratio(-1.0).unwrap();
        let least = stats(&MinCvar::default()).cvar(0.95).unwrap();
        for weights in long_only_methods(&allocator) {
            let other = allocator.stats(&weights).unwrap();
            prop_assert!(lowest <= other.volatility() * (1.0 + 1e-9), "{weights:?}");
            prop_assert!(other.diversification_ratio().unwrap() <= highest * (1.0 + 1e-9), "{weights:?}");
            prop_assert!(other.sharpe_ratio(-1.0).unwrap() <= best * (1.0 + 1e-9), "{weights:?}");
            prop_assert!(least <= other.cvar(0.95).unwrap() + 1e-12 * least.abs().max(1e-3), "{weights:?}");
        }
        let n = allocator.labels().len() as f64;
        for share in stats(&RiskParity::default()).risk_contributions().unwrap() {
            prop_assert!((share - 1.0 / n).abs() <= 1e-9, "{share}");
        }
    }

    #[test]
    fn stats_with_favor_the_methods_that_optimize_them(
        returns in returns(),
        choice in 0usize..3,
        lambda in 0.0f64..1.0,
        half_life in 1.0f64..30.0,
    ) {
        // Under the covariance they were given, min variance has the lowest volatility, max
        // diversification the highest diversification ratio, and risk parity equal shares.
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns).unwrap();
        let config = match choice {
            0 => CovarianceConfig::default().shrinkage(lambda),
            1 => CovarianceConfig::exponentially_weighted(half_life),
            _ => CovarianceConfig::semicovariance(),
        };
        let stats = |weights: &Weights| allocator.stats_with(weights, &config).unwrap();
        let allocate = |method: &dyn Allocation| allocator.allocate(method).unwrap();
        let min_variance = allocate(&MinVariance::default().covariance(config.clone()));
        let max_diversification = allocate(&MaxDiversification::default().covariance(config.clone()));
        let lowest = stats(&min_variance).volatility();
        prop_assume!(lowest > 0.0);
        let highest = stats(&max_diversification).diversification_ratio().unwrap();
        for weights in long_only_methods(&allocator) {
            let other = stats(&weights);
            prop_assert!(lowest <= other.volatility() * (1.0 + 1e-9), "{weights:?}");
            prop_assert!(other.diversification_ratio().unwrap() <= highest * (1.0 + 1e-9), "{weights:?}");
        }
        let n = allocator.labels().len() as f64;
        let risk_parity = allocate(&RiskParity::default().covariance(config.clone()));
        for share in stats(&risk_parity).risk_contributions().unwrap() {
            prop_assert!((share - 1.0 / n).abs() <= 1e-9, "{share}");
        }
    }

    #[test]
    fn stats_scale_with_the_returns(returns in returns(), scale in 0.1f64..10.0) {
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns.clone()).unwrap();
        let scaled = returns_via_flat(&returns * scale).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        let (a, b) = (allocator.stats(&weights).unwrap(), scaled.stats(&weights).unwrap());
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-12 * x.abs().max(y.abs());
        prop_assert!((b.expected_return() - a.expected_return() * scale).abs() <= 1e-12 * scale * returns.amax());
        prop_assert!(close(b.volatility(), a.volatility() * scale));
        prop_assert!((b.cvar(0.9).unwrap() - a.cvar(0.9).unwrap() * scale).abs() <= 1e-12 * scale * returns.amax());
        prop_assert!(close(b.diversification_ratio().unwrap(), a.diversification_ratio().unwrap()));
        let rate = 1e-4;
        let (x, y) = (a.sharpe_ratio(rate).unwrap(), b.sharpe_ratio(rate * scale).unwrap());
        prop_assert!((x - y).abs() <= 1e-10 * x.abs().max(1.0), "{x} vs {y}");
        for (x, y) in a.risk_contributions().unwrap().iter().zip(b.risk_contributions().unwrap()) {
            prop_assert!((x - y).abs() <= 1e-12, "{x} vs {y}");
        }
    }

    #[test]
    fn a_given_covariance_is_used_like_the_sample_one(returns in returns(), lambda in 0.0f64..1.0) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        // This file's covariance, equal to cutup's to rounding.
        let cov = covariance(&returns);
        let given = CovarianceConfig::matrix_flat(cov.transpose().as_slice(), n).unwrap();
        let allocator = returns_via_flat(returns).unwrap();
        for (adjust, sample) in [
            (given.clone(), CovarianceConfig::default()),
            (given.shrinkage(lambda), CovarianceConfig::default().shrinkage(lambda)),
        ] {
            let methods = |c: CovarianceConfig| -> Vec<Box<dyn Allocation>> {
                vec![
                    Box::new(MinVariance::default().covariance(c.clone())),
                    Box::new(MinVariance::default().allow_short().covariance(c.clone())),
                    Box::new(MaxDiversification::default().covariance(c.clone())),
                    Box::new(RiskParity::default().covariance(c)),
                ]
            };
            for (a, b) in methods(adjust.clone()).iter().zip(methods(sample)) {
                let (a, b) = (allocator.allocate(a.as_ref()).unwrap(), allocator.allocate(b.as_ref()).unwrap());
                for (x, y) in a.iter().zip(&b) {
                    prop_assert!((x - y).abs() <= 1e-9, "{a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn ledoit_wolf_matches_this_files_estimate(returns in returns(), exponent in -100i32..100) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let scaled = Returns::new_flat((&returns * 10f64.powi(exponent)).transpose().as_slice(), n)
            .unwrap();
        let allocator = returns_via_flat(returns.clone()).unwrap();
        for target in [ShrinkageTarget::ConstantVariance, ShrinkageTarget::ConstantCorrelation] {
            let mine = ledoit_wolf(&returns, target);
            let given = CovarianceConfig::matrix_flat(mine.transpose().as_slice(), n).unwrap();
            let estimated = CovarianceConfig::default().ledoit_wolf(target);
            let methods = |c: CovarianceConfig| -> Vec<Box<dyn Allocation>> {
                vec![
                    Box::new(MinVariance::default().covariance(c.clone())),
                    Box::new(MinVariance::default().allow_short().covariance(c.clone())),
                    Box::new(MaxDiversification::default().covariance(c.clone())),
                    Box::new(RiskParity::default().covariance(c)),
                ]
            };
            for (method, mine) in methods(estimated).iter().zip(methods(given)) {
                // Against this file's estimate, and at another scale, which changes nothing.
                let a = allocator.allocate(method.as_ref()).unwrap();
                for b in [
                    allocator.allocate(mine.as_ref()).unwrap(),
                    scaled.allocate(method.as_ref()).unwrap(),
                ] {
                    for (x, y) in a.iter().zip(&b) {
                        prop_assert!((x - y).abs() <= 1e-9, "{target:?}: {a:?} vs {b:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn computed_estimates_match_this_files(
        returns in returns(),
        half_life in 0.5f64..50.0,
        threshold in -0.02f64..0.02,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let allocator = returns_via_flat(returns.clone()).unwrap();
        for (config, mine) in [
            (
                CovarianceConfig::exponentially_weighted(half_life),
                exponentially_weighted(&returns, half_life),
            ),
            (
                CovarianceConfig::exponentially_weighted_decay(0.5f64.powf(1.0 / half_life)),
                exponentially_weighted(&returns, half_life),
            ),
            (CovarianceConfig::semicovariance(), semicovariance(&returns, None)),
            (
                CovarianceConfig::semicovariance_below(threshold),
                semicovariance(&returns, Some(threshold)),
            ),
        ] {
            // Positive semidefinite, and conditioned well enough for a 1e-9 comparison.
            let eigenvalues = mine.clone().symmetric_eigen().eigenvalues;
            prop_assert!(eigenvalues.min() >= -1e-12 * eigenvalues.max(), "{mine}");
            if mine.diagonal().min() < 1e-6 || eigenvalues.max() > 1e8 * eigenvalues.min() {
                continue;
            }
            let given = CovarianceConfig::matrix_flat(mine.transpose().as_slice(), n).unwrap();
            for (a, b) in [
                (MinVariance::default().covariance(config.clone()), MinVariance::default().covariance(given.clone())),
                (
                    MinVariance::default().allow_short().covariance(config.clone()),
                    MinVariance::default().allow_short().covariance(given),
                ),
            ] {
                let (a, b) = (allocator.allocate(&a).unwrap(), allocator.allocate(&b).unwrap());
                for (x, y) in a.iter().zip(&b) {
                    prop_assert!((x - y).abs() <= 1e-9, "{config:?}: {a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn products_of_factors_are_covariances(
        factors in prop::collection::vec(-1.0f64..1.0, 36),
        n in 1usize..=6,
        k in 1usize..=6,
        exponent in -150i32..150,
    ) {
        // BBᵀ is semidefinite, of rank k at most, at any scale.
        let b = DMatrix::from_fn(n, k, |i, j| factors[i * 6 + j] * 10f64.powi(exponent));
        let product = &b * b.transpose();
        let rows: Vec<Vec<f64>> = product.row_iter().map(|r| r.iter().copied().collect()).collect();
        prop_assert!(CovarianceConfig::matrix(&rows).is_ok(), "{product}");
        // Taking twice the largest variance off one leaves a portfolio with negative variance.
        let mut indefinite = rows.clone();
        let largest = product.diagonal().amax();
        prop_assume!(largest > 0.0);
        indefinite[0][0] -= 2.0 * largest;
        let result = CovarianceConfig::matrix(&indefinite);
        let negative = matches!(result, Err(CutupError::IndefiniteCovariance { eigenvalue }) if eigenvalue < 0.0);
        prop_assert!(negative, "{result:?}");
    }

    #[test]
    fn long_only_methods_never_short(returns in returns()) {
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns).unwrap();
        let mut long_only = vec![
            allocator.allocate(&EqualWeight).unwrap(),
            allocator.allocate(&InverseVolatility).unwrap(),
            allocator.allocate(&InverseVariance).unwrap(),
            allocator.allocate(&MinVariance::default()).unwrap(),
            allocator
                .allocate(&MeanVariance::risk_aversion(50.0))
                .unwrap(),
            allocator.allocate(&low_rate()).unwrap(),
            allocator.allocate(&MaxDiversification::default()).unwrap(),
            allocator.allocate(&RiskParity::default()).unwrap(),
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
        let allocator = returns_via_flat(returns).unwrap();
        let long_short = allocator.allocate(&MinVariance::default().allow_short()).unwrap();
        let minimum = variance(&cov, &long_short);
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
            config = config.shrinkage(lambda);
            let average = cov.trace() / n as f64;
            cov = cov * (1.0 - lambda) + DMatrix::identity(n, n) * (lambda * average);
        }
        let allocator = returns_via_flat(returns).unwrap();
        let weights = allocator.allocate(&MinVariance::default().covariance(config)).unwrap();
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
        let allocator = returns_via_flat(returns).unwrap();
        let unconstrained = allocator.allocate(&MinVariance::default().allow_short()).unwrap();
        let long_only = allocator.allocate(&MinVariance::default()).unwrap();
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
        let base = risk_only_methods(&returns_via_flat(returns).unwrap());
        let flipped = risk_only_methods(&returns_via_flat(reversed).unwrap());
        for ((name, a), (_, b)) in base.iter().zip(&flipped) {
            for j in 0..n {
                prop_assert!((a[j] - b[n - 1 - j]).abs() < 1e-9, "{name}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn return_scale_does_not_matter(returns in returns(), scale in 0.1f64..10.0) {
        prop_assume!(well_posed(&returns));
        let base = returns_via_flat(returns.clone()).unwrap();
        let scaled = returns_via_flat(returns * scale).unwrap();
        assert_same(&risk_only_methods(&base), &risk_only_methods(&scaled))?;
    }

    #[test]
    fn herc_ignores_column_order(returns in returns()) {
        // HERC splits where the tree branches, not at a position in the ordered list, so
        // unlike HRP it doesn't depend on the order of the columns.
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let reversed = DMatrix::from_fn(returns.nrows(), n, |t, j| returns[(t, n - 1 - j)]);
        let base = herc_methods(&returns_via_flat(returns).unwrap());
        let flipped = herc_methods(&returns_via_flat(reversed).unwrap());
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
            covariance_config = covariance_config.shrinkage(lambda);
        }
        let budgets = DVector::from_column_slice(&budgets[..n]);
        let risk_parity = RiskParity::default()
            .risk_budgets(budgets.as_slice())
            .covariance(covariance_config);
        let allocator = returns_via_flat(returns.clone()).unwrap();
        let weights = allocator.allocate(&risk_parity).unwrap();
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
        let base = returns_via_flat(returns.clone()).unwrap();
        let scaled = returns_via_flat(returns * 2f64.powi(exponent)).unwrap();
        assert_same(&hrp_methods(&base), &hrp_methods(&scaled))?;
        assert_same(&herc_methods(&base), &herc_methods(&scaled))?;
    }

    #[test]
    fn mean_variance_meets_its_objective(returns in returns(), gamma in 1.0f64..500.0) {
        prop_assume!(well_posed(&returns));
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let target = mu.mean() * 1.5;
        let allocator = returns_via_flat(returns).unwrap();

        let targeted = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();
        let w = DVector::from_column_slice(targeted.as_slice());
        prop_assert!((mu.dot(&w) - target).abs() < 1e-9 * (1.0 + target.abs()));

        // At the optimum of μᵀw − (γ/2)wᵀΣw with 1ᵀw = 1, μ − γΣw is a constant vector.
        let averse = allocator
            .allocate(&MeanVariance::risk_aversion(gamma).allow_short())
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
        let allocator = returns_via_flat(returns).unwrap();

        // For a risk aversion, the gradient g = μ − γΣw of the utility is the same on every
        // held asset and no higher on the others: no shift of weight between assets adds
        // utility.
        let averse = allocator
            .allocate(&MeanVariance::risk_aversion(gamma))
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
            .allocate(&MeanVariance::target_return(target))
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
        let allocator = returns_via_flat(returns).unwrap();
        for mean_variance in [
            MeanVariance::risk_aversion(gamma),
            MeanVariance::target_return(target),
        ] {
            let unconstrained = allocator.allocate(&mean_variance.clone().allow_short()).unwrap();
            let long_only = allocator.allocate(&mean_variance).unwrap();
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
        let allocator = returns_via_flat(returns).unwrap();
        let max_sharpe = MaxSharpe::default().risk_free_rate(rate);
        let weights = allocator.allocate(&max_sharpe).unwrap();
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
        let allocator = returns_via_flat(returns).unwrap();
        let max_sharpe = MaxSharpe::default().risk_free_rate(rate);
        let best = sharpe(&cov, &excess, &allocator.allocate(&max_sharpe).unwrap());

        for weights in long_only_methods(&allocator) {
            let other = sharpe(&cov, &excess, &weights);
            prop_assert!(other <= best + 1e-9 * best.abs(), "{weights:?}: {other} > {best}");
        }

        // When the tangency portfolio is long-only, it is the long-only maximum too.
        if let Ok(unconstrained) = allocator.allocate(&max_sharpe.clone().allow_short()) {
            if unconstrained.iter().all(|&w| w >= 0.0) {
                let long_only = allocator.allocate(&max_sharpe).unwrap();
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
        let allocator = returns_via_flat(returns).unwrap();
        let weights = allocator.allocate(&MaxDiversification::default()).unwrap();
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
        let allocator = returns_via_flat(returns).unwrap();
        let ratio = |weights: &Weights| sharpe(&cov, &volatilities, weights);
        let best = ratio(&allocator.allocate(&MaxDiversification::default()).unwrap());
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
        let allocator = returns_via_flat(returns).unwrap();
        let risk_parity = allocator.allocate(&RiskParity::default());
        let max_diversification = allocator.allocate(&MaxDiversification::default());
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
        let allocator = returns_via_flat(returns).unwrap();

        // The maximum exists for any risk-free rate below the minimum variance return.
        let min_variance = allocator.allocate(&MinVariance::default().allow_short()).unwrap();
        let risk_free_rate = mu.dot(&DVector::from_column_slice(min_variance.as_slice())) - margin;
        let sharpe = |weights: &Weights| {
            let w = DVector::from_column_slice(weights.as_slice());
            (mu.dot(&w) - risk_free_rate) / variance(&cov, weights).sqrt()
        };

        let max_sharpe = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let best = sharpe(&allocator.allocate(&max_sharpe.allow_short()).unwrap());
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
        let covariance = CovarianceConfig::default().shrinkage(if shrink { 0.3 } else { 0.0 });
        let mu = DVector::from_column_slice(&custom[..n]);
        let allocator = returns_via_flat(returns.clone()).unwrap();

        // A rate below the minimum variance return (under the same covariance) has a maximum.
        let long_short = MinVariance::default().allow_short().covariance(covariance.clone());
        let min_variance = allocator.allocate(&long_short).unwrap();
        let rate = mu.dot(&DVector::from_column_slice(min_variance.as_slice())) - margin;
        let max_sharpe = |mu: DVector<f64>, rate: f64| {
            MaxSharpe::default()
                .risk_free_rate(rate)
                .expected_returns(mu.as_slice())
                .covariance(covariance.clone())
        };
        let base = allocator.allocate(&max_sharpe(mu.clone(), rate).allow_short()).unwrap();

        let reversed = DMatrix::from_fn(returns.nrows(), n, |t, j| returns[(t, n - 1 - j)]);
        let mu_reversed = DVector::from_fn(n, |j, _| mu[n - 1 - j]);
        let flipped = returns_via_flat(reversed)
            .unwrap()
            .allocate(&max_sharpe(mu_reversed, rate).allow_short())
            .unwrap();

        let s = 2f64.powi(exponent);
        let scaled = returns_via_flat(returns * s)
            .unwrap()
            .allocate(&max_sharpe(mu * s, rate * s).allow_short())
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
            config = config.shrinkage(lambda);
            let average = cov.trace() / n as f64;
            cov = cov * (1.0 - lambda) + DMatrix::identity(n, n) * (lambda * average);
        }
        let allocator = returns_via_flat(returns).unwrap();

        // Min variance has gradient Σw, and risk aversion γΣw − μ.
        let weights = match gamma {
            None => allocator.allocate(&MinVariance::default().covariance(config).bounds(bounds)),
            Some(gamma) => allocator
                .allocate(&MeanVariance::risk_aversion(gamma).covariance(config).bounds(bounds)),
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
        let allocator = returns_via_flat(returns).unwrap();

        // An unreachable target reports the reachable range.
        let Err(CutupError::TargetReturnOutOfRange { lowest, highest, .. }) = allocator
            .allocate(&MeanVariance::target_return(1.0).bounds(bounds.clone()))
        else {
            panic!("a target return of 1 is out of range");
        };
        prop_assert!(lowest <= highest);
        let target = if fraction == 1.0 { highest } else { lowest + fraction * (highest - lowest) };
        let weights = allocator
            .allocate(&MeanVariance::target_return(target).bounds(bounds))
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
        let allocator = returns_via_flat(returns).unwrap();
        let solve = |upper: Vec<f64>| {
            allocator.allocate(&MinVariance::default().bounds(WeightBounds::per_asset(lower.clone(), upper)))
        };
        let loose = solve(upper.clone()).unwrap();

        // Lower the asset's cap below its weight, toward its floor, but no further than the
        // other caps allow, so the caps can still sum to 1. That limit is at most the weight,
        // since the other weights are within their caps.
        let others: f64 = (0..n).filter(|&j| j != asset).map(|j| upper[j]).sum();
        let mut tight = upper;
        tight[asset] = (lower[asset] + squeeze * (loose[asset] - lower[asset]).max(0.0))
            .max(1.0 - others);
        let tight = solve(tight).unwrap();
        let (v_loose, v_tight) = (variance(&cov, &loose), variance(&cov, &tight));
        prop_assert!(v_tight >= v_loose * (1.0 - 1e-12), "{v_tight} < {v_loose}");
    }

    #[test]
    fn bounds_that_never_bind_change_nothing(returns in returns(), gamma in 1.0f64..500.0) {
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns).unwrap();
        let loose = WeightBounds::uniform(0.0, 1.0);
        prop_assert_eq!(
            allocator.allocate(&MinVariance::default().bounds(loose.clone())).unwrap(),
            allocator.allocate(&MinVariance::default()).unwrap()
        );
        let mean_variance = MeanVariance::risk_aversion(gamma);
        prop_assert_eq!(
            allocator.allocate(&mean_variance.clone().bounds(loose)).unwrap(),
            allocator.allocate(&mean_variance).unwrap()
        );
    }

    #[test]
    fn covariance_shortcuts_match_the_covariance_setter(
        returns in returns(),
        shrinkage in 0.0f64..1.0,
        regularization in 1e-8f64..1e-4,
        gamma in 1.0f64..500.0,
    ) {
        // No `well_posed` filter: errors must match too.
        let n = returns.ncols();
        let allocator = returns_via_flat(returns).unwrap();
        let config = CovarianceConfig::default()
            .shrinkage(shrinkage)
            .regularization(regularization);
        // Both orders, so that neither shortcut can undo the other.
        macro_rules! check {
            ($method:expr) => {{
                let method = $method;
                let expected = allocator.allocate(&method.clone().covariance(config.clone()));
                prop_assert_eq!(
                    allocator.allocate(&method.clone().shrinkage(shrinkage).regularization(regularization)),
                    expected.clone()
                );
                prop_assert_eq!(
                    allocator.allocate(&method.regularization(regularization).shrinkage(shrinkage)),
                    expected
                );
            }};
        }
        check!(MinVariance::default());
        check!(MinVariance::default().allow_short());
        check!(MeanVariance::risk_aversion(gamma));
        check!(MeanVariance::risk_aversion(gamma).allow_short());
        check!(low_rate());
        check!(low_rate().allow_short());
        check!(MaxDiversification::default());
        check!(RiskParity::default());
        check!(Hrp::default());
        check!(Herc::default());

        let model = BlackLitterman::new(vec![1.0 / n as f64; n], 2.5);
        let expected = model.clone().covariance(config).expected_returns(&allocator);
        prop_assert_eq!(
            model
                .clone()
                .shrinkage(shrinkage)
                .regularization(regularization)
                .expected_returns(&allocator),
            expected.clone()
        );
        prop_assert_eq!(
            model
                .regularization(regularization)
                .shrinkage(shrinkage)
                .expected_returns(&allocator),
            expected
        );
    }

    #[test]
    fn hrp_with_full_shrinkage_is_equal_weight(returns in returns(), linkage in 0usize..4) {
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns).unwrap();
        // Every variance is equal and every covariance zero, so each half's variance is
        // inversely proportional to its size, and each split shares weight by size.
        let hrp = Hrp::default().linkage(LINKAGES[linkage]);
        let weights = allocator.allocate(&hrp.clone().shrinkage(1.0)).unwrap();
        let equal = allocator.allocate(&EqualWeight).unwrap();
        for (h, e) in weights.iter().zip(&equal) {
            prop_assert!((h - e).abs() < 1e-12, "{weights:?}");
        }
        // No adjustment is the same as no covariance setting at all.
        prop_assert_eq!(
            allocator.allocate(&hrp.clone().covariance(CovarianceConfig::default())),
            allocator.allocate(&hrp)
        );
    }

    #[test]
    fn herc_with_one_cluster_is_inverse_variance(returns in returns(), linkage in 0usize..4) {
        prop_assume!(well_posed(&returns));
        let allocator = returns_via_flat(returns).unwrap();
        let herc = Herc::default().linkage(LINKAGES[linkage]);
        // One cluster holds every asset, so its weight is shared by inverse variance. A gap
        // statistic allowed at most one cluster must choose it.
        let one = allocator.allocate(&herc.clone().clusters(1)).unwrap();
        prop_assert_eq!(&allocator.allocate(&herc.max_clusters(1)).unwrap(), &one);
        let inverse_variance = allocator.allocate(&InverseVariance).unwrap();
        for (h, i) in one.iter().zip(&inverse_variance) {
            prop_assert!((h - i).abs() < 1e-12, "{one:?} vs {inverse_variance:?}");
        }
    }

    #[test]
    fn implied_returns_recover_the_market_weights(
        returns in returns(),
        caps in prop::collection::vec(0.1f64..10.0, 6),
        delta in 0.5f64..10.0,
        risk_free_rate in -0.01f64..0.01,
    ) {
        // Reverse optimization: the market portfolio is mean-variance optimal for the
        // returns it implies, at the same risk aversion, whatever the risk-free rate.
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let allocator = returns_via_flat(returns).unwrap();
        let caps = caps[..n].to_vec();
        let total: f64 = caps.iter().sum();
        let prior = BlackLitterman::new(caps.clone(), delta)
            .risk_free_rate(risk_free_rate)
            .prior_returns(&allocator)
            .unwrap();
        let weights = allocator
            .allocate(&MeanVariance::risk_aversion(delta).expected_returns(prior).allow_short())
            .unwrap();
        for (w, cap) in weights.iter().zip(&caps) {
            prop_assert!((w - cap / total).abs() < 1e-9, "{weights:?} vs {caps:?}");
        }
    }

    #[test]
    fn views_hold_and_move_other_assets_by_their_covariance(
        returns in returns(),
        caps in prop::collection::vec(0.1f64..10.0, 6),
        q in -0.01f64..0.01,
        spread in -0.01f64..0.01,
        confidence in 0.05f64..0.95,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let cov = covariance(&returns);
        let allocator = returns_via_flat(returns).unwrap();
        let model = BlackLitterman::new(caps[..n].to_vec(), 2.5);
        let prior = model.prior_returns(&allocator).unwrap();

        let certain = model
            .clone()
            .view(View::absolute(0, q).variance(0.0))
            .view(View::relative(n - 1, 0, spread).confidence(1.0))
            .expected_returns(&allocator)
            .unwrap();
        let scale = prior.iter().fold(q.abs().max(spread.abs()), |m, p| m.max(p.abs()));
        prop_assert!((certain[0] - q).abs() <= 1e-9 * scale, "{certain:?}");
        prop_assert!((certain[n - 1] - certain[0] - spread).abs() <= 1e-9 * scale, "{certain:?}");

        // A single uncertain view lands between the prior and the view, and moves every
        // other asset by its covariance with the view's: μ − π = τΣp · x for a scalar x, so
        // μᵢ − πᵢ = Σᵢ₀ / Σ₀₀ · (μ₀ − π₀).
        let partial = model
            .view(View::absolute(0, q).confidence(confidence))
            .expected_returns(&allocator)
            .unwrap();
        let (low, high) = (prior[0].min(q), prior[0].max(q));
        prop_assert!(partial[0] >= low && partial[0] <= high, "{} outside [{low}, {high}]", partial[0]);
        let moved = partial[0] - prior[0];
        for i in 1..n {
            let expected = cov[(i, 0)] / cov[(0, 0)] * moved;
            let tolerance = 1e-9 * (expected.abs() + moved.abs() * (cov[(i, i)] / cov[(0, 0)]).sqrt());
            prop_assert!(
                (partial[i] - prior[i] - expected).abs() <= tolerance,
                "asset {i} moved {} not {expected}",
                partial[i] - prior[i]
            );
        }
    }

    #[test]
    fn rebalancing_satisfies_kkt(
        returns in returns(),
        fractions in prop::collection::vec(0.05f64..1.0, 6),
        costs in prop::collection::vec(0.0f64..0.002, 6),
        gamma in 1.0f64..500.0,
        kappa in prop::option::of(0.0f64..0.01),
        capped in prop::bool::ANY,
        long_short in prop::bool::ANY,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let current = current_weights(&fractions, n);
        let costs = costs[..n].to_vec();
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let allocator = returns_via_flat(returns).unwrap();
        let mut method = MeanVariance::risk_aversion(gamma)
            .current_weights(current.clone())
            .transaction_costs_per_asset(costs.clone());
        if let Some(kappa) = kappa {
            method = method.turnover_penalty(kappa);
        }
        // Half of every cap is room enough: (0, 0.5 + 0.5/n) caps on each asset.
        let (lower, upper) = if long_short {
            method = method.allow_short();
            (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n])
        } else if capped {
            let cap = 0.5 + 0.5 / n as f64;
            method = method.bounds(WeightBounds::uniform(0.0, cap));
            (vec![0.0; n], vec![cap; n])
        } else {
            (vec![0.0; n], vec![f64::INFINITY; n])
        };
        let weights = allocator.allocate(&method).unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        // −μᵀw + (γ/2) wᵀΣw + κ‖w − w₀‖² has gradient γΣw − μ + 2κ(w − w₀).
        let pull = (&w - DVector::from_column_slice(&current)) * (2.0 * kappa.unwrap_or(0.0));
        let gradient = &cov * &w * gamma - &mu + pull;
        let tolerance = 1e-8 * gradient.amax().max(mu.amax()).max(costs.iter().cloned().fold(0.0, f64::max));
        assert_kinked_optimum(&w, &gradient, (&current, &costs), (&lower, &upper), tolerance)?;
    }

    #[test]
    fn turnover_caps_hold_bind_and_cost_variance(
        returns in returns(),
        fractions in prop::collection::vec(0.05f64..1.0, 6),
        cap in 0.0f64..1.0,
        looser in 0.0f64..1.0,
        long_short in prop::bool::ANY,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let current = current_weights(&fractions, n);
        let cov = covariance(&returns);
        let allocator = returns_via_flat(returns).unwrap();
        let method = |cap: Option<f64>| {
            let mut method = MinVariance::default().current_weights(current.clone());
            if let Some(cap) = cap {
                method = method.max_turnover(cap);
            }
            if long_short {
                method = method.allow_short();
            }
            method
        };
        let free = allocator.allocate(&method(None)).unwrap();
        let free_turnover = turnover(&free, &current);
        let capped = allocator.allocate(&method(Some(cap))).unwrap();
        let used = turnover(&capped, &current);
        prop_assert!(used <= cap + 1e-12, "turned over {used} past {cap}");
        if free_turnover > cap {
            prop_assert!((used - cap).abs() <= 1e-9, "turned over {used}, cap {cap}");
        } else if long_short {
            // Without a cap that binds, long-short goes through the closed form, and the cap
            // through the solver, which agree to rounding.
            for (a, b) in capped.iter().zip(&free) {
                prop_assert!((a - b).abs() <= 1e-12 * a.abs().max(1.0), "{capped:?} vs {free:?}");
            }
        } else {
            prop_assert_eq!(&capped, &free);
        }
        // A looser cap never has more variance.
        let loose = allocator.allocate(&method(Some(cap + looser))).unwrap();
        let v = |w: &Weights| variance(&cov, w);
        prop_assert!(v(&loose) <= v(&capped) * (1.0 + 1e-12) + 1e-18, "{} > {}", v(&loose), v(&capped));
        prop_assert!(v(&free) <= v(&loose) * (1.0 + 1e-12) + 1e-18);
    }

    #[test]
    fn capped_target_returns_hold_their_target_within_the_cap(
        returns in returns(),
        fractions in prop::collection::vec(0.05f64..1.0, 6),
        cap in 0.0f64..1.0,
        looser in 0.0f64..1.0,
        mix in prop_oneof![Just(0.0), Just(1.0), 0.0f64..1.0],
        long_short in prop::bool::ANY,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let current = current_weights(&fractions, n);
        let mu = returns.row_mean().transpose();
        let cov = covariance(&returns);
        let allocator = returns_via_flat(returns).unwrap();
        // Long-only, every return from the lowest asset mean to the highest is reachable,
        // the ends only by the assets tied there.
        let target = mu.min() + mix * (mu.max() - mu.min());
        // A cap within this of the least turnover counts as at it.
        let rounding = 1e-12;
        let method = |cap: Option<f64>| {
            let mut method = MeanVariance::target_return(target).current_weights(current.clone());
            if let Some(cap) = cap {
                method = method.max_turnover(cap);
            }
            if long_short {
                method = method.allow_short();
            }
            method
        };
        let free = allocator.allocate(&method(None)).unwrap();
        let capped = match allocator.allocate(&method(Some(cap))) {
            Ok(capped) => capped,
            Err(CutupError::InfeasibleTurnover { minimum, cap: given }) => {
                prop_assert!(minimum > given && given == cap);
                // The least turnover is reachable: a cap of exactly that is met, unless the
                // expected returns are so close that the target's constraint is resolved no
                // better than the rounding the cap allows. Rounding the target by an ulp of μ
                // moves the weights by about that over the spread of μ, and the turnover with
                // them, within a few times that over the solver's steps.
                let resolution = f64::EPSILON * mu.amax() / (mu.max() - mu.min());
                match allocator.allocate(&method(Some(minimum))) {
                    Ok(least) => prop_assert!(turnover(&least, &current) <= minimum + 2.0 * rounding),
                    Err(CutupError::NotConverged) => prop_assert!(resolution > rounding / 10.0, "{mu:?}"),
                    Err(other) => return Err(TestCaseError::fail(format!("{other:?}"))),
                }
                return Ok(());
            }
            Err(other) => return Err(TestCaseError::fail(format!("{other:?}"))),
        };
        let used = turnover(&capped, &current);
        prop_assert!(used <= cap + 2.0 * rounding, "turned over {used} past {cap}");
        let achieved = mu.dot(&DVector::from_column_slice(capped.as_slice()));
        prop_assert!((achieved - target).abs() <= 1e-12 * mu.amax().max(1e-3), "{achieved} vs {target}");
        prop_assert!(long_short || capped.iter().all(|&w| w >= 0.0), "{capped:?}");
        if turnover(&free, &current) > cap + 1e-9 {
            prop_assert!((used - cap).abs() <= 1e-9, "turned over {used}, cap {cap}");
        }
        // A looser cap never has more variance, nor less than no cap.
        let loose = allocator.allocate(&method(Some(cap + looser))).unwrap();
        let v = |w: &Weights| variance(&cov, w);
        prop_assert!(v(&loose) <= v(&capped) * (1.0 + 1e-12) + 1e-18, "{} > {}", v(&loose), v(&capped));
        prop_assert!(v(&free) <= v(&loose) * (1.0 + 1e-12) + 1e-18, "{} > {}", v(&free), v(&loose));
    }

    #[test]
    fn trading_settings_that_cost_nothing_change_nothing(
        returns in returns(),
        fractions in prop::collection::vec(0.05f64..1.0, 6),
        gamma in 1.0f64..500.0,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let current = current_weights(&fractions, n);
        let allocator = returns_via_flat(returns).unwrap();
        // No turnover exceeds 2, the most a portfolio of positive weights can turn over.
        let idle = MeanVariance::risk_aversion(gamma)
            .current_weights(current.clone())
            .transaction_costs(0.0)
            .turnover_penalty(0.0);
        prop_assert_eq!(
            allocator.allocate(&idle.max_turnover(2.0)),
            allocator.allocate(&MeanVariance::risk_aversion(gamma))
        );
        prop_assert_eq!(
            allocator.allocate(&MinVariance::default().current_weights(current).max_turnover(2.0)),
            allocator.allocate(&MinVariance::default())
        );
    }

    #[test]
    fn higher_costs_never_trade_more(
        returns in returns(),
        fractions in prop::collection::vec(0.05f64..1.0, 6),
        cost in 0.0f64..0.002,
        more in 0.0f64..0.002,
        gamma in 1.0f64..500.0,
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let current = current_weights(&fractions, n);
        let allocator = returns_via_flat(returns).unwrap();
        let trade = |cost: f64| {
            let method = MeanVariance::risk_aversion(gamma)
                .current_weights(current.clone())
                .transaction_costs(cost);
            turnover(&allocator.allocate(&method).unwrap(), &current)
        };
        let (cheap, dear) = (trade(cost), trade(cost + more));
        prop_assert!(dear <= cheap + 1e-9, "{dear} > {cheap}");
    }

    #[test]
    fn min_cvar_beats_every_long_only_portfolio_tried(
        returns in returns(),
        beta in 0.5f64..0.99,
        tries in prop::collection::vec(prop::collection::vec(0.0f64..1.0, 6), 30),
    ) {
        let n = returns.ncols();
        let allocator = returns_via_flat(returns).unwrap();
        let weights = allocator.allocate(&MinCvar::default().confidence(beta)).unwrap();
        prop_assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        prop_assert!(weights.iter().all(|&w| w >= 0.0));
        let best = allocator.cvar(&weights, beta).unwrap();
        let scale = best.abs().max(1e-3);
        let mut candidates: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect();
        candidates.push(vec![1.0 / n as f64; n]);
        candidates.extend(tries.iter().map(|t| current_weights(&t.iter().map(|x| x + 1e-3).collect::<Vec<_>>(), n)));
        for candidate in candidates {
            let other = allocator.cvar(&Weights::from(candidate.clone()), beta).unwrap();
            prop_assert!(best <= other + 1e-12 * scale, "{best} > {other} at {candidate:?}");
        }
    }

    #[test]
    fn min_cvar_meets_its_bounds_and_target(
        returns in returns(),
        beta in 0.5f64..0.99,
        fractions in bounds(),
        mix in 0.0f64..0.999,
    ) {
        let n = returns.ncols();
        let mu = returns.row_mean();
        let (lower, upper) = weight_bounds(&fractions, n);
        let allocator = returns_via_flat(returns).unwrap();
        let bounded = MinCvar::default()
            .confidence(beta)
            .bounds(WeightBounds::per_asset(lower.clone(), upper.clone()));
        let weights = allocator.allocate(&bounded).unwrap();
        for i in 0..n {
            prop_assert!(weights[i] >= lower[i] - 1e-15 && weights[i] <= upper[i] + 1e-15);
        }
        // The highest return inside the bounds: every asset at its floor, then what's left of
        // the budget to the highest means, each up to its cap. Any target between that and the
        // bounded portfolio's own return is reachable.
        let own: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| mu[j].total_cmp(&mu[i]));
        let mut left = 1.0 - lower.iter().sum::<f64>();
        let mut highest: f64 = lower.iter().zip(mu.iter()).map(|(l, m)| l * m).sum();
        for i in order {
            let share = left.min(upper[i] - lower[i]);
            highest += share * mu[i];
            left -= share;
        }
        let target = own + mix * (highest - own);
        let targeted = allocator.allocate(&bounded.target_return(target)).unwrap();
        for i in 0..n {
            prop_assert!(targeted[i] >= lower[i] - 1e-15 && targeted[i] <= upper[i] + 1e-15);
        }
        let achieved: f64 = targeted.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        prop_assert!((achieved - target).abs() <= 1e-12 * mu.amax().max(1e-3), "{achieved} vs {target}");
        prop_assert!(allocator.cvar(&targeted, beta).unwrap() >= allocator.cvar(&weights, beta).unwrap() - 1e-12);
    }

    #[test]
    fn min_cvar_ignores_scenario_and_column_order(
        returns in returns(),
        beta in 0.5f64..0.99,
    ) {
        let (t, n) = returns.shape();
        let base = returns_via_flat(returns.clone()).unwrap();
        let best = base.cvar(&base.allocate(&MinCvar::default().confidence(beta)).unwrap(), beta).unwrap();
        // The optimal CVaR, not the weights, which needn't be unique.
        let reversed = DMatrix::from_fn(t, n, |s, i| returns[(t - 1 - s, n - 1 - i)]);
        let other = returns_via_flat(reversed).unwrap();
        let again = other.cvar(&other.allocate(&MinCvar::default().confidence(beta)).unwrap(), beta).unwrap();
        prop_assert!((best - again).abs() <= 1e-12 * best.abs().max(1e-3), "{best} vs {again}");
    }

    #[test]
    fn labeled_pairs_in_any_order_match_column_order(
        returns in returns(),
        fractions in prop::collection::vec(0.05f64..1.0, 6),
        caps in bounds(),
        mu in prop::collection::vec(-0.002f64..0.004, 6),
        order in Just((0..6).collect::<Vec<usize>>()).prop_shuffle(),
    ) {
        prop_assume!(well_posed(&returns));
        let n = returns.ncols();
        let labels: Vec<String> = (0..n).map(|i| format!("asset {i}")).collect();
        let allocator = returns_via_flat(returns).unwrap().labeled(&labels).unwrap();
        let order: Vec<usize> = order.into_iter().filter(|&i| i < n).collect();
        let pairs = |values: &[f64]| -> Vec<(String, f64)> {
            order.iter().map(|&i| (labels[i].clone(), values[i])).collect()
        };
        let current = current_weights(&fractions, n);
        let (lower, upper) = weight_bounds(&caps, n);
        let (fractions, mu) = (&fractions[..n], &mu[..n]);
        let costs: Vec<f64> = fractions.iter().map(|f| f * 1e-3).collect();

        let in_order: Vec<Box<dyn Allocation>> = vec![
            Box::new(
                MinVariance::default()
                    .bounds(WeightBounds::per_asset(lower.clone(), upper.clone()))
                    .current_weights(current.clone())
                    .max_turnover(0.5),
            ),
            Box::new(
                MeanVariance::risk_aversion(50.0)
                    .expected_returns(mu)
                    .current_weights(current.clone())
                    .transaction_costs_per_asset(costs.clone()),
            ),
            Box::new(MaxSharpe::default().expected_returns(mu)),
            Box::new(RiskParity::default().risk_budgets(fractions)),
        ];
        let by_label: Vec<Box<dyn Allocation>> = vec![
            Box::new(
                MinVariance::default()
                    .bounds(WeightBounds::per_asset(pairs(&lower), pairs(&upper)))
                    .current_weights(pairs(&current))
                    .max_turnover(0.5),
            ),
            Box::new(
                MeanVariance::risk_aversion(50.0)
                    .expected_returns(pairs(mu))
                    .current_weights(pairs(&current))
                    .transaction_costs_per_asset(pairs(&costs)),
            ),
            Box::new(MaxSharpe::default().expected_returns(pairs(mu))),
            Box::new(RiskParity::default().risk_budgets(pairs(fractions))),
        ];
        for (a, b) in in_order.iter().zip(&by_label) {
            prop_assert_eq!(allocator.allocate(a), allocator.allocate(b));
        }

        let views = |model: BlackLitterman, p: Vec<(String, f64)>| {
            model
                .view(View::absolute(labels[order[0]].as_str(), mu[order[0]]))
                .view(View::portfolio(p, 0.001))
        };
        let portfolio: Vec<f64> = mu.iter().map(|m| m * 100.0).collect();
        prop_assert_eq!(
            views(BlackLitterman::new(fractions, 2.5), pairs(&portfolio)).expected_returns(&allocator),
            BlackLitterman::new(pairs(fractions), 2.5)
                .view(View::absolute(order[0], mu[order[0]]))
                .view(View::portfolio(portfolio, 0.001))
                .expected_returns(&allocator)
        );
    }
}
