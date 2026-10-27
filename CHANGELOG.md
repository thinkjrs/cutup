# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.6] - 2026-10-27

### Added

- Per-asset weight bounds for long-only min variance and mean-variance:
  `PortfolioAllocator::long_only_min_variance_allocation_with_bounds` and
  `PortfolioAllocator::long_only_mean_variance_allocation_with_bounds`, taking a
  `WeightBounds` built with `WeightBounds::uniform` or `WeightBounds::per_asset`. Each
  asset's weight stays between its floor and its cap; `f64::INFINITY` means no cap. The
  active-set solver now holds each variable at its lower bound, its upper bound, or
  between them, so the results are exact, as without bounds.
- `CutupError::InfeasibleBounds`, returned when the floors sum to more than 1 or the caps
  to less, by more than `1e-12`. Bounds within that of the budget, such as ten caps of 0.1,
  are rescaled to sum to 1, which moves each by at most `1e-12` of itself.
- Reference cases with bounds, against support enumeration over each asset at its floor,
  at its cap, or free, and cross-checked against PyPortfolioOpt. Property tests: bounded
  weights satisfy the KKT conditions for min variance and both mean-variance objectives,
  tightening a cap never lowers the variance, and bounds that never bind change nothing.

### Changed

- `CutupError::TargetReturnOutOfRange` reports the lowest and highest return reachable
  within the bounds. Without bounds those are still the lowest and highest expected return
  of any asset.
- The sum-to-one property test checks long-only methods to `1e-12`. Only the long-short
  methods, whose weights can be in the thousands, are checked relative to their largest
  position.

### Fixed

- The `compare` example's method column fits its longest label, so every row's weights line
  up.

## [0.2.5] - 2026-10-27

### Added

- `PortfolioAllocator::long_only_mean_variance_allocation`: mean-variance with no short
  positions, for a risk aversion or a target return, taking the same `MeanVarianceConfig`
  as `mean_variance_allocation`. Solved exactly by the active-set method behind long-only
  min variance. A risk aversion always has an optimum, even on a singular covariance.
- `CutupError::TargetReturnOutOfRange`, returned when a long-only target return is below
  the lowest or above the highest expected return of any asset.
- Reference cases against support enumeration, now shared with long-only min variance, and
  property tests: the weights satisfy the KKT conditions for both objectives, and match
  unconstrained mean-variance when it has no short positions.
- `PortfolioAllocator::long_only_max_sharpe_allocation`: the maximum Sharpe ratio portfolio
  with no short positions, taking the same `MaxSharpeConfig` as `max_sharpe_allocation`.
  Solved exactly by the active-set method, after the change of variables that makes it a
  convex quadratic program. It exists whenever some asset's expected return is above the
  risk-free rate, even where the unconstrained maximum doesn't.
- `CutupError::NoPositiveExcessReturn`, returned when no asset's expected return is above
  the risk-free rate, so no long-only portfolio has a positive Sharpe ratio.
- Reference cases for long-only max Sharpe against support enumeration, with unbounded
  cases decided by linear programming, and property tests: the weights satisfy the KKT
  conditions and have the highest Sharpe ratio of any long-only method.
- `PortfolioAllocator::max_diversification_allocation` and
  `max_diversification_allocation_with_config`: the maximum diversification portfolio
  (Choueifaty and Coignard, 2008), which maximizes the ratio of weighted average volatility
  to portfolio volatility, with optional covariance adjustments. Long-only by definition,
  and solved like long-only max Sharpe on the correlation matrix, so assets whose
  volatilities differ by many orders of magnitude stay accurate. Returns
  `SingularCovariance` when some long-only portfolio has no variance.
- Reference cases for max diversification against the same support enumeration, including
  a dataset with volatilities eight orders of magnitude apart, and property tests: the
  weights satisfy the KKT conditions, have the highest diversification ratio of any
  long-only method, and don't depend on column order, return scale or any one asset's
  units.
- Reference cases on a dataset with volatilities eight orders of magnitude apart for every
  covariance-based method, with the Python references also working in correlation units,
  and property tests: rescaling one asset's returns only rescales its weight in max
  diversification, risk parity and long-only max Sharpe, and risk parity has no answer on a
  rank-one covariance exactly when max diversification has none.

### Changed

- On a singular covariance, min variance, mean-variance and max Sharpe return the
  minimum-norm optimum in correlation units `diag(σ)w` instead of in the weights, so which
  of the tied portfolios is returned doesn't depend on each asset's units. Weights on
  singular covariances can differ from 0.2.4's.

### Fixed

- Min variance, mean-variance, max Sharpe, their long-only versions and risk parity work in
  correlation units, so assets whose volatilities are many orders of magnitude apart stay
  accurate. At a gap of about 1e8, they gave wrong weights, or returned `SingularCovariance`,
  `Unbounded` or `RiskFreeRateTooHigh` where an answer exists. Singular cutoffs use the
  correlation matrix, so they don't depend on units either.
- Max Sharpe's tolerance for a risk-free rate at the minimum variance return is relative to
  that portfolio's terms `wᵢμᵢ`, which don't depend on units, instead of the largest expected
  return.
- Risk parity checks its result against max diversification's null-space cutoff before
  returning, so rounding on a covariance with a zero-variance long-only portfolio can't
  give weights instead of `SingularCovariance`.

## [0.2.4] - 2026-10-26

### Added

- `PortfolioAllocator::long_only_min_variance_allocation` and
  `long_only_min_variance_allocation_with_config`: the minimum variance portfolio with no
  short positions, with optional covariance adjustments. Solved exactly by a primal
  active-set method, written for cutup so it adds no dependencies.
- Reference cases against support enumeration, and property tests: the weights satisfy the
  KKT conditions and have the lowest variance of any long-only method.

### Fixed

- Min variance, mean-variance and max Sharpe could return `SingularCovariance`, or slightly
  inaccurate weights, on valid input: a heavily shrunk covariance, or a rank-deficient one
  from fewer observations than assets. nalgebra's SVD can return factors that don't
  reconstruct the matrix, by up to 1e-2. cutup no longer uses it. Equality-constrained
  problems are solved by the null-space method, with the reduced system and max Sharpe's
  covariance decomposed by Jacobi rotations, which reconstruct it to within rounding. Results are
  unchanged up to rounding, including the minimum-norm weights on a singular covariance.

## [0.2.3] - 2026-10-23

### Added

- `PortfolioAllocator::risk_parity_allocation` and `risk_parity_allocation_with_config` with
  `RiskParityConfig`: the long-only portfolio where every asset contributes the same share
  of variance, or a share set by risk budgets, with optional covariance adjustments. Solved
  with Newton's method (Spinu, 2013); returns `SingularCovariance` when a long-only
  portfolio has zero variance, where no solution exists.
- `CutupError::NotConverged`, returned when an iterative solver stops before reaching a
  solution that exists, which risk parity does only for budgets more than about 10¹⁴ apart.
- `PortfolioAllocator::herc_allocation` and `herc_allocation_with_config` with
  `HercConfig`: Hierarchical Equal Risk Contribution (Raffinot, 2017) as riskfolio-lib
  implements it, with any linkage, and the number of clusters either fixed or chosen by
  the two-difference gap statistic. Results match riskfolio-lib up to floating-point
  rounding and don't depend on column order, except where merges tie at the cut.
- Reference cases for both, against coordinate descent and riskfolio-lib, and property
  tests: risk parity meets its budgets, HERC ignores column order, and both are long-only
  and ignore return scale.

## [0.2.2] - 2026-10-22

### Added

- `PortfolioAllocator::max_sharpe_allocation` with `MaxSharpeConfig`: the unconstrained
  maximum Sharpe ratio (tangency) portfolio for a risk-free rate, with optional expected
  returns and covariance adjustments. On a singular covariance it returns the
  minimum-norm optimum, or decides from Σ's null space whether the Sharpe ratio is unbounded
  or the risk-free rate is too high.
- `CutupError::RiskFreeRateTooHigh`, returned when the risk-free rate is at or above the
  minimum variance portfolio's return, where no maximum exists.
- HRP linkage choice: `HrpConfig` with `Linkage::{Single, Complete, Average, Ward}`, and
  `PortfolioAllocator::hrp_allocation_with_config`. Every linkage follows SciPy's
  `linkage` algorithm and tie-breaking rule step for step, so results match
  PyPortfolioOpt's `HRPOpt` up to floating-point rounding.
- `PortfolioAllocator::hrp_cluster_order`, the order HRP puts the assets in.
- Property tests (`tests/properties.rs`) checking invariants on random returns: weights
  sum to 1, long-only methods never short, minimum variance has the lowest variance,
  results don't depend on column order or return scale, mean-variance meets its objective,
  and max Sharpe has the highest Sharpe ratio.
- `examples/compare.rs`, running every method on the same data, and a table in the README
  for choosing a method.

### Changed

- HRP's clustering moved into a private `cluster` module. Results are unchanged.

### Documented

- HRP, like the original algorithm and `HRPOpt`, can depend on the order of the input
  columns: reordering them can move where the recursive bisection splits. Found by the new
  property tests.

## [0.2.1] - 2026-10-22

### Added

- `PortfolioAllocator::inverse_volatility_allocation`: weights proportional to `1 / σᵢ`.
- `PortfolioAllocator::inverse_variance_allocation`: weights proportional to `1 / σᵢ²`.
- Integration tests that check every allocation method against independent Python
  reference implementations (NumPy closed forms, PyPortfolioOpt for HRP) on fixed
  datasets, including singular and ill-conditioned ones. See `reference/README.md`.

## [0.2.0] - 2026-10-22

0.2.0 fixes how every covariance-based method computed its weights, and
changes the API so that bad input returns an error instead of a meaningless
result. **Expect different weights from every method except equal weight**, and
see [Migrating from 0.1.x](#migrating-from-01x) below.

### Fixed

- **Covariance is computed from returns, not prices.** 0.1.x took the
  covariance of raw price levels, so assets with large prices dominated it and
  minimum variance and HRP weights were wrong. Prices are now converted to
  simple returns (`p_t / p_{t-1} - 1`, as in `pandas.DataFrame.pct_change`)
  first.
- **HRP clusters assets.** 0.1.x bisected assets in column order, so its result
  depended on how the input columns were arranged. HRP now runs the López de
  Prado (2016) algorithm: single-linkage clustering on correlation distance,
  then recursive bisection by inverse-variance cluster risk. Results match
  PyPortfolioOpt's `HRPOpt` up to floating-point rounding.
- **Shrinkage targets the average variance.** `λ·I` became `λ·μ·I` with
  `μ = trace(Σ) / n`, so shrinkage behaves the same at any scale of returns.
  At typical return variances (around 1e-4), the old target overwhelmed Σ even
  at small λ.
- **Singular covariance matrices give a true minimum.** With no more rows of
  returns than assets, or with perfectly collinear assets, minimum variance
  and mean-variance used to invert a singular matrix and return arbitrary
  weights. They now solve the constrained problem exactly, or return an error
  when it has no solution.
- Minimum variance no longer panics when the covariance can't be inverted.

### Added

- `PortfolioAllocator::from_prices` and `PortfolioAllocator::from_returns`
  constructors, which validate their input.
- `CutupError`, returned for empty input, too few rows, NaN or infinite values,
  non-positive prices, zero-variance assets, out-of-range configuration, and
  problems with no solution.
- `Weights`, an ordered result type: one weight per asset, in column order.
- `PortfolioAllocator::mean_variance_allocation` with `MeanVarianceConfig`: the
  Markowitz portfolio for a risk aversion or a target return, using expected
  returns (the sample mean by default).
- `CovarianceConfig`, with `with_shrinkage` and `with_regularization` builders.

### Changed

- `mvo_allocation` is renamed `min_variance_allocation`, and
  `mvo_allocation_with_config` is renamed
  `min_variance_allocation_with_config`. The method always computed the
  global minimum variance portfolio and never used expected returns. "Mean-variance"
  now refers to `mean_variance_allocation`.
- `MvoConfig` is renamed `CovarianceConfig`. It is `#[non_exhaustive]`, so it's
  built with `CovarianceConfig::default().with_shrinkage(..)` rather than a
  struct literal.
- Allocation methods return `Weights` instead of `HashMap<usize, f64>`.
- Constructors, `min_variance_allocation*`, `mean_variance_allocation`,
  `hrp_allocation` and `run_portfolio_allocation` return `Result`.
  `ew_allocation` can't fail and returns `Weights` directly.
- The demo binary moved to `examples/basic.rs`, so `cargo install cutup` no
  longer installs it.

### Removed

- `PortfolioAllocator::new`. Use `PortfolioAllocator::from_prices`, which
  returns a `Result`.
- `mvo_allocation`, `mvo_allocation_with_config` and `MvoConfig` (see Changed).

### Migrating from 0.1.x

```rust
// 0.1.x
let allocator = PortfolioAllocator::new(prices);
let weights = allocator.mvo_allocation_with_config(&MvoConfig {
    regularization: Some(1e-6),
    shrinkage: Some(0.1),
});
let first = weights[&0];

// 0.2.0
let allocator = PortfolioAllocator::from_prices(prices)?;
let config = CovarianceConfig::default()
    .with_regularization(1e-6)
    .with_shrinkage(0.1);
let weights = allocator.min_variance_allocation_with_config(&config)?;
let first = weights[0];
```

- Replace `PortfolioAllocator::new(prices)` with
  `PortfolioAllocator::from_prices(prices)?`.
- Replace `mvo_allocation` with `min_variance_allocation`, and add `?` to the
  allocation calls that now return `Result`.
- Read weights by position (`weights[i]`, `weights.get(i)`, `weights.iter()`),
  or call `weights.to_hashmap()` to keep the old `HashMap<usize, f64>` shape.
- Re-check any stored or expected weights: minimum variance, HRP and shrinkage
  results all change with the fixes above.
- Regularization is still an absolute ε added to the diagonal, so scale it to
  your returns' variances.

## [0.1.4] and earlier

Initial releases. See the [tags](https://github.com/thinkjrs/cutup/tags).

[Unreleased]: https://github.com/thinkjrs/cutup/compare/0.2.6...HEAD
[0.2.6]: https://github.com/thinkjrs/cutup/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/thinkjrs/cutup/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/thinkjrs/cutup/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/thinkjrs/cutup/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/thinkjrs/cutup/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/thinkjrs/cutup/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/thinkjrs/cutup/compare/0.1.4...0.2.0
[0.1.4]: https://github.com/thinkjrs/cutup/releases/tag/0.1.4
