# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- HRP linkage choice: `HrpConfig` with `Linkage::{Single, Complete, Average, Ward}`, and
  `PortfolioAllocator::hrp_allocation_with_config`. Every linkage follows SciPy's
  `linkage` algorithm and tie-breaking rule step for step, so results match
  PyPortfolioOpt's `HRPOpt` up to floating-point rounding.
- `PortfolioAllocator::hrp_cluster_order`, the order HRP puts the assets in.
- Property tests (`tests/properties.rs`) checking invariants on random returns: weights
  sum to 1, long-only methods never short, minimum variance has the lowest variance,
  results don't depend on column order or return scale, and mean-variance meets its
  objective.
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

[0.2.1]: https://github.com/thinkjrs/cutup/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/thinkjrs/cutup/compare/0.1.4...0.2.0
[0.1.4]: https://github.com/thinkjrs/cutup/releases/tag/0.1.4
