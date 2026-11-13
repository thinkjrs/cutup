# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.9] - 2026-11-13

### Added

- `Returns::rolling(window)` allocates over rolling windows of the data, giving weights over
  time: `data.rolling(60).step(5).allocate(method)` takes any `Allocation`, by value or by
  reference, and returns a `RollingAllocation`, an `ExactSizeIterator` and
  `DoubleEndedIterator` of `Window`s, oldest first. Skipping windows, as with `nth`, `skip`,
  `last` or `rev`, allocates only those reached. Each window is the method on its own rows
  alone, with the data's labels. Each
  `Window` has its `periods()`, a range of return rows, and its own `weights()` result, so
  one that fails, such as with `ZeroVariance` over a quiet stretch, doesn't stop the others.
  The last window ends at the last period. A window under 2 periods or longer than the data,
  or a step of 0, is `InvalidConfig`. It's not a backtest: no window's weights feed the next.
- An optional `serde` feature: `Serialize` and `Deserialize` for `Weights`, `Asset`,
  `PerAsset`, `WeightBounds`, `CovarianceConfig`, every method, `BlackLitterman`, `View` and
  `FrontierPoint`, and `Serialize` for `CutupError` and `Window`. Each method and config is
  an object whose keys are its builders, each with its argument, such as
  `{"risk_aversion": 50.0, "bounds": {"upper": 0.6}, "covariance": {"shrinkage": 0.1}}`.
  - **Defaults:** settings at their defaults are left out, and left out they're the defaults.
  - **Errors:** unknown keys, keys that exclude each other, and missing required ones.
  - **No cap:** an upper bound of `f64::INFINITY` is `null`.
  - **Checks:** a given covariance matrix is checked as `CovarianceConfig::matrix` checks it.
  - **Formats:** it needs a self-describing format, such as JSON.
- `examples/rolling.rs` recomputes min variance weights over rolling windows as one asset
  turns volatile, and the README has sections on rolling allocation and `serde`.

## [0.3.8] - 2026-11-13

### Added

- `WeightBounds::group(members, lower, upper)` limits the total weight of several assets at
  once, such as every ad channel on one platform or every signal from one source, by label
  or by column. Groups may overlap, an asset listed twice in a group counts once, and an
  upper limit of `f64::INFINITY` means no cap. `MinVariance`, every `MeanVariance`
  objective, with or without rebalancing, `Returns::frontier` and `MinCvar` take them.
- `CutupError::InfeasibleGroups { group }`: `group` names a group no portfolio summing to 1
  within the bounds can meet, by its members' labels joined with `" + "`, and is `None` when
  each group can be met but not all at once, which the simplex solver finds. A group's
  limits within `1e-12` of what its members can hold are rounding, as for the bounds.
- A `MinVarianceGroups` benchmark, with two binding groups that overlap.
- `examples/groups.rs` splits a budget across five ad channels with caps and floors on two
  platforms' channels, and puts each method's portfolio next to the one without groups. The
  README has a section on groups.

### Changed

- The active-set solver takes two-sided limits on sums of weights, each through a slack
  variable for the sum that it holds at a limit and releases as it does a weight. Free
  slacks are left out of each factored solve, so the Cholesky path still applies. At the
  end, each limit is checked on the weights themselves, to `1e-8` of the sum's terms and the
  limit. Assets with almost no variance, such as constant returns under a shrinkage of
  `1e-309`, are scaled near `1e-155` in the solver, where a limit on them is within its
  rounding. That's `SingularCovariance` rather than weights that miss the limit.
- With groups, the solver starts from a vertex the simplex solver finds, and the least
  turnover into the bounds for a turnover cap comes from the simplex solver too, since
  clipping each weight into its own bounds can break a group.
- With groups, the range of target returns comes from the simplex solver, since filling the
  bounds in order of expected return can break a group. At an end, the portfolios that
  reach it are the linear program's optimal face, read off its reduced costs, so the target
  needs no constraint of its own there, as without groups. Each end's rounding counts the
  asset at its lower bound with room whose reduced cost is least as the next a target moves
  into, the asset the fill would move into next.
- `MinCvar`'s dual gets a multiplier for each finite group limit, as for each bound. Groups
  that can't all be met are found first, by the simplex solver's start, since they'd leave
  the dual unbounded rather than infeasible.

### Fixed

- `MinCvar` solves on returns scaled by a power of two to at most 1 in size, which is exact
  and leaves the answer unchanged. Returns near `1e100` dwarfed the bounds and the budget in
  its linear program, and the simplex method could stop at a wrong vertex: one asset capped
  at 0 came back as `NonFiniteWeight`, and with the cap as a group, with that asset's
  weight at 1. The fuzzer found it once its `allocate` target took groups, and checked that
  each group's total is within its limits.

## [0.3.7] - 2026-11-11

### Added

- `MeanVariance::target_return(ρ)` takes `.transaction_costs(..)`, uniform or per asset,
  rather than returning `NotSupported`. The target is then a floor on the return net of
  them, `μᵀw − Σcᵢ|wᵢ − w₀ᵢ| ≥ ρ`: if the least variance portfolio within the bounds and any
  turnover cap already nets `ρ`, that's the answer, and otherwise the net return is `ρ`. The
  net return is concave, so this is convex, and it's solved by a search on the target's
  multiplier, each step the kinked problem a risk aversion solves. A target above the
  highest net return within the bounds and the cap, found by the simplex solver, is
  `TargetReturnOutOfRange` with that as `highest` and `lowest` −∞, since every target below
  it is reachable. Costs of 0 change nothing: the target stays exact.
- `MeanVariance::target_return(ρ)` takes `.turnover_penalty(κ)` too, minimizing
  `wᵀΣw + κ‖w − w₀‖²` at that return, so a target return no longer returns `NotSupported`
  for any rebalancing setting.

### Changed

- The simplex solver behind `MinCvar` and the least-turnover check for a capped target
  return updates its basis's LU factors in product form, factoring afresh once 32 eta matrices
  have built up or after a pivot below 1e-6 of its column's largest entry, rather than
  factoring every iteration. An optimum or an unbounded direction is always confirmed from
  fresh factors. `MinCvar` takes 1.06 s at 500 assets rather than 10.7 s, 8.6 ms at 100 rather
  than 28 ms, and 39 µs at 10 rather than 46 µs.
- The active-set solver behind long-only `MinVariance`, `MeanVariance`, `MaxSharpe` and
  `MaxDiversification`, and behind the rebalancing settings, keeps a Cholesky factor of the
  covariance's block on the weights it's solving for, updated as weights reach their bounds or
  leave them, rather than taking an eigendecomposition every iteration. A block that's
  singular, or has a pivot below `2⁻²⁶` of its diagonal, still goes through the
  eigendecomposition, as does a step whose solution misses the KKT conditions, after which the
  factor is computed afresh. `MinVariance` takes 0.23 s at 500 assets rather than 18.2 s,
  2.3 ms at 100 rather than 36.7 ms, and 10.7 µs at 10 rather than 31 µs.

### Fixed

- A long-only target return, for `MeanVariance` with or without a turnover cap and for
  `MinCvar`, is measured against the rounding of the end of the range it's near, `1e-12` of
  `Σ|μᵢwᵢ| + |μₖ|`, rather than `1e-12` of the larger end in magnitude. The sum is over that
  end's portfolio, and `k` is the next asset a target just inside it moves weight into. One
  expected return far from the rest, such as `−1000` beside means near `0.002`, no longer
  widens the tolerance at an end that neither holds its asset nor moves into it next. Targets just inside that end were
  moved onto it, which also collapsed a narrow `Returns::frontier`'s inner points onto its last;
  now they're solved as given. A target that far past the end is now
  `TargetReturnOutOfRange` rather than moved onto it.
- The long-only target-return solve starts from a portfolio holding only assets next to each
  other in expected return: the budget left after the lower bounds, slid along the assets
  sorted by expected return until it reaches the target. It started from a mix of the lowest
  and highest return portfolios, which held a little of every asset, so a far expected return
  left free in the first step swamped the target's constraint, and a target near the top
  beside one could fail with `SingularCovariance`.
- The simplex solver behind `MinCvar` and the least-turnover check for a capped target
  return gets off a degenerate vertex, where basic variables sit at their bounds, by widening
  those bounds. After 10 pivots in a row that don't move the solution, each basic variable's
  bounds widen by about `1e-7` of their size, by a different amount for each, and at the
  optimum they're restored, with dual simplex pivots to bring back any variable left beyond
  them. It used to switch to Bland's rule, which could run into the iteration cap: started
  with no scenario in its tail, `MinCvar` returned `NotConverged` after 16 s at 200 assets,
  and now solves in 0.1 s. Bland's rule stays for a vertex the widening doesn't resolve.
  `MinCvar` still starts from its worst scenarios, which is faster, so its times are
  unchanged.
- A long-only `MeanVariance::target_volatility(σ)` counts a variance within rounding of the
  target's as at it. At the minimum variance portfolio's volatility it gives that portfolio:
  squaring the volatility back could miss its variance by an ulp, and the step from it, the
  square root of that, moved the weights by up to 2e-8. With a singular covariance, it gives
  the highest-return portfolio among those with the least variance, where rounding could
  stop the search partway.
- A long-short `MeanVariance::target_volatility(σ)` counts a variance within rounding of the
  minimum variance portfolio's as at it too. A target an ulp above that portfolio's volatility
  stepped from it by the square root of the ulp, which moved the weights by 7e-9.
- A long-only `MeanVariance::target_volatility(σ)` counts a weight within `4nε` of a bound's
  size as at that bound. The solver returns a held weight within rounding of its bound, and
  compared exactly, that could hide every lower bound of a solution, as it did on wasm. The
  search then stopped between two portfolios holding different assets, and was off by 1e-3.

## [0.3.6] - 2026-11-10

### Added

- `MeanVariance::target_volatility(σ)`, the highest expected return at a volatility of at most
  `σ`: the efficient frontier portfolio at that risk, as PyPortfolioOpt's `efficient_risk`.
  Long-short it's exact, the minimum variance portfolio plus the least-variance direction that
  adds expected return. Long-only it searches over the target return until the frontier's ends
  hold the same assets at their bounds, then solves the quadratic between them exactly. Past the
  highest-return portfolio's volatility within the bounds, it returns that portfolio. A `σ`
  below the minimum variance portfolio's volatility is the new
  `CutupError::TargetVolatilityOutOfRange`, with that minimum. Rebalancing settings return
  `NotSupported` with it for now.
- `Returns::frontier(&method, n)`, the efficient frontier: `n` portfolios with the least
  variance at expected returns evenly spaced from the minimum variance portfolio's to the
  highest reachable, as PyPortfolioOpt's `efficient_return` at each. It takes any
  `MeanVariance` and uses its expected returns, covariance settings and bounds, ignoring its
  objective. Each `FrontierPoint` has its `expected_return()`, `volatility()` and `weights()`,
  under the method's expected returns and adjusted covariance. Long-only it ends at the
  highest-return portfolio within the bounds, and long-short at the highest expected return
  of any one asset. With no range of returns to cover, it's the minimum variance portfolio
  alone. Fewer than 2 points is `InvalidConfig`, and rebalancing settings return
  `NotSupported`.

## [0.3.5] - 2026-11-09

### Added

- `Returns::exponentially_weighted_mean(half_life)` and
  `exponentially_weighted_mean_decay(λ)`, each asset's exponentially weighted mean, to pass
  to `MeanVariance` or `MaxSharpe` through `.expected_returns(..)` in place of the sample
  mean. It's pandas' `ewm().mean()`, the mean the exponentially weighted covariance uses: each
  period's weight halves every `half_life` periods back from the last row, so a long half-life
  gives the sample mean and a short one the last period's returns. An invalid half-life or
  decay is `InvalidConfig`, as for the covariance.
- `Returns::capm_returns(market_weights, risk_free_rate)`, each asset's expected return under
  the capital asset pricing model, `r_f + βᵢ(E[r_m] − r_f)`, with its beta to the market
  portfolio from the sample covariance and the market's sample mean, as PyPortfolioOpt's
  `capm_return` computes it per period without compounding. The market weights are checked
  and normalized as `BlackLitterman`'s are, so capitalizations work too. A market with no
  variance is the new `CutupError::RisklessMarket`. For the returns the market implies through
  a risk aversion, `δΣw`, there's `BlackLitterman::prior_returns`.
- `Returns::james_stein_mean()` and `Returns::bayes_stein_mean()`, the sample means shrunk
  toward a common value to reduce their estimation error. James-Stein shrinks toward their
  average with riskfolio-lib's `"JS"` intensity, `(tr Σ − 2λ_max) / (T‖μ − μ̄1‖²)`, clamped to
  `[0, 1]`, so it keeps the average and gives the sample means for fewer than three assets.
  Bayes-Stein (Jorion, 1986) shrinks toward the minimum variance portfolio's mean, keeping that
  portfolio's return, with intensity `(n + 2) / (n + 2 + T(μ − μ₀1)ᵀΣ⁻¹(μ − μ₀1))`, riskfolio-lib's
  `"BS"` with target `"b2"`. It works in correlation units, with each asset scaled on its own,
  and returns `ZeroVariance` for a constant asset or `SingularCovariance` for a correlation
  matrix singular to rounding.
- `Returns::log_returns()`, a new `Returns` of the log returns `ln(1 + r)`, with the same
  labels, for every method to work on in place of simple returns. It's PyPortfolioOpt's
  `returns_from_prices(prices, log_returns=True)`, computed with `ln_1p`, so a return near 0 keeps
  its digits. A return of −1 or below has no log return and is the new
  `CutupError::TotalLoss`, with its row and column.

## [0.3.4] - 2026-11-09

### Added

- `CovarianceConfig::matrix(rows)`, `matrix_flat(values, assets)` and, with the `nalgebra`
  feature, `from_matrix`, for a covariance estimate of your own in place of the sample
  covariance, with one row and column per asset in column order. Every covariance-based method
  takes it through `.covariance(..)`, and shrinkage and regularization apply on top. It must be
  symmetric and positive semidefinite to rounding: entries within `1e-12` of the largest of
  their transposes are averaged, and eigenvalues down to `−10nε` of the largest are allowed.
  HRP and HERC still cluster on the returns' correlations. A covariance of another size than
  the data returns `DimensionMismatch`, and an asset given a variance of 0 `ZeroVariance`,
  unless an adjustment gives it one.
- `CutupError::AsymmetricCovariance` and `CutupError::IndefiniteCovariance`, for a given
  covariance that isn't symmetric or positive semidefinite beyond rounding.
- Ledoit-Wolf shrinkage: `CovarianceConfig::ledoit_wolf(target)`, and a `ledoit_wolf` setter
  on every covariance-based method, shrink the sample covariance with the intensity Ledoit and
  Wolf estimate from the returns, in place of a fixed `λ`. Whichever of `shrinkage` and
  `ledoit_wolf` is set last applies. `ShrinkageTarget::ConstantVariance` is the average
  variance times the identity, the target of `shrinkage`, with scikit-learn's
  `ledoit_wolf_shrinkage` intensity. `ShrinkageTarget::ConstantCorrelation` keeps each
  variance and sets every correlation to the average, with the intensity from Ledoit and
  Wolf's "Honey, I shrunk the sample covariance matrix". The intensity comes from the `1/T`
  moments the papers use, and applies to cutup's `T − 1` sample covariance. With a given
  covariance it's `NotSupported`, and a constant asset is `ZeroVariance` toward a constant
  correlation, or when the intensity is 0.
- Exponentially weighted covariances and semicovariances, as estimates in place of the sample
  covariance, which every covariance-based method takes through `.covariance(..)`, with
  shrinkage and regularization on top:
  - `CovarianceConfig::exponentially_weighted(half_life)`, or
    `exponentially_weighted_decay(λ)` with `λ = 2^(−1/half_life)`, weights the last row most.
    It's pandas' `ewm().cov()`, as in riskfolio-lib and skfolio: a weighted mean and the
    unbiased correction for the weights, so a long half-life gives the sample covariance.
  - `CovarianceConfig::semicovariance()` counts returns below each asset's mean, over `T − 1`,
    as riskfolio-lib does, and `semicovariance_below(threshold)` those below a fixed
    threshold, over `T`, as PyPortfolioOpt does.
  - HRP and HERC still cluster on the returns' correlations. An invalid half-life, decay or
    threshold is `InvalidConfig` when a method uses it, Ledoit-Wolf shrinkage of any of them is
    `NotSupported`, and an asset whose estimated variance is 0, such as one never below the
    threshold, is `ZeroVariance` unless an adjustment gives it one.
- `BlackLitterman::posterior_covariance(&data)`, the posterior covariance of the returns,
  `Σ + τΣ − τΣPᵀ(τPΣPᵀ + Ω)⁻¹PτΣ` (He and Litterman, 1999), as PyPortfolioOpt's `bl_cov` and
  riskfolio-lib's `black_litterman` compute it. It returns a `CovarianceConfig` holding it as a
  given covariance, to pass to a method's `covariance` setter alongside the posterior returns,
  with shrinkage and regularization still available on top. With no views it's `(1 + τ)Σ`. Its
  errors are those of `expected_returns`, and `NonFiniteCovariance` if it overflows.
- `Returns::stats_with(&weights, &covariance)`, the statistics of `Returns::stats` with the
  covariance a `CovarianceConfig` sets, as an allocation method would use it: the volatility,
  the Sharpe ratio's denominator, the diversification ratio and the risk shares come from its
  matrix, while the expected return is still the sample mean and the CVaR historical. With the
  default config it's `Returns::stats` exactly. A variance within `10nε` of the square of the
  positions' volatilities under the covariance, `Σᵢ|wᵢ|σᵢ`, counts as riskless: the rounding a
  covariance matrix's entries carry, the allowance `CovarianceConfig::matrix` gives eigenvalues.

### Fixed

- `InverseVariance` and `InverseVolatility` no longer return weights of 0 or NaN on variances
  near the smallest normal number, about 1e-308. Their inverses summed past `f64::MAX`, so every
  weight was 0, or, smaller, overflowed themselves, which returned `NonFiniteWeight`. Each
  variance is now divided by the smallest first. Found by fuzzing.
- `Returns::stats` counts a portfolio as riskless when its volatility is below `1e-10` of the
  sizes of its positions' largest returns, `Σᵢ|wᵢ| maxₜ|rₜᵢ|`: the rule for a constant asset,
  applied to each position. It was `1e-10` of the portfolio's own largest return, which is the
  same for one asset, but missed positions that cancel to within rounding. On returns near
  1e134, `MinCvar` weights that cancel to returns near 1e116 had a volatility and risk shares
  made of rounding, some of them infinite, and a hedge of an asset that counts as constant had
  risk shares of NaN. Found by fuzzing.
- `Returns::stats` risk contributions sum to 1 when subnormal weights carry the portfolio's
  return. Their products with the returns lose most of their bits, so weights of 1e-322 gave
  shares summing to 0.9995, and a share was multiplied by its weight before being rescaled, so
  one of 1.5e-323 underflowed to 0 and every share was NaN. Each weight is now rescaled first,
  and the shares divided by their sum. Found by fuzzing.
- `Hrp` and `Herc` no longer return weights of 0 on covariances with variances near the smallest
  normal number, about 1e-308: the same overflow as in `InverseVariance` above, in the
  inverse-variance weights they split by within each cluster. Each cluster's smallest variance
  is now divided by each asset's, which keeps every term in (0, 1]. Found by fuzzing, on a
  semicovariance of 0 regularized by 1.4e-308, but a given covariance reached it too. And
  where shrinkage underflows and leaves an asset with a variance of 0, as a shrinkage of 1e-30
  toward an average variance near 1e-300 does, the assets in its cluster with none now split
  its weight, the limit of `1 / σ²`, where `Herc` returned `NonFiniteWeight`.

## [0.3.3] - 2026-11-04

### Added

- `Returns::stats(&weights)`, which returns a `PortfolioStats` of any weights over the data: the
  expected return, the volatility, the Sharpe ratio at a risk-free rate, the diversification
  ratio, each asset's share of the variance, and CVaR at a confidence, the same as
  `Returns::cvar`. They're per period, with the sample mean and covariance, and computed from the
  portfolio's return in each period, scaled by powers of two, so they don't overflow and the
  variance can't round below 0. A volatility below `1e-10` of the portfolio's largest return
  counts as 0, as it does for an asset, and then the ratios and shares are `None`. It returns
  `DimensionMismatch` or `LabelMismatch` for weights that don't fit the data, as `cvar` does, and
  `NonFiniteWeight` for a weight that's NaN or infinite.
- `Weights::clean(cutoff)`, which sets every weight smaller in size than `cutoff` to 0 and
  rescales the rest to sum to 1, so results don't carry weights like 0.0003. Unlike
  PyPortfolioOpt's `clean_weights`, it rescales, so the weights can be passed back as current
  weights; that can move a weight just past its bounds or a turnover cap. It returns
  `InvalidConfig` for a cutoff that's negative or not finite, or one that leaves weights that
  can't be rescaled: it drops them all, or the shorts it keeps outweigh or all but cancel the
  longs. A weight that's NaN or infinite returns `NonFiniteWeight`.

## [0.3.2] - 2026-11-03

### Fixed

- `MinCvar` no longer returns `NotConverged` from about 200 assets. Its linear program started
  every scenario weight at 0, a degenerate start that it didn't leave before the iteration cap.
  It now starts the scenarios where the equal-weight portfolio loses most at their cap, and
  it's faster at every size: 28 ms rather than 0.96 s at 100 assets, and 11 s at 500. Where
  the optimum is unique, the weights are the same. Where several portfolios share the lowest
  CVaR, it can return a different one of them.
- Long-short `MeanVariance::target_return` no longer returns weights that miss the budget at
  extreme scales: one asset with returns near 1e126 and a target near 1e299 got a weight of
  about 8.9e172. The quadratic solver checks its solution against the constraints, and that
  check squared residuals near 1e299, overflowed, and passed. It's now computed without
  overflow, and returns `SingularCovariance` where it still can't be. Found by fuzzing.
- Long-short `MaxSharpe` no longer returns weights when the unnormalized weights sum past
  `f64::MAX`, as a shrinkage of 1.9e-314 on constant assets did. The same overflow made the
  minimum-variance return NaN, which skipped the risk-free rate check, so it returned weights
  where it should have returned `RiskFreeRateTooHigh`; below that rate, they were all zero. Both
  sums are now divided by their largest magnitude first. Found by fuzzing.
- Long-short `MeanVariance` no longer returns weights that miss the budget when one asset's
  variance is vanishingly small next to the others'. A regularization of 3.9e-308 on a
  constant asset left weights summing to about −7.6e75. The quadratic solver's null-space
  basis was accurate only to rounding of 1, so its tiny entry for that asset, near 1e-62, was
  wrong in most of its digits, and converting back to weights multiplied it by about 1e50. It's
  now accurate in its tiny entries, and that input returns the near-riskless asset. Long-short
  `MinVariance` and rebalancing use the same solver. Found by fuzzing.
- `MeanVariance`, `MinVariance`, long-only `MaxSharpe` and `MaxDiversification`, and
  rebalancing now check their solutions against the budget and any target return in the
  original units, and return `SingularCovariance` if they miss.
  Bounded solves were only rescaled to sum to 1, which hid a missed budget, and a target return
  wasn't rechecked: long-only rebalancing to a target of 0 on the input above earned −1e8.
  Long-short, it now meets the target. Long-only, the two constraints are parallel to rounding
  once rescaled, so it returns `SingularCovariance`.
- Long-short `MaxSharpe` no longer misses a fully invested zero-variance portfolio when asset
  volatilities are below about 1e-154. The checks that find one compared norms that
  overflowed, and `∞ > ∞` is false, so on two assets that offset exactly at returns near
  1e-155 it returned weights where it should have returned `RiskFreeRateTooHigh`. They're now
  computed without overflow.

## [0.3.1] - 2026-11-01

### Added

- `wasm32-unknown-unknown` is a tested target. CI builds for it with and without features, and
  runs the unit, integration and reference tests there, so weights match the reference
  implementations on wasm too.

### Fixed

- `Hrp` and `Herc` no longer fail with `NonFiniteCovariance` on returns between about 1e-81
  and 1e-146, where the product of two variances underflows to zero. Correlations are divided
  by each volatility in turn, as the other methods already do, so the weights match the same
  returns at any larger scale. Variances below about 1e-292, `f64::MIN_POSITIVE / f64::EPSILON`,
  still return `NonFiniteCovariance`: below it, covariances between assets are subnormal and
  lose precision, and a cluster's inverse variances can sum past `f64::MAX`.

### Changed

- The code is split into modules: one per method under `methods/`, and `error`, `weights`,
  `covariance`, `bounds` and `returns` for the shared types. `lib.rs` keeps the crate docs,
  the module list and the re-exports. Nothing public moved: every item is still at the crate
  root, and no weights change.
- The crate is `#![forbid(unsafe_code)]`. It had no `unsafe` code before either; now it can't
  gain any.
- The crates.io description lists cutup's allocation methods, where it described a general
  library for dividing resources, and the keywords are `portfolio`, `allocation`,
  `optimization`, `finance` and `quant`.

## [0.3.0] - 2026-10-30

### Added

- `.covariance(..)` on `Hrp` and `Herc`, with the `shrinkage(λ)` and `regularization(ε)`
  shortcuts the other covariance-based methods have. They adjust only the risk split: the
  variances that divide weight between clusters and within them. The hierarchy stays on the
  sample correlations, so `Hrp::cluster_order` and HERC's clusters don't change. Results
  match riskfolio-lib's `HCPortfolio` with `method_cov="shrunk"` and `leaf_order=False`, and
  the reference tests have 35 new cases for them.
- `Returns::from_prices_flat` and `Returns::new_flat`, which take one flat, row-major slice
  and the number of assets, such as a JavaScript `Float64Array`.
- A `nalgebra` feature, off by default, with `Returns::from_prices_matrix` and
  `Returns::from_matrix` for `DMatrix` input and `From<Weights> for DVector<f64>`. It tracks
  the nalgebra version cutup depends on.
- `CutupError::RaggedRows`, for rows of different lengths, or flat input whose length isn't
  a multiple of the number of assets.
- Asset labels. `Returns::labeled([..])` names the assets, for example with their tickers,
  and `Returns::labels` returns them; without labels, each asset's label is its column, `"0"`,
  `"1"`, and so on. `Weights` carries them, with `weight(label)`, `weights[label]`, `labels()`
  and `labeled()`, which iterates over `(label, weight)` pairs. `iter()`, indexing by column
  and `as_slice()` are unchanged.
- Settings can name assets by label. `View::absolute` and `View::relative` take an `Asset`, a
  column or a label, and `WeightBounds::asset(asset, lower, upper)` bounds one asset. Every
  setting with a value per asset takes `PerAsset` values: in column order, as before, or
  pairs of an asset and its value, as an array, a `Vec`, a slice, a `HashMap` or a
  `BTreeMap`, or `Weights`, by their labels, so last period's weights can be the current
  weights even if the assets change order. That's `WeightBounds::per_asset`, `BlackLitterman::new`'s market weights,
  `View::portfolio`, `current_weights`, `expected_returns`, `risk_budgets` and
  `transaction_costs_per_asset`. Pairs must give every asset a value, except current weights
  and view portfolios, where an asset left out is 0.
- `CutupError::UnknownAsset`, with the `Asset` as given, for a label that isn't in the data
  or a column out of range, and `CutupError::DuplicateAsset { column, asset }`, for a label repeated in
  `labeled`, an asset given twice, or a relative view of an asset against itself. `Asset`
  has `label()` and `column()`, and displays as its label or as `column 3`.
- `CutupError::InvalidAssetValue { field, column, asset, value }`, for one asset's value out
  of range in a per-asset setting: an asset's bound, a market weight, view portfolio weight,
  current weight, expected return, risk budget or per-asset transaction cost. Settings that
  aren't per asset, including uniform bounds and costs, still return `InvalidConfig`.
- `Display` for `Weights`: a table of each asset's label and weight, with 4 decimals or the
  precision given, as in `{weights:.2}`.
- `IntoIterator` for `Weights` by value, yielding each weight in column order.
- `Debug`, `Clone` and `PartialEq` for `Returns`, and `Hash` for `Linkage`. A test checks
  that every public type is `Send + Sync`.
- `CutupError::LabelMismatch`, which `Returns::cvar` returns for weights labeled with other
  assets than the data, or in another order. Weights labeled by column, such as
  `Weights::from(vec)`, are still taken in column order.

### Removed

- The API deprecated in 0.2.8: every `PortfolioAllocator::*_allocation*` method,
  `PortfolioAllocator::hrp_cluster_order`, `run_portfolio_allocation`, and the config
  structs only they took: `MeanVarianceConfig`, `MeanVarianceObjective`,
  `MaxSharpeConfig`, `RiskParityConfig`, `HrpConfig` and `HercConfig`. Use
  `Returns::allocate` with the method values, and `Hrp::cluster_order`; the
  README's "Upgrading from 0.2" section maps each one to its replacement.
- `Weights::to_hashmap`, left over from when methods returned a `HashMap`. Use
  `weights.iter().copied().enumerate().collect::<HashMap<_, _>>()`, or `as_slice()`.

### Changed

- `PortfolioAllocator` is now `Returns`: it holds the returns and their covariance, and the
  methods are values passed to `Returns::allocate`. The returns constructor is
  `Returns::new`. `Allocation::allocate` takes `&Returns`.
- `Returns::from_prices` and `Returns::new` take rows, `&[R]` for any
  `R: AsRef<[f64]>`, such as `&[[f64; N]]` or `&[Vec<f64>]`, instead of a nalgebra `DMatrix`,
  so nalgebra is no longer part of cutup's public API without the `nalgebra` feature.
- `CovarianceConfig`'s setters are `shrinkage(λ)` and `regularization(ε)`, without `with_`,
  like every other setter. Its fields are private: set them through the setters, and read
  them with the new `lambda()` and `epsilon()`, which return the λ and ε set, or `None`.
- The minimum supported Rust version is 1.85, up from 1.78, and the crate is on edition 2024.
  CI tests on 1.85 rather than 1.78. A crate's edition doesn't affect the crates that depend
  on it.
- `Hrp` and `Herc` are no longer `Copy` or `Eq`, since they now hold a `CovarianceConfig`.
  They're still `Debug`, `Clone`, `PartialEq` and `Default`; clone one to reuse it.
- `InverseVolatility` and `InverseVariance` are no longer `#[non_exhaustive]`, so they can be
  written as plain `InverseVolatility` and `InverseVariance`, as `EqualWeight` can.
  `default()` still works.
- The per-asset setters take `impl Into<PerAsset>` instead of `impl Into<Vec<f64>>`. Every
  type of `f64`s that converted into a `Vec<f64>` still works: arrays, `Vec`s, slices,
  `Box<[f64]>`, `VecDeque<f64>` and `Cow<[f64]>`. `WeightBounds::per_asset` took `Vec<f64>`s,
  so a bare `.collect()` passed to it no longer infers its type; collect into a `Vec<f64>`.
- A `View` on an asset column out of range returns `CutupError::UnknownAsset`, and a relative
  view of an asset against itself `CutupError::DuplicateAsset`, instead of
  `CutupError::InvalidConfig` with the field `view_asset`.
- Errors name assets by label as well as column, so their payloads can stay fixed through
  1.x. `ZeroVariance` and `NonFiniteWeight` are `{ column, asset }`, with `asset` the label,
  instead of `{ asset }` with the column, and their messages give both, or only the column
  for data without labels. A per-asset value out of range is an `InvalidAssetValue` rather
  than an `InvalidConfig`.
- `CutupError::DimensionMismatch` has a `field` naming the setting, or `weights` or `labels`,
  and its message names it.
- `Weights` compare their labels as well as their values, so weights from labeled data
  aren't equal to `Weights::from(vec)` with the same values, which are labeled by column.

### Deprecated

- `PortfolioAllocator`, now an alias for `Returns`, and `Returns::from_returns`, the old name
  of `Returns::new`. Both still compile, with a warning, and are removed in 1.0.0.

## [0.2.8] - 2026-10-29

### Added

- `MinCvar`, the long-only portfolio with the lowest conditional value at risk (expected
  shortfall) over the historical return scenarios, at a confidence of 0.95 by default, with
  `.bounds(..)` and `.target_return(..)`. It's solved exactly as the dual of the
  Rockafellar-Uryasev linear program, which has a row per asset rather than per scenario.
  `PortfolioAllocator::cvar` measures any portfolio's CVaR.
- `MeanVariance::target_return(..)` takes `.max_turnover(..)`: the lowest variance at the
  target return within the turnover cap, long-only, within bounds or long-short.
  `CutupError::InfeasibleTurnover` reports the least turnover that reaches the target, from
  a linear program. `.turnover_penalty(..)` and `.transaction_costs(..)` with a target
  return still return `CutupError::NotSupported`: both are in the units of the returns, and
  the objective is a variance.
- An internal linear program solver, `src/lp.rs`: a two-phase revised simplex method with
  bounded variables, taking the most improving variable and falling back to Bland's rule
  when degenerate pivots stall it. It's checked against `scipy.optimize.linprog` on 70
  random problems (`reference/lp.py`, run in CI), and by a property test that every optimum
  satisfies the optimality conditions.
- Min CVaR reference cases on every dataset, against `linprog` on the primal program and
  cross-checked against PyPortfolioOpt and riskfolio-lib, and property tests: the optimum
  beats every long-only portfolio tried, meets its bounds and target, and doesn't depend on
  the order of scenarios or assets.
- `Hrp::cluster_order`, the order HRP puts the assets in, replacing
  `PortfolioAllocator::hrp_cluster_order`.
- An "Upgrading from the method API" table in the README, from each deprecated method and
  config setting to its replacement.
- Target-return rebalancing reference cases, with the target as a second equality in the
  support enumeration and the least turnover from `linprog`, cross-checked against
  PyPortfolioOpt, and a property test that a capped target return meets its target within
  the cap.

### Changed

- A long-only target return within rounding of the lowest or highest return the bounds
  allow, just inside the range, is now treated as that end, as one just outside it already
  was, and in a range narrower than rounding, to the nearer end. Only a sliver of portfolios
  next to the end reaches such a target, and the solvers resolve it no better than rounding.
- A turnover cap that the solver's arithmetic breaks down short of, on extreme inputs,
  returns `CutupError::NotConverged`, as documented, rather than an error such as
  `NonFiniteCovariance` from the breakdown.
- The method docs, with each method's algorithm, errors and examples, are on the method
  values, such as `MinVariance` and `Hrp`, rather than on the deprecated methods.
- The reference, integration and unit tests run through `allocate`. A property test checks
  that each deprecated method, which now builds a method value from its config, returns
  what that value returns when built with its own setters, defaults included.

### Deprecated

- Every `PortfolioAllocator::*_allocation*` method, `PortfolioAllocator::hrp_cluster_order`
  and `run_portfolio_allocation`, with the config structs only they take:
  `MeanVarianceConfig`, `MeanVarianceObjective`, `MaxSharpeConfig`, `RiskParityConfig`,
  `HrpConfig` and `HercConfig`. Use `PortfolioAllocator::allocate` with the method values
  instead. Each deprecation note names the replacement, and each method still returns
  exactly what its replacement does. They'll be removed in 0.3.0. `CovarianceConfig` and
  `WeightBounds` stay, since the method values take them.

## [0.2.7] - 2026-10-28

### Added

- The `Allocation` trait and `PortfolioAllocator::allocate`, one entry point for every
  method. Each method is a value built from `default()` and chained settings:
  `EqualWeight`, `InverseVolatility`, `InverseVariance`, `MinVariance`, `MeanVariance`,
  `MaxSharpe`, `MaxDiversification`, `RiskParity`, `Hrp` and `Herc`, as in
  `allocator.allocate(&MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.3)))`.
  `MinVariance`, `MeanVariance` and `MaxSharpe` are long-only unless `.allow_short()` is
  called. `MeanVariance` has no default, since a sensible risk aversion depends on the
  period of the returns: start from `MeanVariance::risk_aversion` or
  `MeanVariance::target_return`. Expected returns and risk budgets take plain `Vec<f64>`s or
  arrays. Each value gives exactly the weights of the `PortfolioAllocator` method it wraps,
  and those methods are unchanged. `Allocation` is implemented for `&A` and `Box<A>`, so a
  `Vec<Box<dyn Allocation>>` runs without `.as_ref()`.
- `From<Vec<f64>> for Weights`, so you can implement `Allocation` for methods of your own.
  `allocate` checks there's one weight per asset (`CutupError::DimensionMismatch`
  otherwise) and that each is finite (the new `CutupError::NonFiniteWeight` otherwise).
- The `basic` and `compare` examples use `allocate`.
- A property test that every method value gives the same weights or error as the method it
  wraps, including through the `shrinkage` and `regularization` shortcuts, and integration
  tests that mix methods, including one defined outside the crate, in a
  `Vec<Box<dyn Allocation>>`, and that `allocate` rejects malformed weights.
- Black-Litterman expected returns: `BlackLitterman::new(market_weights, delta)` gives the
  market-implied prior `π = δΣw_mkt + r_f` (`prior_returns`), and `expected_returns` blends
  it with views, `π + τΣPᵀ(τPΣPᵀ + Ω)⁻¹(Q − Pπ)`. Views are `View::absolute`,
  `View::relative` and `View::portfolio`. Each view's variance defaults to `τpᵀΣp` (He and
  Litterman), or is set with `.variance(ω)`, or as an Idzorek confidence from 0 to 1 with
  `.confidence(c)`. `τ` defaults to 0.05, and the covariance takes the same shrinkage and
  regularization setters as the allocation methods. Market capitalizations are normalized
  to weights. The result is a `Vec<f64>` for `MeanVariance` or `MaxSharpe`.
- Rebalancing from current weights, on `MinVariance` and on `MeanVariance` with a risk
  aversion: `.current_weights(w₀)` with `.max_turnover(T)`, capping `Σ|wᵢ − w₀ᵢ|` at `T`, and
  `.turnover_penalty(κ)`, adding `κ‖w − w₀‖²`. `MeanVariance` also takes
  `.transaction_costs(c)` and `.transaction_costs_per_asset([..])`, maximizing the expected
  return net of `Σcᵢ|wᵢ − w₀ᵢ|`. Min variance doesn't take costs, which are in units of
  return; mean-variance with zero expected returns gives min variance with costs. All work
  long-only, within bounds, or long-short. The active-set solver now cuts each weight's
  range at its current weight, where costs make the objective kinked, so the results are
  exact; a cap is solved through its multiplier, a cost added to every asset, found by
  bisection.
- `CutupError::MissingCurrentWeights`, `CutupError::InfeasibleTurnover` for a cap below the
  turnover needed to move the current weights into the bounds, and
  `CutupError::NotSupported`, returned for trading settings with a target return.
- Rebalancing reference cases, from support enumeration over five states per asset and
  cross-checked against PyPortfolioOpt on cvxpy to 1e-6, and property tests: rebalanced
  weights satisfy the KKT conditions with costs, caps hold and bind, settings that cost
  nothing change nothing, and higher costs never trade more.
- `CutupError::DegenerateViews`, returned when certain views repeat or contradict each
  other, or nearly do, or when a view with the default variance or a confidence is on a
  portfolio with no variance, which would make it certain.
- Black-Litterman reference cases against PyPortfolioOpt's `BlackLittermanModel`, to 1e-15,
  and property tests: mean-variance on the implied returns, at the same risk aversion,
  recovers the market weights, and certain views hold exactly.

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

[Unreleased]: https://github.com/thinkjrs/cutup/compare/0.3.9...HEAD
[0.3.9]: https://github.com/thinkjrs/cutup/compare/0.3.8...0.3.9
[0.3.8]: https://github.com/thinkjrs/cutup/compare/0.3.7...0.3.8
[0.3.7]: https://github.com/thinkjrs/cutup/compare/0.3.6...0.3.7
[0.3.6]: https://github.com/thinkjrs/cutup/compare/0.3.5...0.3.6
[0.3.5]: https://github.com/thinkjrs/cutup/compare/0.3.4...0.3.5
[0.3.4]: https://github.com/thinkjrs/cutup/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/thinkjrs/cutup/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/thinkjrs/cutup/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/thinkjrs/cutup/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/thinkjrs/cutup/compare/0.2.8...0.3.0
[0.2.8]: https://github.com/thinkjrs/cutup/compare/0.2.7...0.2.8
[0.2.7]: https://github.com/thinkjrs/cutup/compare/0.2.6...0.2.7
[0.2.6]: https://github.com/thinkjrs/cutup/compare/0.2.5...0.2.6
[0.2.5]: https://github.com/thinkjrs/cutup/compare/0.2.4...0.2.5
[0.2.4]: https://github.com/thinkjrs/cutup/compare/0.2.3...0.2.4
[0.2.3]: https://github.com/thinkjrs/cutup/compare/0.2.2...0.2.3
[0.2.2]: https://github.com/thinkjrs/cutup/compare/0.2.1...0.2.2
[0.2.1]: https://github.com/thinkjrs/cutup/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/thinkjrs/cutup/compare/0.1.4...0.2.0
[0.1.4]: https://github.com/thinkjrs/cutup/releases/tag/0.1.4
