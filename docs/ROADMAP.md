# cutup release plan: 0.3.x to 1.0.0

Status: draft · Last updated: 2026-10-29 · Current release: 0.2.8

This plans the releases from 0.3.0 to 1.0.0. The history before 0.2.8, and the
first version of the plan for these releases, are in `docs/PLAN.md`.

## Release rules

- **Breaking changes go in 0.3.0 only.**
  - Before 1.0, Cargo treats 0.3.1 as compatible with 0.3.0, so anyone who
    depends on `cutup = "0.3"` gets every 0.3.x on their next
    `cargo update`.
  - A breaking change in 0.3.1 or later would break their build.
  - Every release after 0.3.0 is additive, and none changes a default.
- **One theme per release.** Removal, cleanup and features ship separately,
  so each CHANGELOG entry has one theme and each release is easy to bisect.
- **1.0.0 is a freeze, not a feature release.** It's tagged when 0.3.x has
  gone a few releases without needing a breaking change. The API at that point
  becomes 1.0.0.
- **Every release** passes the full suite on stable and on the MSRV, clippy
  and `cargo doc` with `-D warnings`, the reference drift checks, and three
  property runs at `PROPTEST_CASES=50000`.

## Releases

| Release | Theme | Breaking |
| --- | --- | --- |
| 0.3.0 | Remove the deprecated API, make every breaking change planned before 1.0, add `.covariance(..)` to `Hrp` and `Herc`, and move to MSRV 1.85 with edition 2024 | Yes |
| 0.3.1 | Split `lib.rs` into modules | No |
| 0.3.2 | Tidying and tooling: tests, docs, packaging, CI checks, benchmarks | No |
| 0.3.3 | Portfolio statistics and `Weights::clean` | No |
| 0.3.4 | Covariance input and estimators | No |
| 0.3.5 | Efficient frontier and target volatility | No |
| 0.3.6 | Solver work: costs with a target return, LP speed | No |
| 0.3.7 | Group constraints | No |
| 0.3.8 | Rolling allocation and the `serde` feature | No |
| 0.3.9 | Sample campaign allocation app, native and WebAssembly | No |
| Later 0.3.x or 1.x | Mean absolute deviation and CDaR, NCO, examples and the selection guide | No |
| 1.0.0 | API freeze | No |

This order is agreed. A release can still split or merge if it grows, as long
as each keeps one theme.

## 0.3.0: remove the deprecated API and make the breaking changes

### Removed

- `src/deprecated.rs` and its `pub use` in `lib.rs`:
  - the 21 `PortfolioAllocator::*_allocation*` methods;
  - `PortfolioAllocator::hrp_cluster_order`;
  - `run_portfolio_allocation`;
  - `MeanVarianceConfig`, `MeanVarianceObjective`, `MaxSharpeConfig`,
    `RiskParityConfig`, `HrpConfig` and `HercConfig`.
- **The tests that exist only for them:**
  - the `deprecated_methods_match_their_replacements` property;
  - the `run_portfolio_allocation` tests;
  - the `hrp_cluster_order` test;
  - the `compile_fail` doctests in `deprecated.rs`.
- **README:** the two upgrade tables from 0.2.8 move into the "Upgrading from
  0.2" section described under PRs below, and link to the 0.2.8 docs on
  docs.rs, which keep the deprecation notes.
- **CHANGELOG:** a `### Removed` section.

### Breaking changes

1. **`CovarianceConfig` setters.** `with_shrinkage` and `with_regularization`
   become `shrinkage` and `regularization`, matching every other setter, and
   its fields become private.
2. **`InverseVolatility` and `InverseVariance` lose `#[non_exhaustive]`.**
   Then `&InverseVolatility` works the way `&EqualWeight` does. Removing the
   attribute isn't breaking in itself, but it belongs with this cleanup.
3. **`Weights::to_hashmap` is removed.** It's left over from when results
   were `HashMap`s; `iter()` and `into_vec()` cover the same ground.

### Carried over from the earlier plan

`docs/PLAN.md` planned these for 0.3.0 too, and they're agreed. Each is
breaking, so 0.3.0 is the only release before 2.0 that can make it.

4. **`PortfolioAllocator` becomes `Returns`.** It holds the data, not a
   method, now that methods are values.
   - `PortfolioAllocator` stays as a deprecated alias through 0.3.x, so
     upgrading to 0.3.0 doesn't also force a rename.
   - The returns constructor is `Returns::new`, since `Returns::from_returns`
     repeats itself, with `new_flat` and, with the `nalgebra` feature,
     `from_matrix`. `from_returns` stays as a deprecated name for `new`; the
     flat and matrix forms were never released under their old names.
   - Both deprecated names live in `src/deprecated.rs`. 1.0.0 deletes the
     file, the one removal in 1.0.0, and each deprecation note says so.
5. **Plain Rust inputs; nalgebra becomes internal.**
   - Constructors take `&[f64]` in row-major order with the asset count, or
     `&[Vec<f64>]` rows, instead of a `DMatrix`.
   - An optional `nalgebra` feature adds constructors from `DMatrix<f64>` and
     a conversion from `Weights` to `DVector<f64>`, documented as tracking
     nalgebra's version. Vectors going in need no feature: the setters take
     `impl Into<Vec<f64>>`, which `v.as_slice()` satisfies, and the orphan
     rule rules out a `From<DVector<f64>> for Vec<f64>` in cutup.
   - Then a nalgebra major release never forces a cutup major after 1.0.
6. **Asset labels.**
   - `Returns::labeled([..])` names the columns, without `with_`, like every
     other setter. Without labels, they're named `"0"`, `"1"`, …, so there's
     one code path.
   - `Weights` carries the labels, with `weight("SPY")`, `weights["SPY"]`,
     `labels()` and `labeled()`, which iterates over `(label, weight)` pairs.
     The sample app returns weights keyed by campaign name.
   - `iter()`, indexing and `as_slice()` still go by column, as for a slice,
     so labels were added without changing `Weights`' iteration type.
   - Every setting that names an asset takes an `Asset`, a column or a
     label: `View::absolute` and `View::relative`, and
     `WeightBounds::asset(..)` for one asset's bounds. Every setting with a
     value per asset takes `PerAsset` values, in column order or as pairs of
     an asset and its value: bounds, market weights, view portfolios, current
     weights, expected returns, risk budgets and transaction costs.
   - New errors: `UnknownAsset` and `DuplicateAsset`.
7. **Standard traits on every public type.**
   - `Debug`, `Clone` and `PartialEq` everywhere, including `Returns`, `Default`
     on every method value with a sensible default, and `Copy` where it's free,
     such as `Linkage`. `MeanVariance`, `BlackLitterman` and `View` have no
     `Default`: they need an objective, market weights or a view.
   - `Send + Sync`, asserted by a compile-time test in `tests/integration.rs`.
   - `Display` for `Weights`, as a labeled table, and `IntoIterator` by value.
   - Checked against the Rust API guidelines: C-COMMON-TRAITS, C-GETTER,
     C-CONV, C-ITER, C-SEND-SYNC and C-GOOD-ERR.
8. **Settled `CutupError` payloads.**
   - Each variant's fields are fixed before 0.3.0. Errors about one asset at
     allocation time name it by column and label: `ZeroVariance`,
     `NonFiniteWeight` and the new `InvalidAssetValue`. `DimensionMismatch`
     names its setting, and `UnknownAsset` carries the `Asset` as given.
     `InfeasibleBounds` stays as sums, since it's about every asset at once,
     and `NonFiniteValue` and `NonPositivePrice` keep their row and column,
     since they're found before any labels exist.
   - The enum stays `#[non_exhaustive]`, so 1.x can add variants; changing a
     payload after 0.3.0 would be breaking. Variants aren't
     `#[non_exhaustive]`, so an `Allocation` of your own can build them.

### Added in this release

9. **`.covariance(..)` on `Hrp` and `Herc`**, with the `shrinkage` and
   `regularization` shortcuts every other covariance-based method has.
   - **Breaking:** `Hrp` and `Herc` lose `Copy` and `Eq`, since a
     `CovarianceConfig` holds `f64`s. They keep `Debug`, `Clone`,
     `PartialEq` and `Default`.
   - **What it changes:** only the risk split, meaning the cluster variances
     and the inverse-variance weights within clusters. The hierarchy, and so
     `Hrp::cluster_order`, stays on the sample correlations. That's what
     riskfolio-lib (`method_cov="shrunk"`) does. PyPortfolioOpt's `HRPOpt`
     ignores `cov_matrix` when it's given the returns. Shrinking toward a scaled
     identity also pulls the correlations toward zero, so it could reshape the
     tree.
   - **Constant assets** still return `ZeroVariance`, even with shrinkage:
     their correlation distance is undefined whatever the covariance.
   - **Reference cases:**
     - HRP against riskfolio-lib's `HCPortfolio.optimization(model="HRP",
       rm="MV", method_cov="shrunk", dict_cov={"alpha": λ},
       leaf_order=False)`. `leaf_order=False` matches the tree's plain leaf
       order, which cutup and PyPortfolioOpt use; riskfolio's default applies
       SciPy's optimal leaf ordering.
     - Cross-checked against PyPortfolioOpt's `HRPOpt` steps, its linkage,
       `_get_quasi_diag` and `_raw_hrp_allocation`, run on the sample
       correlations and the adjusted covariance.
     - HERC against riskfolio-lib with the same shrunk covariance.
     - Regularization, which riskfolio has no option for, through its
       `custom_cov` with the adjusted matrix computed in NumPy.
     - sklearn's `ShrunkCovariance`, which riskfolio uses, divides by `n`
       rather than `n − 1`. HRP and HERC weights don't change when the
       covariance is scaled, so shrinkage still matches; regularization
       depends on scale, so its cases pass the `n − 1` matrix.
     - Three λ values on four existing datasets, including the singular and
       ill-conditioned ones, with single linkage, plus a Ward HRP case, fixed
       numbers of HERC clusters, and two ε values, alone and with shrinkage.
   - **Tests:** shrinkage changes the weights but not the cluster order, full
     shrinkage matches clustering with equal variances, invalid settings
     return `InvalidConfig`, and the HRP and HERC properties run with random
     shrinkage.

10. **MSRV 1.85 and edition 2024**, up from 1.78 and edition 2021.
    - **Why now:** raising the MSRV is breaking for anyone on an older
      compiler, so 0.3.0 is the only release before 1.0 that can do it. The
      MSRV set here holds through 0.3.x and into 1.0.
    - **What 1.85 brings:**
      - edition 2024, the current edition, which 1.85 is the first to support.
        Let chains (`if let Some(x) = a && x > 0`) need it too, but they're
        stable only from 1.88, so they stay off-limits while the MSRV is 1.85;
      - `#[expect(lint)]` (1.81), which warns once the lint it silences stops
        firing, in place of `#[allow]`;
      - the MSRV-aware dependency resolver (1.84), which picks dependency
        versions that build on the declared MSRV.
    - **Why 1.85 and not newer:** Debian 13 (trixie) ships rustc 1.85, so
      users who build with their distribution's compiler are covered. It's
      also well over a year old, past the six-month minimum lag of the MSRV
      policy.
    - **No dependency forces it:** the highest requirement among them is
      `matrixmultiply`'s 1.75.
    - **The work:**
      - `rust-version = "1.85.0"` and `edition = "2024"` in `Cargo.toml`;
      - the CI matrix moves from 1.78.0 to 1.85.0;
      - the step in `tests.yml` that regenerated `Cargo.lock` for the 1.78 job
        if it was in the version 4 format goes. 1.78 can't read version 4 and
        1.85 can. The committed lockfile is version 3, so the step never ran.
        The edition 2024 resolver doesn't change a committed lockfile; it
        picks versions that build on the MSRV only when the lockfile is
        regenerated or updated, so the 1.85 job is what catches a locked
        version that needs a newer Rust;
      - `cargo fix --edition` for the mechanical changes, in a commit of
        their own so the review can skip it;
      - the README's MSRV, and a CHANGELOG entry.

Done already in 0.2.7 and 0.2.8: methods as values, setters without `with_`,
long-only by default with `.allow_short()`, and the deprecation of the method
API.

### PRs

In this order, since later ones build on earlier ones:

- **PR A:** the removal.
- **PR B:** changes 1 to 3: the `CovarianceConfig` setters, the unit
  structs, and `Weights::to_hashmap`.
- **PR C:** change 10, MSRV 1.85 and edition 2024, so every later PR is
  written against the new edition.
- **PR D:** change 9, `.covariance(..)` on `Hrp` and `Herc`, after PR B has
  renamed the `CovarianceConfig` setters it builds on.
- **PR E:** change 5, plain inputs, so the rename and labels are built on the
  new constructors.
- **PR F:** change 4, `Returns` and the deprecated alias.
- **PR G:** change 6, asset labels.
- **PR H:** changes 7 and 8, the traits and error payloads, as the last pass
  over the public API.

The README gets an "Upgrading from 0.2" section covering every change above,
the removed methods, `with_` setters, `Returns`, plain inputs, labels and the
MSRV, and
the CHANGELOG a table from each 0.2 form to its 0.3 form.

## 0.3.1: split `lib.rs`

`lib.rs` is 4,400 lines. Each method's code lives there while its type lives in
`allocation.rs`. Once 0.3.0 has removed the deprecated items, the split is pure
moves, with no logic or API changes, so the review checks only that.

- **Shared types:** `error.rs` (`CutupError`), `weights.rs` (`Weights`),
  `covariance.rs` (`CovarianceConfig` and the adjusted covariance), and
  `bounds.rs` (`WeightBounds`, `ReturnRange` and `interior_start`).
- **Methods:** one module per method under `methods/`, holding its type, its
  private solver code and its unit tests:
  - `equal_weight`, `inverse` (volatility and variance), `min_variance`,
    `mean_variance`, `max_sharpe`, `max_diversification`, `risk_parity`,
    `hrp` and `herc`;
  - `cvar`, `black_litterman` and `turnover` stay where they are.
- **`allocation.rs`** keeps the `Allocation` trait and its impls for `&A` and
  `Box<A>`.
- **`lib.rs`** keeps the crate docs, the module list and the re-exports.

## 0.3.2: tidying and tooling

No API changes.

### Tidying

- **Conversion leftovers:** the conversion to `allocate` left variables named
  `config` that hold method values, and clones that aren't needed. The tests
  named after the old methods were renamed in 0.3.0.
- **`docs/PLAN.md`:** fold anything still open into this document, and keep
  the rest as history or remove it.
- **`Cargo.toml`:** a description and keywords that fit a portfolio
  allocation crate.

### Tooling

- **Lint jobs in CI:** clippy and `cargo doc` with `-D warnings` on stable
  and the MSRV, and `cargo fmt --check` on stable. The release rule above is
  checked by hand until then. Beta and nightly stay test-only, so a new lint
  there doesn't fail a PR.
- **`#![forbid(unsafe_code)]`.** cutup has no `unsafe` code; this makes that
  a guarantee.
- **`cargo-semver-checks` in CI** on every PR to `main`, so 0.3.x stays
  additive. Moved here from the 1.0.0 list.
- **WebAssembly as a supported target:**
  - a CI build for `wasm32-unknown-unknown`, which already succeeds with no
    changes as of 0.2.8;
  - the test suite run under wasm with `wasm-bindgen-test`, so anything that
    breaks only on wasm shows up here rather than in the sample app.
- **Benchmarks** with `criterion`, for the active-set solver, the LP, the
  clustering methods and risk parity, at 10, 100 and 500 assets. Moved here
  from the 1.0.0 list so 0.3.6's solver work is measured before and after.
- **Mutation testing** with `cargo-mutants`, which reports code changes that
  no test notices: weak assertions that coverage doesn't show. Run before each
  release, with a CI job if it's fast enough.
- **Fuzzing** of the constructors and setters with `cargo-fuzz`, to show no
  input panics. The `MinCvar` clamp panic from the #23 review is the kind of
  bug it finds.
- **`cargo-deny`** for dependency licenses and security advisories.

## 0.3.3: portfolio statistics and `Weights::clean`

- **`Weights::stats(&data)`:** expected return, volatility, Sharpe ratio,
  diversification ratio, each asset's risk contribution, and CVaR at a given
  confidence. The methods already compute most of these internally; today
  users recompute them.
- **`Weights::clean(cutoff)`:** sets weights below `cutoff` to zero and
  renormalizes, like PyPortfolioOpt's `clean_weights`, so results don't carry
  weights like 0.0003.

## 0.3.4: covariance input and estimators

- **Covariance input:** a way to give a method a covariance estimate rather
  than only adjustments to the sample covariance.
- **Covariance estimators** on that input:
  - Ledoit-Wolf, which estimates the shrinkage intensity from the data
    rather than taking a fixed λ, toward the identity and constant-correlation
    targets. Moved here from the 1.x candidates.
  - an exponentially weighted (EWMA) covariance;
  - semicovariance, for downside risk.
- **The Black-Litterman posterior covariance**,
  `Σ_BL = Σ + τΣ − τΣPᵀ(τPΣPᵀ + Ω)⁻¹PτΣ` ("note 2", deferred since 0.2.7).
- **Expected-return estimators:** an exponentially weighted mean, CAPM
  implied returns, and James-Stein shrinkage.
- **Log returns**, as an option on the data type. Simple returns stay the
  default.

## 0.3.5: efficient frontier and target volatility

- **`MeanVariance::frontier(n)`:** `n` portfolios along the efficient
  frontier, from the minimum variance portfolio to the highest reachable
  return, within the bounds.
- **`MeanVariance::target_volatility(σ)`:** the highest expected return at a
  volatility of `σ`, PyPortfolioOpt's `efficient_risk`, found by searching
  over the risk aversion.

## 0.3.6: solver work

- **Costs with a target return:** transaction costs as a net-return target,
  `μᵀw − Σcᵢ|wᵢ − w₀ᵢ| ≥ ρ`, so `MeanVariance::target_return(..)` takes
  `.transaction_costs(..)` instead of returning `NotSupported`.
- **LP speed:** LU updates in the simplex solver rather than refactoring the
  basis every iteration, and computing the reduced-cost magnitudes in the same
  pass over the columns. Measured with the 0.3.2 benchmarks.
- **Factorization reuse** in the active-set solver, if the benchmarks show it
  matters.

## 0.3.7: group constraints

- **Caps and floors on groups of assets**, such as sectors or ad channels:
  "Search campaigns at most 40%", "Video at least 20%". Groups are given by
  index, or by label.
- **Solver work:** the active-set solver handles equality constraints and
  bounds, but not general inequalities `Aw ≤ b`. Supporting them is the bulk
  of this release. Moved here from the 1.x candidates.
- **Which methods:** the QP methods first, `MinVariance` and `MeanVariance`,
  then `MinCvar` through the LP.

## 0.3.8: rolling allocation and `serde`

- **Rolling allocation:** weights recomputed over rolling windows of the
  data, for example `data.rolling(60).allocate(&method)`, giving weights over
  time. Pure computation, not backtesting.
- **An optional `serde` feature** for `Weights`, the method values and
  `CutupError`. Moved here from the 1.0.0 list, since the sample app's
  backends pass JSON.

## 0.3.9: sample campaign allocation app

A Next.js app that allocates a budget across a basket of ad campaigns. The
same Rust allocation code runs either on Vercel's servers as a native binary or
in the visitor's browser as WebAssembly, and the frontend has a toggle to pick
which one. Its data, and the modeling choices that depend on it, are settled
when this release starts.

### Layout

```
apps/campaigns/
  engine/   shared Rust code: JSON request in, JSON response out
  api/      allocate.rs: the native vercel-rust handler
  wasm/     the wasm-bindgen wrapper
  app/      the Next.js frontend
```

- **`engine/`** holds everything both backends share:
  - the request and response types;
  - turning campaign series into a returns matrix;
  - picking the method and its settings, calling cutup, and computing the
    statistics.

  Its one entry point is `engine::allocate(request) -> Result<Response, Error>`.
  Both backends are thin wrappers around it, so they can't diverge.
- **`api/allocate.rs`:** the same setup as `thinkjrs/vercel-rust-runtime`. A
  native Rust binary, deployed as a Vercel serverless function by the
  `vercel-rust` runtime that `vercel.json` names. It parses the JSON body,
  calls the engine and returns JSON.
  - The example's `allocate.rs` targets cutup 0.1.4 and its `"mvo"`
    strategy; this one uses the method values.
- **`wasm/`:** `#[wasm_bindgen] fn allocate(request: JsValue)` around the
  same engine, built with `wasm-pack --target web`.
- **Dependencies:** each crate takes cutup as a path dependency, so the app
  always builds against the current code. The crate's `include` list already
  keeps `apps/` out of the published package.
- **Local development:** run `vercel dev` after `cargo build --release`.
  `npm run dev` serves only the frontend.

### Frontend engines

- **One interface, two implementations:**
  `Engine { allocate(request): Promise<Response> }`.
  - **Server (native Rust):** `fetch('/api/allocate')`. This is the default.
  - **Browser (WebAssembly):** the wasm module, run in a Web Worker so a large
    allocation doesn't block the page, and loaded only when it's first
    selected.
- **The toggle** has exactly these two choices, with no "auto". The choice is
  remembered per visitor.
- **Each result** shows which engine produced it and how long it took. The
  server time includes the network round trip, and the browser time includes
  the module's first load.
- **Shared types:** the TypeScript request and response types are generated
  from the engine's Rust types with `ts-rs`, so there's one definition.

### Parity between the engines

- A CI test runs the same request fixtures through the native engine and
  through wasm, with `wasm-pack test --node`, and compares the outputs.
- Basic arithmetic and `sqrt` give identical results on both. `ln`, which
  risk parity uses, comes from a different math library on wasm and may differ
  in the last bit. So the test compares to within about 1e-14 rather than
  exactly.

### Deployment

- **The risk:** the `vercel-rust` runtime installs Rust for the functions,
  but not necessarily for the Next.js build step that runs `wasm-pack`.
- **First try:** install the toolchain in the build command. Builds are
  slower, but everything is built from source.
- **Fallback:** build the wasm package in CI and commit it, or publish it as
  an artifact the build downloads.
- **When:** a test deploy of both engines early in this release, before the
  UI is built on top.
- **Bundle size:** cutup and nalgebra compile to a few hundred KB of wasm.
  Building with `opt-level = "z"` and `wasm-opt` shrinks it, and lazy loading
  keeps it off the page for visitors who stay on the server engine.

### Model

- **Assets:** each campaign is an asset, and the weights are shares of the
  budget, keyed by campaign name.
- **Series:** each campaign gets a per-period yield series built from its
  clicks, views and impressions. Which metric, and whether spend is included,
  is settled when the data arrives.
- **Data:** an anonymized sample of the campaign data, committed to the repo.
- **Settings that fit:**
  - `WeightBounds` for a minimum and maximum share per campaign;
  - group constraints (0.3.7) for channel caps;
  - rebalancing from the current budget split, with `max_turnover` limiting
    how much budget moves at once;
  - `MinCvar` against the worst periods;
  - Black-Litterman for a marketer's own views on campaigns;
  - risk parity, HRP, min variance and max Sharpe compared side by side.
- **Caveat, stated in the app:** these methods assume results scale linearly
  with budget, so saturation isn't modeled.

### UI, tests and CI

- **UI:**
  - load the sample, then pick the metric, method, constraints and engine;
  - see each campaign's metrics over time, the budget weights with their
    statistics (0.3.3), the efficient frontier (0.3.5), and the weights over
    time (0.3.8);
  - a table compares methods.
- **Tests:** unit tests for the engine and the handler, the parity test, and
  a TypeScript typecheck.
- **CI:** a job that builds the native functions and the wasm package, and
  runs `next build`.
- **The crate's README** links to the app.

## Later 0.3.x or 1.x

- **Mean absolute deviation and CDaR** (conditional drawdown at risk). Both
  reduce to linear programs, so they reuse the `MinCvar` approach.
- **Nested Clustered Optimization** (López de Prado): any method inside and
  across the clusters HRP and HERC already build.
- **Examples:** one per topic in `examples/`, covering rebalancing with
  costs, Black-Litterman views, `MinCvar`, bounds and group constraints.
- **A method-selection guide** in the crate docs: which method fits which
  question, which use expected returns, and which are long-only.
- **A "verified against" table** in the README: the reference suite checks
  against PyPortfolioOpt, riskfolio-lib and SciPy, which is currently noted
  only in the Testing section.

## 1.0.0: API freeze

Worked through during 0.3.x. Nothing new lands in 1.0.0 itself.
`cargo-semver-checks` and the benchmarks moved to 0.3.2, and `serde` to
0.3.8.

- **An MSRV policy in the README:**
  - the MSRV is 1.85 from 0.3.0, and stays there through 0.3.x and 1.0.0;
  - after 1.0, an MSRV bump is a minor release, never a patch;
  - the MSRV stays at least six months behind the latest stable Rust.
- **Documented size limits**, from the benchmarks: how many assets and
  scenarios each method handles in practice.
- **Docs:**
  - the method-selection guide, if it hasn't landed in 0.3.x;
  - an asserted doctest on every public item.
- **Removal:** `src/deprecated.rs`, with the `PortfolioAllocator` alias and
  `Returns::from_returns`.
- **Audit:** `cargo public-api` diffed against the last 0.3.x before
  tagging.

## After 1.0.0

- **1.0.x patches:** bug and doc fixes only.
  - A fix that changes the weights for some input is allowed only when the old
    weights were wrong, and the CHANGELOG says which inputs are affected.
- **1.x minors:** additive changes, meaning new methods, estimators, setters,
  features and error variants, plus MSRV bumps.
- **Deprecations:**
  - a deprecated item stays for at least one minor release, with a note naming
    its replacement;
  - it's removed only in 2.0.
- **Candidates for 1.x minors:** whatever of "Later 0.3.x or 1.x" hasn't
  shipped, plus discrete allocation (whole shares or spend increments),
  resampled and robust optimization, and fixed or tiered transaction costs.
  Cardinality limits ("at most k assets") need a mixed-integer solver and stay
  out of scope.

## Open decisions

- **The sample app's data and model:** settled when 0.3.9 starts.
