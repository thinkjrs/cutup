# Reference implementations

`tests/reference.rs` checks every cutup allocation method against independent Python
implementations on fixed inputs. This directory holds the Python side. It isn't part of
the published crate, and `cargo test` never needs Python: it reads the committed
fixtures in `tests/fixtures/`.

| File | Purpose |
| --- | --- |
| `inputs.py` | Builds the input datasets and freezes them to `tests/fixtures/datasets.json`. |
| `generate.py` | Computes the expected weights for every case and writes `tests/fixtures/expected.json`. |
| `lp.py` | Solves random linear programs with `scipy.optimize.linprog` (HiGHS) and writes `tests/fixtures/lp.json`, which the unit tests in `src/lp.rs` check cutup's simplex solver against: the outcome (optimal, infeasible or unbounded) and the optimal objective, to 1e-9. |
| `stats.py` | Computes the portfolio statistics of fixed weights on every dataset, also under three covariance configs, and writes `tests/fixtures/stats.json`, which `tests/reference.rs` checks `Returns::stats` and `Returns::stats_with` against. |
| `covariance.py` | Computes the Ledoit-Wolf intensities and shrunk covariances, the exponentially weighted covariances, the semicovariances and Black-Litterman posterior covariances on fixed datasets and writes `tests/fixtures/covariance.json`, which the unit tests in `src/covariance.rs` and `src/black_litterman.rs` check `CovarianceConfig`'s estimates and `BlackLitterman::posterior_covariance` against, to 1e-12 of each covariance's largest entry. |
| `expected_returns.py` | Computes the expected-return estimates on every dataset and writes `tests/fixtures/expected_returns.json`, which `tests/reference.rs` checks `Returns`' estimators against, to 1e-13 of each dataset's largest return, looser in proportion to the condition number for Bayes-Stein. |
| `log_returns.py` | Computes each dataset's log returns and writes `tests/fixtures/log_returns.json`, which the unit tests in `src/returns.rs` check `Returns::log_returns` against, to 4ε of each log return. |
| `fixtures.py` | Shared paths and JSON formatting. |
| `pyproject.toml`, `uv.lock` | Pinned Python, NumPy, pandas, SciPy, PyPortfolioOpt and riskfolio-lib, and scikit-learn through them. |

## References

| Method | Reference |
| --- | --- |
| Equal weight | `1 / n` |
| Inverse volatility, inverse variance | `1 / σ` and `1 / σ²` from `np.var(ddof=1)`, normalized |
| Minimum variance | NumPy closed form `Σ⁻¹1 / 1ᵀΣ⁻¹1`; minimum-norm KKT solution from `numpy.linalg.lstsq` when Σ is singular |
| Long-only minimum variance and mean-variance | Support enumeration: the KKT solution on every subset of assets with the rest held at zero, keeping the best feasible one with no negative weights. Each subset's system is solved through the Schur complement of the covariance block, which stays accurate when the constraints' entries are many orders of magnitude apart. Unlike cutup's active-set solver, it never checks multipliers' signs or iterates. Singular covariances are used only with shrinkage, since their optimum needn't be unique. A target return outside the assets' expected returns is an expected error. With weight bounds, each asset is held at its floor, at its cap, or left free; the range of reachable target returns comes from `scipy.optimize.linprog`, bounds that can't sum to 1 are an expected error, and each result is cross-checked against PyPortfolioOpt's `EfficientFrontier` with the same `weight_bounds` to 1e-5, except where Σ is too badly scaled for cvxpy. |
| Mean-variance | NumPy closed forms for risk aversion and target return (Merton, 1972); minimum-norm KKT solution when Σ is singular. Expected errors are decided from the null space of Σ. |
| Long-only max Sharpe | The same support enumeration, after the change of variables `y = w / (μ − r_f)ᵀw` that makes it minimizing `yᵀΣy` subject to `(μ − r_f)ᵀy = 1` and `y ≥ 0`. An unbounded Sharpe ratio is decided by `scipy.optimize.linprog` finding a long-only portfolio with zero variance and positive excess return. |
| Max diversification | The same support enumeration and linear program as long-only max Sharpe, with the volatilities in place of the excess returns. An unbounded ratio is the expected `SingularCovariance`. |
| Max Sharpe | NumPy closed form `Σ⁻¹(μ − r_f) / 1ᵀΣ⁻¹(μ − r_f)`, or `numpy.linalg.pinv` when Σ is singular. Expected errors are decided from the minimum variance portfolio's own variance and return, and the null space of Σ. |
| Risk parity | Cyclical coordinate descent (Griveau-Billion et al., 2013) on the correlation matrix, run to convergence, unlike cutup's Newton iteration, with equal and custom risk budgets. The expected error is decided by `scipy.optimize.nnls` finding a long-only portfolio with zero variance. |
| HRP | PyPortfolioOpt `HRPOpt`, for each linkage. With shrinkage or regularization, riskfolio-lib `HCPortfolio.optimization(model="HRP", rm="MV", leaf_order=False)`, cross-checked against `HRPOpt`'s clustering and bisection run on the adjusted covariance, since `HRPOpt` ignores `cov_matrix` when given the returns. |
| HERC | riskfolio-lib `HCPortfolio.optimization(model="HERC", rm="MV", leaf_order=False)`, for each linkage, for fixed numbers of clusters, and with shrinkage or regularization. Datasets with two assets are skipped, because riskfolio-lib's gap statistic can't choose a number of clusters for them. |
| HRP and HERC covariance | Shrinkage alone through riskfolio-lib's `method_cov="shrunk"`, sklearn's `ShrunkCovariance`. It divides by `n` rather than `n − 1`, which HRP and HERC weights don't depend on. Regularization depends on scale, so it goes through `custom_cov` with the `n − 1` matrix adjusted in NumPy. |
| Rebalancing (turnover and costs) | Support enumeration with five states per asset: at its lower bound, at its upper bound, at its current weight `w₀ᵢ`, or free above or below it, where the cost `cᵢ|wᵢ − w₀ᵢ|` is the linear term `±cᵢ`. With a turnover cap, each support also gets a candidate with the cap as an equality. The best feasible candidate is the optimum. The objective is written from each method's definition, not cutup's. Each result is cross-checked against PyPortfolioOpt, with an L1 cost, a squared-distance objective and an L1 constraint on cvxpy with Clarabel at tight tolerances, to 1e-6. A cap below the turnover needed to move `w₀` into the bounds is an expected error. With a target return, mean-variance is min variance with `μᵀw = ρ` as a second equality in each support, the least turnover that reaches the target comes from `scipy.optimize.linprog`, and the cross-check adds the same equality to PyPortfolioOpt's `min_volatility`. |
| Min CVaR | `scipy.optimize.linprog` (HiGHS) on the primal Rockafellar-Uryasev program, with a constraint per scenario, independent of the dual cutup solves. The optimum needn't be unique, so the test compares CVaR: HiGHS stops within about 1e-10, so cutup's must be no worse than the reference weights' CVaR, and no more than 1e-9 better. Each case without a target is cross-checked against PyPortfolioOpt's `EfficientCVaR` and, without bounds, riskfolio-lib's `rm="CVaR"`, by CVaR, to 1e-6. |
| Portfolio statistics | Closed forms on NumPy's sample mean and covariance (`np.cov`, `ddof=1`), independent of cutup's computation from the portfolio's return in each period: `μᵀw`, `sqrt(wᵀΣw)`, the Sharpe ratio at two risk-free rates, `σᵀw / sqrt(wᵀΣw)`, each asset's share of the variance, and CVaR at three confidences, written out from its definition. Each case is cross-checked against PyPortfolioOpt's `portfolio_performance` to 1e-12, riskfolio-lib's `Risk_Contribution` with `rm="MV"`, which uses central differences, to 1e-6, and its `CVaR_Hist` to 1e-12. Weights are equal, long-short summing to 1, and unnormalized, plus a riskless hedge on `collinear2`. The test compares them to about 1e-12. |
| Portfolio statistics under a covariance | The same weights under a shrinkage of 0.2, the exponentially weighted covariance at a half-life of 5 and a given `(S + diag(S)) / 2`: the volatility, Sharpe ratios, diversification ratio and risk shares from their closed forms on that matrix, cross-checked against PyPortfolioOpt's `portfolio_performance` to 1e-12 and riskfolio-lib's `Risk_Contribution` to 1e-6. A variance within `10nε` of `(Σᵢ|wᵢ|σᵢ)²` counts as riskless, as cutup counts it. The test checks the expected return and CVaR are `Returns::stats`'s. |
| Ledoit-Wolf shrinkage | Toward a constant variance, scikit-learn's `ledoit_wolf_shrinkage`, with the shrunk covariance checked against its `ledoit_wolf` to 1e-12. Toward a constant correlation, the intensity written out from Ledoit and Wolf's `covCor.m`, checked against PyPortfolioOpt's `_ledoit_wolf_constant_correlation` given the `1/T` sample covariance the paper uses, to 1e-12: on its own it uses the `ddof=1` one with `1/T` moments, which moves the intensity by O(1/T). Both shrink NumPy's `ddof=1` covariance, as cutup does, so each case is `T / (T − 1)` times the papers' estimate. Where the constant correlation target is the sample covariance but for rounding, as it always is for two assets, the intensity is null and only the covariance is compared. |
| Exponentially weighted covariance | Written out from its definition, at half-lives of 1, 5 and 20: weights `λ^(T−1−t)` with `λ = 2^(−1/h)`, the weighted mean, and `Σwₜ(rₜ − m)(rₜ − m)ᵀ / (Σwₜ − Σwₜ² / Σwₜ)`. Cross-checked against pandas' `ewm(halflife=h).cov()` and riskfolio-lib's `covar_matrix(method="ewma1", d=λ)` to 1e-12. |
| Semicovariance | Below each asset's mean over `T − 1`, cross-checked against riskfolio-lib's `covar_matrix(method="semi")`, and below 0 over `T`, against PyPortfolioOpt's `semicovariance(benchmark=0)`, to 1e-12. |
| Black-Litterman | PyPortfolioOpt `market_implied_prior_returns` and `BlackLittermanModel.bl_returns`, with absolute, relative and portfolio views, and its `"default"` (He-Litterman) and `"idzorek"` view uncertainties. Cases that mix kinds of uncertainty pass a diagonal Ω built row by row with `BlackLittermanModel.default_omega` and `idzorek_method`. These cases hold expected returns rather than weights, checked to 1e-15. |
| Black-Litterman posterior covariance | `Σ + τΣ − τΣPᵀ(τPΣPᵀ + Ω)⁻¹PτΣ` written out in NumPy, on four datasets with equal market weights: default views at `τ = 1/T`, an explicit variance, a confidence and a certain view at `τ = 0.05`, and no views. Cross-checked against PyPortfolioOpt's `BlackLittermanModel.bl_cov` to 1e-12, and the default views against riskfolio-lib's `black_litterman`, which fixes `τ` at `1/T`. riskfolio-lib inverts `τΣ`, so it's checked to 1e-15 times Σ's condition number where that's looser than 1e-12, and not on `scales4`, whose condition number is about 1e16. |
| Exponentially weighted mean | Written out from its definition, at half-lives of 1, 5 and 20: weights `λ^(T−1−t)` with `λ = 2^(−1/h)` and `Σwₜrₜ / Σwₜ`. Cross-checked against pandas' `ewm(halflife=h).mean()`, riskfolio-lib's `mean_vector(method="ewma1", d=λ)` and PyPortfolioOpt's `ema_historical_return` with `span = 2 / (1 − λ) − 1`, per period and without compounding, to 1e-12 of the dataset's largest return. |
| CAPM returns | `r_f + βᵢ(E[r_m] − r_f)` written out, with `r_m` the market portfolio's return each period and `βᵢ` from the `ddof=1` covariance of each asset with it, for equal market weights at a rate of 0 and weights `n, n − 1, …, 1` at `1e-4`. Cross-checked against PyPortfolioOpt's `capm_return`, per period and without compounding, with its default equal-weighted market or given the market's returns, to 1e-12 of the dataset's largest return. A market with no variance by cutup's rule for `Returns::stats` is an expected error. |
| James-Stein and Bayes-Stein means | Written out in NumPy: James-Stein `(1 − α)μ + α·μ̄1` with `α = (tr Σ − 2λ_max) / (T‖μ − μ̄1‖²)` clamped to `[0, 1]`, checked against riskfolio-lib's `mean_vector(method="JS")` where its unclamped intensity is in `[0, 1]`; Bayes-Stein `(1 − α)μ + α·μ₀1` with `μ₀ = 1ᵀΣ⁻¹μ / 1ᵀΣ⁻¹1` and `α = (n + 2) / (n + 2 + T(μ − μ₀1)ᵀΣ⁻¹(μ − μ₀1))` in correlation units, checked against riskfolio-lib's `mean_vector(method="BS", target="b2")`, which inverts Σ, to 1e-12 of the largest return or 1e-15 times Σ's condition number, and not above 1e10. A correlation matrix with an eigenvalue at most `nε` of its largest is an expected error. The test's Bayes-Stein tolerance is 1e-16 times the correlation matrix's condition number where that's looser than 1e-13. |
| Log returns | NumPy's `log1p` of the simple returns, from pandas' `pct_change` for prices. Cross-checked against PyPortfolioOpt's `returns_from_prices(log_returns=True)`, given the prices or, for a dataset of returns, the prices that compound them from 1, and against `ln(pₜ / pₜ₋₁)`, to 1e-12 of the dataset's largest log return: both take the log of `1 + r` or of the ratio, which loses the digits of a small return that `log1p` keeps. |

Every covariance-based reference works in correlation units `u = diag(σ)w`, as cutup
does: with `Σ = diag(σ) C diag(σ)`, linear solves, pseudo-inverses, null spaces and
minimum norms are those of the correlation matrix `C`. Then nothing depends on each
asset's units, and the `scales4` dataset, with volatilities from about 1e-2 to 1e-10,
checks that every method stays accurate.

Each case's tolerance is relative to the size of the weight. Ill-conditioned
correlation matrices get a looser tolerance in proportion to their condition number, which
is recorded in the case. Numerically singular ones keep the tight tolerance, because both
sides truncate the null space and agree on the minimum-norm solution.

`--check` fails if a weight moves by more than a tenth of its tolerance (a NaN counts as
moved), if a case's shape or expected error changes, or if a tolerance changes by more
than a factor of two.

## Commands

Requires [uv](https://docs.astral.sh/uv/). Run from the repository root.

```bash
# Check that the committed fixtures still match the reference (what CI runs)
uv run --project reference python reference/generate.py --check
uv run --project reference python reference/lp.py --check
uv run --project reference python reference/stats.py --check
uv run --project reference python reference/covariance.py --check
uv run --project reference python reference/expected_returns.py --check

# Rewrite the expected weights after adding or changing a case
uv run --project reference python reference/generate.py

# Rebuild the input datasets (only when adding or changing a dataset)
uv run --project reference python reference/inputs.py
```

Then run `cargo test --test reference`, and `cargo test --lib` for `lp.json` and
`covariance.json`, to check cutup against the new fixtures.

## Adding a method

1. Add a reference implementation and its cases to `generate.py`. Prefer a well-known
   library or a closed form over reimplementing cutup's algorithm.
2. Map the method name to the cutup call in `run` in `tests/reference.rs`, and add it
   to the list in `every_method_and_dataset_is_covered`.
3. Regenerate the fixtures and commit them with the code.
