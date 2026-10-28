# Reference implementations

`tests/reference.rs` checks every cutup allocation method against independent Python
implementations on fixed inputs. This directory holds the Python side. It isn't part of
the published crate, and `cargo test` never needs Python: it reads the committed
fixtures in `tests/fixtures/`.

| File | Purpose |
| --- | --- |
| `inputs.py` | Builds the input datasets and freezes them to `tests/fixtures/datasets.json`. |
| `generate.py` | Computes the expected weights for every case and writes `tests/fixtures/expected.json`. |
| `fixtures.py` | Shared paths and JSON formatting. |
| `pyproject.toml`, `uv.lock` | Pinned Python, NumPy, pandas, SciPy, PyPortfolioOpt and riskfolio-lib. |

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
| HRP | PyPortfolioOpt `HRPOpt`, for each linkage |
| HERC | riskfolio-lib `HCPortfolio.optimization(model="HERC", rm="MV")`, for each linkage and for fixed numbers of clusters. Datasets with two assets are skipped, because riskfolio-lib's gap statistic can't choose a number of clusters for them. |
| Rebalancing (turnover and costs) | Support enumeration with five states per asset: at its lower bound, at its upper bound, at its current weight `w₀ᵢ`, or free above or below it, where the cost `cᵢ|wᵢ − w₀ᵢ|` is the linear term `±cᵢ`. With a turnover cap, each support also gets a candidate with the cap as an equality. The best feasible candidate is the optimum. The objective is written from each method's definition, not cutup's. Each result is cross-checked against PyPortfolioOpt, with an L1 cost, a squared-distance objective and an L1 constraint on cvxpy with Clarabel at tight tolerances, to 1e-6. A cap below the turnover needed to move `w₀` into the bounds is an expected error. |
| Black-Litterman | PyPortfolioOpt `market_implied_prior_returns` and `BlackLittermanModel.bl_returns`, with absolute, relative and portfolio views, and its `"default"` (He-Litterman) and `"idzorek"` view uncertainties. Cases that mix kinds of uncertainty pass a diagonal Ω built row by row with `BlackLittermanModel.default_omega` and `idzorek_method`. These cases hold expected returns rather than weights, checked to 1e-15. |

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

# Rewrite the expected weights after adding or changing a case
uv run --project reference python reference/generate.py

# Rebuild the input datasets (only when adding or changing a dataset)
uv run --project reference python reference/inputs.py
```

Then run `cargo test --test reference` to check cutup against the new fixtures.

## Adding a method

1. Add a reference implementation and its cases to `generate.py`. Prefer a well-known
   library or a closed form over reimplementing cutup's algorithm.
2. Map the method name to the cutup call in `run` in `tests/reference.rs`, and add it
   to the list in `every_method_and_dataset_is_covered`.
3. Regenerate the fixtures and commit them with the code.
