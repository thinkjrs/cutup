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
| Long-only minimum variance and mean-variance | Support enumeration: the KKT solution on every subset of assets with the rest held at zero, keeping the best feasible one with no negative weights. Unlike cutup's active-set solver, it never checks multipliers' signs or iterates. Singular covariances are used only with shrinkage, since their optimum needn't be unique. A target return outside the assets' expected returns is an expected error. |
| Mean-variance | NumPy closed forms for risk aversion and target return (Merton, 1972); minimum-norm KKT solution when Σ is singular. Expected errors are decided from the null space of Σ. |
| Max Sharpe | NumPy closed form `Σ⁻¹(μ − r_f) / 1ᵀΣ⁻¹(μ − r_f)`, or `numpy.linalg.pinv` when Σ is singular. Expected errors are decided from the minimum variance portfolio's own variance and return, and the null space of Σ. |
| Risk parity | Cyclical coordinate descent (Griveau-Billion et al., 2013) run to convergence, unlike cutup's Newton iteration, with equal and custom risk budgets. The expected error is decided by `scipy.optimize.nnls` finding a long-only portfolio with zero variance. |
| HRP | PyPortfolioOpt `HRPOpt`, for each linkage |
| HERC | riskfolio-lib `HCPortfolio.optimization(model="HERC", rm="MV")`, for each linkage and for fixed numbers of clusters. Datasets with two assets are skipped, because riskfolio-lib's gap statistic can't choose a number of clusters for them. |

Each case's tolerance is relative to the size of the weight. Ill-conditioned
covariances get a looser tolerance in proportion to their condition number, which is
recorded in the case. Numerically singular ones keep the tight tolerance, because both
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
