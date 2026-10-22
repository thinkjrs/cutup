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
| `pyproject.toml`, `uv.lock` | Pinned Python, NumPy, pandas, SciPy and PyPortfolioOpt. |

## References

| Method | Reference |
| --- | --- |
| Equal weight | `1 / n` |
| Inverse volatility, inverse variance | `1 / σ` and `1 / σ²` from `np.var(ddof=1)`, normalized |
| Minimum variance | NumPy closed form `Σ⁻¹1 / 1ᵀΣ⁻¹1`; minimum-norm KKT solution from `numpy.linalg.lstsq` when Σ is singular |
| Mean-variance | NumPy closed forms for risk aversion and target return (Merton, 1972); minimum-norm KKT solution when Σ is singular. Expected errors are decided from the null space of Σ. |
| HRP | PyPortfolioOpt `HRPOpt` |

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
