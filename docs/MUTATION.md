# Mutation testing

[cargo-mutants](https://mutants.rs) makes small changes to the code, one at a
time, such as `<` for `<=`, `*` for `/`, or a function body replaced with a
default value, and runs the tests on each. A change that no test notices is a
_missed_ mutant: either the tests don't pin that behavior, or the change can't
alter it. Coverage shows which lines run; this shows which lines the tests
actually check.

## Running

```bash
cargo install cargo-mutants
cargo mutants -j 4
```

Results go to `mutants.out/`, with the missed mutants in `missed.txt`.
`cargo mutants -f src/lp.rs` mutates one file, `--iterate` retests only the
mutants that weren't caught last time, and `--in-diff <file>` only the lines a
diff changes. `.cargo/mutants.toml` turns on every feature, so the `nalgebra`
constructors are tested too, and sets a test timeout of at least 60 s.

The whole crate has 3,213 mutants and takes about 2 hours at four jobs. A
mutant that makes a solver loop forever times out, which counts as caught.
Run it on an otherwise idle machine: on a busy one, caught and missed mutants
alike run past the timeout, and some mutants allocate without bound.

## CI

`.github/workflows/mutants.yml` runs on every PR, mutating only the lines the PR
changes, and fails if any of those mutants are missed. Either add a test that
catches each one, or explain in the PR why it can't change the result; the
kinds below are the usual reasons.

The whole crate runs weekly and by hand, split across eight jobs. It reports
the missed and timed-out mutants in each job's summary and uploads
`mutants.out`, but doesn't fail: the mutants below are known to survive.
GitHub runs scheduled workflows from `main` only.

## Baseline

At 0.3.2, before and after the tests added with this setup:

| Outcome | Before | After |
| --- | --- | --- |
| Caught | 1,284 | 1,389 |
| Timed out | 209 | 19 |
| Missed | 104 | 187 |
| Unviable: doesn't compile | 561 | 563 |

Most of the first run's timeouts weren't loops. With the 20 s timeout
cargo-mutants sets by default, they were slow tests on a busy machine, and
given 60 s, most of them turned out to be missed. The 19 left loop forever.

## Known survivors

187 of the 1,595 viable mutants survive, so 88% are caught. 127 of the
survivors are in the solvers, `qp.rs` and `lp.rs`. They fall into these kinds:

- **Boundaries of a tolerance.** `<` for `<=`, or `>` for `>=`, where a value
  is compared with a tolerance or a rounding threshold. The two differ only
  when the value lands exactly on it.
- **The size of a tolerance.** `*` for `/` or `+` in the arithmetic that scales
  a tolerance, such as `1e-9 * scale` or `noise / k_scale`. At the scales the
  tests use, both versions accept and reject the same steps.
- **The path, not the result.** Which variables the simplex method starts at
  an upper bound, when it switches to Bland's rule, how the turnover-cap search
  picks its next λ, and how the active-set solver counts a step as having
  moved. Each changes how the solver reaches the optimum, not which optimum.
- **Caps and sizes.** Iteration caps such as `1000 + 100(m + n)`, which the
  tests never reach, and buffer sizes such as `2n − 1`, which may grow.
- **Guards another check covers.** For example, `risk_budgets` and
  `expected_returns` of the wrong length are rejected before the length check
  in the method itself, and an exact tie in the gap statistic would need
  distances that SciPy and riskfolio-lib can't tell apart.
- **Invariances.** `MaxDiversification` divides the volatilities by the
  largest before maximizing a ratio that doesn't depend on their scale.

The tests added with this setup target the mutants that did change results:
merges checked against SciPy and a gap statistic checked against riskfolio-lib,
Jacobi rotations on their own, per-asset bounds and the rounding allowance,
views on nearly riskless portfolios, a single asset under `MinCvar`, and linear
programs that exercise the simplex method's bounds, its infeasibility
tolerance and its guard against a long step carrying a variable past a bound.
Two lines that no mutant could affect were removed: a simplex update the next
pivot recomputes, and a volatility check that the correlation check covers.
