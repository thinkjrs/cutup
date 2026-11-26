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
constructors are tested too, and sets a test timeout of at least 60 s. It also
leaves out the return-value replacements for `lp::scaled` and
`methods::correlation`, which return tuples of matrices: cargo-mutants tries
every combination of its guesses for each matrix, 1,100 mutants in all, and
none of them compiles.

The property tests draw random cases, so a mutant that only some cases catch
is caught in one run and missed in the next. To compare two runs, give both
the same seed, as CI does:

```bash
PROPTEST_RNG_SEED=20261126 cargo mutants -j 4
```

The whole crate has 4,182 mutants and takes about 2 hours on GitHub's eight
jobs. A mutant that makes a solver loop forever times out, which counts as
caught.
Run it on an otherwise idle machine: on a busy one, caught and missed mutants
alike run past the timeout, and some mutants allocate without bound. On Linux,
`ulimit -v 8388608` first caps each process at 8 GiB of address space, so that
such a mutant fails its allocation, which counts as caught, rather than taking
the machine down. macOS doesn't enforce that limit.

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

Full runs, on the code as it was at each:

| Outcome | 0.3.2 | 0.3.12 | 0.3.13 |
| --- | --- | --- | --- |
| Caught | 1,389 | 2,889 | 3,173 |
| Timed out | 19 | 34 | 37 |
| Missed | 187 | 142 | 49 |
| Unviable: doesn't compile | 563 | 1,995 | 923 |

- **0.3.2** was the first, after the tests added with this setup. Before them
  it had 104 missed and 209 timed out, most of which were slow tests on a busy
  machine under cargo-mutants' default timeout of 20 s, and missed at 60 s.
- **0.3.12** counts the 1,100 mutants that are now left out among its
  unviable ones, and its property tests weren't seeded: about 20 of its 142
  were in files that hadn't changed since 0.3.11's run, which caught them.
- **0.3.13** is seeded. A run before its tests, on the same code, missed 143.
  Tests for 95 of those brought it to 49, and one of the 49 was caught by a
  test added after the run, which leaves the 48 below.

## Known survivors

48 of the 3,259 viable mutants survive, so 98.5% are caught. None of them can
change a result, and each is one of these kinds:

- **An exact boundary: 21.** `<` for `<=`, or `>` for `>=`, where the two
  differ only when a value lands exactly on the other side, and it either
  can't or nothing changes when it does. A sum of bounds of exactly 1 is
  divided by 1. A change of exactly 0 in the simplex method never counts. A
  noise threshold in `max_sharpe` is never 0. Risk parity compares a power of
  two with 1e-10, which isn't one.
- **The path, not the result: 10.** Which of two valid cuts `MinCdar` adds
  in a round, and how risk parity scales its start, accepts a step length or
  gives up on a line search. Each changes how the solver reaches its optimum,
  or how many rounds that takes, and not the optimum: risk parity's is
  unique. They move a result only past its documented limit, budgets more
  than about 1e14 apart, where rounding already decides it.
- **A value that can't occur, or one nothing reads: 13.** Spare slots in the
  clustering's buffers, and distances it rewrites for clusters it has
  retired. A `sign` in the simplex method that's only ever 1 or −1, and its
  artificial variables, already 0 where one would be set to 0. A multiplier
  in the active-set solver that's still its initial 0 where it's read.
- **An invariance: 2.** `max_sharpe` negates `Σ⁺1` and `Σ⁺(μ − r_f)`
  together, and every use is a ratio of the two, an absolute value, or
  normalized by its own sum. Risk parity starts from `1/√v` of a correlation
  matrix's diagonal, which is 1.
- **A guard another check covers: 2.** `risk_budgets` and `expected_returns`
  of the wrong length are rejected when they're resolved, before the length
  check in the method itself.

The kinds that 0.3.2 listed here and that tests now catch: the size of a
tolerance, which variables the simplex method starts at an upper bound, when
it switches to Bland's rule, how the active-set solver counts a step as having
moved, iteration caps, a tie in the gap statistic, and the scale
`MaxDiversification` divides its volatilities by.

## Lines no test runs

Coverage, from `cargo llvm-cov`, is 98.7% of lines. Outside the test modules
57 lines aren't run by any test, and each is one of these:

- **A solver that doesn't converge: 12.** The `NotConverged` returns in
  `lp.rs`, `qp.rs`, `dual.rs`, `cdar.rs`, `turnover.rs` and `risk_parity.rs`,
  behind iteration caps that no test reaches.
- **Arithmetic that isn't finite, or a system that's singular: 15.** The
  `SingularCovariance`, `NonFiniteCovariance` and `DegenerateViews` returns in
  `qp.rs`, `max_sharpe.rs`, `black_litterman.rs` and `methods/mod.rs`, and
  the fallbacks in `eigen.rs` for an eigendecomposition that fails or turns
  up a NaN. They guard inputs at the edge of what `f64` holds.
- **A guard another check covers: 9.** The two length checks above.
- **An error that can't be passed on: 7.** A `?` on a call whose errors the
  caller has already ruled out, in `mean_variance.rs`, `min_variance.rs`,
  `methods/mod.rs` and `turnover.rs`.
- **An `unreachable!` arm: 4.** In `cluster.rs`, `mean_variance.rs` and
  `qp.rs`, for states the code before them rules out.
- **A branch of a search: 10.** Two steps of the target-volatility search
  that its tests' frontiers don't take, a return range's fall-through, and
  the closing lines of two branches that always match.
