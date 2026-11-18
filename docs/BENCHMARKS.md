# Benchmarks

`benches/allocate.rs` times `allocate` with [criterion](https://docs.rs/criterion)
for the methods that run a solver or a clustering step:

| Method | What it measures |
| --- | --- |
| `MinVariance` | The active-set QP solver, long-only |
| `MinVarianceGroups` | The same with two overlapping groups that bind: the first half of the assets at most 30% together, and every third asset at least 40% |
| `MinCvar` | The two-phase simplex LP |
| `MinCdar` | Cutting planes, each round a simplex LP with a row per cut |
| `Hrp` | Single-linkage clustering and recursive bisection |
| `Herc` | Single-linkage clustering, the gap statistic and equal risk contribution across clusters |
| `RiskParity` | Newton's method on `½yᵀΣy − Σᵢ bᵢ ln yᵢ` |

```bash
cargo bench
cargo bench -- MinVariance/100
```

## Data

The returns are generated from a fixed seed, so every run sees the same inputs:

- three common factors, each asset with its own exposures between 0.5 and 1.5;
- a daily drift between 0.02% and 0.08%;
- noise with a standard deviation between 0.5% and 2%;
- two periods per asset, which keeps the sample covariance well conditioned.

Each timing covers the whole of `allocate`, including the sample covariance.

## Baseline

Measured at 0.3.1 on an Apple M5 with Rust 1.92. Each time is criterion's
point estimate.

| Method | 10 assets | 50 assets | 100 assets | 500 assets |
| --- | --- | --- | --- | --- |
| `MinVariance` | 28 µs | | 35 ms | 18 s |
| `MinCvar` | 54 µs | 24 ms | 0.96 s | Not run |
| `Hrp` | 1.5 µs | | 62 µs | 1.4 ms |
| `Herc` | 2.2 µs | | 0.19 ms | 4.6 ms |
| `RiskParity` | 2.6 µs | | 0.24 ms | 20 ms |

- **`MinCvar`:** at 0.3.1 it was benchmarked at 10, 50 and 100 assets. At 200,
  its LP's phase 1 stalled on a degenerate vertex: every pivot left `x` where it
  was, until the iteration cap returned `NotConverged` after about 75 seconds.
- **`MinVariance`:** a solve takes about 500 times longer from 100 to 500
  assets, where the size alone would account for about 125 times.

## Changes since the baseline

### `MinCvar`'s starting point

The LP used to start every scenario weight at 0, which left its asset rows
without a residual: the degenerate start above. It now starts the scenarios
where the equal-weight portfolio loses most at their cap.

| `MinCvar` | 10 assets | 100 assets | 500 assets |
| --- | --- | --- | --- |
| 0.3.1 | 54 µs | 0.96 s | Not run |
| Now | 47 µs | 28 ms | 11 s |

### Groups

Measured on 2026-11-12, on the same machine, when `MinVarianceGroups` was added.
`MinVariance` is the same benchmark as above, after the solver work of 0.3.2 to
0.3.7.

| Method | 10 assets | 100 assets | 500 assets |
| --- | --- | --- | --- |
| `MinVariance` | 12 µs | 2.4 ms | 0.23 s |
| `MinVarianceGroups` | 20 µs | 0.32 ms | 7.1 ms |

From 100 assets it's faster than without groups: the binding groups leave the
solver fewer free weights to factor. A group that doesn't bind costs a slack
variable, which the solver leaves out of its factored solves.

### `MinCdar`

Measured on 2026-11-15, on the same machine, when `MinCdar` was added. Its full
linear program has rows for every period's peak and drawdown, so it grows with
the periods: at 1,000 periods it took 86 seconds with 5 assets and 96 with 20.
`MinCdar` solves it by cutting planes instead, and the table follows how they
got fast enough.

| `MinCdar` | 10 assets | 100 assets | 500 assets |
| --- | --- | --- | --- |
| Each round's program as `MinCvar`'s dual, a row per asset | 0.22 ms | 1.5 s | 602 s |
| As the primal, a row per cut | | | 10.8 s |
| Each cut halfway from the best so far | 59 µs | 0.15 s | 5.5 s |
| Sorting only the tail, with no temporaries | 29 µs | 77 ms | 2.8 s |
| Each round from the last's basis | 24 µs | 8.7 ms | 0.16 s |

- **The dual form** has a row per asset whatever the rounds, so at 500 assets
  each round took about 2 seconds, from the first.
- **The primal form** has a row per cut, so the early rounds are tiny, and the
  bounds are the weights' own rather than columns of the program. The rounds
  and their bounds were the same.
- **In-out stabilization** takes each cut halfway from the best portfolio so far
  to the program's minimizer, where Kelley's cuts at the minimizer alone
  zig-zag: about 110 rounds at 500 assets rather than 190.
- **Each cut** sorts only the worst drawdowns that can be in the tail, as
  `tail_loss` does for CVaR, rather than all `T`, and adds each one's gap into
  the cut in place rather than through temporary vectors.
- **Warm starts:** each round's program is the last's with a row more, for the
  new cut, and a slack basic in it, so the last basis is still optimal for the
  costs, if not feasible. The simplex solver starts from it, and a few dual
  pivots bring the new cut in, where each round had solved from scratch, from a
  basis of artificial variables. Anything the warm start can't do falls back to
  that, so it can only be slower, never wrong.

The 10.8 s row is from a test with other generated returns at that size, which
took 2.2 s with the stabilization, so the last two rows aren't measured alike.

## Running time

- **`cargo bench`:** about eight minutes, most of it `MinVariance`, `MinCvar`
  and `MinCdar` at 500 assets. Those sizes, where a solve takes over a tenth of a
  second, get criterion's minimum of 10 samples rather than its default of 100.
- **`cargo test --all-targets`:** runs each benchmark once, unoptimized, at the
  smallest size only, so CI checks that the benchmarks build and run without
  timing them. Timings on shared CI runners are too noisy to compare.
