# Benchmarks

`benches/allocate.rs` times `allocate` with [criterion](https://docs.rs/criterion)
for the methods that run a solver or a clustering step:

| Method | What it measures |
| --- | --- |
| `MinVariance` | The active-set QP solver, long-only |
| `MinCvar` | The two-phase simplex LP |
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

## Running time

- **`cargo bench`:** about seven minutes, most of it `MinVariance` and
  `MinCvar` at 500 assets. Those two sizes, where a solve takes over a tenth of
  a second, get criterion's minimum of 10 samples rather than its default of
  100.
- **`cargo test --all-targets`:** runs each benchmark once, unoptimized, at the
  smallest size only, so CI checks that the benchmarks build and run without
  timing them. Timings on shared CI runners are too noisy to compare.
