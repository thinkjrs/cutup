# Fuzzing

`fuzz/` holds [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) targets for
the public API. Each feeds arbitrary input to cutup, and any input may return an
error, but none may panic.

| Target | What it covers |
| --- | --- |
| `returns` | Every `Returns` constructor on arbitrary shapes and floats, including NaN, infinities, subnormals and huge values, then labeling, `cvar` and label lookups |
| `allocate` | Every allocation method with arbitrary settings, including bounds, current weights, turnover, costs, risk budgets, Black-Litterman views, covariance settings and clustering. Weights it returns must be finite, one per asset, and sum to 1 |
| `weights` | `Weights`, `PerAsset` and `Asset` from arbitrary values and labels: lookups by column and label, iteration and `Display`, which must agree |

## Running

The fuzzer needs nightly Rust:

```bash
cargo install cargo-fuzz
cd fuzz
cargo +nightly fuzz run allocate -- -max_total_time=600
```

A crash writes its input to `fuzz/artifacts/<target>/`.
`cargo +nightly fuzz fmt <target> <file>` prints it as the target's input type,
and `cargo +nightly fuzz run <target> <file>` replays it.

`fuzz/` is its own workspace, so the crate's `cargo test`, clippy and `fmt`
don't build it. Its `Cargo.lock` is committed.

## CI

`.github/workflows/fuzz.yml` builds every target on each PR, and runs `fmt` and
clippy on `fuzz/` with stable Rust. The fuzzer itself runs weekly, five minutes
a target, and uploads any crashing input, so a new finding never fails an
unrelated PR. GitHub runs scheduled workflows from `main` only.

## Findings

Each finding gets a regression test in `src` and a CHANGELOG entry.

- **`allocate`:** long-short `MeanVariance::target_return` on one asset, with
  returns near 1e126 and a target near 1e299, returned a weight of about
  8.9e172. The KKT solver's consistency check squared residuals near 1e299,
  which overflowed, and `∞ > ∞` is false, so the check passed.
- **`allocate`:** long-short `MaxSharpe` with a shrinkage of 1.9e-314, on two
  periods with two constant assets, returned all-zero weights. The shrinkage left
  those assets with variances near 1e-308, so the unnormalized weights were
  huge, their sum overflowed, and dividing by it gave zeros. The same overflow
  made the minimum-variance return NaN, so the risk-free rate check passed when
  it should have returned `RiskFreeRateTooHigh`.
- **`allocate`:** long-short `MeanVariance::risk_aversion(2.6e207)` with a
  regularization of 3.9e-308, on six assets and three periods of returns near 1e8
  to 1e12, returned weights summing to about −7.6e75. One asset was constant, so
  its regularized variance was some 1e332 times smaller than the others'. The KKT
  solver checked its solution after rescaling each asset to unit variance, where
  that asset's rounding is tiny, and converting back multiplied it by about 1e50.
  The budget is now also checked in the original units.

Fixing that one also turned up an overflow the fuzzer hadn't reached: long-short
`MaxSharpe` compared norms of `1/σ` that overflow below a volatility of about
1e-154, which skipped its check for a fully invested zero-variance portfolio.
