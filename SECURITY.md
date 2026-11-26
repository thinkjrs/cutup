# Security

## Reporting a vulnerability

Please report it privately, not in a public issue: [open a draft security advisory](https://github.com/thinkjrs/cutup/security/advisories/new), which is the "Report a vulnerability" button under the repository's Security tab. Say which version of cutup, the input that does it, as code or as JSON, and what happens.

A fix is released as a new patch version, and the advisory is published with it.

## Supported versions

Fixes go into the latest release. Older releases aren't patched.

## What counts

cutup computes on what it's given. It opens no files, makes no network requests and starts no processes. So a vulnerability is something input can do to the program that calls it, through the documented API or the JSON the `serde` feature reads:

- **A panic or an abort,** from any data, setting or JSON. Each should be a `CutupError`, or an error from the reader.
- **Memory or time out of proportion to the input,** such as a few bytes that reserve gigabytes.

A wrong result isn't a vulnerability in that sense. It's a bug, and worth an issue.

## What cutup doesn't limit

cutup has no cap on the size of a problem, and its time and memory grow with it: about as the cube of the assets for the solver methods. The README's [size limits](README.md#size-limits) has the timings. Where the data or the settings come from someone you don't trust, cap these before cutup sees them:

- **The JSON's length.** A request is read whole before any of it is checked. A covariance matrix given in one is checked as it's read, in time that grows as the cube of its side.
- **Assets and periods,** which set the size of every solve.
- **Groups in the bounds.** Each is a row of every solve.
- **Frontier points and rolling windows.** Each is a solve, or several.

The sample app does this in [`apps/campaigns/engine`](apps/campaigns/engine/src/lib.rs), over data of its own: at most 16 KiB of JSON, 32 groups, 100 frontier points and 100 rolling windows.

## How it's kept

- **No `unsafe`:** the crate is `#![forbid(unsafe_code)]`.
- **Fuzzing:** three targets build data, weights and every method with its settings from arbitrary bytes, and write and read their JSON, weekly in CI and before each release. [`docs/FUZZING.md`](docs/FUZZING.md) says what each checks.
- **Advisories:** cargo-deny checks the dependencies' advisories, licenses and sources on every PR, and weekly on `main`. [`deny.toml`](deny.toml) has the one advisory that's ignored, and why.

## The dependencies' `unsafe`

cutup's own code has none, and its dependencies do. These are the crates in a build, with the lines of code that have the `unsafe` keyword, counted in the versions in `Cargo.lock` at 0.3.13. It's a size, not an audit.

Without features, cutup depends on nalgebra alone:

| Crate | Lines with `unsafe` | What it is |
| --- | --- | --- |
| nalgebra | 460 | The matrices |
| matrixmultiply, rawpointer | 95 and 20 | The matrix product |
| simba | 21 | The number traits nalgebra is generic over |
| wide, safe_arch, bytemuck | 312, 859 and 323 | SIMD number types under simba. cutup's matrices are of `f64` |
| num-complex, num-traits | 2 and 1 | Number traits |
| approx, num-integer, num-rational, typenum | 0 | |
| paste | 0 | A macro, run at build time |

The features add:

- **`serde`:** serde and serde_core, with 2 lines each, and serde_derive, run at build time, with syn, quote, proc-macro2 and unicode-ident under it.
- **`ts-rs`:** ts-rs and thiserror, with none, and their macros, run at build time. On Windows those macros bring in winapi-util and windows-sys, which are bindings to the system's API.
