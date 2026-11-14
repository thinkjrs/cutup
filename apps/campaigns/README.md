# Campaign allocation app

A sample app for [cutup](../../README.md) that splits an ad budget across platforms. The same Rust allocation code runs either on Vercel's servers, as native serverless functions, or in the visitor's browser, as WebAssembly. None of it is published with the crate.

## Layout

```
apps/campaigns/
  Cargo.toml   the native functions' package, and the workspace
  api/         allocate.rs and platforms.rs: the functions, one [[bin]] each
  engine/      shared Rust: a JSON request in, a JSON reply out
  wasm/        the engine for the browser, with wasm-bindgen
  data/        the sample: six platforms' daily totals
  fixtures/    requests and their replies, for the parity check
  vercel.json  the functions' runtime, vercel-rust
```

## The model

Each ad platform is an asset, and the weights are shares of the budget.
- **Returns:** for each metric, impressions, clicks, link clicks or views, a platform's return on a day is what a unit of spend bought there. Each is divided by its average over the request's platforms and days.
- **One allocation per metric:** every method allocates once for each metric.
- **Days:** only days on which every chosen platform spent count.
- **Saturation isn't modeled:** these methods take results to grow in proportion to spend.

The sample has six platforms' daily spend, impressions, clicks, link clicks and views, from 2025-09-01 to 2026-09-30. Every volume is scaled by the same constant, so each yield per unit of spend is the real one and the absolute amounts aren't.

## Requests and replies

`POST /api/allocate` and the wasm `allocate` take the same JSON request:

```json
{
  "platforms": ["Meta", "Google", "TikTok"],
  "from": "2026-01-01",
  "metrics": ["clicks", "link_clicks"],
  "method": {"name": "min_variance", "bounds": {"upper": 0.6}},
  "rolling": {"window": 60, "step": 7}
}
```

- **Defaults:** everything but `method` is optional. The platforms default to Meta, Google and TikTok, the three that spent on every day, and the metrics to all four.
- **Methods:** a method is cutup's own JSON with a `name`, such as `{"name": "mean_variance", "risk_aversion": 5.0}`. A frontier, `"frontier": 10`, takes a mean-variance method.
- **Replies:** a response, with each metric's weights and statistics or its own error, comes with status 200. A request that isn't JSON for one is 400, and one that can't be allocated is 422. Both come as `{"error": ..}`.

`GET /api/platforms`, and the wasm `platforms`, list the sample's platforms, each with its first and last dates and how many days it spent on.

## Development

From `apps/campaigns`:

```bash
# Format, lint and test everything natively
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# The engine's and the wrapper's tests on wasm, the parity check included
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test --target wasm32-unknown-unknown -p campaign-engine -p campaign-wasm

# Print each metric's allocation for a few methods
cargo run -p campaign-engine --example sample
```

- **Parity:** `fixtures/requests.json` holds requests, and `fixtures/replies.json` their replies. The engine's tests check every reply natively and on wasm, to 1e-14, since `ln` can differ in the last bit between the two. After a change that moves the replies, regenerate them natively with `UPDATE_FIXTURES=1 cargo test -p campaign-engine parity`.
- **Builds:**
  - The native functions build with `cargo build --release`.
  - The browser package builds with the `wasm-release` profile, which optimizes for size, then the `wasm-bindgen` CLI at the version in `Cargo.lock`. It comes to about 300 KB of wasm gzipped, with the sample in it:

    ```bash
    cargo build --profile wasm-release --target wasm32-unknown-unknown -p campaign-wasm
    wasm-bindgen --target web --out-dir wasm/pkg \
      target/wasm32-unknown-unknown/wasm-release/campaign_wasm.wasm
    ```
  - The native functions need Rust 1.88 for `vercel_runtime`. The engine and the wrapper keep cutup's 1.85.
- **Running locally:** `vercel dev` runs the functions.
