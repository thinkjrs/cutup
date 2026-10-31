# Cutup: A Rust Portfolio Allocation Library

[![Crates.io](https://img.shields.io/crates/v/cutup)](https://crates.io/crates/cutup)
[![Build Status](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/thinkjrs/cutup/branch/main/graph/badge.svg)](https://codecov.io/gh/thinkjrs/cutup)
[![Docs.rs](https://docs.rs/cutup/badge.svg)](https://docs.rs/cutup)
[![License](https://img.shields.io/crates/l/cutup)](https://crates.io/crates/cutup)

Cutup is a Rust library for portfolio allocation strategies, providing implementations for various allocation methods. 

It's designed to be efficient and easy to use.

Here are some of the allocation methods included in this library:

- **Minimum Variance (GMV)**, unconstrained or long-only
- **Mean-Variance Optimization (MVO)**, unconstrained or long-only
- **Maximum Sharpe Ratio (tangency portfolio)**, unconstrained or long-only
- **Equal Weight Allocation (EW)**
- **Inverse Volatility and Inverse Variance**
- **Maximum Diversification**
- **Risk Parity (Equal Risk Contribution) and Risk Budgeting**
- **Hierarchical Risk Parity (HRP)**
- **Hierarchical Equal Risk Contribution (HERC)**
- **Minimum CVaR (expected shortfall)**, from historical scenarios
- **Black-Litterman expected returns**, to feed mean-variance or max Sharpe
- **Rebalancing** from current weights: turnover caps, turnover penalties and transaction costs
- **Asset labels**, such as tickers, on the data and the weights, and in every per-asset setting

Inputs are plain Rust slices, so you don't need a matrix library. cutup uses `nalgebra` internally, and its optional `nalgebra` feature accepts `DMatrix` input directly.

## Features

- **Minimum Variance**: The global minimum variance portfolio, using only the covariance matrix. Optional shrinkage and regularization via `CovarianceConfig`.
- **Long-Only Minimum Variance**: The lowest-variance portfolio with no short positions, solved exactly by a built-in active-set solver. Cap or floor each asset's weight with `WeightBounds`.
- **Mean-Variance Optimization**: The Markowitz portfolio for a given risk aversion or target return, using expected returns (the sample mean by default) and the covariance matrix.
- **Long-Only Mean-Variance**: The same objectives with no short positions, solved exactly by the active-set solver. Also takes `WeightBounds`.
- **Maximum Sharpe Ratio**: The tangency portfolio for a given risk-free rate, using expected returns and the covariance matrix.
- **Long-Only Maximum Sharpe Ratio**: The highest Sharpe ratio with no short positions, solved exactly by the active-set solver.
- **EW Allocation**: Assigns equal weights to all assets.
- **Inverse Volatility / Inverse Variance**: Weights each asset by `1 / σ` or `1 / σ²`, ignoring correlations. Simple risk-based weights that are always positive.
- **Maximum Diversification**: The long-only portfolio with the highest ratio of weighted average volatility to portfolio volatility, using only the covariance matrix.
- **Risk Parity**: Long-only weights where every asset contributes the same share of portfolio variance, accounting for correlations. Set unequal shares with `RiskParity::risk_budgets`.
- **HRP Allocation**: Uses hierarchical clustering and recursive bisection for risk-based allocation. Choose single, complete, average or Ward linkage with `Hrp::linkage`, and inspect the cluster order with `Hrp::cluster_order`.
- **HERC Allocation**: Cuts the same clustering into groups, chosen by a gap statistic or set with `Herc::clusters`, then splits risk down the tree and by inverse variance within each group.
- **Minimum CVaR**: The long-only portfolio with the lowest average loss over the worst `1 − β` of the historical return scenarios, solved exactly as a linear program by a built-in simplex solver. Takes weight bounds and a target return, and `cvar` measures any portfolio's CVaR.
- **Rebalancing**: From your current weights, cap the turnover of min variance and mean-variance, with a risk aversion or a target return, penalize it, or, for mean-variance, charge proportional transaction costs against the expected return. Solved exactly by the same active-set solver.
- **Black-Litterman**: Expected returns that blend the returns the market portfolio implies with your views on some assets, each with a variance or an Idzorek confidence. Pass them to `MeanVariance` or `MaxSharpe`.
- **Fully Unit-Tested**: Includes test cases for correctness verification.

## Choosing a method

Each method is a value you pass to `Returns::allocate`. Methods that can short are long-only unless you call `.allow_short()`.

| Method | Uses | Long-only | Notes |
| --- | --- | --- | --- |
| `EqualWeight` | nothing | ✅ | A baseline that needs no estimates and can't fail. |
| `InverseVolatility` | volatilities | ✅ | Riskier assets get less; correlations are ignored. |
| `InverseVariance` | variances | ✅ | A stronger tilt than inverse volatility. Equals min variance when assets are uncorrelated. |
| `MinVariance` | covariance | ✅ by default | The lowest-variance portfolio. Long-only, it usually concentrates in a few low-risk assets and holds none of the rest; `.bounds(..)` caps or floors each asset. Long-short, it's sensitive to estimation error, so consider `.shrinkage(..)`. From `.current_weights(..)`, `.max_turnover(..)` and `.turnover_penalty(..)` limit trading. |
| `MeanVariance` | covariance, expected returns | ✅ by default | Markowitz, built from `MeanVariance::risk_aversion(..)` or `MeanVariance::target_return(..)`. Most sensitive to the expected returns you supply. Long-only, a target return must lie between the lowest and highest expected return, or the lowest and highest return the bounds allow; `.bounds(..)` caps or floors each asset. With a risk aversion, `.transaction_costs(..)`, `.max_turnover(..)` and `.turnover_penalty(..)` rebalance from `.current_weights(..)`; with a target return, `.max_turnover(..)` does. |
| `MaxSharpe` | covariance, expected returns, risk-free rate | ✅ by default | The tangency portfolio. Long-only, it only needs some asset's expected return above the risk-free rate; long-short, the risk-free rate must be below the min variance portfolio's return. |
| `MinCvar` | return scenarios | ✅ | The lowest average loss in the worst `1 − β` of the scenarios, 5% by default. Looks at the losses themselves rather than a covariance, so it reacts to skew and fat tails; needs enough scenarios for a tail, at least 20 at 95%. `.bounds(..)` and `.target_return(..)` work as for mean-variance. |
| `MaxDiversification` | covariance | ✅ | The most diversified portfolio: the highest weighted average volatility per unit of portfolio volatility. Inverse volatility when assets are uncorrelated. Needs every long-only portfolio to have some variance, so consider shrinkage when data is short. |
| `RiskParity` | covariance | ✅ | Every asset contributes the same risk, or the shares set by `.risk_budgets(..)`. Diversifies more than inverse volatility when correlations differ. Needs every long-only portfolio to have some variance, so consider shrinkage when data is short. |
| `Hrp` | covariance, correlations | ✅ | Clusters correlated assets, then splits risk between clusters. Never inverts the covariance, so it's robust when data is short. |
| `Herc` | covariance, correlations | ✅ | Like HRP, but splits where the clustering tree branches and groups similar assets, so the result doesn't depend on column order. |

Long-only methods never return negative weights. The others can. To see every method on the same data:

```bash
cargo run --example compare
```

## Installation

Add `cutup` to your `Cargo.toml`:

```toml
[dependencies]
cutup = "0.3.0"
```

cutup 0.3 needs Rust 1.85 or newer. cutup 0.2 needs 1.78.

To pass nalgebra matrices in and get `DVector` weights out, turn on the `nalgebra` feature. It tracks the nalgebra version cutup depends on, currently 0.33:

```toml
[dependencies]
cutup = { version = "0.3.0", features = ["nalgebra"] }
```

## Usage

Pass prices to `Returns::from_prices`, one row per period and one value per asset in each row. Rows can be arrays, `Vec`s or slices. Prices are converted to simple returns, and every covariance-based method works on the covariance of those returns. If you already have returns, use `Returns::new` instead.

For data in one flat, row-major slice, such as a JavaScript `Float64Array` or a CSV read row by row, use `from_prices_flat(&values, assets)` or `new_flat(&values, assets)`. With the `nalgebra` feature, `from_prices_matrix` and `from_matrix` take a `DMatrix`.

Build a method from `default()`, or for mean-variance from its objective, and chained settings, and pass it to `allocate`. Any type that implements the `Allocation` trait works, including your own. The `PortfolioAllocator` methods from 0.2, such as `min_variance_allocation`, were removed in 0.3.0: see [Upgrading from 0.2](#upgrading-from-02).

Allocation methods return `Weights`: one weight per asset, in the same order as the columns of your input, and labeled with the assets: see [Labels](#labels). Bad input (no assets, too few rows, NaN or infinite values, non-positive prices) returns a `CutupError` instead of producing meaningless weights.

```rust
use cutup::{
    CutupError, EqualWeight, Hrp, InverseVolatility, MeanVariance, MinVariance,
    Returns, WeightBounds,
};

fn main() -> Result<(), CutupError> {
    let prices = [
        [125.0, 1500.0, 210.0, 600.0],
        [123.0, 1520.0, 215.0, 620.0],
        [130.0, 1510.0, 220.0, 610.0],
        [128.0, 1530.0, 225.0, 630.0],
        [131.0, 1525.0, 223.0, 640.0],
        [129.5, 1550.0, 228.0, 635.0],
        [133.0, 1545.0, 226.0, 650.0],
        [135.5, 1560.0, 231.0, 645.0],
        [134.0, 1580.0, 229.0, 660.0],
        [137.0, 1575.0, 234.0, 668.0],
        [136.0, 1600.0, 238.0, 662.0],
        [139.5, 1595.0, 236.0, 675.0],
        [141.0, 1615.0, 241.0, 680.0],
    ];

    let data = Returns::from_prices(&prices)?;

    let min_variance_weights = data.allocate(&MinVariance::default().allow_short())?;
    // No asset above 40%.
    let capped_weights =
        data.allocate(&MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.4)))?;
    // Risk aversion is in the units of your returns: per-period returns have small variances,
    // so meaningful values are large. Shrinking the sample covariance toward its average
    // variance reduces estimation error.
    let mean_variance_weights =
        data.allocate(&MeanVariance::risk_aversion(100.0).shrinkage(0.1))?;
    let ew_weights = data.allocate(&EqualWeight)?;
    let inverse_volatility_weights = data.allocate(&InverseVolatility)?;
    let hrp_weights = data.allocate(&Hrp::default())?;

    println!("Min Variance Weights: {:?}", min_variance_weights.as_slice());
    println!("Capped Min Variance Weights: {:?}", capped_weights.as_slice());
    println!("Mean-Variance Weights: {:?}", mean_variance_weights.as_slice());
    println!("EW Weights: {:?}", ew_weights.as_slice());
    println!("Inverse Volatility Weights: {:?}", inverse_volatility_weights.as_slice());
    println!("HRP Weights: {:?}", hrp_weights.as_slice());

    Ok(())
}
```

### Labels

Name the assets with `labeled`, for example with their tickers. The weights carry the labels, and every setting that takes an asset or a value per asset takes them too: bounds, Black-Litterman views and market weights, current weights, expected returns, risk budgets and transaction costs. Values per asset can be in column order, as before, or pairs of a label and a value, in any order, as an array, a `Vec` or a `HashMap`. Without labels, each asset's label is its column: `"0"`, `"1"`, and so on.

```rust
use cutup::{CutupError, MinVariance, Returns, WeightBounds};

fn main() -> Result<(), CutupError> {
    let returns = [
        [ 0.010, -0.020,  0.005],
        [-0.004,  0.015,  0.002],
        [ 0.007,  0.003, -0.006],
        [ 0.002, -0.001,  0.004],
        [-0.003,  0.006,  0.001],
    ];
    let data = Returns::new(&returns)?.labeled(["SPY", "TLT", "GLD"])?;

    let weights = data.allocate(
        &MinVariance::default()
            // No asset above 60%, and at most 10% in gold.
            .bounds(WeightBounds::uniform(0.0, 0.6).asset("GLD", 0.0, 0.1))
            // Current weights left out are 0.
            .current_weights([("SPY", 0.6), ("TLT", 0.4)])
            .max_turnover(0.2),
    )?;
    assert!(weights["GLD"] <= 0.1 + 1e-12);
    for (ticker, weight) in weights.labeled() {
        println!("{ticker}: {weight:.4}");
    }
    Ok(())
}
```

`println!("{weights}")` prints the same labels and weights as a table, with 4 decimals, or `{weights:.2}` for 2. `weights.iter()`, `weights[i]` and `as_slice()` still go by column. A label that isn't in the data returns `CutupError::UnknownAsset`, and one given twice `CutupError::DuplicateAsset`. Errors about one asset, such as `ZeroVariance`, give both its label and its column.

### Rebalancing

Give `MinVariance` or `MeanVariance` your current weights to rebalance from them. Turnover is the total weight bought and sold, `Σ|wᵢ − w₀ᵢ|`. Transaction costs are in units of return, so only mean-variance takes them; for min variance with costs, use mean-variance with zero expected returns.

```rust
use cutup::{CutupError, MeanVariance, MinVariance, Returns};

fn main() -> Result<(), CutupError> {
    let returns = [
        [ 0.010, -0.020,  0.005],
        [-0.004,  0.015,  0.002],
        [ 0.007,  0.003, -0.006],
        [ 0.002, -0.001,  0.004],
        [-0.003,  0.006,  0.001],
    ];
    let data = Returns::new(&returns)?;
    let current = [0.5, 0.3, 0.2];

    // The lowest-variance portfolio within 10% turnover of the current one.
    let capped = data.allocate(&MinVariance::default().current_weights(current).max_turnover(0.1))?;
    let turnover: f64 = capped.iter().zip(current).map(|(w, c)| (w - c).abs()).sum();
    assert!(turnover <= 0.1 + 1e-12);

    // Mean-variance net of 10 basis points a trade.
    let net = data.allocate(
        &MeanVariance::risk_aversion(100.0)
            .current_weights(current)
            .transaction_costs(0.001),
    )?;
    println!("Rebalanced weights: {:?}", net.as_slice());
    Ok(())
}
```

### Black-Litterman

Black-Litterman estimates expected returns rather than allocating. Start from the market weights (or capitalizations) and the market's risk aversion δ, add views, and pass the result to a method that uses expected returns:

```rust
use cutup::{BlackLitterman, CutupError, MeanVariance, Returns, View};

fn main() -> Result<(), CutupError> {
    let returns = [
        [ 0.010, -0.020,  0.005],
        [-0.004,  0.015,  0.002],
        [ 0.007,  0.003, -0.006],
        [ 0.002, -0.001,  0.004],
        [-0.003,  0.006,  0.001],
    ];
    let data = Returns::new(&returns)?;

    let mu = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
        // Asset 0 returns 0.1% a period, and you're 60% sure of it.
        .view(View::absolute(0, 0.001).confidence(0.6))
        // Asset 1 beats asset 2 by 0.05% a period.
        .view(View::relative(1, 2, 0.0005))
        .expected_returns(&data)?;

    let weights = data.allocate(&MeanVariance::risk_aversion(2.5).expected_returns(mu))?;
    println!("Black-Litterman weights: {:?}", weights.as_slice());
    Ok(())
}
```

### Upgrading from 0.2

0.3.0 removed the `PortfolioAllocator` methods, `run_portfolio_allocation` and the config structs that were deprecated in 0.2.8. Their [0.2.8 docs](https://docs.rs/cutup/0.2.8/cutup/) still have the deprecation notes. In 0.2.8, each one returned exactly what `allocate` returns for its replacement, so switching changes no weights. The old min variance, mean-variance and max Sharpe methods are long-short unless their name starts with `long_only_`, while `MinVariance`, `MeanVariance` and `MaxSharpe` are long-only unless you call `.allow_short()`. The other methods were always long-only.

| Removed | Replacement |
| --- | --- |
| `allocator.ew_allocation()` | `allocator.allocate(&EqualWeight)?` |
| `inverse_volatility_allocation()` | `allocate(&InverseVolatility)` |
| `inverse_variance_allocation()` | `allocate(&InverseVariance)` |
| `min_variance_allocation()` | `allocate(&MinVariance::default().allow_short())` |
| `min_variance_allocation_with_config(&c)` | `allocate(&MinVariance::default().allow_short().covariance(c))` |
| `long_only_min_variance_allocation()` | `allocate(&MinVariance::default())` |
| `long_only_min_variance_allocation_with_config(&c)` | `allocate(&MinVariance::default().covariance(c))` |
| `long_only_min_variance_allocation_with_bounds(&c, &b)` | `allocate(&MinVariance::default().covariance(c).bounds(b))` |
| `mean_variance_allocation(&MeanVarianceConfig::risk_aversion(γ))` | `allocate(&MeanVariance::risk_aversion(γ).allow_short())` |
| `long_only_mean_variance_allocation(&MeanVarianceConfig::target_return(ρ))` | `allocate(&MeanVariance::target_return(ρ))` |
| `long_only_mean_variance_allocation_with_bounds(&config, &b)` | `allocate(&MeanVariance::risk_aversion(γ).bounds(b))`, or `target_return(ρ)` |
| `max_sharpe_allocation(&config)` | `allocate(&MaxSharpe::default().allow_short())` |
| `long_only_max_sharpe_allocation(&config)` | `allocate(&MaxSharpe::default())` |
| `max_diversification_allocation()`, `_with_config(&c)` | `allocate(&MaxDiversification::default())`, `.covariance(c)` |
| `risk_parity_allocation()`, `_with_config(&config)` | `allocate(&RiskParity::default())` |
| `hrp_allocation()`, `_with_config(&config)` | `allocate(&Hrp::default())` |
| `hrp_cluster_order(&config)` | `Hrp::default().linkage(l).cluster_order(&allocator)` |
| `herc_allocation()`, `_with_config(&config)` | `allocate(&Herc::default())` |
| `run_portfolio_allocation(prices)` | `Returns::from_prices(&prices)?.allocate(&MinVariance::default().allow_short())` |

The config structs' settings are setters on the values, without the `with_`:

| Removed | Replacement |
| --- | --- |
| `.with_expected_returns(dvector![..])` | `.expected_returns([..])`, which takes a `Vec<f64>`, an array or a slice, such as `mu.as_slice()` for a `DVector` |
| `.with_risk_budgets(dvector![..])` | `.risk_budgets([..])` |
| `.with_covariance(c)` | `.covariance(c)`, or `.shrinkage(λ)` and `.regularization(ε)` directly |
| `.with_risk_free_rate(r)` | `.risk_free_rate(r)` |
| `.with_linkage(l)`, `.with_clusters(k)`, `.with_max_clusters(m)` | `.linkage(l)`, `.clusters(k)`, `.max_clusters(m)` |

Seven other changes in 0.3.0:

| 0.2 | 0.3 |
| --- | --- |
| `CovarianceConfig::default().with_shrinkage(λ).with_regularization(ε)` | `CovarianceConfig::default().shrinkage(λ).regularization(ε)`. Its fields are private: set them with these, and read them with `config.lambda()` and `config.epsilon()` instead of `config.shrinkage` and `config.regularization`. |
| `weights.to_hashmap()` | `weights.iter().copied().enumerate().collect::<HashMap<_, _>>()`, `weights.labeled().collect::<HashMap<_, _>>()` by label, or `weights.as_slice()` |
| `InverseVolatility::default()`, `InverseVariance::default()` | Unchanged, and now also plain `InverseVolatility` and `InverseVariance`, as with `EqualWeight` |
| `PortfolioAllocator` | `Returns`, since it holds the data and the methods are values. `PortfolioAllocator` stays as a deprecated alias until 1.0.0, so code that names the type, such as an `Allocation` impl, still compiles, with a warning. |
| `from_prices(dmatrix![..])`, `from_returns(dmatrix![..])` | `Returns::from_prices(&[[..], ..])` and `Returns::new(&[[..], ..])` with rows, or `from_prices_flat(&values, assets)` and `new_flat(&values, assets)` with one row-major slice. To keep passing a `DMatrix`, turn on the `nalgebra` feature and call `from_prices_matrix(m)` or `from_matrix(m)`. `from_returns` stays as a deprecated name for `new` until 1.0.0. |
| `Hrp` and `Herc` are `Copy` and `Eq` | They're `Clone` and `PartialEq`, since they now take `.covariance(..)`, `.shrinkage(λ)` and `.regularization(ε)`. Clone a value to reuse it. |
| Rust 1.78 or newer | Rust 1.85 or newer. cutup itself moves to edition 2024, which doesn't affect crates that depend on it. |

## Testing

```bash
cargo test
```

Besides unit tests and doctests, `tests/reference.rs` checks every allocation method against independent Python implementations: NumPy closed forms for minimum variance, mean-variance and max Sharpe, support enumeration for long-only minimum variance, mean-variance and max Sharpe and for max diversification (with weight bounds, cross-checked against PyPortfolioOpt), coordinate descent for risk parity, linear programs from SciPy's HiGHS for minimum CVaR (cross-checked against PyPortfolioOpt and riskfolio-lib) and for the simplex solver itself, support enumeration for rebalancing with turnover caps, transaction costs and target returns (cross-checked against PyPortfolioOpt), PyPortfolioOpt for HRP and Black-Litterman, and riskfolio-lib for HERC and for HRP and HERC with shrinkage and regularization. These run on fixed datasets, including singular and ill-conditioned ones. The expected weights are committed, so `cargo test` doesn't need Python. See [`reference/README.md`](reference/README.md) to regenerate them.

## License

This project is licensed under the MIT License.
