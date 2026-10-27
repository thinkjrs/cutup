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
- **Maximum Sharpe Ratio (tangency portfolio)**
- **Equal Weight Allocation (EW)**
- **Inverse Volatility and Inverse Variance**
- **Risk Parity (Equal Risk Contribution) and Risk Budgeting**
- **Hierarchical Risk Parity (HRP)**
- **Hierarchical Equal Risk Contribution (HERC)**

This library leverages `nalgebra` for efficient matrix operations and is designed for performance and extensibility.

## Features

- **Minimum Variance**: The global minimum variance portfolio, using only the covariance matrix. Optional shrinkage and regularization via `CovarianceConfig`.
- **Long-Only Minimum Variance**: The lowest-variance portfolio with no short positions, solved exactly by a built-in active-set solver.
- **Mean-Variance Optimization**: The Markowitz portfolio for a given risk aversion or target return, using expected returns (the sample mean by default) and the covariance matrix.
- **Long-Only Mean-Variance**: The same objectives with no short positions, solved exactly by the active-set solver.
- **Maximum Sharpe Ratio**: The tangency portfolio for a given risk-free rate, using expected returns and the covariance matrix.
- **EW Allocation**: Assigns equal weights to all assets.
- **Inverse Volatility / Inverse Variance**: Weights each asset by `1 / σ` or `1 / σ²`, ignoring correlations. Simple risk-based weights that are always positive.
- **Risk Parity**: Long-only weights where every asset contributes the same share of portfolio variance, accounting for correlations. Set unequal shares with risk budgets in `RiskParityConfig`.
- **HRP Allocation**: Uses hierarchical clustering and recursive bisection for risk-based allocation. Choose single, complete, average or Ward linkage with `HrpConfig`, and inspect the cluster order with `hrp_cluster_order`.
- **HERC Allocation**: Cuts the same clustering into groups, chosen by a gap statistic or set with `HercConfig`, then splits risk down the tree and by inverse variance within each group.
- **Fully Unit-Tested**: Includes test cases for correctness verification.

## Choosing a method

| Method | Uses | Long-only | Notes |
| --- | --- | --- | --- |
| `ew_allocation` | nothing | ✅ | A baseline that needs no estimates and can't fail. |
| `inverse_volatility_allocation` | volatilities | ✅ | Riskier assets get less; correlations are ignored. |
| `inverse_variance_allocation` | variances | ✅ | A stronger tilt than inverse volatility. Equals min variance when assets are uncorrelated. |
| `min_variance_allocation` | covariance | ❌ | The lowest-variance fully invested portfolio. Sensitive to estimation error, so consider shrinkage. |
| `long_only_min_variance_allocation` | covariance | ✅ | The lowest-variance portfolio without shorting. Usually concentrates in a few low-risk assets and holds none of the rest. |
| `mean_variance_allocation` | covariance, expected returns | ❌ | Markowitz, for a risk aversion or a target return. Most sensitive to the expected returns you supply. |
| `long_only_mean_variance_allocation` | covariance, expected returns | ✅ | Markowitz without shorting. A target return must lie between the lowest and highest expected return. Low risk aversion concentrates in the highest expected returns. |
| `max_sharpe_allocation` | covariance, expected returns, risk-free rate | ❌ | The tangency portfolio. Requires the risk-free rate to be below the min variance portfolio's return. As sensitive to expected returns as mean-variance. |
| `risk_parity_allocation` | covariance | ✅ | Every asset contributes the same risk. Diversifies more than inverse volatility when correlations differ. Needs every long-only portfolio to have some variance, so consider shrinkage when data is short. |
| `hrp_allocation` | covariance, correlations | ✅ | Clusters correlated assets, then splits risk between clusters. Never inverts the covariance, so it's robust when data is short. |
| `herc_allocation` | covariance, correlations | ✅ | Like HRP, but splits where the clustering tree branches and groups similar assets, so the result doesn't depend on column order. |

Long-only methods never return negative weights. The others can. To see every method on the same data:

```bash
cargo run --example compare
```

## Installation

Add `cutup` to your `Cargo.toml`:

```toml
[dependencies]
cutup = "0.2.4"
```

## Usage

Pass a matrix of prices (rows: time, columns: assets) to `PortfolioAllocator::from_prices`. Prices are converted to simple returns, and every covariance-based method works on the covariance of those returns. If you already have returns, use `PortfolioAllocator::from_returns` instead.

Minimum variance, mean-variance and max Sharpe weights can be negative (short positions). Without shorting, use `long_only_min_variance_allocation` or `long_only_mean_variance_allocation`.

Allocation methods return `Weights`: one weight per asset, in the same order as the columns of your input. Bad input (no assets, too few rows, NaN or infinite values, non-positive prices) returns a `CutupError` instead of producing meaningless weights.

```rust
use cutup::{
    run_portfolio_allocation, CovarianceConfig, CutupError, MeanVarianceConfig, PortfolioAllocator,
};
use nalgebra::DMatrix;

fn main() -> Result<(), CutupError> {
    let prices = DMatrix::from_row_slice(
        13,
        4,
        &[
            125.0, 1500.0, 210.0, 600.0,
            123.0, 1520.0, 215.0, 620.0,
            130.0, 1510.0, 220.0, 610.0,
            128.0, 1530.0, 225.0, 630.0,
            131.0, 1525.0, 223.0, 640.0,
            129.5, 1550.0, 228.0, 635.0,
            133.0, 1545.0, 226.0, 650.0,
            135.5, 1560.0, 231.0, 645.0,
            134.0, 1580.0, 229.0, 660.0,
            137.0, 1575.0, 234.0, 668.0,
            136.0, 1600.0, 238.0, 662.0,
            139.5, 1595.0, 236.0, 675.0,
            141.0, 1615.0, 241.0, 680.0,
        ],
    );

    let allocator = PortfolioAllocator::from_prices(prices.clone())?;

    let min_variance_weights = allocator.min_variance_allocation()?;
    // Risk aversion is in the units of your returns: per-period returns have small variances,
    // so meaningful values are large. Shrinking the sample covariance toward its average
    // variance reduces estimation error.
    let config = MeanVarianceConfig::risk_aversion(100.0)
        .with_covariance(CovarianceConfig::default().with_shrinkage(0.1));
    let mean_variance_weights = allocator.mean_variance_allocation(&config)?;
    let ew_weights = allocator.ew_allocation();
    let inverse_volatility_weights = allocator.inverse_volatility_allocation()?;
    let hrp_weights = allocator.hrp_allocation()?;

    println!("Min Variance Weights: {:?}", min_variance_weights.as_slice());
    println!("Mean-Variance Weights: {:?}", mean_variance_weights.as_slice());
    println!("EW Weights: {:?}", ew_weights.as_slice());
    println!("Inverse Volatility Weights: {:?}", inverse_volatility_weights.as_slice());
    println!("HRP Weights: {:?}", hrp_weights.as_slice());

    // or run a minimum variance allocation in one go

    let weights = run_portfolio_allocation(prices)?;
    println!("Portfolio Weights: {:?}", weights.as_slice());

    Ok(())
}
```

## Testing

```bash
cargo test
```

Besides unit tests and doctests, `tests/reference.rs` checks every allocation method against independent Python implementations: NumPy closed forms for minimum variance, mean-variance and max Sharpe, support enumeration for long-only minimum variance and mean-variance, coordinate descent for risk parity, PyPortfolioOpt for HRP, and riskfolio-lib for HERC. These run on fixed datasets, including singular and ill-conditioned ones. The expected weights are committed, so `cargo test` doesn't need Python. See [`reference/README.md`](reference/README.md) to regenerate them.

## License

This project is licensed under the MIT License.
