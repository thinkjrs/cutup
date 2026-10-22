# Cutup: A Rust Portfolio Allocation Library

[![Crates.io](https://img.shields.io/crates/v/cutup)](https://crates.io/crates/cutup)
[![Build Status](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/thinkjrs/cutup/branch/main/graph/badge.svg)](https://codecov.io/gh/thinkjrs/cutup)
[![Docs.rs](https://docs.rs/cutup/badge.svg)](https://docs.rs/cutup)
[![License](https://img.shields.io/crates/l/cutup)](https://crates.io/crates/cutup)

Cutup is a Rust library for portfolio allocation strategies, providing implementations for various allocation methods. 

It's designed to be efficient and easy to use.

Here are some of the allocation methods included in this library:

- **Minimum Variance (GMV)**
- **Mean-Variance Optimization (MVO)**
- **Equal Weight Allocation (EW)**
- **Hierarchical Risk Parity (HRP)**

This library leverages `nalgebra` for efficient matrix operations and is designed for performance and extensibility.

## Features

- **Minimum Variance**: The global minimum variance portfolio, using only the covariance matrix. Optional shrinkage and regularization via `CovarianceConfig`.
- **Mean-Variance Optimization**: The Markowitz portfolio for a given risk aversion or target return, using expected returns (the sample mean by default) and the covariance matrix.
- **EW Allocation**: Assigns equal weights to all assets.
- **HRP Allocation**: Uses hierarchical clustering and recursive bisection for risk-based allocation.
- **Fully Unit-Tested**: Includes test cases for correctness verification.

## Installation

Add `cutup` to your `Cargo.toml`:

```toml
[dependencies]
cutup = "0.2.0"
```

## Usage

Pass a matrix of prices (rows: time, columns: assets) to `PortfolioAllocator::from_prices`. Prices are converted to simple returns, and every covariance-based method works on the covariance of those returns. If you already have returns, use `PortfolioAllocator::from_returns` instead.

Minimum variance and mean-variance weights can be negative (short positions); there is no long-only constraint yet.

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
    let hrp_weights = allocator.hrp_allocation()?;

    println!("Min Variance Weights: {:?}", min_variance_weights.as_slice());
    println!("Mean-Variance Weights: {:?}", mean_variance_weights.as_slice());
    println!("EW Weights: {:?}", ew_weights.as_slice());
    println!("HRP Weights: {:?}", hrp_weights.as_slice());

    // or run a minimum variance allocation in one go

    let weights = run_portfolio_allocation(prices)?;
    println!("Portfolio Weights: {:?}", weights.as_slice());

    Ok(())
}
```

## License

This project is licensed under the MIT License.
