# Cutup: A Rust Portfolio Allocation Library

[![Crates.io](https://img.shields.io/crates/v/cutup)](https://crates.io/crates/cutup)
[![Build Status](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/thinkjrs/cutup/branch/main/graph/badge.svg)](https://codecov.io/gh/thinkjrs/cutup)
[![Docs.rs](https://docs.rs/cutup/badge.svg)](https://docs.rs/cutup)
[![License](https://img.shields.io/crates/l/cutup)](https://crates.io/crates/cutup)

Cutup is a Rust library for portfolio allocation strategies, providing implementations for various allocation methods. 

It's designed to be efficient and easy to use.

Here are some of the allocation methods included in this library:

- **Mean-Variance Optimization (MVO)**
- **Equal Weight Allocation (EW)**
- **Hierarchical Risk Parity (HRP)**

This library leverages `nalgebra` for efficient matrix operations and is designed for performance and extensibility.

## Features

- **MVO Allocation**: Computes portfolio weights using mean-variance optimization with covariance matrix regularization.
- **EW Allocation**: Assigns equal weights to all assets.
- **HRP Allocation**: Uses hierarchical clustering and recursive bisection for risk-based allocation.
- **Fully Unit-Tested**: Includes test cases for correctness verification.

## Installation

Add `cutup` to your `Cargo.toml`:

```toml
[dependencies]
cutup = "0.1.4"
```

## Usage

Pass a matrix of prices (rows: time, columns: assets) to `PortfolioAllocator::from_prices`. Prices are converted to simple returns, and every covariance-based method works on the covariance of those returns. If you already have returns, use `PortfolioAllocator::from_returns` instead.

Allocation methods return `Weights`: one weight per asset, in the same order as the columns of your input. Bad input (no assets, too few rows, NaN or infinite values, non-positive prices) returns a `CutupError` instead of producing meaningless weights.

```rust
use cutup::{run_portfolio_allocation, CutupError, PortfolioAllocator};
use nalgebra::DMatrix;

fn main() -> Result<(), CutupError> {
    let prices = DMatrix::from_row_slice(
        4,
        4,
        &[
            125.0, 1500.0, 210.0, 600.0,
            123.0, 1520.0, 215.0, 620.0,
            130.0, 1510.0, 220.0, 610.0,
            128.0, 1530.0, 225.0, 630.0,
        ],
    );

    let allocator = PortfolioAllocator::from_prices(prices.clone())?;

    let mvo_weights = allocator.mvo_allocation()?;
    let ew_weights = allocator.ew_allocation();
    let hrp_weights = allocator.hrp_allocation()?;

    println!("MVO Weights: {:?}", mvo_weights.as_slice());
    println!("EW Weights: {:?}", ew_weights.as_slice());
    println!("HRP Weights: {:?}", hrp_weights.as_slice());

    // or do it all in one go

    let weights = run_portfolio_allocation(prices)?;
    println!("Portfolio Weights: {:?}", weights.as_slice());

    Ok(())
}
```

## License

This project is licensed under the MIT License.
