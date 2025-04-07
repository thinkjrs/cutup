# Cutup: A Rust Portfolio Allocation Library

[![Crates.io](https://img.shields.io/crates/v/cutup)](https://crates.io/crates/cutup)
[![Build Status](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml/badge.svg)](https://github.com/thinkjrs/cutup/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/thinkjrs/cutup/branch/main/graph/badge.svg)](https://codecov.io/gh/thinkjrs/cutup)

Cutup is a Rust library for portfolio allocation strategies, providing implementations for:

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
cutup = "0.1.0"
```

## Usage

```rust
use nalgebra::DMatrix;
use cutup::PortfolioAllocator;

fn main() {
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

    let allocator = PortfolioAllocator::new(prices);

    let mvo_weights = allocator.mvo_allocation();
    let ew_weights = allocator.ew_allocation();
    let hrp_weights = allocator.hrp_allocation();

    println!("MVO Weights: {:?}", mvo_weights);
    println!("EW Weights: {:?}", ew_weights);
    println!("HRP Weights: {:?}", hrp_weights);

    // or do it all in one go

    let weights = run_portfolio_allocation(prices);
    println!("Portfolio Weights: {:?}", weights);
}
```

## License

This project is licensed under the MIT License.
