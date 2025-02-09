# Cutup: A Rust Portfolio Allocation Library

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
    let prices = DMatrix::from_row_slice(3, 3, &[
        100.0, 200.0, 300.0,
        110.0, 210.0, 310.0,
        120.0, 220.0, 320.0,
    ]);

    let allocator = PortfolioAllocator::new(prices);
    
    let mvo_weights = allocator.mvo_allocation();
    let ew_weights = allocator.ew_allocation();
    let hrp_weights = allocator.hrp_allocation();
    
    println!("MVO Weights: {:?}", mvo_weights);
    println!("EW Weights: {:?}", ew_weights);
    println!("HRP Weights: {:?}", hrp_weights);
}
```

## License
This project is licensed under the MIT License.

