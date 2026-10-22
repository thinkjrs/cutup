//! Runs every allocation method on the same prices and prints the weights side by side.
//!
//! ```bash
//! cargo run --example compare
//! ```

use cutup::{
    CovarianceConfig, CutupError, HrpConfig, Linkage, MeanVarianceConfig, PortfolioAllocator,
};
use nalgebra::DMatrix;

fn main() -> Result<(), CutupError> {
    #[rustfmt::skip]
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
    let allocator = PortfolioAllocator::from_prices(prices)?;
    let shrunk = CovarianceConfig::default().with_shrinkage(0.2);

    let methods = [
        ("Equal weight", allocator.ew_allocation()),
        (
            "Inverse volatility",
            allocator.inverse_volatility_allocation()?,
        ),
        ("Inverse variance", allocator.inverse_variance_allocation()?),
        ("Min variance", allocator.min_variance_allocation()?),
        (
            "Min variance, 20% shrinkage",
            allocator.min_variance_allocation_with_config(&shrunk)?,
        ),
        (
            "Mean-variance, γ = 100",
            allocator.mean_variance_allocation(&MeanVarianceConfig::risk_aversion(100.0))?,
        ),
        ("HRP, single linkage", allocator.hrp_allocation()?),
        (
            "HRP, Ward linkage",
            allocator
                .hrp_allocation_with_config(&HrpConfig::default().with_linkage(Linkage::Ward))?,
        ),
    ];

    println!(
        "{:<30}{:>9}{:>9}{:>9}{:>9}",
        "Method", "Asset 0", "Asset 1", "Asset 2", "Asset 3"
    );
    for (name, weights) in &methods {
        print!("{name:<30}");
        for w in weights {
            print!("{w:>9.3}");
        }
        println!();
    }
    Ok(())
}
