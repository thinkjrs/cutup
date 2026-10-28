//! Runs every allocation method on the same prices and prints the weights side by side.
//!
//! ```bash
//! cargo run --example compare
//! ```

use cutup::{
    Allocation, CutupError, EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility, Linkage,
    MaxDiversification, MaxSharpe, MeanVariance, MinVariance, PortfolioAllocator, RiskParity,
    WeightBounds,
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

    let methods: Vec<(&str, Box<dyn Allocation>)> = vec![
        ("Equal weight", Box::new(EqualWeight)),
        ("Inverse volatility", Box::new(InverseVolatility::default())),
        ("Inverse variance", Box::new(InverseVariance::default())),
        (
            "Min variance",
            Box::new(MinVariance::default().allow_short()),
        ),
        (
            "Min variance, 20% shrinkage",
            Box::new(MinVariance::default().allow_short().shrinkage(0.2)),
        ),
        ("Long-only min variance", Box::new(MinVariance::default())),
        (
            "Long-only min variance, ≤ 30% each",
            Box::new(MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.3))),
        ),
        (
            "Mean-variance, γ = 100",
            Box::new(MeanVariance::risk_aversion(100.0).allow_short()),
        ),
        (
            "Long-only mean-variance, γ = 100",
            Box::new(MeanVariance::risk_aversion(100.0)),
        ),
        (
            "Max Sharpe, r_f = 0",
            Box::new(MaxSharpe::default().allow_short()),
        ),
        (
            "Long-only max Sharpe, r_f = 0",
            Box::new(MaxSharpe::default()),
        ),
        (
            "Max diversification",
            Box::new(MaxDiversification::default()),
        ),
        ("Risk parity", Box::new(RiskParity::default())),
        ("HRP, single linkage", Box::new(Hrp::default())),
        (
            "HRP, Ward linkage",
            Box::new(Hrp::default().linkage(Linkage::Ward)),
        ),
        ("HERC, single linkage", Box::new(Herc::default())),
        (
            "HERC, Ward linkage",
            Box::new(Herc::default().linkage(Linkage::Ward)),
        ),
    ];

    // Padding counts characters, so the widest label sets the column.
    let width = methods
        .iter()
        .map(|(name, _)| name.chars().count())
        .max()
        .unwrap_or(0)
        + 2;
    println!(
        "{:<width$}{:>9}{:>9}{:>9}{:>9}",
        "Method", "Asset 0", "Asset 1", "Asset 2", "Asset 3"
    );
    for (name, method) in &methods {
        print!("{name:<width$}");
        for w in &allocator.allocate(method)? {
            print!("{w:>9.3}");
        }
        println!();
    }
    Ok(())
}
