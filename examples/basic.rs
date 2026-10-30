use cutup::{CutupError, MinVariance, Returns};

fn main() -> Result<(), CutupError> {
    // 13 periods of 4 assets' prices, one period after another.
    #[rustfmt::skip]
    let prices = [
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
    ];

    let allocator = Returns::from_prices_flat(&prices, 4)?;
    let weights = allocator.allocate(&MinVariance::default())?;
    println!("Long-only min variance weights: {:?}", weights.as_slice());

    Ok(())
}
