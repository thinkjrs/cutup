//! Recomputes min variance weights over rolling windows of 20 periods, 5 apart, and prints how
//! they move as one asset turns volatile halfway through.
//!
//! ```bash
//! cargo run --example rolling
//! ```

use cutup::{CutupError, MinVariance, Returns, WeightBounds};

const ASSETS: [&str; 3] = ["steady", "turns", "other"];

/// 60 periods of returns: three assets with their own cycles, the second with five times its
/// volatility from period 30 on.
fn returns() -> Vec<[f64; 3]> {
    (0..60)
        .map(|t| {
            let t = t as f64;
            let shock = if t >= 30.0 { 5.0 } else { 1.0 };
            [
                0.004 * (0.9 * t).sin() + 0.002 * (2.3 * t).cos(),
                shock * (0.003 * (1.7 * t + 1.0).sin() + 0.001 * (0.4 * t).cos()),
                0.006 * (0.6 * t + 2.0).sin() - 0.002 * (1.1 * t).cos(),
            ]
        })
        .collect()
}

fn main() -> Result<(), CutupError> {
    let data = Returns::new(&returns())?.labeled(ASSETS)?;
    let method = MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.8));

    print!("{:<10}", "Periods");
    for asset in ASSETS {
        print!("{asset:>9}");
    }
    println!();
    for window in data.rolling(20).step(5).allocate(&method)? {
        let periods = window.periods();
        print!("{:<10}", format!("{}..{}", periods.start, periods.end));
        match window.weights() {
            Ok(weights) => {
                for w in weights {
                    print!("{w:>9.3}");
                }
                println!();
            }
            Err(error) => println!("  {error}"),
        }
    }

    // The whole history at once, for comparison: one set of weights for all 60 periods.
    print!("{:<10}", "all");
    for w in &data.allocate(&method)? {
        print!("{w:>9.3}");
    }
    println!();
    Ok(())
}
