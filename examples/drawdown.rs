//! Compares min variance, min MAD and min CDaR on two assets with the same spread of returns, in
//! a different order: one moves in long streaks, the other back and forth.
//!
//! ```bash
//! cargo run --example drawdown
//! ```
//!
//! Variance and MAD measure how far each period's return is from the mean, so they can't tell
//! the two apart. A drawdown is how far the running total is below its best so far, which a
//! streak of losses deepens and a loss after a gain doesn't. CDaR at `β` is the average of the
//! worst `1 − β` of the drawdowns, so min CDaR is the only method of the three to see it. Each
//! row prints the weights, the volatility, the MAD, the mean shortfall below 0, the maximum
//! drawdown and the 95% CDaR, all per period or in the returns' units.

use cutup::{CutupError, MinCdar, MinMad, MinVariance, Returns, Weights};

const ASSETS: [&str; 3] = ["seasonal", "streaky", "choppy"];

/// 240 periods of returns. Streaky and choppy have the same mean and amplitude, but streaky's
/// cycle is 60 periods long, so it loses for 30 in a row, and choppy's is about 2. Seasonal is
/// in between, on a cycle of 12.
fn returns() -> Vec<[f64; 3]> {
    use std::f64::consts::TAU;
    (0..240)
        .map(|t| {
            let t = t as f64;
            [
                0.0004 + 0.005 * (TAU * t / 12.0).sin() + 0.001 * (2.9 * t).cos(),
                0.0006 + 0.006 * (TAU * t / 60.0).sin() + 0.001 * (1.9 * t).cos(),
                0.0006 + 0.006 * (2.7 * t).sin() + 0.001 * (1.9 * t).cos(),
            ]
        })
        .collect()
}

fn main() -> Result<(), CutupError> {
    let data = Returns::new(&returns())?.labeled(ASSETS)?;

    print!("{:<24}", "");
    for asset in ASSETS {
        print!("{asset:>9}");
    }
    println!(
        "{:>12}{:>9}{:>11}{:>10}{:>9}",
        "volatility", "MAD", "shortfall", "max DD", "CDaR 95"
    );
    row(
        &data,
        "Min variance",
        &data.allocate(&MinVariance::default())?,
    )?;
    row(&data, "Min MAD", &data.allocate(&MinMad::default())?)?;
    // Only shortfalls below a threshold count, here any loss at all.
    let downside = MinMad::default().downside(0.0);
    row(&data, "  shortfall below 0", &data.allocate(&downside)?)?;
    // Min CDaR moves to choppy, and takes on more volatility for drawdowns a fifth as deep.
    row(&data, "Min CDaR", &data.allocate(&MinCdar::default())?)?;
    // Near β = 1 the tail is the single worst drawdown, so this minimizes the maximum.
    let deepest = MinCdar::default().confidence(0.999);
    row(&data, "  max drawdown", &data.allocate(&deepest)?)?;
    Ok(())
}

/// Prints `weights` in a row after `name`, then their volatility, MAD, mean shortfall below 0,
/// maximum drawdown and 95% CDaR.
fn row(data: &Returns, name: &str, weights: &Weights) -> Result<(), CutupError> {
    let stats = data.stats(weights)?;
    print!("{name:<24}");
    for w in weights {
        print!("{w:>9.3}");
    }
    println!(
        "{:>12.5}{:>9.5}{:>11.5}{:>10.4}{:>9.4}",
        stats.volatility(),
        stats.mad(),
        stats.downside_mad(0.0)?,
        stats.max_drawdown(),
        stats.cdar(0.95)?,
    );
    Ok(())
}
