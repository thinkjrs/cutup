//! Compares min variance with min CVaR on three assets, one of which earns a steady carry and
//! now and then loses a lot at once.
//!
//! ```bash
//! cargo run --example tail_risk
//! ```
//!
//! Variance counts gains and losses alike and squares them, so an asset whose volatility looks
//! modest can still hide rare, large losses. CVaR at `β` is the average loss over the worst
//! `1 − β` of the periods, so it looks at those losses directly. Each row prints the weights,
//! the volatility, the 95% CVaR and the worst period, all per period.

use cutup::{CutupError, MinCvar, MinVariance, Returns, WeightBounds, Weights};

const ASSETS: [&str; 3] = ["bonds", "carry", "equity"];

/// 500 periods of returns. Carry earns 0.08% a period with little noise but loses 3% one
/// period in 50. Bonds are steady, and equity is riskier with symmetric swings.
fn returns() -> Vec<[f64; 3]> {
    (0..500)
        .map(|t| {
            let s = t as f64;
            let crash = if t % 50 == 17 { -0.03 } else { 0.0 };
            [
                0.0001 + 0.003 * (1.3 * s).sin() + 0.001 * (3.7 * s).cos(),
                0.0008 + 0.002 * (2.9 * s).cos() + crash,
                0.0004 + 0.008 * (0.7 * s).sin() + 0.004 * (2.3 * s).cos(),
            ]
        })
        .collect()
}

fn main() -> Result<(), CutupError> {
    let data = Returns::new(&returns())?.labeled(ASSETS)?;

    print!("{:<30}", "");
    for asset in ASSETS {
        print!("{asset:>9}");
    }
    println!("{:>12}{:>10}{:>10}", "volatility", "CVaR 95", "worst");
    row(
        &data,
        "Min variance",
        &data.allocate(&MinVariance::default())?,
    )?;
    // The worst 5% of periods and the worst 1%. Each needs enough periods for its tail: 500
    // periods leave 25 in the first and 5 in the second.
    for beta in [0.95, 0.99] {
        let method = MinCvar::default().confidence(beta);
        row(
            &data,
            &format!("Min CVaR at {beta}"),
            &data.allocate(&method)?,
        )?;
    }
    // With an expected return of 0.03% a period, by the sample mean, and a cap on each asset,
    // as for mean-variance: the least CVaR along the way to a higher return.
    let method = MinCvar::default()
        .target_return(0.0003)
        .bounds(WeightBounds::uniform(0.0, 0.6));
    row(
        &data,
        "  0.03% a period, each ≤ 60%",
        &data.allocate(&method)?,
    )?;
    Ok(())
}

/// Prints `weights` in a row after `name`, then their volatility, 95% CVaR and worst period.
fn row(data: &Returns, name: &str, weights: &Weights) -> Result<(), CutupError> {
    let stats = data.stats(weights)?;
    print!("{name:<30}");
    for w in weights {
        print!("{w:>9.3}");
    }
    // CVaR at a confidence near 1 is the worst single period.
    println!(
        "{:>12.5}{:>10.5}{:>10.5}",
        stats.volatility(),
        stats.cvar(0.95)?,
        stats.cvar(0.999)?,
    );
    Ok(())
}
