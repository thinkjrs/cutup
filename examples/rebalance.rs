//! Rebalances a four-asset portfolio from its current weights: min variance with no limit, with
//! a turnover cap and with a turnover penalty, then mean-variance net of transaction costs.
//!
//! ```bash
//! cargo run --example rebalance
//! ```
//!
//! Turnover is the total weight bought and sold, `Σ|wᵢ − w₀ᵢ|`, so moving 10% from one asset
//! to another is a turnover of 0.2. Each row prints the weights, their turnover from the
//! current ones and their volatility, all per period.

use cutup::{CutupError, MeanVariance, MinVariance, Returns, Weights};

const ASSETS: [&str; 4] = ["bonds", "credit", "equity", "commodities"];

/// The weights held now, which every row trades from.
const CURRENT: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

/// 120 periods of returns: bonds steady, credit a little riskier, equity and commodities the
/// riskiest, each with its own cycles so that they're imperfectly correlated.
fn returns() -> Vec<[f64; 4]> {
    (0..120)
        .map(|t| {
            let t = t as f64;
            let market = 0.010 * (0.7 * t).sin();
            [
                0.0002 + 0.002 * (1.3 * t).cos(),
                0.0003 + 0.3 * market + 0.003 * (2.1 * t).sin(),
                0.0006 + market + 0.008 * (0.4 * t).cos(),
                0.0004 - 0.4 * market + 0.012 * (1.7 * t).sin(),
            ]
        })
        .collect()
}

fn main() -> Result<(), CutupError> {
    let data = Returns::new(&returns())?.labeled(ASSETS)?;

    print!("{:<34}", "");
    for asset in ASSETS {
        print!("{asset:>12}");
    }
    println!("{:>10}{:>12}", "turnover", "volatility");
    // A turnover cap of 0 keeps the current weights, which gives their volatility.
    let current = MinVariance::default()
        .current_weights(CURRENT)
        .max_turnover(0.0);
    row(&data, "Current", &data.allocate(&current)?)?;

    // Min variance as if from cash: it moves as far as it takes.
    let free = data.allocate(&MinVariance::default())?;
    row(&data, "Min variance", &free)?;

    // The least variance within a turnover cap: each trade goes where it cuts the most.
    for cap in [0.2, 0.05] {
        let method = MinVariance::default()
            .current_weights(CURRENT)
            .max_turnover(cap);
        row(
            &data,
            &format!("  turnover at most {cap}"),
            &data.allocate(&method)?,
        )?;
    }

    // A penalty κ‖w − w₀‖² on trading instead: it trades less the higher κ, but has no cap.
    // κ is in units of variance, so it's small, as variances per period are.
    for kappa in [0.00001, 0.0001] {
        let method = MinVariance::default()
            .current_weights(CURRENT)
            .turnover_penalty(kappa);
        row(
            &data,
            &format!("  penalty {kappa}"),
            &data.allocate(&method)?,
        )?;
    }

    // Mean-variance net of proportional costs, in units of the expected return: a trade has to
    // earn its cost back, so higher costs leave the weights nearer the current. The returns are
    // per period, so a cost is a trade's cost spread over the periods it's held: 0.0001 is 1%
    // held for 100 periods.
    let method = MeanVariance::risk_aversion(10.0);
    row(&data, "Mean-variance", &data.allocate(&method)?)?;
    for cost in [0.00003, 0.0001] {
        let method = MeanVariance::risk_aversion(10.0)
            .current_weights(CURRENT)
            .transaction_costs(cost);
        row(&data, &format!("  costs {cost}"), &data.allocate(&method)?)?;
    }
    Ok(())
}

/// Prints `weights` in a row after `name`, then their turnover and volatility.
fn row(data: &Returns, name: &str, weights: &Weights) -> Result<(), CutupError> {
    let turnover: f64 = weights
        .iter()
        .zip(CURRENT)
        .map(|(w, c)| (w - c).abs())
        .sum();
    print!("{name:<34}");
    for w in weights {
        print!("{w:>12.3}");
    }
    println!(
        "{turnover:>10.3}{:>12.5}",
        data.stats(weights)?.volatility()
    );
    Ok(())
}
