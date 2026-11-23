//! Blends the returns a market portfolio implies with two views, and compares mean-variance on
//! the sample mean, on the market's returns and on the blend.
//!
//! ```bash
//! cargo run --example black_litterman
//! ```
//!
//! The sample mean of a short history is a noisy estimate, and mean-variance on it bets hard on
//! the noise. Black-Litterman starts instead from the returns that make the market portfolio
//! optimal, `π = δΣw_mkt`, and moves them toward your views as far as your confidence in each.

use cutup::{BlackLitterman, CutupError, MeanVariance, Returns, View};

const ASSETS: [&str; 4] = ["us", "europe", "japan", "emerging"];

/// Each asset's share of the market's value.
const MARKET: [f64; 4] = [0.6, 0.2, 0.1, 0.1];

/// The market's risk aversion δ, used both to imply its returns and to allocate.
const DELTA: f64 = 2.5;

/// 60 periods of returns, about monthly in scale, driven by a common factor with each region's
/// own cycle on top.
fn returns() -> Vec<[f64; 4]> {
    (0..60)
        .map(|t| {
            let t = t as f64;
            let world = 0.03 * (0.5 * t).sin();
            [
                0.008 + world + 0.015 * (1.1 * t).cos(),
                0.004 + 0.9 * world + 0.020 * (1.7 * t).sin(),
                0.002 + 0.7 * world + 0.025 * (0.8 * t).cos(),
                0.006 + 1.2 * world + 0.035 * (2.3 * t).sin(),
            ]
        })
        .collect()
}

fn main() -> Result<(), CutupError> {
    let rows = returns();
    let data = Returns::new(&rows)?.labeled(ASSETS)?;

    let market = BlackLitterman::new(MARKET, DELTA);
    let model = market
        .clone()
        // Japan returns 0.15% a month, and you're 50% sure of it.
        .view(View::absolute("japan", 0.0015).confidence(0.5))
        // Emerging markets beat Europe by 0.05% a month, with the default uncertainty, in
        // proportion to that bet's own variance.
        .view(View::relative("emerging", "europe", 0.0005));

    let sample = data.allocate(&MeanVariance::risk_aversion(DELTA))?;
    let prior = market.prior_returns(&data)?;
    let posterior = model.expected_returns(&data)?;

    println!("Expected returns, % a month");
    header("");
    values("Sample mean", &sample_mean(&rows), 100.0);
    values("Market-implied", &prior, 100.0);
    values("With the views", &posterior, 100.0);

    println!("\nMean-variance weights at δ = {DELTA}");
    header("");
    // On the sample mean, mean-variance bets everything on the asset with the best history.
    values("Sample mean", sample.as_slice(), 1.0);
    // With no views, the market's own returns give back its weights.
    let implied = MeanVariance::risk_aversion(DELTA).expected_returns(prior);
    values("Market-implied", data.allocate(&implied)?.as_slice(), 1.0);
    // The views tilt from there, toward Japan and emerging markets, and away from Europe.
    let blended = MeanVariance::risk_aversion(DELTA).expected_returns(posterior);
    values("With the views", data.allocate(&blended)?.as_slice(), 1.0);
    // The posterior covariance, `Σ + τΣ` less what the views resolve, in place of the sample's.
    // At the default τ of 0.05, it moves the weights by 0.02 at most here.
    let posterior_risk = blended.covariance(model.posterior_covariance(&data)?);
    values(
        "  posterior covariance",
        data.allocate(&posterior_risk)?.as_slice(),
        1.0,
    );
    Ok(())
}

/// Each asset's mean return, which mean-variance uses by default.
fn sample_mean(rows: &[[f64; 4]]) -> Vec<f64> {
    (0..ASSETS.len())
        .map(|i| rows.iter().map(|row| row[i]).sum::<f64>() / rows.len() as f64)
        .collect()
}

/// Prints the asset names after `name`'s width.
fn header(name: &str) {
    print!("{name:<26}");
    for asset in ASSETS {
        print!("{asset:>10}");
    }
    println!();
}

/// Prints `values`, each times `scale`, in a row after `name`.
fn values(name: &str, values: &[f64], scale: f64) {
    print!("{name:<26}");
    for v in values {
        print!("{:>10.3}", v * scale);
    }
    println!();
}
