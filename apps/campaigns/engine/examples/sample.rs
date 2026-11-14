//! Prints each metric's allocation over the bundled sample's default platforms, for a few
//! methods.
//!
//! ```bash
//! cargo run --manifest-path apps/campaigns/Cargo.toml -p campaign-engine --example sample
//! ```

use campaign_engine::{Request, allocate};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for method in [
        r#"{"name": "min_variance"}"#,
        r#"{"name": "max_sharpe"}"#,
        r#"{"name": "risk_parity"}"#,
        r#"{"name": "hrp"}"#,
        r#"{"name": "min_cvar"}"#,
    ] {
        let request: Request = serde_json::from_str(&format!(r#"{{"method": {method}}}"#))?;
        let response = allocate(&request)?;
        println!("{method} over {} days", response.days.len());
        for result in &response.results {
            let metric = format!("{:?}", result.metric);
            match (&result.weights, &result.error) {
                (Some(weights), _) => {
                    let shares: Vec<String> = weights
                        .labeled()
                        .map(|(platform, w)| format!("{platform} {w:.3}"))
                        .collect();
                    println!("  {metric:<12} {}", shares.join(", "));
                }
                (_, Some(error)) => println!("  {metric:<12} {error}"),
                _ => {}
            }
        }
    }
    Ok(())
}
