//! Splits a budget across five ad channels with caps and floors on groups of them, and puts
//! each portfolio next to the one with the same method and no groups.
//!
//! ```bash
//! cargo run --example groups
//! ```
//!
//! Each column is a channel's daily return on spend: what a unit spent that day brought back,
//! less the unit. Search and shopping run on one platform, and video and social on another, so
//! the groups limit what goes to each platform, on top of a cap on each channel.

use cutup::{
    Allocation, CutupError, MeanVariance, MinCvar, MinVariance, Returns, WeightBounds, Weights,
};

const CHANNELS: [&str; 5] = ["search", "shopping", "video", "social", "display"];

/// An average daily return on spend of 0.3%, by the sample mean, for mean-variance.
const TARGET: f64 = 0.003;

/// A method, built with the bounds it's given.
type Method = fn(WeightBounds) -> Box<dyn Allocation>;

fn main() -> Result<(), CutupError> {
    #[rustfmt::skip]
    let returns = [
        [ 0.0076,  0.0108,  0.0351,  0.0251,  0.0066],
        [ 0.0012,  0.0053, -0.0408, -0.0059, -0.0027],
        [ 0.0044, -0.0036,  0.0632, -0.0111,  0.0337],
        [ 0.0017,  0.0035,  0.0634, -0.0286,  0.0175],
        [ 0.0082, -0.0025,  0.0321, -0.0061,  0.0178],
        [ 0.0012,  0.0097,  0.0092,  0.0043,  0.0068],
        [-0.0040,  0.0060,  0.0101,  0.0216,  0.0115],
        [ 0.0074, -0.0024,  0.0255,  0.0072,  0.0078],
        [ 0.0012, -0.0063, -0.0380, -0.0450,  0.0189],
        [ 0.0020, -0.0005,  0.0216, -0.0054, -0.0049],
        [ 0.0005,  0.0020, -0.0100, -0.0144,  0.0093],
        [ 0.0120,  0.0017,  0.0519,  0.0345,  0.0011],
        [-0.0071, -0.0036,  0.0011,  0.0203, -0.0014],
        [-0.0021,  0.0090,  0.0009, -0.0035,  0.0055],
        [-0.0066,  0.0060, -0.0554, -0.0101, -0.0278],
        [-0.0038,  0.0042,  0.0451,  0.0181,  0.0074],
        [-0.0086,  0.0087, -0.0220,  0.0071,  0.0038],
        [ 0.0029, -0.0077, -0.0131,  0.0079, -0.0206],
        [ 0.0023,  0.0002, -0.0484,  0.0079,  0.0018],
        [ 0.0062, -0.0013,  0.0268,  0.0028,  0.0143],
        [-0.0070,  0.0019, -0.0205,  0.0153, -0.0044],
        [ 0.0010,  0.0012,  0.0221, -0.0080,  0.0104],
        [ 0.0006, -0.0012, -0.0297,  0.0185,  0.0021],
        [ 0.0024,  0.0074, -0.0083,  0.0110, -0.0041],
    ];
    let data = Returns::new(&returns)?.labeled(CHANNELS)?;

    // No channel above 35%. With groups, at most 40% on search and shopping's platform, and
    // at least 30% on video and social's.
    let plain = WeightBounds::uniform(0.0, 0.35);
    let grouped =
        plain
            .clone()
            .group(["search", "shopping"], 0.0, 0.4)
            .group(["video", "social"], 0.3, 1.0);

    let methods: [(&str, Method); 3] = [
        ("Min variance", |bounds| {
            Box::new(MinVariance::default().bounds(bounds))
        }),
        ("Mean-variance, 0.3% a day", |bounds| {
            Box::new(MeanVariance::target_return(TARGET).bounds(bounds))
        }),
        ("Min CVaR, worst 25% of days", |bounds| {
            Box::new(MinCvar::default().confidence(0.75).bounds(bounds))
        }),
    ];

    print!("{:<32}", "Method");
    for channel in CHANNELS {
        print!("{channel:>9}");
    }
    println!("{:>14}{:>14}", "search+shop", "video+social");
    for (name, method) in methods {
        row(name, &data.allocate(&*method(plain.clone()))?);
        row("  with groups", &data.allocate(&*method(grouped.clone()))?);
    }

    // Groups no portfolio can meet are an error rather than a portfolio that misses them: one
    // group past what its members' caps allow, and then two that can each be met, but not
    // together.
    for bounds in [
        plain.clone().group(["search", "shopping"], 0.75, 1.0),
        plain
            .clone()
            .group(["search", "shopping"], 0.6, 1.0)
            .group(["video", "social"], 0.5, 1.0),
    ] {
        if let Err(error) = data.allocate(&MinVariance::default().bounds(bounds)) {
            println!("\n{error}");
        }
    }
    Ok(())
}

/// Prints `weights` in a row after `name`, then the two groups' totals.
fn row(name: &str, weights: &Weights) {
    print!("{name:<32}");
    for w in weights {
        print!("{w:>9.3}");
    }
    println!(
        "{:>14.3}{:>14.3}",
        weights["search"] + weights["shopping"],
        weights["video"] + weights["social"]
    );
}
