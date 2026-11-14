//! The campaign allocation app's engine: everything the native handler and the wasm wrapper
//! share, so they can't diverge. Its one entry point is [`allocate`]: a [`Request`] in, a
//! [`Response`] out, both JSON through serde, with TypeScript types through ts-rs.
//!
//! Each ad platform is an asset, and the weights are shares of the budget. For each metric,
//! each platform's return on a day is what a unit of spend bought there: impressions, clicks,
//! link clicks or views per unit. Those are divided by their average over the platforms and
//! days in the request, so every metric's returns are near 1 and a method's settings, such as a
//! risk aversion, mean the same for each. Every method allocates once per metric.
//!
//! These methods take a platform's results to grow in proportion to its spend, so they don't
//! model saturation.

mod sample;

pub use sample::{Metric, PlatformSummary, Sample, Totals};

use cutup::{
    Allocation, CutupError, EqualWeight, FrontierPoint, Herc, Hrp, InverseVariance,
    InverseVolatility, MaxDiversification, MaxSharpe, MeanVariance, MinCvar, MinVariance, Returns,
    RiskParity, Weights, Window,
};
use serde::{Deserialize, Serialize};
use ts_rs::TS;

/// The platforms a request allocates across when it names none: the three with spend on every
/// day of the sample.
pub const DEFAULT_PLATFORMS: [&str; 3] = ["Meta", "Google", "TikTok"];

/// The confidence of each result's CVaR.
pub const CVAR_CONFIDENCE: f64 = 0.95;

/// An allocation method, by `"name"`, with its settings as cutup reads them, such as
/// `{"name": "min_variance", "bounds": {"upper": 0.6}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum Method {
    EqualWeight,
    InverseVolatility,
    InverseVariance,
    MinVariance(MinVariance),
    MeanVariance(MeanVariance),
    MaxSharpe(MaxSharpe),
    MaxDiversification(MaxDiversification),
    RiskParity(RiskParity),
    Hrp(Hrp),
    Herc(Herc),
    MinCvar(MinCvar),
}

impl Method {
    /// The cutup method, boxed.
    fn allocation(&self) -> Box<dyn Allocation> {
        match self {
            Method::EqualWeight => Box::new(EqualWeight),
            Method::InverseVolatility => Box::new(InverseVolatility),
            Method::InverseVariance => Box::new(InverseVariance),
            Method::MinVariance(m) => Box::new(m.clone()),
            Method::MeanVariance(m) => Box::new(m.clone()),
            Method::MaxSharpe(m) => Box::new(m.clone()),
            Method::MaxDiversification(m) => Box::new(m.clone()),
            Method::RiskParity(m) => Box::new(m.clone()),
            Method::Hrp(m) => Box::new(m.clone()),
            Method::Herc(m) => Box::new(m.clone()),
            Method::MinCvar(m) => Box::new(m.clone()),
        }
    }
}

/// Rolling windows to allocate over too: `window` days each, `step` apart.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, TS)]
#[serde(deny_unknown_fields)]
pub struct RollingRequest {
    pub window: usize,
    #[serde(default = "one")]
    #[ts(as = "Option<usize>", optional)]
    pub step: usize,
}

fn one() -> usize {
    1
}

/// What to allocate: across which platforms, between which dates, for which metrics, with
/// which method, and what to add.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// The platforms to allocate across, [`DEFAULT_PLATFORMS`] if left out.
    #[serde(default = "default_platforms")]
    #[ts(as = "Option<Vec<String>>", optional)]
    pub platforms: Vec<String>,
    /// The first and last dates to use, `YYYY-MM-DD`, inclusive. Either may be left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub from: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub to: Option<String>,
    /// The metrics to allocate for, every one if left out.
    #[serde(default = "all_metrics")]
    #[ts(as = "Option<Vec<Metric>>", optional)]
    pub metrics: Vec<Metric>,
    pub method: Method,
    /// Points along the efficient frontier, with a mean-variance method only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub frontier: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub rolling: Option<RollingRequest>,
}

fn default_platforms() -> Vec<String> {
    DEFAULT_PLATFORMS.map(String::from).to_vec()
}

fn all_metrics() -> Vec<Metric> {
    Metric::ALL.to_vec()
}

/// A portfolio's statistics, from [`cutup::PortfolioStats`], in the units of the normalized
/// returns: an expected return of 1.05 is 5% more of the metric per unit of spend than the
/// average.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct Stats {
    pub expected_return: f64,
    pub volatility: f64,
    /// The ratio of expected return to volatility, with no risk-free rate, if there's any
    /// volatility.
    pub sharpe_ratio: Option<f64>,
    pub diversification_ratio: Option<f64>,
    /// Each platform's share of the variance.
    pub risk_contributions: Option<Vec<f64>>,
    /// The average of the worst 5% of days, as a loss: the negative of their return.
    pub cvar: f64,
}

/// One metric's allocation: its weights and statistics, or why the method couldn't allocate,
/// and the frontier and rolling windows if they were asked for.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct MetricResult {
    pub metric: Metric,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub weights: Option<Weights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub stats: Option<Stats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub error: Option<CutupError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub frontier: Option<Vec<FrontierPoint>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub rolling: Option<Vec<Window>>,
}

/// Each metric's allocation, over the days every platform spent on.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct Response {
    pub platforms: Vec<String>,
    /// The days used, oldest first.
    pub days: Vec<String>,
    pub results: Vec<MetricResult>,
}

/// Why a request couldn't be allocated at all. A method that can't allocate one metric is that
/// metric's [`MetricResult::error`] instead.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
#[serde(rename_all = "snake_case")]
pub enum EngineError {
    /// The sample's CSV is malformed at `line`, counting from 1.
    Sample {
        line: usize,
        message: String,
    },
    NoPlatforms,
    UnknownPlatform {
        platform: String,
    },
    DuplicatePlatform {
        platform: String,
    },
    NoMetrics,
    InvalidDate {
        date: String,
    },
    /// Fewer than 2 days on which every platform spent, too few for a covariance.
    TooFewDays {
        days: usize,
    },
    /// A frontier was asked for with a method other than mean-variance.
    FrontierNeedsMeanVariance,
    /// The rolling windows don't fit the days, or a step is 0.
    Rolling {
        error: CutupError,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Sample { line, message } => write!(f, "sample line {line}: {message}"),
            EngineError::NoPlatforms => write!(f, "a request needs at least one platform"),
            EngineError::UnknownPlatform { platform } => {
                write!(f, "`{platform}` isn't a platform in the sample")
            }
            EngineError::DuplicatePlatform { platform } => {
                write!(f, "`{platform}` is named twice")
            }
            EngineError::NoMetrics => write!(f, "a request needs at least one metric"),
            EngineError::InvalidDate { date } => write!(f, "`{date}` isn't a YYYY-MM-DD date"),
            EngineError::TooFewDays { days } => write!(
                f,
                "only {days} days on which every platform spent; at least 2 are needed"
            ),
            EngineError::FrontierNeedsMeanVariance => {
                write!(f, "a frontier needs a mean-variance method")
            }
            EngineError::Rolling { error } => write!(f, "rolling windows: {error}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Allocates `request` over the bundled sample.
///
/// # Errors
///
/// [`EngineError`] for a request that can't be allocated at all.
pub fn allocate(request: &Request) -> Result<Response, EngineError> {
    allocate_over(Sample::bundled(), request)
}

/// Allocates `request` over `sample`.
///
/// # Errors
///
/// [`EngineError`] for a request that can't be allocated at all.
pub fn allocate_over(sample: &Sample, request: &Request) -> Result<Response, EngineError> {
    let platforms = &request.platforms;
    check_platforms(sample, platforms)?;
    if request.metrics.is_empty() {
        return Err(EngineError::NoMetrics);
    }
    for date in [&request.from, &request.to].into_iter().flatten() {
        if !sample::is_date(date) {
            return Err(EngineError::InvalidDate { date: date.clone() });
        }
    }
    let mean_variance = match (&request.method, request.frontier) {
        (Method::MeanVariance(m), _) => Some(m),
        (_, Some(_)) => return Err(EngineError::FrontierNeedsMeanVariance),
        _ => None,
    };
    let (days, totals): (Vec<&str>, Vec<Vec<Totals>>) = sample
        .days_with_spend(platforms, request.from.as_deref(), request.to.as_deref())
        .unzip();
    if days.len() < 2 {
        return Err(EngineError::TooFewDays { days: days.len() });
    }
    let method = request.method.allocation();
    let mut results = Vec::with_capacity(request.metrics.len());
    for &metric in &request.metrics {
        let data = returns(&totals, metric, platforms);
        let mut result = MetricResult {
            metric,
            weights: None,
            stats: None,
            error: None,
            frontier: None,
            rolling: None,
        };
        match data.allocate(method.as_ref()) {
            Ok(weights) => {
                result.stats = data.stats(&weights).ok().map(|s| stats(&s));
                result.weights = Some(weights);
            }
            Err(error) => result.error = Some(error),
        }
        if let (Some(points), Some(m)) = (request.frontier, mean_variance) {
            result.frontier = data.frontier(m, points).ok();
        }
        if let Some(rolling) = request.rolling {
            let windows = data
                .rolling(rolling.window)
                .step(rolling.step)
                .allocate(method.as_ref())
                .map_err(|error| EngineError::Rolling { error })?;
            result.rolling = Some(windows.collect());
        }
        results.push(result);
    }
    Ok(Response {
        platforms: platforms.clone(),
        days: days.into_iter().map(String::from).collect(),
        results,
    })
}

/// Checks `platforms` are some of the sample's, each once.
fn check_platforms(sample: &Sample, platforms: &[String]) -> Result<(), EngineError> {
    if platforms.is_empty() {
        return Err(EngineError::NoPlatforms);
    }
    let known = sample.platforms();
    for (i, platform) in platforms.iter().enumerate() {
        if !known.iter().any(|p| &p.platform == platform) {
            return Err(EngineError::UnknownPlatform {
                platform: platform.clone(),
            });
        }
        if platforms[..i].contains(platform) {
            return Err(EngineError::DuplicatePlatform {
                platform: platform.clone(),
            });
        }
    }
    Ok(())
}

/// Each platform's yield of `metric` on each day, divided by their average over every
/// platform and day, labeled with the platforms.
fn returns(totals: &[Vec<Totals>], metric: Metric, platforms: &[String]) -> Returns {
    let yields: Vec<Vec<f64>> = totals
        .iter()
        .map(|day| day.iter().map(|t| t.yield_of(metric)).collect())
        .collect();
    let count = (yields.len() * platforms.len()) as f64;
    let mean = yields.iter().flatten().sum::<f64>() / count;
    // A metric none of the platforms delivered has no yield to compare, so it stays at 0.
    let mean = if mean > 0.0 { mean } else { 1.0 };
    let rows: Vec<Vec<f64>> = yields
        .into_iter()
        .map(|day| day.into_iter().map(|y| y / mean).collect())
        .collect();
    Returns::new(&rows)
        .and_then(|data| data.labeled(platforms.iter().cloned()))
        .expect("finite yields over at least 2 days, one per platform")
}

/// The statistics a result reports, from cutup's.
fn stats(stats: &cutup::PortfolioStats) -> Stats {
    Stats {
        expected_return: stats.expected_return(),
        volatility: stats.volatility(),
        sharpe_ratio: stats.sharpe_ratio(0.0),
        diversification_ratio: stats.diversification_ratio(),
        risk_contributions: stats.risk_contributions().map(<[f64]>::to_vec),
        cvar: stats.cvar(CVAR_CONFIDENCE).unwrap_or(f64::NAN),
    }
}

#[cfg(test)]
mod tests;
