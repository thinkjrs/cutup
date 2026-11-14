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

/// The most points a frontier can ask for. Each is a solve, once per metric.
pub const MAX_FRONTIER_POINTS: usize = 100;

/// The most rolling windows a request can ask for. Each is a solve, once per metric, and the
/// native function answers within a time limit.
pub const MAX_ROLLING_WINDOWS: usize = 100;

/// An allocation method, by `"name"`, with its settings as cutup reads them, such as
/// `{"name": "min_variance", "bounds": {"upper": 0.6}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, TS)]
#[serde(tag = "name", rename_all = "snake_case")]
pub enum Method {
    // cutup's methods with no settings, not unit variants: serde ignores any other fields given
    // with a unit variant, so `{"name": "equal_weight", "bounds": ..}` would allocate without
    // the bounds, where cutup's own types refuse them. Their TypeScript type is `{}`, since
    // cutup's `Record<string, never>` would leave no type for `"name"`.
    EqualWeight(#[ts(type = "{}")] EqualWeight),
    InverseVolatility(#[ts(type = "{}")] InverseVolatility),
    InverseVariance(#[ts(type = "{}")] InverseVariance),
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
            Method::EqualWeight(m) => Box::new(*m),
            Method::InverseVolatility(m) => Box::new(*m),
            Method::InverseVariance(m) => Box::new(*m),
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
    /// Points along the efficient frontier, from 2 to [`MAX_FRONTIER_POINTS`], with a
    /// mean-variance method only.
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
    /// Why the frontier couldn't be traced, when one was asked for: a setting it doesn't take,
    /// such as rebalancing, or one it can't use, such as data it can't allocate on.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub frontier_error: Option<CutupError>,
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
    /// The request isn't JSON for a [`Request`]: malformed, a key unknown or missing, or a value
    /// of the wrong type. `message` is serde's.
    Request {
        message: String,
    },
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
    /// A frontier was asked for with fewer than 2 points or more than
    /// [`MAX_FRONTIER_POINTS`].
    FrontierPoints {
        points: usize,
    },
    /// The rolling windows don't fit the days, or a step is 0.
    Rolling {
        error: CutupError,
    },
    /// More rolling windows than [`MAX_ROLLING_WINDOWS`]: each is a solve, once per metric.
    RollingWindows {
        windows: usize,
    },
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Request { message } => write!(f, "the request: {message}"),
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
            EngineError::FrontierPoints { points } => write!(
                f,
                "a frontier of {points} points; it needs from 2 to {MAX_FRONTIER_POINTS}"
            ),
            EngineError::Rolling { error } => write!(f, "rolling windows: {error}"),
            EngineError::RollingWindows { windows } => write!(
                f,
                "{windows} rolling windows; at most {MAX_ROLLING_WINDOWS}, so take a longer step"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

/// A reply to a request, for either backend to send as it is: an HTTP status and a JSON body,
/// a [`Response`] or `{"error": EngineError}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    /// 200 for a response, 400 for a request that isn't JSON for a [`Request`], and 422 for one
    /// that is but can't be allocated.
    pub status: u16,
    pub body: String,
}

/// A reply's JSON body when it's an error.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct ErrorBody {
    pub error: EngineError,
}

/// The JSON reply to a JSON request, over the bundled sample: what both backends send, so
/// they can't diverge.
pub fn respond(request: &str) -> Reply {
    let result = serde_json::from_str::<Request>(request)
        .map_err(|e| EngineError::Request {
            message: e.to_string(),
        })
        .and_then(|request| allocate(&request));
    let (status, body) = match result {
        Ok(response) => (200, serde_json::to_string(&response)),
        Err(error) => {
            let status = if matches!(error, EngineError::Request { .. }) {
                400
            } else {
                422
            };
            (status, serde_json::to_string(&ErrorBody { error }))
        }
    };
    Reply {
        status,
        body: body.expect("responses and errors serialize"),
    }
}

/// A reply refusing a request before it's read as one, such as a 405 for the wrong HTTP method
/// or a 400 for a body that isn't UTF-8: `status`, with an [`EngineError::Request`] saying
/// `message`.
pub fn reject(status: u16, message: &str) -> Reply {
    let error = EngineError::Request {
        message: message.to_owned(),
    };
    Reply {
        status,
        body: serde_json::to_string(&ErrorBody { error }).expect("errors serialize"),
    }
}

/// The bundled sample's platforms, as JSON: a [`PlatformSummary`] for each, by name.
pub fn platforms() -> String {
    serde_json::to_string(&Sample::bundled().platforms()).expect("summaries serialize")
}

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
        (Method::MeanVariance(_), Some(points)) if !(2..=MAX_FRONTIER_POINTS).contains(&points) => {
            return Err(EngineError::FrontierPoints { points });
        }
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
    // Windows that fit are counted up front, so too many are refused before any is solved.
    // Windows that don't fit are cutup's error, from the first metric.
    if let Some(RollingRequest { window, step }) = request.rolling {
        if step > 0 && (2..=days.len()).contains(&window) {
            let windows = (days.len() - window) / step + 1;
            if windows > MAX_ROLLING_WINDOWS {
                return Err(EngineError::RollingWindows { windows });
            }
        }
    }
    let method = request.method.allocation();
    let mut results = Vec::with_capacity(request.metrics.len());
    for &metric in &request.metrics {
        let mut result = MetricResult {
            metric,
            weights: None,
            stats: None,
            error: None,
            frontier: None,
            frontier_error: None,
            rolling: None,
        };
        match returns(&totals, metric, platforms) {
            Ok(data) => {
                allocate_metric(&mut result, &data, method.as_ref(), request, mean_variance)?
            }
            Err(error) => result.error = Some(error),
        }
        results.push(result);
    }
    Ok(Response {
        platforms: platforms.clone(),
        days: days.into_iter().map(String::from).collect(),
        results,
    })
}

/// Fills in `result` from one metric's returns. The rolling windows are checked first, since
/// they're lazy, so a request whose windows don't fit fails before anything is solved.
fn allocate_metric(
    result: &mut MetricResult,
    data: &Returns,
    method: &dyn Allocation,
    request: &Request,
    mean_variance: Option<&MeanVariance>,
) -> Result<(), EngineError> {
    let rolling = match request.rolling {
        Some(rolling) => Some(
            data.rolling(rolling.window)
                .step(rolling.step)
                .allocate(method)
                .map_err(|error| EngineError::Rolling { error })?,
        ),
        None => None,
    };
    match data.allocate(method) {
        Ok(weights) => {
            result.stats = data.stats(&weights).and_then(|s| stats(&s)).ok();
            result.weights = Some(weights);
        }
        Err(error) => result.error = Some(error),
    }
    if let (Some(points), Some(m)) = (request.frontier, mean_variance) {
        match data.frontier(m, points) {
            Ok(frontier) => result.frontier = Some(frontier),
            Err(error) => result.frontier_error = Some(error),
        }
    }
    result.rolling = rolling.map(Iterator::collect);
    Ok(())
}

/// Checks `platforms` are some of the sample's, each once.
fn check_platforms(sample: &Sample, platforms: &[String]) -> Result<(), EngineError> {
    if platforms.is_empty() {
        return Err(EngineError::NoPlatforms);
    }
    for (i, platform) in platforms.iter().enumerate() {
        if !sample.has_platform(platform) {
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
///
/// # Errors
///
/// [`CutupError::NonFiniteValue`] if a yield isn't finite, such as a total over a spend so
/// small that their ratio overflows.
fn returns(
    totals: &[Vec<Totals>],
    metric: Metric,
    platforms: &[String],
) -> Result<Returns, CutupError> {
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
    Returns::new(&rows).and_then(|data| data.labeled(platforms.iter().cloned()))
}

/// The statistics a result reports, from cutup's.
fn stats(stats: &cutup::PortfolioStats) -> Result<Stats, CutupError> {
    Ok(Stats {
        expected_return: stats.expected_return(),
        volatility: stats.volatility(),
        sharpe_ratio: stats.sharpe_ratio(0.0),
        diversification_ratio: stats.diversification_ratio(),
        risk_contributions: stats.risk_contributions().map(<[f64]>::to_vec),
        cvar: stats.cvar(CVAR_CONFIDENCE)?,
    })
}

#[cfg(test)]
mod parity;
#[cfg(test)]
mod tests;
