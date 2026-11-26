//! The campaign allocation app's engine: everything the native handler and the wasm wrapper
//! share, so they can't diverge. Both send [`respond`]'s reply: a JSON [`Request`] in, and a
//! [`Response`] or an error out with its HTTP status. [`allocate`] does the work. The types are
//! JSON through serde, with TypeScript types through ts-rs.
//!
//! Each ad platform is an asset, and the weights are shares of the budget. For each metric,
//! each platform's return on a day is how far what a unit of spend bought there, impressions,
//! clicks, link clicks or views per unit, is above or below the average yield, as a share of
//! it: 0.05 is 5% more than the average. The average is the benchmark, as an index is for a
//! fund, and it's over the platforms and days the allocation sees: the request's, or a rolling
//! window's own, so no window is measured against days after it. The returns average 0, so a
//! method's settings, such as a risk aversion, mean the same for every metric, but a platform
//! below the benchmark stays below it. Every method allocates once per metric.
//!
//! These methods take a platform's results to grow in proportion to its spend, so they don't
//! model saturation.

mod sample;

pub use sample::{Metric, PlatformSummary, Sample, Totals};

use cutup::{
    Allocation, CutupError, EqualWeight, FrontierPoint, Herc, Hrp, InverseVariance,
    InverseVolatility, MaxDiversification, MaxSharpe, MeanVariance, MinCdar, MinCvar, MinMad,
    MinVariance, Returns, RiskParity, Weights,
};
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use ts_rs::TS;

/// The platforms a request allocates across when it names none: the three with spend on every
/// day of the sample.
pub const DEFAULT_PLATFORMS: [&str; 3] = ["Meta", "Google", "TikTok"];

/// The confidence of each result's CVaR and CDaR.
pub const CVAR_CONFIDENCE: f64 = 0.95;

/// The most points a frontier can ask for. Each is a solve, once per metric.
pub const MAX_FRONTIER_POINTS: usize = 100;

/// The most rolling windows a request can ask for. Each is a solve, once per metric, and the
/// native function answers within a time limit.
pub const MAX_ROLLING_WINDOWS: usize = 100;

/// The most groups a method's bounds can have. Each is a row of every solve, and a solve's
/// time grows with about the cube of them: 400 take a frontier of 100 points half a minute.
pub const MAX_GROUPS: usize = 32;

/// The most bytes a request's text can be. A request is read whole before any of it is
/// checked, and a covariance matrix given in one is checked as it's read, in time that grows
/// with the cube of its side: a megabyte of one takes seconds. The largest of the parity
/// fixtures' requests is under 200 bytes.
pub const MAX_REQUEST_BYTES: usize = 16 * 1024;

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
    MinMad(MinMad),
    MinCdar(MinCdar),
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
            Method::MinMad(m) => Box::new(m.clone()),
            Method::MinCdar(m) => Box::new(m.clone()),
        }
    }

    /// How many groups the method's bounds have, from the method's own JSON, where its
    /// `bounds` list them: cutup's bounds don't say. `None` for a method that doesn't
    /// serialize: it has a setting that isn't finite, which no request's JSON carries and
    /// `allocate` refuses by itself.
    fn groups(&self) -> Option<usize> {
        let method = serde_json::to_value(self).ok()?;
        let groups = method.get("bounds").and_then(|bounds| bounds.get("groups"));
        let groups = groups.and_then(|groups| groups.as_array());
        Some(groups.map_or(0, Vec::len))
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

/// A portfolio's statistics, from [`cutup::PortfolioStats`], in the units of the returns: an
/// expected return of 0.05 is 5% more of the metric per unit of spend than the average, and a
/// loss is a shortfall from it.
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
    /// The average of the worst 5% of days, as a loss: how far they fall short of the average.
    pub cvar: f64,
    /// The mean absolute deviation of each day's return from their mean.
    pub mad: f64,
    /// How far each day's return falls short of the average on average, the days above it
    /// counting as 0: the mean shortfall that `MinMad`'s downside minimizes, at a threshold of 0.
    pub downside_mad: f64,
    /// The average of the worst 5% of the drawdowns: how far the running sum of the returns,
    /// from 0 before the first day, is below its highest so far, after each day. That's how
    /// far the budget has fallen behind the benchmark since its best, in days' worth of the
    /// average yield. As on any P&L, it includes the drift: a portfolio below the benchmark
    /// falls further behind it every day, besides any run of bad days.
    pub cdar: f64,
    pub max_drawdown: f64,
    pub average_drawdown: f64,
}

/// One rolling window's allocation: the days it covers, as a range of the response's days, and
/// its weights or why the method couldn't allocate over them. Its returns are measured from the
/// average over its own days, so none of them depend on days after it.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct RollingWindow {
    pub start: usize,
    pub end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub weights: Option<Weights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[ts(optional)]
    pub error: Option<CutupError>,
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
    pub rolling: Option<Vec<RollingWindow>>,
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
    /// A metric named twice would be allocated twice, frontier and rolling windows included.
    DuplicateMetric {
        metric: Metric,
    },
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
    /// More groups in the method's bounds than [`MAX_GROUPS`]: each is a row of every solve.
    Groups {
        groups: usize,
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
            EngineError::DuplicateMetric { metric } => {
                // As a request names it, quotes included.
                let name = serde_json::to_string(metric).map_err(|_| std::fmt::Error)?;
                write!(f, "the metric {name} is named twice")
            }
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
            EngineError::Groups { groups } => write!(
                f,
                "{groups} groups in the method's bounds; at most {MAX_GROUPS}"
            ),
        }
    }
}

impl std::error::Error for EngineError {}

/// A reply to a request, for either backend to send as it is: an HTTP status and a JSON body,
/// a [`Response`] or `{"error": EngineError}`.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    /// 200 for a response, 400 for a request that isn't JSON for a [`Request`] or a body that
    /// isn't UTF-8, 405 for the wrong HTTP method, 413 for a request longer than
    /// [`MAX_REQUEST_BYTES`], and 422 for a request that can't be allocated.
    pub status: u16,
    pub body: String,
}

/// A reply's JSON body when it's an error.
#[derive(Debug, Clone, PartialEq, Serialize, TS)]
pub struct ErrorBody {
    pub error: EngineError,
}

/// The JSON reply to a JSON request, over the bundled sample: what both backends send, so
/// they can't diverge. A request longer than [`MAX_REQUEST_BYTES`] is refused before it's read.
pub fn respond(request: &str) -> Reply {
    if request.len() > MAX_REQUEST_BYTES {
        let bytes = request.len();
        return reject(
            413,
            &format!("a request of {bytes} bytes; at most {MAX_REQUEST_BYTES}"),
        );
    }
    let request = match serde_json::from_str::<Request>(request) {
        Ok(request) => request,
        Err(e) => return reject(400, &e.to_string()),
    };
    match allocate(&request) {
        Ok(response) => Reply {
            status: 200,
            body: serde_json::to_string(&response).expect("responses serialize"),
        },
        Err(error) => error_reply(422, error),
    }
}

/// A reply refusing a request before it's read as one, such as a 405 for the wrong HTTP method,
/// a 400 for a body that isn't UTF-8 or a 413 for one that's too long: `status`, with an
/// [`EngineError::Request`] saying `message`.
pub fn reject(status: u16, message: &str) -> Reply {
    let error = EngineError::Request {
        message: message.to_owned(),
    };
    error_reply(status, error)
}

/// `status`, with `{"error": error}`.
fn error_reply(status: u16, error: EngineError) -> Reply {
    Reply {
        status,
        body: serde_json::to_string(&ErrorBody { error }).expect("errors serialize"),
    }
}

/// The bundled sample's platforms, as JSON: a [`PlatformSummary`] for each, by name. Built
/// once, since the sample doesn't change.
pub fn platforms() -> String {
    static PLATFORMS: OnceLock<String> = OnceLock::new();
    PLATFORMS
        .get_or_init(|| {
            serde_json::to_string(&Sample::bundled().platforms()).expect("summaries serialize")
        })
        .clone()
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
    check_metrics(&request.metrics)?;
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
    // Groups that can't be counted are those of a method with a setting that isn't finite.
    // Every allocation refuses the setting before it solves, but a frontier doesn't take the
    // objective, so one that isn't finite would be traced with however many groups there are.
    let groups = request.method.groups();
    if let Some(groups) = groups.filter(|&groups| groups > MAX_GROUPS) {
        return Err(EngineError::Groups { groups });
    }
    let counted = groups.is_some();
    let (days, totals): (Vec<&str>, Vec<Vec<Totals>>) = sample
        .days_with_spend(platforms, request.from.as_deref(), request.to.as_deref())
        .unzip();
    if days.len() < 2 {
        return Err(EngineError::TooFewDays { days: days.len() });
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
        let yields = yields(&totals, metric);
        match excess(&yields, platforms) {
            Ok(data) => allocate_metric(
                &mut result,
                &data,
                &yields,
                method.as_ref(),
                request,
                mean_variance,
                counted,
            )?,
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

/// Fills in `result` from one metric's returns, `data`, and the yields they come from. The
/// rolling windows are checked first, through cutup's, which count them without solving, so a
/// request whose windows don't fit, or are more than [`MAX_ROLLING_WINDOWS`], fails before
/// anything is solved. A method whose groups weren't `counted` gets no frontier: its
/// allocation's error is the frontier's too.
fn allocate_metric(
    result: &mut MetricResult,
    data: &Returns,
    yields: &[Vec<f64>],
    method: &dyn Allocation,
    request: &Request,
    mean_variance: Option<&MeanVariance>,
    counted: bool,
) -> Result<(), EngineError> {
    let rolling = match request.rolling {
        Some(rolling) => {
            let count = data
                .rolling(rolling.window)
                .step(rolling.step)
                .allocate(method)
                .map_err(|error| EngineError::Rolling { error })?
                .len();
            if count > MAX_ROLLING_WINDOWS {
                return Err(EngineError::RollingWindows { windows: count });
            }
            Some((rolling, count))
        }
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
        if counted {
            match data.frontier(m, points) {
                Ok(frontier) => result.frontier = Some(frontier),
                Err(error) => result.frontier_error = Some(error),
            }
        } else {
            result.frontier_error = result.error.clone();
        }
    }
    result.rolling = rolling.map(|(rolling, count)| {
        windows(yields.len(), rolling, count)
            .map(|periods| {
                let weights = excess(&yields[periods.clone()], data.labels())
                    .and_then(|window| window.allocate(method));
                RollingWindow {
                    start: periods.start,
                    end: periods.end,
                    weights: weights.as_ref().ok().cloned(),
                    error: weights.err(),
                }
            })
            .collect()
    });
    Ok(())
}

/// The days each of `count` rolling windows covers, oldest first, as cutup lays them out:
/// `window` days each, `step` apart, the last ending at the last of `days`.
fn windows(
    days: usize,
    rolling: RollingRequest,
    count: usize,
) -> impl Iterator<Item = std::ops::Range<usize>> {
    let last = days - rolling.window;
    (0..count).map(move |k| {
        let start = last - (count - 1 - k) * rolling.step;
        start..start + rolling.window
    })
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

/// Checks there's at least one of `metrics`, each once: every one is allocated on its own, so
/// one named again would repeat all of its solves.
fn check_metrics(metrics: &[Metric]) -> Result<(), EngineError> {
    if metrics.is_empty() {
        return Err(EngineError::NoMetrics);
    }
    match (1..metrics.len()).find(|&i| metrics[..i].contains(&metrics[i])) {
        Some(i) => Err(EngineError::DuplicateMetric { metric: metrics[i] }),
        None => Ok(()),
    }
}

/// Each platform's yield of `metric` on each day: what a unit of spend bought.
fn yields(totals: &[Vec<Totals>], metric: Metric) -> Vec<Vec<f64>> {
    totals
        .iter()
        .map(|day| day.iter().map(|t| t.yield_of(metric)).collect())
        .collect()
}

/// Each of `yields`, one row per day, divided by their average over every platform and day,
/// less 1, labeled with the platforms: how far above or below the average each is, as a share
/// of it.
///
/// # Errors
///
/// [`CutupError::NonFiniteValue`] if a yield isn't finite, such as a total over a spend so
/// small that their ratio overflows, and cutup's errors for too few days.
fn excess(yields: &[Vec<f64>], platforms: &[String]) -> Result<Returns, CutupError> {
    let count = (yields.len() * platforms.len()) as f64;
    // Yields whose sum overflows are each divided by the count first: their average is
    // finite, where an infinite one would leave every return at −1.
    let sum = yields.iter().flatten().sum::<f64>();
    let mean = if sum.is_finite() {
        sum / count
    } else {
        yields.iter().flatten().map(|y| y / count).sum()
    };
    // A metric none of the platforms delivered has no yield to compare, so its returns are 0.
    let rows: Vec<Vec<f64>> = yields
        .iter()
        .map(|day| {
            day.iter()
                .map(|&y| if mean > 0.0 { y / mean - 1.0 } else { 0.0 })
                .collect()
        })
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
        mad: stats.mad(),
        downside_mad: stats.downside_mad(0.0)?,
        cdar: stats.cdar(CVAR_CONFIDENCE)?,
        max_drawdown: stats.max_drawdown(),
        average_drawdown: stats.average_drawdown(),
    })
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod bindings;
#[cfg(test)]
mod parity;
#[cfg(test)]
mod tests;
