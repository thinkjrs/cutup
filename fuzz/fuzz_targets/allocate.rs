//! Every allocation method, with arbitrary settings, on returns built from arbitrary data. Any
//! of them may return an error, but none may panic, and weights they return must be finite, one
//! per asset, and sum to 1, and with groups each group's total must be within its limits, to
//! 1e-9. Their statistics must not be NaN, and their ratios and risk
//! contributions must be finite, with the contributions summing to 1. The expected return,
//! volatility and CVaR can overflow, when the weights times returns near `f64::MAX` do. Their
//! statistics under an arbitrary covariance config, if it's valid for the data, must keep the
//! expected return and CVaR, and have shares summing to 1 and no NaN; there, a Sharpe ratio past
//! `f64::MAX` is infinite, against a volatility far smaller than the returns. An expected-return
//! estimate may fail only for its parameters, a riskless market, a covariance Bayes-Stein
//! can't invert, or one whose eigenvalues a Stein mean can't compute, and is otherwise one value
//! per asset, never NaN, and finite but for a CAPM return past `f64::MAX`.
//!
//! Every method, written to JSON with no `null` in it, reads back equal, and so do its
//! weights. Rolling windows of an arbitrary length and step either fail at once for a window or
//! step out of range, or are as many as fit, each as long as asked, `step` apart and ending at
//! the last period, and the windows that allocate have weights as above.

#![no_main]

use arbitrary::Arbitrary;
use cutup::{
    Allocation, Asset, BlackLitterman, CovarianceConfig, CutupError, EqualWeight, Herc, Hrp,
    InverseVariance, InverseVolatility, Linkage, MaxDiversification, MaxSharpe, MeanVariance,
    MinCvar, MinVariance, PerAsset, Returns, RiskParity, ShrinkageTarget, View, WeightBounds,
    Weights,
};
use libfuzzer_sys::fuzz_target;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;

/// The most assets: enough for clusters and binding bounds, few enough to stay fast.
const MAX_ASSETS: usize = 8;

/// Returns as small integers times a power of ten, so that most inputs are finite and pass the
/// constructor, while the exponent still reaches subnormal and overflowing scales.
#[derive(Arbitrary, Debug)]
struct Data {
    assets: u8,
    values: Vec<i16>,
    exponent: i16,
    labels: Option<Vec<String>>,
}

#[derive(Arbitrary, Clone, Debug)]
enum Key {
    Column(u8),
    Label(String),
}

impl From<Key> for Asset {
    fn from(key: Key) -> Self {
        match key {
            Key::Column(column) => Asset::from(usize::from(column)),
            Key::Label(label) => Asset::from(label),
        }
    }
}

/// A value per asset, in column order or by asset.
#[derive(Arbitrary, Debug)]
enum Values {
    InOrder(Vec<f64>),
    ByAsset(Vec<(Key, f64)>),
}

impl From<Values> for PerAsset {
    fn from(values: Values) -> Self {
        match values {
            Values::InOrder(values) => PerAsset::from(values),
            Values::ByAsset(pairs) => {
                let pairs: Vec<(Asset, f64)> = pairs
                    .into_iter()
                    .map(|(k, v)| (Asset::from(k), v))
                    .collect();
                PerAsset::from(pairs)
            }
        }
    }
}

#[derive(Arbitrary, Debug)]
enum Bounds {
    Uniform(f64, f64),
    PerAsset(Values, Values),
    Asset(f64, f64, Key, f64, f64),
}

impl From<Bounds> for WeightBounds {
    fn from(bounds: Bounds) -> Self {
        match bounds {
            Bounds::Uniform(lower, upper) => WeightBounds::uniform(lower, upper),
            Bounds::PerAsset(lower, upper) => WeightBounds::per_asset(lower, upper),
            Bounds::Asset(lower, upper, key, l, u) => {
                WeightBounds::uniform(lower, upper).asset(key, l, u)
            }
        }
    }
}

/// Limits on the total weight of `members`.
#[derive(Arbitrary, Clone, Debug)]
struct Group {
    members: Vec<Key>,
    lower: f64,
    upper: f64,
}

/// Bounds, or the default ones, with groups on top.
#[derive(Arbitrary, Debug)]
struct Grouped {
    bounds: Option<Bounds>,
    groups: Vec<Group>,
}

impl From<Grouped> for WeightBounds {
    fn from(grouped: Grouped) -> Self {
        let bounds = grouped
            .bounds
            .map_or_else(WeightBounds::default, Into::into);
        grouped.groups.into_iter().fold(bounds, |bounds, g| {
            bounds.group(g.members, g.lower, g.upper)
        })
    }
}

/// The groups `method`'s bounds have, if any, to check its weights against.
fn groups(method: &Method) -> Vec<Group> {
    match method {
        Method::MinVariance { bounds, .. }
        | Method::MeanVariance { bounds, .. }
        | Method::MinCvar { bounds, .. } => bounds
            .as_ref()
            .map_or_else(Vec::new, |grouped| grouped.groups.clone()),
        _ => Vec::new(),
    }
}

/// Checks that each group's members, by column or by label and each counted once, hold a
/// total within its limits, to 1e-9.
fn check_groups(data: &Returns, weights: &[f64], groups: &[Group]) {
    for group in groups {
        let mut members: Vec<usize> = group
            .members
            .iter()
            .map(|key| match key {
                Key::Column(column) => usize::from(*column),
                Key::Label(label) => data
                    .labels()
                    .iter()
                    .position(|l| l == label)
                    .expect("weights from allocate have every member"),
            })
            .collect();
        members.sort_unstable();
        members.dedup();
        let total: f64 = members.iter().map(|&i| weights[i]).sum();
        assert!(
            group.lower - 1e-9 <= total && total <= group.upper + 1e-9,
            "{group:?} holds {total}: {weights:?}"
        );
    }
}

#[derive(Arbitrary, Debug)]
struct Covariance {
    shrinkage: Option<f64>,
    regularization: Option<f64>,
    given: Option<Given>,
    computed: Option<Computed>,
    ledoit_wolf: Option<LedoitWolf>,
}

/// An expected-return estimate from the returns, with any parameter.
#[derive(Arbitrary, Debug)]
enum Estimate {
    HalfLife(f64),
    Decay(f64),
    Capm(Values, f64),
    JamesStein,
    BayesStein,
}

impl Estimate {
    fn estimate(self, data: &Returns) -> Result<Vec<f64>, CutupError> {
        match self {
            Estimate::HalfLife(h) => data.exponentially_weighted_mean(h),
            Estimate::Decay(lambda) => data.exponentially_weighted_mean_decay(lambda),
            Estimate::Capm(market, rate) => data.capm_returns(market, rate),
            Estimate::JamesStein => data.james_stein_mean(),
            Estimate::BayesStein => data.bayes_stein_mean(),
        }
    }
}

/// A covariance estimated from the returns in place of the sample one, with any parameter.
#[derive(Arbitrary, Debug)]
enum Computed {
    HalfLife(f64),
    Decay(f64),
    Semicovariance,
    Below(f64),
}

impl Computed {
    fn config(self) -> CovarianceConfig {
        match self {
            Computed::HalfLife(h) => CovarianceConfig::exponentially_weighted(h),
            Computed::Decay(lambda) => CovarianceConfig::exponentially_weighted_decay(lambda),
            Computed::Semicovariance => CovarianceConfig::semicovariance(),
            Computed::Below(threshold) => CovarianceConfig::semicovariance_below(threshold),
        }
    }
}

/// Ledoit-Wolf shrinkage, set before or after a fixed intensity: the last one set applies.
#[derive(Arbitrary, Debug)]
struct LedoitWolf {
    constant_correlation: bool,
    last: bool,
}

impl LedoitWolf {
    fn target(&self) -> ShrinkageTarget {
        if self.constant_correlation {
            ShrinkageTarget::ConstantCorrelation
        } else {
            ShrinkageTarget::ConstantVariance
        }
    }
}

/// A covariance given in place of the sample one, for `CovarianceConfig::matrix_flat`.
#[derive(Arbitrary, Debug)]
enum Given {
    /// `BBᵀ` for `B` of small integers times a power of ten, as for `Data`: semidefinite, but
    /// for rounding.
    Factors {
        assets: u8,
        values: Vec<i16>,
        exponent: i16,
    },
    /// Arbitrary values, which the constructor must check without panicking.
    Raw { assets: u8, values: Vec<f64> },
}

/// The covariance config for `given`, if its constructor accepts it.
fn given(given: Given) -> Option<CovarianceConfig> {
    let (assets, values) = match given {
        Given::Factors {
            assets,
            values,
            exponent,
        } => {
            let n = usize::from(assets) % MAX_ASSETS + 1;
            let scale = 10f64.powi(i32::from(exponent % 330));
            let k = (values.len() / n).min(MAX_ASSETS);
            let b = |i: usize, j: usize| f64::from(values[i * k + j]) * scale;
            let product = (0..n * n)
                .map(|e| (0..k).map(|j| b(e / n, j) * b(e % n, j)).sum())
                .collect();
            (n, product)
        }
        Given::Raw { assets, values } => (usize::from(assets) % MAX_ASSETS + 1, values),
    };
    CovarianceConfig::matrix_flat(&values, assets).ok()
}

#[derive(Arbitrary, Debug)]
struct Trading {
    current: Option<Values>,
    max_turnover: Option<f64>,
    turnover_penalty: Option<f64>,
}

#[derive(Arbitrary, Debug)]
enum Objective {
    RiskAversion(f64),
    TargetReturn(f64),
    TargetVolatility(f64),
}

#[derive(Arbitrary, Debug)]
enum Costs {
    Uniform(f64),
    PerAsset(Values),
}

#[derive(Arbitrary, Debug)]
enum ViewInput {
    Absolute(Key, f64),
    Relative(Key, Key, f64),
    Portfolio(Values, f64),
}

#[derive(Arbitrary, Debug)]
struct ViewSpec {
    view: ViewInput,
    variance: Option<f64>,
    confidence: Option<f64>,
}

#[derive(Arbitrary, Debug)]
struct BlackLittermanSpec {
    market_weights: Values,
    delta: f64,
    views: Vec<ViewSpec>,
    risk_free_rate: Option<f64>,
    tau: Option<f64>,
    covariance: Covariance,
}

#[derive(Arbitrary, Debug)]
enum Method {
    EqualWeight,
    InverseVolatility,
    InverseVariance,
    MinVariance {
        allow_short: bool,
        bounds: Option<Grouped>,
        trading: Trading,
        covariance: Covariance,
    },
    MeanVariance {
        objective: Objective,
        expected_returns: Option<Values>,
        allow_short: bool,
        bounds: Option<Grouped>,
        trading: Trading,
        costs: Option<Costs>,
        covariance: Covariance,
        /// Also checks the method's efficient frontier, with this many points.
        frontier: Option<u8>,
    },
    MaxSharpe {
        risk_free_rate: Option<f64>,
        expected_returns: Option<Values>,
        allow_short: bool,
        covariance: Covariance,
    },
    MaxDiversification {
        covariance: Covariance,
    },
    RiskParity {
        risk_budgets: Option<Values>,
        covariance: Covariance,
    },
    Hrp {
        linkage: u8,
        covariance: Covariance,
    },
    Herc {
        linkage: u8,
        clusters: Option<u8>,
        max_clusters: Option<u8>,
        covariance: Covariance,
    },
    MinCvar {
        confidence: Option<f64>,
        bounds: Option<Grouped>,
        target_return: Option<f64>,
    },
    /// Black-Litterman expected returns, into mean-variance or max Sharpe, with its posterior
    /// covariance or the method's own.
    BlackLitterman {
        spec: BlackLittermanSpec,
        max_sharpe: bool,
        posterior_covariance: bool,
    },
    /// Estimated expected returns, into mean-variance or max Sharpe.
    Estimated {
        estimate: Estimate,
        max_sharpe: bool,
        covariance: Covariance,
    },
}

#[derive(Arbitrary, Debug)]
struct Input {
    data: Data,
    method: Method,
    /// The covariance for `Returns::stats_with`.
    stats_covariance: Covariance,
    /// Rolling windows to allocate over too: their length and step.
    rolling: Option<(u8, u8)>,
}

/// The covariance config `covariance` sets, for `Returns::stats_with` and, through the
/// `covariance!` macro, any method's `covariance` setter.
fn config(covariance: Covariance) -> CovarianceConfig {
    let mut config = CovarianceConfig::default();
    if let Some(computed) = covariance.computed {
        config = computed.config();
    }
    if let Some(given) = covariance.given.and_then(given) {
        config = given;
    }
    let ledoit_wolf = covariance.ledoit_wolf;
    if let Some(lw) = ledoit_wolf.as_ref().filter(|lw| !lw.last) {
        config = config.ledoit_wolf(lw.target());
    }
    if let Some(lambda) = covariance.shrinkage {
        config = config.shrinkage(lambda);
    }
    if let Some(lw) = ledoit_wolf.as_ref().filter(|lw| lw.last) {
        config = config.ledoit_wolf(lw.target());
    }
    if let Some(epsilon) = covariance.regularization {
        config = config.regularization(epsilon);
    }
    config
}

/// The returns `data` builds, and how many periods they have.
fn returns(data: Data) -> Option<(Returns, usize)> {
    let assets = usize::from(data.assets) % MAX_ASSETS + 1;
    let scale = 10f64.powi(i32::from(data.exponent % 330));
    let values: Vec<f64> = data.values.iter().map(|&v| f64::from(v) * scale).collect();
    let periods = values.len() / assets;
    let returns = Returns::new_flat(&values[..periods * assets], assets).ok()?;
    let returns = match data.labels {
        Some(labels) => returns.labeled(labels).ok()?,
        None => returns,
    };
    Some((returns, periods))
}

fn linkage(choice: u8) -> Linkage {
    match choice % 4 {
        0 => Linkage::Single,
        1 => Linkage::Complete,
        2 => Linkage::Average,
        _ => Linkage::Ward,
    }
}

/// Applies the covariance settings with any method's `covariance` setter, as [`config`] builds
/// them: a given covariance takes the place of a computed one, and one the constructor rejects
/// is left out.
macro_rules! covariance {
    ($method:expr, $covariance:expr) => {
        $method.covariance(config($covariance))
    };
}

/// Applies current weights and a turnover cap and penalty.
macro_rules! trading {
    ($method:expr, $trading:expr) => {{
        let mut method = $method;
        if let Some(current) = $trading.current {
            method = method.current_weights(current);
        }
        if let Some(cap) = $trading.max_turnover {
            method = method.max_turnover(cap);
        }
        if let Some(kappa) = $trading.turnover_penalty {
            method = method.turnover_penalty(kappa);
        }
        method
    }};
}

fn black_litterman(spec: BlackLittermanSpec) -> BlackLitterman {
    let mut model = BlackLitterman::new(spec.market_weights, spec.delta);
    for spec in spec.views {
        let mut view = match spec.view {
            ViewInput::Absolute(asset, q) => View::absolute(asset, q),
            ViewInput::Relative(long, short, q) => View::relative(long, short, q),
            ViewInput::Portfolio(weights, q) => View::portfolio(weights, q),
        };
        if let Some(omega) = spec.variance {
            view = view.variance(omega);
        }
        if let Some(confidence) = spec.confidence {
            view = view.confidence(confidence);
        }
        model = model.view(view);
    }
    if let Some(rate) = spec.risk_free_rate {
        model = model.risk_free_rate(rate);
    }
    if let Some(tau) = spec.tau {
        model = model.tau(tau);
    }
    covariance!(model, spec.covariance)
}

/// `method`, boxed, after checking it reads back from its JSON equal to itself. A `null` in
/// the JSON stands for an infinite or NaN value, which reads back as infinite or unset, so
/// methods with one are only written.
fn boxed<M>(method: M) -> Box<dyn Allocation>
where
    M: Allocation + Serialize + DeserializeOwned + PartialEq + Debug + 'static,
{
    let json = serde_json::to_string(&method).expect("a method serializes");
    if !json.contains("null") {
        let back: M = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{json}: {e}"));
        assert_eq!(back, method, "{json}");
    }
    Box::new(method)
}

/// Checks that `weights` are finite, one per asset of `data` with its labels, and sum to 1.
fn check_weights(data: &Returns, weights: &Weights) {
    assert_eq!(weights.labels(), data.labels());
    assert!(weights.iter().all(|w| w.is_finite()), "{weights:?}");
    // Divided by the largest magnitude, so that huge weights can't overflow the gross sum and
    // make the tolerance infinite.
    let largest = weights.iter().fold(1.0, |m: f64, w| m.max(w.abs()));
    let sum: f64 = weights.iter().map(|w| w / largest).sum();
    let gross: f64 = weights.iter().map(|w| (w / largest).abs()).sum();
    assert!(
        (sum - 1.0 / largest).abs() <= 1e-6 * gross.max(1.0 / largest),
        "weights sum to {}: {weights:?}",
        sum * largest
    );
}

/// Checks rolling windows of `window` periods, `step` apart, over `data`: an error at once if
/// either is out of range, and otherwise as many windows as fit, the last ending at the last
/// period. Checks the first, second and last, so that a long run of windows doesn't allocate
/// each one.
fn check_rolling(
    data: &Returns,
    periods: usize,
    method: &dyn Allocation,
    window: usize,
    step: usize,
) {
    let windows = match data.rolling(window).step(step).allocate(method) {
        Ok(windows) => windows,
        Err(e) => {
            assert!(window < 2 || window > periods || step == 0, "{e:?}");
            assert!(matches!(e, CutupError::InvalidConfig { .. }), "{e:?}");
            return;
        }
    };
    assert!(window >= 2 && window <= periods && step >= 1);
    let count = (periods - window) / step + 1;
    assert_eq!(windows.len(), count);
    let mut windows = windows;
    // The first two in order, and the last, without allocating the ones between.
    let (first, second) = (windows.next(), windows.next());
    let last = windows.last();
    let start = periods - window - (count - 1) * step;
    let reached = [(0, first), (1, second), (count - 1, last)];
    let reached: Vec<_> = reached
        .into_iter()
        .filter_map(|(k, w)| w.map(|w| (k, w)))
        .collect();
    assert_eq!(reached.len(), count.min(3));
    for (k, reached) in reached {
        assert_eq!(
            reached.periods(),
            start + k * step..start + k * step + window
        );
        if let Ok(weights) = reached.weights() {
            check_weights(data, weights);
        }
        serde_json::to_string(&reached).expect("a window serializes");
    }
}

/// Checks that `method`'s efficient frontier, if it has one, is `points` finite portfolios, or
/// one, each with a return that isn't NaN and a volatility that isn't negative.
fn check_frontier(data: &Returns, method: &MeanVariance, points: usize) {
    let Ok(frontier) = data.frontier(method, points) else {
        return;
    };
    assert!(points >= 2, "{points} points");
    assert!(
        frontier.len() == points || frontier.len() == 1,
        "{frontier:?}"
    );
    for point in &frontier {
        assert_eq!(point.weights().len(), data.labels().len());
        assert!(point.weights().iter().all(|w| w.is_finite()), "{point:?}");
        assert!(!point.expected_return().is_nan(), "{point:?}");
        assert!(point.volatility() >= 0.0, "{point:?}");
    }
}

fn method(method: Method, data: &Returns) -> Option<Box<dyn Allocation>> {
    Some(match method {
        Method::EqualWeight => boxed(EqualWeight),
        Method::InverseVolatility => boxed(InverseVolatility),
        Method::InverseVariance => boxed(InverseVariance),
        Method::MinVariance {
            allow_short,
            bounds,
            trading,
            covariance,
        } => {
            let mut m = MinVariance::default();
            if allow_short {
                m = m.allow_short();
            }
            if let Some(bounds) = bounds {
                m = m.bounds(bounds.into());
            }
            boxed(covariance!(trading!(m, trading), covariance))
        }
        Method::MeanVariance {
            objective,
            expected_returns,
            allow_short,
            bounds,
            trading,
            costs,
            covariance,
            frontier,
        } => {
            let mut m = match objective {
                Objective::RiskAversion(gamma) => MeanVariance::risk_aversion(gamma),
                Objective::TargetReturn(target) => MeanVariance::target_return(target),
                Objective::TargetVolatility(sigma) => MeanVariance::target_volatility(sigma),
            };
            if let Some(mu) = expected_returns {
                m = m.expected_returns(mu);
            }
            if allow_short {
                m = m.allow_short();
            }
            if let Some(bounds) = bounds {
                m = m.bounds(bounds.into());
            }
            match costs {
                Some(Costs::Uniform(cost)) => m = m.transaction_costs(cost),
                Some(Costs::PerAsset(costs)) => m = m.transaction_costs_per_asset(costs),
                None => {}
            }
            let m = covariance!(trading!(m, trading), covariance);
            if let Some(points) = frontier {
                // Up to 15 points: each is a solve, and more only repeat the same work.
                check_frontier(data, &m, usize::from(points % 16));
            }
            boxed(m)
        }
        Method::MaxSharpe {
            risk_free_rate,
            expected_returns,
            allow_short,
            covariance,
        } => {
            let mut m = MaxSharpe::default();
            if let Some(rate) = risk_free_rate {
                m = m.risk_free_rate(rate);
            }
            if let Some(mu) = expected_returns {
                m = m.expected_returns(mu);
            }
            if allow_short {
                m = m.allow_short();
            }
            boxed(covariance!(m, covariance))
        }
        Method::MaxDiversification { covariance } => {
            boxed(covariance!(MaxDiversification::default(), covariance))
        }
        Method::RiskParity {
            risk_budgets,
            covariance,
        } => {
            let mut m = RiskParity::default();
            if let Some(budgets) = risk_budgets {
                m = m.risk_budgets(budgets);
            }
            boxed(covariance!(m, covariance))
        }
        Method::Hrp {
            linkage: choice,
            covariance,
        } => boxed(covariance!(
            Hrp::default().linkage(linkage(choice)),
            covariance
        )),
        Method::Herc {
            linkage: choice,
            clusters,
            max_clusters,
            covariance,
        } => {
            let mut m = Herc::default().linkage(linkage(choice));
            if let Some(k) = clusters {
                m = m.clusters(usize::from(k));
            }
            if let Some(k) = max_clusters {
                m = m.max_clusters(usize::from(k));
            }
            boxed(covariance!(m, covariance))
        }
        Method::MinCvar {
            confidence,
            bounds,
            target_return,
        } => {
            let mut m = MinCvar::default();
            if let Some(beta) = confidence {
                m = m.confidence(beta);
            }
            if let Some(bounds) = bounds {
                m = m.bounds(bounds.into());
            }
            if let Some(target) = target_return {
                m = m.target_return(target);
            }
            boxed(m)
        }
        Method::Estimated {
            estimate,
            max_sharpe,
            covariance,
        } => {
            // An estimate fails only for its parameters, a riskless market, a covariance
            // Bayes-Stein can't invert, or one whose eigenvalues a Stein mean can't compute.
            // Otherwise it's one per asset, never NaN, and finite but for a CAPM return past
            // f64::MAX.
            let capm = matches!(estimate, Estimate::Capm(..));
            let bayes = matches!(estimate, Estimate::BayesStein);
            let stein = bayes || matches!(estimate, Estimate::JamesStein);
            let mu = match estimate.estimate(data) {
                Ok(mu) => mu,
                Err(
                    CutupError::InvalidConfig { .. }
                    | CutupError::InvalidAssetValue { .. }
                    | CutupError::DimensionMismatch { .. }
                    | CutupError::UnknownAsset { .. }
                    | CutupError::DuplicateAsset { .. }
                    | CutupError::RisklessMarket,
                ) if capm => return None,
                Err(CutupError::ZeroVariance { .. }) if bayes => return None,
                Err(CutupError::SingularCovariance) if stein => return None,
                Err(CutupError::InvalidConfig { .. }) if !stein => return None,
                Err(e) => panic!("{e:?}"),
            };
            assert_eq!(mu.len(), data.labels().len());
            assert!(
                mu.iter().all(|m| m.is_finite() || capm && !m.is_nan()),
                "{mu:?}"
            );
            if max_sharpe {
                boxed(covariance!(
                    MaxSharpe::default().expected_returns(mu),
                    covariance
                ))
            } else {
                boxed(covariance!(
                    MeanVariance::risk_aversion(1.0).expected_returns(mu),
                    covariance
                ))
            }
        }
        Method::BlackLitterman {
            spec,
            max_sharpe,
            posterior_covariance,
        } => {
            let model = black_litterman(spec);
            let _ = model.prior_returns(data);
            let covariance = model.posterior_covariance(data);
            let mu = model.expected_returns(data).ok()?;
            // Both solve the same views, so they fail together, except that the covariance
            // can overflow where the returns don't.
            assert!(
                covariance.is_ok() || covariance == Err(CutupError::NonFiniteCovariance),
                "{covariance:?}"
            );
            let covariance = if posterior_covariance {
                covariance.ok()?
            } else {
                CovarianceConfig::default()
            };
            if max_sharpe {
                boxed(
                    MaxSharpe::default()
                        .expected_returns(mu)
                        .covariance(covariance),
                )
            } else {
                boxed(
                    MeanVariance::risk_aversion(1.0)
                        .expected_returns(mu)
                        .covariance(covariance),
                )
            }
        }
    })
}

fuzz_target!(|input: Input| {
    let Some((data, periods)) = returns(input.data) else {
        return;
    };
    let groups = groups(&input.method);
    let Some(method) = method(input.method, &data) else {
        return;
    };
    if let Some((window, step)) = input.rolling {
        let (window, step) = (usize::from(window), usize::from(step) % 8);
        check_rolling(&data, periods, method.as_ref(), window, step);
    }
    let Ok(weights) = data.allocate(method.as_ref()) else {
        return;
    };
    check_weights(&data, &weights);
    check_groups(&data, weights.as_slice(), &groups);
    let json = serde_json::to_string(&weights).expect("weights serialize");
    let back: Weights = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{json}: {e}"));
    assert_eq!(back, weights);

    let stats = data
        .stats(&weights)
        .expect("weights from allocate fit their data");
    assert!(!stats.expected_return().is_nan(), "{stats:?}");
    assert!(stats.volatility() >= 0.0, "{stats:?}");
    assert!(stats.cvar(0.9).is_ok_and(|c| !c.is_nan()), "{stats:?}");
    assert_eq!(stats.sharpe_ratio(0.0).is_some(), stats.volatility() > 0.0);
    assert!(
        stats.sharpe_ratio(0.0).is_none_or(f64::is_finite),
        "{stats:?}"
    );
    assert!(
        stats.diversification_ratio().is_none_or(f64::is_finite),
        "{stats:?}"
    );
    if let Some(shares) = stats.risk_contributions() {
        assert!(shares.iter().all(|s| s.is_finite()), "{stats:?}");
        let largest = shares.iter().fold(1.0, |m: f64, s| m.max(s.abs()));
        let total: f64 = shares.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-6 * largest,
            "risk contributions sum to {total}"
        );
    }

    let Ok(with) = data.stats_with(&weights, &config(input.stats_covariance)) else {
        return;
    };
    assert_eq!(
        with.expected_return().to_bits(),
        stats.expected_return().to_bits()
    );
    assert_eq!(
        with.cvar(0.9).map(f64::to_bits),
        stats.cvar(0.9).map(f64::to_bits)
    );
    assert!(with.volatility() >= 0.0, "{with:?}");
    assert_eq!(with.sharpe_ratio(0.0).is_some(), with.volatility() > 0.0);
    assert!(
        with.sharpe_ratio(0.0).is_none_or(|s| !s.is_nan()),
        "{with:?}"
    );
    assert!(
        with.diversification_ratio().is_none_or(f64::is_finite),
        "{with:?}"
    );
    if let Some(shares) = with.risk_contributions() {
        assert!(shares.iter().all(|s| s.is_finite()), "{with:?}");
        let largest = shares.iter().fold(1.0, |m: f64, s| m.max(s.abs()));
        let total: f64 = shares.iter().sum();
        assert!(
            (total - 1.0).abs() <= 1e-6 * largest,
            "risk contributions under a config sum to {total}"
        );
    }
});
