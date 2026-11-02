//! Every allocation method, with arbitrary settings, on returns built from arbitrary data. Any
//! of them may return an error, but none may panic, and weights they return must be finite, one
//! per asset, and sum to 1.

#![no_main]

use arbitrary::Arbitrary;
use cutup::{
    Allocation, Asset, BlackLitterman, EqualWeight, Herc, Hrp, InverseVariance, InverseVolatility,
    Linkage, MaxDiversification, MaxSharpe, MeanVariance, MinCvar, MinVariance, PerAsset, Returns,
    RiskParity, View, WeightBounds,
};
use libfuzzer_sys::fuzz_target;

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

#[derive(Arbitrary, Debug)]
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

#[derive(Arbitrary, Debug)]
struct Covariance {
    shrinkage: Option<f64>,
    regularization: Option<f64>,
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
        bounds: Option<Bounds>,
        trading: Trading,
        covariance: Covariance,
    },
    MeanVariance {
        objective: Objective,
        expected_returns: Option<Values>,
        allow_short: bool,
        bounds: Option<Bounds>,
        trading: Trading,
        costs: Option<Costs>,
        covariance: Covariance,
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
        bounds: Option<Bounds>,
        target_return: Option<f64>,
    },
    /// Black-Litterman expected returns, into mean-variance or max Sharpe.
    BlackLitterman {
        spec: BlackLittermanSpec,
        max_sharpe: bool,
    },
}

#[derive(Arbitrary, Debug)]
struct Input {
    data: Data,
    method: Method,
}

fn returns(data: Data) -> Option<Returns> {
    let assets = usize::from(data.assets) % MAX_ASSETS + 1;
    let scale = 10f64.powi(i32::from(data.exponent % 330));
    let values: Vec<f64> = data.values.iter().map(|&v| f64::from(v) * scale).collect();
    let periods = values.len() / assets;
    let returns = Returns::new_flat(&values[..periods * assets], assets).ok()?;
    match data.labels {
        Some(labels) => returns.labeled(labels).ok(),
        None => Some(returns),
    }
}

fn linkage(choice: u8) -> Linkage {
    match choice % 4 {
        0 => Linkage::Single,
        1 => Linkage::Complete,
        2 => Linkage::Average,
        _ => Linkage::Ward,
    }
}

/// Applies the covariance settings with any method's `shrinkage` and `regularization`.
macro_rules! covariance {
    ($method:expr, $covariance:expr) => {{
        let mut method = $method;
        if let Some(lambda) = $covariance.shrinkage {
            method = method.shrinkage(lambda);
        }
        if let Some(epsilon) = $covariance.regularization {
            method = method.regularization(epsilon);
        }
        method
    }};
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

fn method(method: Method, data: &Returns) -> Option<Box<dyn Allocation>> {
    Some(match method {
        Method::EqualWeight => Box::new(EqualWeight),
        Method::InverseVolatility => Box::new(InverseVolatility),
        Method::InverseVariance => Box::new(InverseVariance),
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
            Box::new(covariance!(trading!(m, trading), covariance))
        }
        Method::MeanVariance {
            objective,
            expected_returns,
            allow_short,
            bounds,
            trading,
            costs,
            covariance,
        } => {
            let mut m = match objective {
                Objective::RiskAversion(gamma) => MeanVariance::risk_aversion(gamma),
                Objective::TargetReturn(target) => MeanVariance::target_return(target),
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
            Box::new(covariance!(trading!(m, trading), covariance))
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
            Box::new(covariance!(m, covariance))
        }
        Method::MaxDiversification { covariance } => {
            Box::new(covariance!(MaxDiversification::default(), covariance))
        }
        Method::RiskParity {
            risk_budgets,
            covariance,
        } => {
            let mut m = RiskParity::default();
            if let Some(budgets) = risk_budgets {
                m = m.risk_budgets(budgets);
            }
            Box::new(covariance!(m, covariance))
        }
        Method::Hrp {
            linkage: choice,
            covariance,
        } => Box::new(covariance!(
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
            Box::new(covariance!(m, covariance))
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
            Box::new(m)
        }
        Method::BlackLitterman { spec, max_sharpe } => {
            let model = black_litterman(spec);
            let _ = model.prior_returns(data);
            let mu = model.expected_returns(data).ok()?;
            if max_sharpe {
                Box::new(MaxSharpe::default().expected_returns(mu))
            } else {
                Box::new(MeanVariance::risk_aversion(1.0).expected_returns(mu))
            }
        }
    })
}

fuzz_target!(|input: Input| {
    let Some(data) = returns(input.data) else {
        return;
    };
    let Some(method) = method(input.method, &data) else {
        return;
    };
    let Ok(weights) = data.allocate(method.as_ref()) else {
        return;
    };
    assert_eq!(weights.len(), data.labels().len());
    assert!(weights.iter().all(|w| w.is_finite()), "{weights:?}");
    let sum: f64 = weights.iter().sum();
    let gross: f64 = weights.iter().map(|w| w.abs()).sum();
    assert!(
        (sum - 1.0).abs() <= 1e-6 * gross.max(1.0),
        "weights sum to {sum}: {weights:?}"
    );
});
