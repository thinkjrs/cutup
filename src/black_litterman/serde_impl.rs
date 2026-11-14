//! `Serialize` and `Deserialize` for [`BlackLitterman`] and [`View`], by their builders:
//!
//! ```json
//! {
//!   "market_weights": {"SPY": 0.6, "TLT": 0.4},
//!   "risk_aversion": 2.5,
//!   "views": [{"relative": ["SPY", "TLT"], "q": 0.0005, "confidence": 0.5}]
//! }
//! ```
//!
//! `market_weights` and `risk_aversion` are [`BlackLitterman::new`]'s and required; then
//! `risk_free_rate`, `tau`, `views` and `covariance`. A view is one of `absolute`, an asset,
//! `relative`, `[long, short]`, or `portfolio`, values per asset, with its return `q`, and at
//! most one of `variance` and `confidence`.

use super::{Assets, BlackLitterman, Uncertainty, View};
use crate::serde_support::is_default;
use crate::{Asset, CovarianceConfig, PerAsset};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// [`BlackLitterman`]'s default `τ`.
const TAU: f64 = 0.05;

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "BlackLitterman"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    market_weights: PerAsset,
    risk_aversion: f64,
    #[serde(default, skip_serializing_if = "is_default")]
    risk_free_rate: f64,
    #[serde(default = "tau", skip_serializing_if = "is_tau")]
    tau: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    views: Vec<View>,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
}

fn tau() -> f64 {
    TAU
}

fn is_tau(tau: &f64) -> bool {
    *tau == TAU
}

impl Serialize for BlackLitterman {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            market_weights: self.market_weights.clone(),
            risk_aversion: self.risk_aversion,
            risk_free_rate: self.risk_free_rate,
            tau: self.tau,
            views: self.views.clone(),
            covariance: self.covariance.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for BlackLitterman {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(BlackLitterman {
            market_weights: wire.market_weights,
            risk_aversion: wire.risk_aversion,
            risk_free_rate: wire.risk_free_rate,
            tau: wire.tau,
            views: wire.views,
            covariance: wire.covariance,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "View"))]
#[serde(deny_unknown_fields)]
pub(super) struct ViewWire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    absolute: Option<Asset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    relative: Option<(Asset, Asset)>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    portfolio: Option<PerAsset>,
    q: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    variance: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    confidence: Option<f64>,
}

impl Serialize for View {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut wire = ViewWire {
            absolute: None,
            relative: None,
            portfolio: None,
            q: self.q,
            variance: None,
            confidence: None,
        };
        match &self.assets {
            Assets::Absolute(asset) => wire.absolute = Some(asset.clone()),
            Assets::Relative(long, short) => wire.relative = Some((long.clone(), short.clone())),
            Assets::Portfolio(weights) => wire.portfolio = Some(weights.clone()),
        }
        match self.uncertainty {
            Uncertainty::Default => {}
            Uncertainty::Variance(omega) => wire.variance = Some(omega),
            Uncertainty::Confidence(confidence) => wire.confidence = Some(confidence),
        }
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for View {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = ViewWire::deserialize(deserializer)?;
        let view = match (wire.absolute, wire.relative, wire.portfolio) {
            (Some(asset), None, None) => View::absolute(asset, wire.q),
            (None, Some((long, short)), None) => View::relative(long, short, wire.q),
            (None, None, Some(weights)) => View::portfolio(weights, wire.q),
            _ => {
                return Err(D::Error::custom(
                    "a view takes exactly one of `absolute`, `relative` and `portfolio`",
                ));
            }
        };
        match (wire.variance, wire.confidence) {
            (None, None) => Ok(view),
            (Some(omega), None) => Ok(view.variance(omega)),
            (None, Some(confidence)) => Ok(view.confidence(confidence)),
            (Some(_), Some(_)) => Err(D::Error::custom(
                "a view takes `variance` or `confidence`, not both",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn each_setting_and_view_is_its_builder() {
        assert_eq!(BlackLitterman::new([0.5, 0.5], 2.5).tau, TAU);
        for (model, json) in [
            (
                BlackLitterman::new([0.6, 0.4], 2.5),
                json!({"market_weights": [0.6, 0.4], "risk_aversion": 2.5}),
            ),
            (
                BlackLitterman::new([("SPY", 0.6), ("TLT", 0.4)], 3.0)
                    .risk_free_rate(0.0001)
                    .tau(0.025)
                    .view(View::absolute("SPY", 0.001))
                    .view(View::relative("SPY", "TLT", 0.0005).confidence(0.5))
                    .view(View::portfolio([0.5, -0.5], 0.0002).variance(1e-6))
                    .shrinkage(0.1),
                json!({
                    "market_weights": {"SPY": 0.6, "TLT": 0.4},
                    "risk_aversion": 3.0,
                    "risk_free_rate": 0.0001,
                    "tau": 0.025,
                    "views": [
                        {"absolute": "SPY", "q": 0.001},
                        {"relative": ["SPY", "TLT"], "q": 0.0005, "confidence": 0.5},
                        {"portfolio": [0.5, -0.5], "q": 0.0002, "variance": 1e-6}
                    ],
                    "covariance": {"shrinkage": 0.1}
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&model).unwrap(), json);
            let text = serde_json::to_string(&model).unwrap();
            assert_eq!(
                serde_json::from_str::<BlackLitterman>(&text).unwrap(),
                model
            );
        }
    }

    #[test]
    fn a_view_takes_one_kind_and_one_uncertainty() {
        for json in [
            json!({"q": 0.001}),
            json!({"absolute": "SPY", "relative": ["SPY", "TLT"], "q": 0.001}),
            json!({"absolute": "SPY", "q": 0.001, "variance": 1e-6, "confidence": 0.5}),
            json!({"absolute": "SPY"}),
        ] {
            assert!(
                serde_json::from_value::<View>(json.clone()).is_err(),
                "{json}"
            );
        }
        assert!(serde_json::from_value::<BlackLitterman>(json!({"risk_aversion": 2.5})).is_err());
    }
}
