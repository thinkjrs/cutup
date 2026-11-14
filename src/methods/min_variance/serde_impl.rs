//! `Serialize` and `Deserialize` for [`MinVariance`], by its builders:
//!
//! ```json
//! {"bounds": {"lower": 0.0, "upper": 0.4}, "current_weights": [0.5, 0.5], "max_turnover": 0.2}
//! ```
//!
//! `bounds` or `allow_short: true`, then `covariance`, `current_weights`, `max_turnover` and
//! `turnover_penalty`. `{}` is [`MinVariance::default`].

use super::MinVariance;
use crate::serde_support::{bounds_of, is_default, long_only_fields};
use crate::turnover::Trading;
use crate::{CovarianceConfig, PerAsset, WeightBounds};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Not;

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "MinVariance"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bounds: Option<WeightBounds>,
    #[serde(default, skip_serializing_if = "Not::not")]
    allow_short: bool,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_weights: Option<PerAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_turnover: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turnover_penalty: Option<f64>,
}

impl Serialize for MinVariance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (bounds, allow_short) = long_only_fields(&self.bounds);
        Wire {
            bounds,
            allow_short,
            covariance: self.covariance.clone(),
            current_weights: self.trading.current.clone(),
            max_turnover: self.trading.max_turnover,
            turnover_penalty: self.trading.penalty,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MinVariance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(MinVariance {
            bounds: bounds_of(wire.bounds, wire.allow_short)?,
            covariance: wire.covariance,
            trading: Trading {
                current: wire.current_weights,
                max_turnover: wire.max_turnover,
                penalty: wire.turnover_penalty,
                costs: None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn round_trip(method: &MinVariance, json: serde_json::Value) {
        assert_eq!(serde_json::to_value(method).unwrap(), json);
        assert_eq!(
            &serde_json::from_value::<MinVariance>(json).unwrap(),
            method
        );
    }

    #[test]
    fn each_setting_is_its_builder() {
        round_trip(&MinVariance::default(), json!({}));
        round_trip(
            &MinVariance::default().allow_short(),
            json!({"allow_short": true}),
        );
        round_trip(
            &MinVariance::default()
                .bounds(WeightBounds::uniform(0.0, 0.4))
                .shrinkage(0.1)
                .current_weights([0.5, 0.5])
                .max_turnover(0.2)
                .turnover_penalty(1e-4),
            json!({
                "bounds": {"lower": 0.0, "upper": 0.4},
                "covariance": {"shrinkage": 0.1},
                "current_weights": [0.5, 0.5],
                "max_turnover": 0.2,
                "turnover_penalty": 1e-4
            }),
        );
    }

    #[test]
    fn bounds_and_shorting_together_are_rejected() {
        let error = serde_json::from_value::<MinVariance>(
            json!({"bounds": {"upper": 0.4}, "allow_short": true}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("exclude each other"), "{error}");
        assert!(serde_json::from_value::<MinVariance>(json!({"shrinkage": 0.1})).is_err());
    }
}
