//! `Serialize` and `Deserialize` for [`MaxSharpe`], by its builders: `risk_free_rate`,
//! `expected_returns`, `covariance` and `allow_short: true`. `{}` is [`MaxSharpe::default`].

use super::MaxSharpe;
use crate::serde_support::is_default;
use crate::{CovarianceConfig, PerAsset};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Not;

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "MaxSharpe"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "is_default")]
    risk_free_rate: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_returns: Option<PerAsset>,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
    #[serde(default, skip_serializing_if = "Not::not")]
    allow_short: bool,
}

impl Serialize for MaxSharpe {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            risk_free_rate: self.risk_free_rate,
            expected_returns: self.expected_returns.clone(),
            covariance: self.covariance.clone(),
            allow_short: !self.long_only,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MaxSharpe {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(MaxSharpe {
            risk_free_rate: wire.risk_free_rate,
            expected_returns: wire.expected_returns,
            covariance: wire.covariance,
            long_only: !wire.allow_short,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn each_setting_is_its_builder() {
        for (method, json) in [
            (MaxSharpe::default(), json!({})),
            (
                MaxSharpe::default()
                    .risk_free_rate(0.0001)
                    .expected_returns([0.001, 0.002])
                    .regularization(1e-8)
                    .allow_short(),
                json!({
                    "risk_free_rate": 0.0001,
                    "expected_returns": [0.001, 0.002],
                    "covariance": {"regularization": 1e-8},
                    "allow_short": true
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            assert_eq!(serde_json::from_value::<MaxSharpe>(json).unwrap(), method);
        }
    }
}
