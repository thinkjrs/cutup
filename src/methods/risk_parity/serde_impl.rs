//! `Serialize` and `Deserialize` for [`RiskParity`], by its builders: `risk_budgets` and
//! `covariance`. `{}` is [`RiskParity::default`].

use super::RiskParity;
use crate::serde_support::is_default;
use crate::{CovarianceConfig, PerAsset};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "RiskParity"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    risk_budgets: Option<PerAsset>,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
}

impl Serialize for RiskParity {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            risk_budgets: self.risk_budgets.clone(),
            covariance: self.covariance.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RiskParity {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(RiskParity {
            risk_budgets: wire.risk_budgets,
            covariance: wire.covariance,
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
            (RiskParity::default(), json!({})),
            (
                RiskParity::default()
                    .risk_budgets([("SPY", 0.7), ("TLT", 0.3)])
                    .covariance(crate::CovarianceConfig::semicovariance()),
                json!({
                    "risk_budgets": {"SPY": 0.7, "TLT": 0.3},
                    "covariance": {"semicovariance": true}
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            let text = serde_json::to_string(&method).unwrap();
            assert_eq!(serde_json::from_str::<RiskParity>(&text).unwrap(), method);
        }
    }
}
