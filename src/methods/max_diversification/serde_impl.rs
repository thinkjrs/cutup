//! `Serialize` and `Deserialize` for [`MaxDiversification`], by its builders: `covariance`.
//! `{}` is [`MaxDiversification::default`].

use super::MaxDiversification;
use crate::CovarianceConfig;
use crate::serde_support::is_default;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
#[cfg_attr(
    feature = "ts-rs",
    derive(ts_rs::TS),
    ts(rename = "MaxDiversification")
)]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
}

impl Serialize for MaxDiversification {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            covariance: self.covariance.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MaxDiversification {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(MaxDiversification {
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
            (MaxDiversification::default(), json!({})),
            (
                MaxDiversification::default().shrinkage(0.2),
                json!({"covariance": {"shrinkage": 0.2}}),
            ),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            assert_eq!(
                serde_json::from_value::<MaxDiversification>(json).unwrap(),
                method
            );
        }
    }
}
