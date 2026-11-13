//! `Serialize` and `Deserialize` for [`Hrp`], by its builders: `linkage`, such as `"ward"`, and
//! `covariance`. `{}` is [`Hrp::default`].

use super::Hrp;
use crate::serde_support::is_default;
use crate::{CovarianceConfig, Linkage};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    #[serde(default, skip_serializing_if = "is_default")]
    linkage: Linkage,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
}

impl Serialize for Hrp {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            linkage: self.linkage,
            covariance: self.covariance.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Hrp {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(Hrp {
            linkage: wire.linkage,
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
            (Hrp::default(), json!({})),
            (
                Hrp::default().linkage(Linkage::Ward),
                json!({"linkage": "ward"}),
            ),
            (
                Hrp::default().linkage(Linkage::Complete),
                json!({"linkage": "complete"}),
            ),
            (
                Hrp::default().linkage(Linkage::Average),
                json!({"linkage": "average"}),
            ),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            assert_eq!(serde_json::from_value::<Hrp>(json).unwrap(), method);
        }
        assert!(serde_json::from_value::<Hrp>(json!({"linkage": "Ward"})).is_err());
    }
}
