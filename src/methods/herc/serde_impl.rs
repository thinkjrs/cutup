//! `Serialize` and `Deserialize` for [`Herc`], by its builders: `linkage`, `clusters`,
//! `max_clusters` and `covariance`. `{}` is [`Herc::default`].

use super::{HERC_MAX_CLUSTERS, Herc};
use crate::serde_support::is_default;
use crate::{CovarianceConfig, Linkage};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "Herc"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "is_default")]
    linkage: Linkage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    clusters: Option<usize>,
    #[serde(default = "max_clusters", skip_serializing_if = "is_max_clusters")]
    max_clusters: usize,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
}

fn max_clusters() -> usize {
    HERC_MAX_CLUSTERS
}

fn is_max_clusters(value: &usize) -> bool {
    *value == HERC_MAX_CLUSTERS
}

impl Serialize for Herc {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            linkage: self.linkage,
            clusters: self.clusters,
            max_clusters: self.max_clusters,
            covariance: self.covariance.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Herc {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(Herc {
            linkage: wire.linkage,
            clusters: wire.clusters,
            max_clusters: wire.max_clusters,
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
            (Herc::default(), json!({})),
            (
                Herc::default().linkage(Linkage::Ward).clusters(3),
                json!({"linkage": "ward", "clusters": 3}),
            ),
            (Herc::default().max_clusters(5), json!({"max_clusters": 5})),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            assert_eq!(serde_json::from_value::<Herc>(json).unwrap(), method);
        }
    }
}
