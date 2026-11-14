//! `Serialize` and `Deserialize` for [`WeightBounds`]:
//!
//! ```json
//! {
//!   "lower": 0.0,
//!   "upper": 0.3,
//!   "assets": [{"asset": "GLD", "lower": 0.0, "upper": 0.1}],
//!   "groups": [{"members": ["search", "shopping"], "lower": 0.0, "upper": 0.4}]
//! }
//! ```
//!
//! `lower` and `upper` are each a number for every asset or [values per asset](crate::PerAsset).
//! An upper bound of `null`, or a missing `upper`, is no cap, `f64::INFINITY`, and a missing
//! `lower` is 0, as for [`WeightBounds::default`]. `assets` and `groups` are left out when
//! there are none. An asset's or a group's `upper` may be `null` too, but neither limit may be
//! left out.

use super::{Limit, WeightBounds};
use crate::asset::serde_impl::Limit as Value;
use crate::{Asset, PerAsset};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// One side of the bounds: a value for every asset, or values per asset.
#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "BoundsSide"))]
#[serde(untagged)]
enum Side {
    Uniform(Value),
    PerAsset(PerAsset),
}

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS))]
#[serde(deny_unknown_fields)]
struct AssetBounds {
    asset: Asset,
    lower: f64,
    upper: Value,
}

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS))]
#[serde(deny_unknown_fields)]
struct GroupLimits {
    members: Vec<Asset>,
    lower: f64,
    upper: Value,
}

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "WeightBounds"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default = "no_floor")]
    #[cfg_attr(feature = "ts-rs", ts(as = "Option<_>", optional))]
    lower: Side,
    #[serde(default = "no_cap")]
    #[cfg_attr(feature = "ts-rs", ts(as = "Option<_>", optional))]
    upper: Side,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    assets: Vec<AssetBounds>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    groups: Vec<GroupLimits>,
}

fn no_floor() -> Side {
    Side::Uniform(Value(0.0))
}

fn no_cap() -> Side {
    Side::Uniform(Value(f64::INFINITY))
}

impl From<&Limit> for Side {
    fn from(limit: &Limit) -> Self {
        match limit {
            Limit::Uniform(value) => Side::Uniform(Value(*value)),
            Limit::PerAsset(values) => Side::PerAsset(values.clone()),
        }
    }
}

impl From<Side> for Limit {
    fn from(side: Side) -> Self {
        match side {
            Side::Uniform(Value(value)) => Limit::Uniform(value),
            Side::PerAsset(values) => Limit::PerAsset(values),
        }
    }
}

impl Serialize for WeightBounds {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            lower: (&self.lower).into(),
            upper: (&self.upper).into(),
            assets: self
                .assets
                .iter()
                .map(|(asset, lower, upper)| AssetBounds {
                    asset: asset.clone(),
                    lower: *lower,
                    upper: Value(*upper),
                })
                .collect(),
            groups: self
                .groups
                .iter()
                .map(|(members, lower, upper)| GroupLimits {
                    members: members.clone(),
                    lower: *lower,
                    upper: Value(*upper),
                })
                .collect(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightBounds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(WeightBounds {
            lower: wire.lower.into(),
            upper: wire.upper.into(),
            assets: wire
                .assets
                .into_iter()
                .map(|a| (a.asset, a.lower, a.upper.0))
                .collect(),
            groups: wire
                .groups
                .into_iter()
                .map(|g| (g.members, g.lower, g.upper.0))
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn round_trip(bounds: &WeightBounds, json: serde_json::Value) {
        assert_eq!(serde_json::to_value(bounds).unwrap(), json);
        let text = serde_json::to_string(bounds).unwrap();
        assert_eq!(
            &serde_json::from_str::<WeightBounds>(&text).unwrap(),
            bounds
        );
    }

    #[test]
    fn the_default_has_no_cap() {
        round_trip(
            &WeightBounds::default(),
            json!({"lower": 0.0, "upper": null}),
        );
        // Left out, the bounds are the default.
        assert_eq!(
            serde_json::from_value::<WeightBounds>(json!({})).unwrap(),
            WeightBounds::default()
        );
    }

    #[test]
    fn every_kind_of_bound_keeps_its_form() {
        round_trip(
            &WeightBounds::uniform(0.0, 0.3)
                .asset("GLD", 0.0, 0.1)
                .asset(2, 0.05, f64::INFINITY)
                .group(["search", "shopping"], 0.0, 0.4)
                .group([5, 6], 0.2, f64::INFINITY),
            json!({
                "lower": 0.0,
                "upper": 0.3,
                "assets": [
                    {"asset": "GLD", "lower": 0.0, "upper": 0.1},
                    {"asset": 2, "lower": 0.05, "upper": null}
                ],
                "groups": [
                    {"members": ["search", "shopping"], "lower": 0.0, "upper": 0.4},
                    {"members": [5, 6], "lower": 0.2, "upper": null}
                ]
            }),
        );
        round_trip(
            &WeightBounds::per_asset(vec![0.0, 0.1], vec![0.5, f64::INFINITY]),
            json!({"lower": [0.0, 0.1], "upper": [0.5, null]}),
        );
        round_trip(
            &WeightBounds::per_asset([("SPY", 0.1)], [("SPY", 0.6)]),
            json!({"lower": {"SPY": 0.1}, "upper": {"SPY": 0.6}}),
        );
    }

    #[test]
    fn unknown_or_missing_fields_are_rejected() {
        // A misspelled field, an asset's missing cap, a group's missing floor, and a floor of
        // null, which only a cap can be.
        for json in [
            json!({"lower": 0.0, "uper": 0.3}),
            json!({"assets": [{"asset": "GLD", "lower": 0.0}]}),
            json!({"groups": [{"members": ["A"], "upper": 0.4}]}),
            json!({"groups": [{"members": ["A"], "lower": null, "upper": 0.4}]}),
        ] {
            assert!(
                serde_json::from_value::<WeightBounds>(json.clone()).is_err(),
                "{json}"
            );
        }
    }
}
