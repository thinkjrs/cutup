//! `Serialize` and `Deserialize` for [`MinMad`], by its builders: `bounds`, `target_return`
//! and `downside`. `{}` is [`MinMad::default`].

use super::MinMad;
use crate::WeightBounds;
use crate::serde_support::{finite, is_default};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "MinMad"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "is_default")]
    bounds: WeightBounds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_return: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    downside: Option<f64>,
}

impl Serialize for MinMad {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        finite(&[
            ("target_return", self.target_return),
            ("downside", self.downside),
        ])?;
        Wire {
            bounds: self.bounds.clone(),
            target_return: self.target_return,
            downside: self.downside,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MinMad {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(MinMad {
            bounds: wire.bounds,
            target_return: wire.target_return,
            downside: wire.downside,
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
            (MinMad::default(), json!({})),
            (
                MinMad::default()
                    .bounds(WeightBounds::uniform(0.0, 0.5).group([0, 1], 0.0, 0.6))
                    .target_return(0.001)
                    .downside(-0.002),
                json!({
                    "bounds": {
                        "lower": 0.0,
                        "upper": 0.5,
                        "groups": [{"members": [0, 1], "lower": 0.0, "upper": 0.6}]
                    },
                    "target_return": 0.001,
                    "downside": -0.002
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            assert_eq!(serde_json::from_value::<MinMad>(json).unwrap(), method);
        }
        assert!(serde_json::from_value::<MinMad>(json!({"downsides": 0.0})).is_err());
    }
}
