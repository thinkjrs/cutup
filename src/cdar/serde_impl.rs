//! `Serialize` and `Deserialize` for [`MinCdar`], by its builders: `confidence`, `bounds` and
//! `target_return`. `{}` is [`MinCdar::default`].

use super::MinCdar;
use crate::WeightBounds;
use crate::serde_support::is_default;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// [`MinCdar`]'s default confidence.
const CONFIDENCE: f64 = 0.95;

#[derive(Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "MinCdar"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default = "confidence", skip_serializing_if = "is_confidence")]
    confidence: f64,
    #[serde(default, skip_serializing_if = "is_default")]
    bounds: WeightBounds,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_return: Option<f64>,
}

fn confidence() -> f64 {
    CONFIDENCE
}

fn is_confidence(beta: &f64) -> bool {
    *beta == CONFIDENCE
}

impl Serialize for MinCdar {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            confidence: self.confidence,
            bounds: self.bounds.clone(),
            target_return: self.target_return,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MinCdar {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(MinCdar {
            confidence: wire.confidence,
            bounds: wire.bounds,
            target_return: wire.target_return,
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
        assert_eq!(MinCdar::default().confidence, CONFIDENCE);
        for (method, json) in [
            (MinCdar::default(), json!({})),
            (
                MinCdar::default()
                    .confidence(0.9)
                    .bounds(WeightBounds::uniform(0.0, 0.5).group([0, 1], 0.0, 0.6))
                    .target_return(0.001),
                json!({
                    "confidence": 0.9,
                    "bounds": {
                        "lower": 0.0,
                        "upper": 0.5,
                        "groups": [{"members": [0, 1], "lower": 0.0, "upper": 0.6}]
                    },
                    "target_return": 0.001
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&method).unwrap(), json);
            assert_eq!(serde_json::from_value::<MinCdar>(json).unwrap(), method);
        }
    }
}
