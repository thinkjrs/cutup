//! `Serialize` and `Deserialize` for [`Weights`]: an object from each asset's label to its
//! weight, in column order.

use super::Weights;
use serde::de::{self, MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fmt;

impl Serialize for Weights {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.values.len()))?;
        for (label, value) in self.labels.iter().zip(&self.values) {
            map.serialize_entry(label, value)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Weights {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct WeightsVisitor;

        impl<'de> Visitor<'de> for WeightsVisitor {
            type Value = Weights;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object from each asset's label to its weight")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Weights, M::Error> {
                let n = map.size_hint().unwrap_or(0);
                let (mut labels, mut values) = (Vec::with_capacity(n), Vec::with_capacity(n));
                let mut seen = HashSet::new();
                while let Some((label, value)) = map.next_entry::<String, f64>()? {
                    if !seen.insert(label.clone()) {
                        return Err(de::Error::custom(format!("asset `{label}` appears twice")));
                    }
                    labels.push(label);
                    values.push(value);
                }
                Ok(Weights::new(values, labels.into()))
            }
        }

        deserializer.deserialize_map(WeightsVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InverseVolatility, Returns};
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn weights_are_an_object_from_label_to_weight_in_column_order() {
        let data = Returns::new(&[[0.01, -0.02], [-0.004, 0.015], [0.007, 0.003]])
            .unwrap()
            .labeled(["TLT", "SPY"])
            .unwrap();
        let weights = data.allocate(&InverseVolatility).unwrap();
        let text = serde_json::to_string(&weights).unwrap();
        assert!(text.starts_with("{\"TLT\":"), "{text}");
        let back: Weights = serde_json::from_str(&text).unwrap();
        assert_eq!(back, weights);
        assert_eq!(back.labels(), ["TLT", "SPY"]);
    }

    #[test]
    fn unlabeled_weights_go_by_column() {
        let weights = Weights::from(vec![0.25, 0.75]);
        assert_eq!(
            serde_json::to_value(&weights).unwrap(),
            json!({"0": 0.25, "1": 0.75})
        );
    }

    #[test]
    fn anything_but_an_object_says_what_was_expected() {
        let error = serde_json::from_value::<Weights>(json!([0.5, 0.5])).unwrap_err();
        assert!(
            error
                .to_string()
                .ends_with("expected an object from each asset's label to its weight"),
            "{error}"
        );
    }

    #[test]
    fn a_label_twice_is_rejected() {
        let error = serde_json::from_str::<Weights>(r#"{"SPY": 0.5, "SPY": 0.5}"#).unwrap_err();
        assert!(error.to_string().contains("`SPY` appears twice"), "{error}");
    }
}
