//! `Serialize` and `Deserialize` for [`Weights`]: an object from each asset's label to its
//! weight, in column order.
//!
//! Read back, the columns are in the order of the keys. A reader that doesn't keep that order,
//! as `serde_json::Value` doesn't without serde_json's `preserve_order` feature, hands the
//! weights back in another: used with the data, they're a
//! [`LabelMismatch`](crate::CutupError::LabelMismatch).

use super::Weights;
use crate::serde_support::reserved;
use serde::de::{self, MapAccess, Visitor};
use serde::ser::{self, SerializeMap};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fmt;

impl Serialize for Weights {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // A format without NaN or infinities, such as JSON, would write such a weight as
        // `null`, which doesn't read back as one.
        let labeled = self.labels.iter().zip(&self.values);
        if let Some((label, value)) = labeled.clone().find(|(_, value)| !value.is_finite()) {
            return Err(ser::Error::custom(format!(
                "the weight of `{label}` is {value}, which can't be written: it wouldn't read \
                 back"
            )));
        }
        let mut map = serializer.serialize_map(Some(self.values.len()))?;
        for (label, value) in labeled {
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
                let n = reserved(map.size_hint());
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
    fn a_weight_that_isnt_finite_doesnt_serialize() {
        // JSON would write it as `null`, which fails to read back as a weight.
        for not in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let error = serde_json::to_string(&Weights::from(vec![0.5, not])).unwrap_err();
            let named = format!("the weight of `1` is {not}, which can't be written");
            assert!(error.to_string().contains(&named), "{error}");
        }
        // The largest weights there are still serialize, and read back.
        let weights = Weights::from(vec![f64::MAX, f64::MIN]);
        let text = serde_json::to_string(&weights).unwrap();
        assert_eq!(serde_json::from_str::<Weights>(&text).unwrap(), weights);
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

    /// A map with no entries that says it has `usize::MAX`, as the length a format writes
    /// ahead of its entries can.
    struct Claimed;

    impl<'de> MapAccess<'de> for Claimed {
        type Error = de::value::Error;

        fn next_key_seed<K: de::DeserializeSeed<'de>>(
            &mut self,
            _: K,
        ) -> Result<Option<K::Value>, Self::Error> {
            Ok(None)
        }

        fn next_value_seed<V: de::DeserializeSeed<'de>>(
            &mut self,
            _: V,
        ) -> Result<V::Value, Self::Error> {
            unreachable!("the map has no entries")
        }

        fn size_hint(&self) -> Option<usize> {
            Some(usize::MAX)
        }
    }

    #[test]
    fn a_map_that_claims_more_entries_than_it_has_reserves_no_more_than_a_few() {
        let weights = Weights::deserialize(de::value::MapAccessDeserializer::new(Claimed));
        assert!(weights.unwrap().is_empty());
        assert_eq!(reserved(Some(usize::MAX)), 4096);
        assert_eq!((reserved(Some(3)), reserved(None)), (3, 0));
    }

    #[test]
    fn a_label_twice_is_rejected() {
        let error = serde_json::from_str::<Weights>(r#"{"SPY": 0.5, "SPY": 0.5}"#).unwrap_err();
        assert!(error.to_string().contains("`SPY` appears twice"), "{error}");
    }
}
