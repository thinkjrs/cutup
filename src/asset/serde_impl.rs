//! `Serialize` and `Deserialize` for [`Asset`] and [`PerAsset`].
//!
//! An asset is its column, a number, or its label, a string. Values per asset are an array in
//! column order, an object from label to value, or an array of `[asset, value]` pairs, which can
//! name assets by column. A value may be `null`, which is `f64::INFINITY`, as for an upper bound
//! with no cap.

use super::{Asset, Key, PerAsset, Values};
use serde::de::{self, MapAccess, Visitor};
use serde::ser::{self, SerializeMap, SerializeSeq};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::HashSet;
use std::fmt;

impl Serialize for Asset {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.0 {
            Key::Column(column) => serializer.serialize_u64(*column as u64),
            Key::Label(label) => serializer.serialize_str(label),
        }
    }
}

impl<'de> Deserialize<'de> for Asset {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AssetVisitor;

        impl Visitor<'_> for AssetVisitor {
            type Value = Asset;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a column number or a label")
            }

            fn visit_u64<E: de::Error>(self, column: u64) -> Result<Asset, E> {
                usize::try_from(column)
                    .map(Asset::from)
                    .map_err(|_| E::custom("column out of range"))
            }

            fn visit_i64<E: de::Error>(self, column: i64) -> Result<Asset, E> {
                usize::try_from(column)
                    .map(Asset::from)
                    .map_err(|_| E::custom("a column can't be negative"))
            }

            fn visit_str<E: de::Error>(self, label: &str) -> Result<Asset, E> {
                Ok(Asset::from(label))
            }

            fn visit_string<E: de::Error>(self, label: String) -> Result<Asset, E> {
                Ok(Asset::from(label))
            }
        }

        deserializer.deserialize_any(AssetVisitor)
    }
}

/// A value that's `None` for `f64::INFINITY`, written as `null`, so that formats without
/// infinities, such as JSON, can carry it.
pub(crate) fn finite_or_none(value: f64) -> Option<f64> {
    (value != f64::INFINITY).then_some(value)
}

/// A value read as `null` for `f64::INFINITY`, or a number. Only an explicit `null` is
/// infinite: a field left out is still missing, as it wouldn't be through `Option<f64>`. That
/// takes a self-describing format, such as JSON.
#[derive(Clone, Copy)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(type = "number | null"))]
pub(crate) struct Limit(pub(crate) f64);

impl<'de> Deserialize<'de> for Limit {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LimitVisitor;

        impl<'de> Visitor<'de> for LimitVisitor {
            type Value = Limit;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a number, or null for no limit")
            }

            // A limit is written as an `Option`, and a format that spells out `Some`, such as
            // RON, hands it back as one.
            fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Limit, D::Error> {
                f64::deserialize(deserializer).map(Limit)
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Limit, E> {
                Ok(Limit(value))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Limit, E> {
                Ok(Limit(value as f64))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Limit, E> {
                Ok(Limit(value as f64))
            }

            fn visit_unit<E: de::Error>(self) -> Result<Limit, E> {
                Ok(Limit(f64::INFINITY))
            }

            fn visit_none<E: de::Error>(self) -> Result<Limit, E> {
                Ok(Limit(f64::INFINITY))
            }
        }

        deserializer.deserialize_any(LimitVisitor)
    }
}

impl Serialize for Limit {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Only `f64::INFINITY` is `null`. A format without NaN or infinities, such as JSON,
        // would write NaN and −∞ as `null` too, and they'd read back as no cap, or as an
        // infinite value, where the builders refuse them.
        if self.0.is_nan() || self.0 == f64::NEG_INFINITY {
            return Err(ser::Error::custom(format!(
                "{} can't be written: it would read back as infinity",
                self.0
            )));
        }
        finite_or_none(self.0).serialize(serializer)
    }
}

impl Serialize for PerAsset {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match &self.0 {
            Values::Ordered(values) => {
                let mut seq = serializer.serialize_seq(Some(values.len()))?;
                for &value in values {
                    seq.serialize_element(&Limit(value))?;
                }
                seq.end()
            }
            Values::Keyed(pairs) => {
                let mut seen = HashSet::new();
                let labels: Option<Vec<&str>> = pairs
                    .iter()
                    .map(|(asset, _)| match &asset.0 {
                        Key::Label(label) if seen.insert(label.as_str()) => Some(label.as_str()),
                        _ => None,
                    })
                    .collect();
                match labels {
                    // Every asset by a label, each once: an object from label to value.
                    Some(labels) => {
                        let mut map = serializer.serialize_map(Some(pairs.len()))?;
                        for (label, (_, value)) in labels.into_iter().zip(pairs) {
                            map.serialize_entry(label, &Limit(*value))?;
                        }
                        map.end()
                    }
                    // Otherwise `[asset, value]` pairs, which keep columns and repeats.
                    None => {
                        let mut seq = serializer.serialize_seq(Some(pairs.len()))?;
                        for (asset, value) in pairs {
                            seq.serialize_element(&(asset, Limit(*value)))?;
                        }
                        seq.end()
                    }
                }
            }
        }
    }
}

/// An object from label to value, in the order given.
struct Labeled(Vec<(Asset, f64)>);

impl<'de> Deserialize<'de> for Labeled {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct LabeledVisitor;

        impl<'de> Visitor<'de> for LabeledVisitor {
            type Value = Labeled;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object from label to value")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Labeled, M::Error> {
                let mut pairs = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((label, Limit(value))) = map.next_entry::<String, Limit>()? {
                    pairs.push((Asset::from(label), value));
                }
                Ok(Labeled(pairs))
            }
        }

        deserializer.deserialize_map(LabeledVisitor)
    }
}

/// The forms values per asset can take. `expecting` is what a value in none of them is told,
/// and an error reading the input itself, such as JSON that ends early, is still its own.
#[derive(Deserialize)]
#[serde(
    untagged,
    expecting = "expected values per asset: an array in column order, an object from label to \
                 value, or an array of [asset, value] pairs"
)]
enum Wire {
    Ordered(Vec<Limit>),
    Pairs(Vec<(Asset, Limit)>),
    Labeled(Labeled),
}

impl<'de> Deserialize<'de> for PerAsset {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(PerAsset(match wire {
            Wire::Ordered(values) => Values::Ordered(values.into_iter().map(|v| v.0).collect()),
            Wire::Pairs(pairs) => {
                Values::Keyed(pairs.into_iter().map(|(asset, v)| (asset, v.0)).collect())
            }
            Wire::Labeled(Labeled(pairs)) => Values::Keyed(pairs),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    /// `value` as `json`, and back, through a string, which keeps an object's order where
    /// `serde_json::Value` sorts it.
    fn round_trip(value: &PerAsset, json: serde_json::Value) {
        assert_eq!(serde_json::to_value(value).unwrap(), json);
        let text = serde_json::to_string(value).unwrap();
        assert_eq!(&serde_json::from_str::<PerAsset>(&text).unwrap(), value);
    }

    #[test]
    fn an_asset_is_its_column_or_its_label() {
        assert_eq!(serde_json::to_value(Asset::from(2)).unwrap(), json!(2));
        assert_eq!(
            serde_json::to_value(Asset::from("SPY")).unwrap(),
            json!("SPY")
        );
        assert_eq!(
            serde_json::from_value::<Asset>(json!(2)).unwrap(),
            Asset::from(2)
        );
        assert_eq!(
            serde_json::from_value::<Asset>(json!("SPY")).unwrap(),
            Asset::from("SPY")
        );
        assert!(serde_json::from_value::<Asset>(json!(-1)).is_err());
        assert!(serde_json::from_value::<Asset>(json!(1.5)).is_err());
    }

    #[test]
    fn values_per_asset_keep_their_form() {
        round_trip(&PerAsset::from(vec![0.6, 0.4]), json!([0.6, 0.4]));
        round_trip(
            &PerAsset::from(vec![("TLT", 0.4), ("SPY", 0.6)]),
            json!({"TLT": 0.4, "SPY": 0.6}),
        );
        // A column, or a label twice, can't be an object's key, so they stay pairs.
        round_trip(
            &PerAsset::from(vec![(Asset::from(0), 0.6), (Asset::from("TLT"), 0.4)]),
            json!([[0, 0.6], ["TLT", 0.4]]),
        );
        round_trip(
            &PerAsset::from(vec![("SPY", 0.6), ("SPY", 0.4)]),
            json!([["SPY", 0.6], ["SPY", 0.4]]),
        );
    }

    #[test]
    fn null_is_no_limit() {
        round_trip(
            &PerAsset::from(vec![0.4, f64::INFINITY]),
            json!([0.4, null]),
        );
        round_trip(
            &PerAsset::from(vec![("SPY", f64::INFINITY)]),
            json!({"SPY": null}),
        );
    }

    /// A format that spells out `Some`, as RON does: it hands a number to `visit_some`.
    struct Spelled(f64);

    impl<'de> Deserializer<'de> for Spelled {
        type Error = de::value::Error;

        fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
            visitor.visit_some(de::value::F64Deserializer::new(self.0))
        }

        serde::forward_to_deserialize_any! {
            bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes byte_buf
            option unit unit_struct newtype_struct seq tuple tuple_struct map struct enum
            identifier ignored_any
        }
    }

    #[test]
    fn a_limit_spelled_out_as_some_is_its_number() {
        assert_eq!(Limit::deserialize(Spelled(0.3)).unwrap().0, 0.3);
    }

    #[test]
    fn input_that_ends_early_or_is_malformed_is_its_own_error() {
        let early = serde_json::from_str::<PerAsset>("[0.5, ").unwrap_err();
        assert!(early.is_eof(), "{early}");
        let malformed = serde_json::from_str::<PerAsset>(r#"{"SPY": 0.5, }"#).unwrap_err();
        assert!(malformed.is_syntax(), "{malformed}");
        // The same inside a method, where it used to pass for the wrong form.
        let method = r#"{"risk_budgets": [0.5, }"#;
        let malformed = serde_json::from_str::<crate::RiskParity>(method).unwrap_err();
        assert!(malformed.is_syntax(), "{malformed}");
    }

    #[test]
    fn a_wrong_type_says_what_was_expected() {
        /// The error reading `json` as a `T`.
        fn error<T: for<'de> Deserialize<'de>>(json: serde_json::Value) -> String {
            serde_json::from_value::<T>(json).err().unwrap().to_string()
        }
        assert!(error::<Asset>(json!(true)).ends_with("expected a column number or a label"));
        assert!(error::<Limit>(json!("high")).ends_with("expected a number, or null for no limit"));
        assert!(error::<Labeled>(json!([0.5])).ends_with("expected an object from label to value"));
    }

    #[test]
    fn anything_else_is_rejected() {
        for json in [
            json!(0.5),
            json!("SPY"),
            json!([0.5, "SPY"]),
            json!({"SPY": "high"}),
        ] {
            let error = serde_json::from_value::<PerAsset>(json).unwrap_err();
            assert!(
                error.to_string().starts_with("expected values per asset"),
                "{error}"
            );
        }
    }
}
