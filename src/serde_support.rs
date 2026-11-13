//! What the `serde` impls of the methods share. Each method is an object whose keys are its
//! builders, each with its argument, and `true` for a builder without one, such as
//! `allow_short`. A setting left out is the default, and an unknown key is an error.

use crate::WeightBounds;
use serde::de::Error;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Whether `value` is its type's default, so that it's left out.
pub(crate) fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// A long-only method's `bounds` and `allow_short`, from its bounds, `None` when short positions
/// are allowed. Bounds that are the default are left out.
pub(crate) fn long_only_fields(bounds: &Option<WeightBounds>) -> (Option<WeightBounds>, bool) {
    match bounds {
        None => (None, true),
        Some(bounds) if *bounds == WeightBounds::default() => (None, false),
        Some(bounds) => (Some(bounds.clone()), false),
    }
}

/// A method's bounds from its `bounds` and `allow_short`, which exclude each other as
/// `.bounds(..)` and `.allow_short()` replace each other: `None` when short positions are
/// allowed, and the default bounds when neither is given.
pub(crate) fn bounds_of<E: Error>(
    bounds: Option<WeightBounds>,
    allow_short: bool,
) -> Result<Option<WeightBounds>, E> {
    match (bounds, allow_short) {
        (Some(_), true) => Err(E::custom("`bounds` and `allow_short` exclude each other")),
        (_, true) => Ok(None),
        (bounds, false) => Ok(Some(bounds.unwrap_or_default())),
    }
}

/// A method with no settings, written as `{}`.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct NoSettings {}

/// `Serialize` and `Deserialize` as `{}`, for a method with no settings.
macro_rules! no_settings {
    ($method:ident) => {
        impl Serialize for crate::$method {
            fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                NoSettings {}.serialize(serializer)
            }
        }

        impl<'de> Deserialize<'de> for crate::$method {
            fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                NoSettings::deserialize(deserializer).map(|_| crate::$method)
            }
        }
    };
}

no_settings!(EqualWeight);
no_settings!(InverseVolatility);
no_settings!(InverseVariance);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EqualWeight, InverseVariance, InverseVolatility};
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn a_method_with_no_settings_is_an_empty_object() {
        assert_eq!(serde_json::to_value(EqualWeight).unwrap(), json!({}));
        assert_eq!(serde_json::to_value(InverseVolatility).unwrap(), json!({}));
        assert_eq!(serde_json::to_value(InverseVariance).unwrap(), json!({}));
        assert_eq!(
            serde_json::from_value::<EqualWeight>(json!({})).unwrap(),
            EqualWeight
        );
        assert!(serde_json::from_value::<InverseVariance>(json!({"shrinkage": 0.1})).is_err());
    }

    #[test]
    fn an_error_is_its_variant_in_snake_case() {
        use crate::{Asset, CutupError};
        assert_eq!(
            serde_json::to_value(CutupError::NoAssets).unwrap(),
            json!("no_assets")
        );
        assert_eq!(
            serde_json::to_value(CutupError::InvalidConfig {
                field: "risk_aversion",
                value: -1.0
            })
            .unwrap(),
            json!({"invalid_config": {"field": "risk_aversion", "value": -1.0}})
        );
        assert_eq!(
            serde_json::to_value(CutupError::UnknownAsset {
                asset: Asset::from("QQQ")
            })
            .unwrap(),
            json!({"unknown_asset": {"asset": "QQQ"}})
        );
        // No lower end, as for a target above the highest net return.
        assert_eq!(
            serde_json::to_value(CutupError::TargetReturnOutOfRange {
                target_return: 0.01,
                lowest: f64::NEG_INFINITY,
                highest: 0.005
            })
            .unwrap(),
            json!({"target_return_out_of_range":
                {"target_return": 0.01, "lowest": null, "highest": 0.005}})
        );
    }

    #[test]
    fn a_method_from_json_allocates_as_its_builders_do() {
        use crate::{MeanVariance, MinVariance, Returns};
        let data = Returns::new(&[
            [0.010, -0.020, 0.005],
            [-0.004, 0.015, 0.002],
            [0.007, 0.003, -0.006],
            [0.002, -0.001, 0.004],
            [-0.003, 0.006, 0.001],
        ])
        .unwrap()
        .labeled(["SPY", "TLT", "GLD"])
        .unwrap();
        // The README's example, which is `ignore`d there since it needs the feature.
        let built = MeanVariance::risk_aversion(50.0)
            .bounds(WeightBounds::uniform(0.0, 0.6).group(["SPY", "TLT"], 0.0, 0.7))
            .shrinkage(0.1);
        let read: MeanVariance = serde_json::from_str(
            r#"{
    "risk_aversion": 50.0,
    "bounds": {
        "upper": 0.6,
        "groups": [{"members": ["SPY", "TLT"], "lower": 0.0, "upper": 0.7}]
    },
    "covariance": {"shrinkage": 0.1}
}"#,
        )
        .unwrap();
        assert_eq!(read, built);
        assert_eq!(data.allocate(&read), data.allocate(&built));
        let capped: MinVariance = serde_json::from_value(
            json!({"bounds": {"assets": [{"asset": "GLD", "lower": 0.0, "upper": 0.1}]}}),
        )
        .unwrap();
        let weights = data.allocate(&capped).unwrap();
        assert!(weights["GLD"] <= 0.1 + 1e-12);
        let back: crate::Weights =
            serde_json::from_str(&serde_json::to_string(&weights).unwrap()).unwrap();
        assert_eq!(back, weights);
    }

    #[test]
    fn bounds_and_shorting_exclude_each_other() {
        let custom = WeightBounds::uniform(0.0, 0.4);
        assert_eq!(long_only_fields(&None), (None, true));
        assert_eq!(
            long_only_fields(&Some(WeightBounds::default())),
            (None, false)
        );
        assert_eq!(
            long_only_fields(&Some(custom.clone())),
            (Some(custom.clone()), false)
        );
        let of = bounds_of::<serde_json::Error>;
        assert_eq!(of(None, false).ok(), Some(Some(WeightBounds::default())));
        assert_eq!(of(None, true).ok(), Some(None));
        assert_eq!(
            of(Some(custom.clone()), false).ok(),
            Some(Some(custom.clone()))
        );
        let both = of(Some(custom), true).unwrap_err();
        assert!(both.to_string().contains("exclude each other"));
    }
}
