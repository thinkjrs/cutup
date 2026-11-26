//! What the `serde` impls of the methods share. Each method is an object whose keys are its
//! builders, each with its argument, and `true` for a builder without one, such as
//! `allow_short`. A setting left out is the default, and an unknown key is an error.

use crate::WeightBounds;
use serde::de::{Error, MapAccess, Visitor, value::MapAccessDeserializer};
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser};
use std::fmt;
use std::marker::PhantomData;

/// Whether `value` is its type's default, so that it's left out.
pub(crate) fn is_default<T: Default + PartialEq>(value: &T) -> bool {
    *value == T::default()
}

/// An error for the first of `settings`, each a key and its value, that's set to a number that
/// isn't finite.
///
/// A format without NaN or infinities, such as JSON, writes one as `null`. For a setting that
/// can be left out, `null` reads back as not set. The builders refuse such a setting when a
/// method uses it, so written and read back, a config that fails would allocate, without the
/// cap or the estimate it named. For a setting that's always a number, such as a confidence,
/// `null` doesn't read back at all.
pub(crate) fn finite<E: ser::Error>(settings: &[(&str, Option<f64>)]) -> Result<(), E> {
    for &(key, value) in settings {
        if let Some(value) = value.filter(|value| !value.is_finite()) {
            return Err(E::custom(format!(
                "`{key}` is {value}, which can't be written: it would read back as not set, or \
                 not at all"
            )));
        }
    }
    Ok(())
}

/// A `T` read from an object, and from nothing else.
///
/// A derived struct also reads from an array, its fields by position: `[[0, 1]]` was a
/// method with bounds, and `[]` any method's defaults. Nothing documents that form, and a
/// field added or moved would change what it means without an error.
pub(crate) fn object<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    struct Object<T>(PhantomData<T>);

    impl<'de, T: Deserialize<'de>> Visitor<'de> for Object<T> {
        type Value = T;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an object")
        }

        fn visit_map<A: MapAccess<'de>>(self, map: A) -> Result<T, A::Error> {
            T::deserialize(MapAccessDeserializer::new(map))
        }
    }

    deserializer.deserialize_map(Object(PhantomData))
}

/// How many entries to reserve for a map that says it has `hint`: what it says, up to 4096.
/// The hint is the input's own word, and a format that writes a length first, such as
/// MessagePack or CBOR, can claim billions of entries in a few bytes.
pub(crate) fn reserved(hint: Option<usize>) -> usize {
    hint.unwrap_or(0).min(4096)
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
                object::<NoSettings, D>(deserializer).map(|_| crate::$method)
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

    #[test]
    fn a_setting_that_isnt_finite_doesnt_serialize() {
        use crate::{
            BlackLitterman, CovarianceConfig, MaxSharpe, MeanVariance, MinCdar, MinCvar, MinMad,
            MinVariance, View,
        };

        // JSON writes NaN and the infinities as `null`, and `null` reads back as not set. A
        // turnover cap of NaN, which `allocate` refuses, read back with no cap and allocated.
        let written = |value: Result<serde_json::Value, serde_json::Error>, key: &str| {
            let error = value.unwrap_err().to_string();
            assert!(error.contains(&format!("`{key}` is ")), "{key}: {error}");
        };
        for not in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let least = MinVariance::default().current_weights([0.5, 0.5]);
            written(
                serde_json::to_value(least.clone().max_turnover(not)),
                "max_turnover",
            );
            written(
                serde_json::to_value(least.turnover_penalty(not)),
                "turnover_penalty",
            );
            let mean = MeanVariance::risk_aversion(2.0).current_weights([0.5, 0.5]);
            written(
                serde_json::to_value(mean.clone().max_turnover(not)),
                "max_turnover",
            );
            written(
                serde_json::to_value(mean.clone().turnover_penalty(not)),
                "turnover_penalty",
            );
            written(
                serde_json::to_value(mean.transaction_costs(not)),
                "transaction_costs",
            );
            written(
                serde_json::to_value(MeanVariance::risk_aversion(not)),
                "risk_aversion",
            );
            written(
                serde_json::to_value(MeanVariance::target_return(not)),
                "target_return",
            );
            written(
                serde_json::to_value(MeanVariance::target_volatility(not)),
                "target_volatility",
            );
            written(
                serde_json::to_value(MinCvar::default().target_return(not)),
                "target_return",
            );
            written(
                serde_json::to_value(MinCdar::default().target_return(not)),
                "target_return",
            );
            written(
                serde_json::to_value(MinMad::default().target_return(not)),
                "target_return",
            );
            written(
                serde_json::to_value(MinMad::default().downside(not)),
                "downside",
            );
            let covariance = CovarianceConfig::default;
            written(
                serde_json::to_value(CovarianceConfig::exponentially_weighted(not)),
                "exponentially_weighted",
            );
            written(
                serde_json::to_value(CovarianceConfig::exponentially_weighted_decay(not)),
                "exponentially_weighted_decay",
            );
            written(
                serde_json::to_value(CovarianceConfig::semicovariance_below(not)),
                "semicovariance_below",
            );
            written(
                serde_json::to_value(covariance().shrinkage(not)),
                "shrinkage",
            );
            written(
                serde_json::to_value(covariance().regularization(not)),
                "regularization",
            );
            // Inside a method too, and a model's views.
            written(
                serde_json::to_value(MinVariance::default().shrinkage(not)),
                "shrinkage",
            );
            written(
                serde_json::to_value(View::absolute(0, 0.01).variance(not)),
                "variance",
            );
            written(
                serde_json::to_value(View::absolute(0, 0.01).confidence(not)),
                "confidence",
            );
            let model =
                BlackLitterman::new([0.5, 0.5], 2.5).view(View::absolute(0, 0.01).variance(not));
            written(serde_json::to_value(model), "variance");
            // The settings that are always numbers too: their `null` didn't read back at all.
            written(
                serde_json::to_value(MinCvar::default().confidence(not)),
                "confidence",
            );
            written(
                serde_json::to_value(MinCdar::default().confidence(not)),
                "confidence",
            );
            written(
                serde_json::to_value(MaxSharpe::default().risk_free_rate(not)),
                "risk_free_rate",
            );
            written(
                serde_json::to_value(BlackLitterman::new([0.5, 0.5], not)),
                "risk_aversion",
            );
            let model = BlackLitterman::new([0.5, 0.5], 2.5);
            written(
                serde_json::to_value(model.clone().risk_free_rate(not)),
                "risk_free_rate",
            );
            written(serde_json::to_value(model.tau(not)), "tau");
            written(serde_json::to_value(View::absolute(0, not)), "q");
            written(
                serde_json::to_value(WeightBounds::default().asset(0, not, 1.0)),
                "lower",
            );
            written(
                serde_json::to_value(WeightBounds::default().group([0, 1], not, 1.0)),
                "lower",
            );
        }
        // A finite one is written as before.
        let capped = MinVariance::default()
            .current_weights([0.5, 0.5])
            .max_turnover(0.2);
        assert_eq!(
            serde_json::to_value(capped).unwrap(),
            json!({"current_weights": [0.5, 0.5], "max_turnover": 0.2})
        );
    }

    #[test]
    fn only_an_infinite_cap_is_written_as_null() {
        use crate::PerAsset;

        // NaN and −∞ would be `null` in JSON too, and read back as no cap: an upper bound
        // the builders refuse came back as none at all.
        for not in [f64::NAN, f64::NEG_INFINITY] {
            for bounds in [
                WeightBounds::uniform(0.0, not),
                WeightBounds::uniform(not, 1.0),
                WeightBounds::per_asset(vec![0.0, 0.0], vec![0.5, not]),
                WeightBounds::per_asset(vec![0.0, not], vec![0.5, 0.5]),
                WeightBounds::default().asset(0, 0.0, not),
                WeightBounds::default().group([0, 1], 0.0, not),
            ] {
                let error = serde_json::to_value(&bounds).unwrap_err().to_string();
                assert!(error.contains("can't be written"), "{bounds:?}: {error}");
            }
            let values = PerAsset::from(vec![0.01, not]);
            assert!(serde_json::to_value(&values).is_err(), "{not}");
            let labeled = PerAsset::from([("SPY", 0.01), ("TLT", not)]);
            assert!(serde_json::to_value(&labeled).is_err(), "{not}");
        }
        // No cap still is.
        assert_eq!(
            serde_json::to_value(WeightBounds::uniform(0.0, f64::INFINITY)).unwrap(),
            json!({"lower": 0.0, "upper": null})
        );
        assert_eq!(
            serde_json::to_value(WeightBounds::per_asset(
                vec![0.0, 0.0],
                vec![0.5, f64::INFINITY]
            ))
            .unwrap(),
            json!({"lower": [0.0, 0.0], "upper": [0.5, null]})
        );
    }

    #[test]
    fn a_config_written_as_an_array_is_refused() {
        use crate::{
            BlackLitterman, CovarianceConfig, FrontierPoint, Herc, Hrp, MaxDiversification,
            MaxSharpe, MeanVariance, MinCdar, MinCvar, MinMad, MinVariance, RiskParity, View,
        };
        use serde::de::DeserializeOwned;

        // A derived struct reads from an array too, its fields by position: `[]` was any
        // config's defaults, and `[[0, 1]]` a method with bounds.
        fn refused<T: DeserializeOwned>(what: &str) {
            for text in ["[]", "[null]", "[[0, 1]]", "[0.5, 0.5]"] {
                let Err(error) = serde_json::from_str::<T>(text) else {
                    panic!("{what} read {text}");
                };
                let error = error.to_string();
                assert!(error.contains("expected an object"), "{what}: {error}");
            }
        }
        refused::<EqualWeight>("EqualWeight");
        refused::<InverseVolatility>("InverseVolatility");
        refused::<InverseVariance>("InverseVariance");
        refused::<MinVariance>("MinVariance");
        refused::<MeanVariance>("MeanVariance");
        refused::<MaxSharpe>("MaxSharpe");
        refused::<MaxDiversification>("MaxDiversification");
        refused::<RiskParity>("RiskParity");
        refused::<Hrp>("Hrp");
        refused::<Herc>("Herc");
        refused::<MinCvar>("MinCvar");
        refused::<MinCdar>("MinCdar");
        refused::<MinMad>("MinMad");
        refused::<WeightBounds>("WeightBounds");
        refused::<CovarianceConfig>("CovarianceConfig");
        refused::<BlackLitterman>("BlackLitterman");
        refused::<View>("View");
        refused::<FrontierPoint>("FrontierPoint");
        // Through a reader that isn't text, too.
        let error = serde_json::from_value::<MinVariance>(json!([[0, 1]])).unwrap_err();
        assert!(error.to_string().contains("expected an object"), "{error}");

        // An asset's bounds and a group's limits, inside the bounds, are objects as well.
        for text in [
            r#"{"assets": [[0, 0.0, 0.5]]}"#,
            r#"{"groups": [[[0, 1], 0.0, 0.5]]}"#,
        ] {
            let error = serde_json::from_str::<WeightBounds>(text).unwrap_err();
            assert!(error.to_string().contains("expected an object"), "{error}");
        }
        let bounds: WeightBounds = serde_json::from_str(
            r#"{"assets": [{"asset": 0, "lower": 0.0, "upper": 0.5}],
                "groups": [{"members": [0, 1], "lower": 0.1, "upper": 0.9}]}"#,
        )
        .unwrap();
        assert_eq!(
            bounds,
            WeightBounds::default()
                .asset(0, 0.0, 0.5)
                .group([0, 1], 0.1, 0.9)
        );
        // And an unknown key in either is still an error that names it.
        let error = serde_json::from_str::<WeightBounds>(
            r#"{"assets": [{"asset": 0, "lower": 0.0, "uper": 0.5}]}"#,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains("unknown field `uper`"),
            "{error}"
        );
    }
}
