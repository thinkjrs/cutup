//! Arbitrary text read as JSON for every type that deserializes: weights, assets, values per
//! asset, bounds, covariance configs, every method, Black-Litterman models and their views,
//! and frontier points. Reading may return an error, but mustn't panic, whatever the text.
//! What's read must serialize, since JSON carries no number that isn't finite, and must read
//! back from that JSON equal to itself.

#![no_main]

use cutup::{
    Asset, BlackLitterman, CovarianceConfig, EqualWeight, FrontierPoint, Herc, Hrp,
    InverseVariance, InverseVolatility, Linkage, MaxDiversification, MaxSharpe, MeanVariance,
    MinCdar, MinCvar, MinMad, MinVariance, PerAsset, RiskParity, ShrinkageTarget, View,
    WeightBounds, Weights,
};
use libfuzzer_sys::fuzz_target;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;

/// Reads `text` as a `T`. If it is one, it must serialize and read back equal to itself.
fn read<T: Serialize + DeserializeOwned + PartialEq + Debug>(text: &str) {
    let Ok(value) = serde_json::from_str::<T>(text) else {
        return;
    };
    let json = serde_json::to_string(&value).unwrap_or_else(|e| panic!("{value:?}: {e}"));
    let back: T = serde_json::from_str(&json).unwrap_or_else(|e| panic!("{json}: {e}"));
    assert_eq!(back, value, "{text} as {json}");
}

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    read::<Weights>(text);
    read::<Asset>(text);
    read::<PerAsset>(text);
    read::<WeightBounds>(text);
    read::<CovarianceConfig>(text);
    read::<ShrinkageTarget>(text);
    read::<Linkage>(text);
    read::<EqualWeight>(text);
    read::<InverseVolatility>(text);
    read::<InverseVariance>(text);
    read::<MinVariance>(text);
    read::<MeanVariance>(text);
    read::<MaxSharpe>(text);
    read::<MaxDiversification>(text);
    read::<RiskParity>(text);
    read::<Hrp>(text);
    read::<Herc>(text);
    read::<MinCvar>(text);
    read::<MinCdar>(text);
    read::<MinMad>(text);
    read::<View>(text);
    read::<BlackLitterman>(text);
    read::<FrontierPoint>(text);
});
