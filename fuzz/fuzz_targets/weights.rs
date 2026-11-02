//! `Weights`, `PerAsset` and `Asset` from arbitrary values and labels: lookups by column and
//! by label, iteration and `Display` mustn't panic, and must agree with each other.

#![no_main]

use arbitrary::Arbitrary;
use cutup::{Asset, EqualWeight, PerAsset, Returns, Weights};
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
enum Lookup {
    Column(usize),
    Label(String),
}

#[derive(Arbitrary, Debug)]
struct Input {
    values: Vec<f64>,
    /// Labels for a labeled `Weights`, from equal weights on that many assets.
    labels: Option<Vec<String>>,
    lookups: Vec<Lookup>,
    precision: Option<u8>,
}

fuzz_target!(|input: Input| {
    let weights = match input.labels {
        Some(labels) if !labels.is_empty() => {
            let row = vec![0.01; labels.len()];
            let data = Returns::new(&[row.clone(), row]).and_then(|data| data.labeled(labels));
            match data.and_then(|data| data.allocate(&EqualWeight)) {
                Ok(weights) => weights,
                Err(_) => return,
            }
        }
        _ => Weights::from(input.values),
    };

    assert_eq!(weights.len(), weights.iter().count());
    assert_eq!(weights.is_empty(), weights.iter().next().is_none());
    assert_eq!(weights.labels().len(), weights.len());
    for (i, (label, value)) in weights.labeled().enumerate() {
        assert_eq!(weights.labels()[i], label);
        assert_eq!(weights.get(i).map(f64::to_bits), Some(value.to_bits()));
        assert_eq!(weights[i].to_bits(), value.to_bits());
    }

    for lookup in input.lookups {
        let asset = match lookup {
            Lookup::Column(column) => {
                assert_eq!(weights.get(column).is_some(), column < weights.len());
                Asset::from(column)
            }
            Lookup::Label(label) => {
                if let Some(value) = weights.weight(&label) {
                    assert_eq!(weights[label.as_str()].to_bits(), value.to_bits());
                }
                Asset::from(label)
            }
        };
        let _ = (asset.label(), asset.column(), asset.to_string());
    }

    let _ = PerAsset::from(&weights);
    let _ = match input.precision {
        Some(precision) => format!("{weights:.*}", usize::from(precision)),
        None => weights.to_string(),
    };
    let _ = weights.into_vec();
});
