//! Every `Returns` constructor, on arbitrary shapes and floats: NaN, infinities, subnormals and
//! huge values included. Each may return an error, but none may panic. On success, labeling,
//! `cvar` and the label lookups mustn't panic either, and `log_returns` must keep the labels or
//! fail only for a return of −1 or below.

#![no_main]

use arbitrary::Arbitrary;
use cutup::{CutupError, EqualWeight, Returns, Weights};
use libfuzzer_sys::fuzz_target;
use nalgebra::DMatrix;

#[derive(Arbitrary, Debug)]
enum Constructor {
    New,
    NewFlat,
    FromPrices,
    FromPricesFlat,
    FromMatrix,
    FromPricesMatrix,
    FromReturns,
}

#[derive(Arbitrary, Debug)]
struct Input {
    constructor: Constructor,
    /// The rows, for the row constructors. They may be ragged.
    rows: Vec<Vec<f64>>,
    /// The values and column count, for the flat and matrix constructors.
    values: Vec<f64>,
    assets: u8,
    labels: Option<Vec<String>>,
    weights: Vec<f64>,
    beta: f64,
    lookup: String,
}

fuzz_target!(|input: Input| {
    let assets = usize::from(input.assets);
    let matrix = || {
        let columns = assets.max(1);
        let rows = input.values.len() / columns;
        DMatrix::from_row_slice(rows, columns, &input.values[..rows * columns])
    };
    #[allow(deprecated)]
    let data = match input.constructor {
        Constructor::New => Returns::new(&input.rows),
        Constructor::NewFlat => Returns::new_flat(&input.values, assets),
        Constructor::FromPrices => Returns::from_prices(&input.rows),
        Constructor::FromPricesFlat => Returns::from_prices_flat(&input.values, assets),
        Constructor::FromMatrix => Returns::from_matrix(matrix()),
        Constructor::FromPricesMatrix => Returns::from_prices_matrix(matrix()),
        Constructor::FromReturns => Returns::from_returns(&input.rows),
    };
    let Ok(mut data) = data else { return };

    if let Some(labels) = input.labels {
        match data.clone().labeled(labels) {
            Ok(labeled) => data = labeled,
            Err(_) => return,
        }
    }
    let _ = data.labels();
    match data.log_returns() {
        Ok(logs) => assert_eq!(logs.labels(), data.labels()),
        Err(e) => assert!(matches!(e, CutupError::TotalLoss { .. }), "{e:?}"),
    }
    let _ = data.cvar(&Weights::from(input.weights), input.beta);
    if let Ok(equal) = data.allocate(&EqualWeight) {
        let _ = data.cvar(&equal, input.beta);
        let _ = equal.weight(&input.lookup);
    }
});
