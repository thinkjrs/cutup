//! `Serialize` for [`Window`]: its periods, as the rows `start` up to but not including `end`,
//! and its `weights`, or the `error` the method returned for it.
//!
//! ```json
//! {"start": 1, "end": 6, "weights": {"SPY": 0.6, "TLT": 0.4}}
//! {"start": 3, "end": 8, "error": {"zero_variance": {"column": 0, "asset": "SPY"}}}
//! ```
//!
//! A window doesn't deserialize, since a [`CutupError`] doesn't.

use super::Window;
use crate::{CutupError, Weights};
use serde::{Serialize, Serializer};

#[derive(Serialize)]
struct Wire<'a> {
    start: usize,
    end: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    weights: Option<&'a Weights>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<&'a CutupError>,
}

impl Serialize for Window {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            start: self.periods.start,
            end: self.periods.end,
            weights: self.weights.as_ref().ok(),
            error: self.weights.as_ref().err(),
        }
        .serialize(serializer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MinVariance, Returns};
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn a_window_is_its_periods_and_its_weights_or_its_error() {
        // B's returns don't move over the first three periods, so the first window of three
        // has no variance for it.
        let data = Returns::new(&[
            [0.010, 0.001],
            [-0.004, 0.001],
            [0.007, 0.001],
            [0.002, -0.003],
        ])
        .unwrap()
        .labeled(["A", "B"])
        .unwrap();
        let windows: Vec<Window> = data
            .rolling(3)
            .allocate(MinVariance::default())
            .unwrap()
            .collect();
        assert_eq!(
            serde_json::to_value(&windows[0]).unwrap(),
            json!({"start": 0, "end": 3, "error": {"zero_variance": {"column": 1, "asset": "B"}}})
        );
        let weights = windows[1].weights().as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&windows[1]).unwrap(),
            json!({"start": 1, "end": 4, "weights": {"A": weights[0], "B": weights[1]}})
        );
    }
}
