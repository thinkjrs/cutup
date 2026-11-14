//! `Serialize` and `Deserialize` for [`CovarianceConfig`]: an object whose keys are its
//! builders, each with its argument.
//!
//! ```json
//! {"exponentially_weighted": 11.2, "shrinkage": 0.1, "regularization": 1e-8}
//! ```
//!
//! At most one estimate: `matrix`, rows of a given covariance; `exponentially_weighted`, a
//! half-life; `exponentially_weighted_decay`, a `λ`; `semicovariance`, `true`; or
//! `semicovariance_below`, a threshold. Without one it's the sample covariance. At most one of
//! `shrinkage`, an intensity, and `ledoit_wolf`, `"constant_variance"` or
//! `"constant_correlation"`. Then `regularization`. The default, `{}`, is the sample covariance
//! as it is. A given matrix is checked as [`CovarianceConfig::matrix`] checks it, and other
//! values when a method uses them.

use super::{CovarianceConfig, Decay, Estimate, Shrinkage};
use crate::ShrinkageTarget;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Default, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-rs", derive(ts_rs::TS), ts(rename = "CovarianceConfig"))]
#[serde(deny_unknown_fields)]
pub(super) struct Wire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    matrix: Option<Vec<Vec<f64>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exponentially_weighted: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    exponentially_weighted_decay: Option<f64>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    semicovariance: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    semicovariance_below: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shrinkage: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ledoit_wolf: Option<ShrinkageTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    regularization: Option<f64>,
}

impl Serialize for CovarianceConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut wire = Wire {
            regularization: self.regularization,
            ..Wire::default()
        };
        match &self.estimate {
            Estimate::Sample => {}
            Estimate::Given(matrix) => {
                wire.matrix = Some(
                    matrix
                        .row_iter()
                        .map(|row| row.iter().copied().collect())
                        .collect(),
                );
            }
            Estimate::ExponentiallyWeighted(Decay::HalfLife(h)) => {
                wire.exponentially_weighted = Some(*h);
            }
            Estimate::ExponentiallyWeighted(Decay::Lambda(lambda)) => {
                wire.exponentially_weighted_decay = Some(*lambda);
            }
            Estimate::Semi(None) => wire.semicovariance = true,
            Estimate::Semi(Some(threshold)) => wire.semicovariance_below = Some(*threshold),
        }
        match self.shrinkage {
            None => {}
            Some(Shrinkage::Fixed(lambda)) => wire.shrinkage = Some(lambda),
            Some(Shrinkage::LedoitWolf(target)) => wire.ledoit_wolf = Some(target),
        }
        wire.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CovarianceConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        let estimates = [
            wire.matrix.is_some(),
            wire.exponentially_weighted.is_some(),
            wire.exponentially_weighted_decay.is_some(),
            wire.semicovariance,
            wire.semicovariance_below.is_some(),
        ];
        if estimates.into_iter().filter(|&set| set).count() > 1 {
            return Err(D::Error::custom(
                "a covariance config takes at most one of `matrix`, `exponentially_weighted`, \
                 `exponentially_weighted_decay`, `semicovariance` and `semicovariance_below`",
            ));
        }
        if wire.shrinkage.is_some() && wire.ledoit_wolf.is_some() {
            return Err(D::Error::custom(
                "a covariance config takes `shrinkage` or `ledoit_wolf`, not both",
            ));
        }
        let mut config = if let Some(rows) = wire.matrix {
            CovarianceConfig::matrix(&rows).map_err(D::Error::custom)?
        } else if let Some(h) = wire.exponentially_weighted {
            CovarianceConfig::exponentially_weighted(h)
        } else if let Some(lambda) = wire.exponentially_weighted_decay {
            CovarianceConfig::exponentially_weighted_decay(lambda)
        } else if wire.semicovariance {
            CovarianceConfig::semicovariance()
        } else if let Some(threshold) = wire.semicovariance_below {
            CovarianceConfig::semicovariance_below(threshold)
        } else {
            CovarianceConfig::default()
        };
        if let Some(lambda) = wire.shrinkage {
            config = config.shrinkage(lambda);
        }
        if let Some(target) = wire.ledoit_wolf {
            config = config.ledoit_wolf(target);
        }
        if let Some(epsilon) = wire.regularization {
            config = config.regularization(epsilon);
        }
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn round_trip(config: &CovarianceConfig, json: serde_json::Value) {
        assert_eq!(serde_json::to_value(config).unwrap(), json);
        assert_eq!(
            &serde_json::from_value::<CovarianceConfig>(json).unwrap(),
            config
        );
    }

    #[test]
    fn each_setting_is_its_builder() {
        round_trip(&CovarianceConfig::default(), json!({}));
        round_trip(
            &CovarianceConfig::matrix(&[[1e-4, 2e-5], [2e-5, 4e-4]]).unwrap(),
            json!({"matrix": [[1e-4, 2e-5], [2e-5, 4e-4]]}),
        );
        round_trip(
            &CovarianceConfig::exponentially_weighted(11.2).shrinkage(0.1),
            json!({"exponentially_weighted": 11.2, "shrinkage": 0.1}),
        );
        round_trip(
            &CovarianceConfig::exponentially_weighted_decay(0.94).regularization(1e-8),
            json!({"exponentially_weighted_decay": 0.94, "regularization": 1e-8}),
        );
        round_trip(
            &CovarianceConfig::semicovariance(),
            json!({"semicovariance": true}),
        );
        round_trip(
            &CovarianceConfig::semicovariance_below(0.0),
            json!({"semicovariance_below": 0.0}),
        );
        round_trip(
            &CovarianceConfig::default().ledoit_wolf(ShrinkageTarget::ConstantCorrelation),
            json!({"ledoit_wolf": "constant_correlation"}),
        );
        assert_eq!(
            serde_json::from_value::<CovarianceConfig>(json!({"semicovariance": false})).unwrap(),
            CovarianceConfig::default()
        );
    }

    #[test]
    fn a_given_matrix_is_checked() {
        let error = serde_json::from_value::<CovarianceConfig>(
            json!({"matrix": [[1e-4, 2e-4], [2e-4, 1e-4]]}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("negative eigenvalue"), "{error}");
    }

    #[test]
    fn settings_that_exclude_each_other_are_rejected() {
        for json in [
            json!({"semicovariance": true, "exponentially_weighted": 5.0}),
            json!({"shrinkage": 0.1, "ledoit_wolf": "constant_variance"}),
            json!({"shrinkage_intensity": 0.1}),
            json!({"ledoit_wolf": "constant"}),
        ] {
            assert!(
                serde_json::from_value::<CovarianceConfig>(json.clone()).is_err(),
                "{json}"
            );
        }
    }
}
