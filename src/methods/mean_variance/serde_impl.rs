//! `Serialize` and `Deserialize` for [`MeanVariance`], by its builders, and for
//! [`FrontierPoint`]:
//!
//! ```json
//! {"target_return": 0.001, "expected_returns": {"SPY": 0.0012, "TLT": 0.0004}}
//! ```
//!
//! Exactly one objective, `risk_aversion`, `target_return` or `target_volatility`. Then
//! `expected_returns`, `bounds` or `allow_short: true`, `covariance`, `current_weights`,
//! `max_turnover`, `turnover_penalty`, and `transaction_costs` or `transaction_costs_per_asset`.

use super::{FrontierPoint, MeanVariance, Objective};
use crate::serde_support::{bounds_of, is_default, long_only_fields};
use crate::turnover::{Costs, Trading};
use crate::{CovarianceConfig, PerAsset, WeightBounds, Weights};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Not;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    risk_aversion: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_return: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_volatility: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_returns: Option<PerAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    bounds: Option<WeightBounds>,
    #[serde(default, skip_serializing_if = "Not::not")]
    allow_short: bool,
    #[serde(default, skip_serializing_if = "is_default")]
    covariance: CovarianceConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    current_weights: Option<PerAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_turnover: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    turnover_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transaction_costs: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    transaction_costs_per_asset: Option<PerAsset>,
}

impl Serialize for MeanVariance {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let (bounds, allow_short) = long_only_fields(&self.bounds);
        let (transaction_costs, transaction_costs_per_asset) = match &self.trading.costs {
            None => (None, None),
            Some(Costs::Uniform(cost)) => (Some(*cost), None),
            Some(Costs::PerAsset(costs)) => (None, Some(costs.clone())),
        };
        let (risk_aversion, target_return, target_volatility) = match self.objective {
            Objective::RiskAversion(gamma) => (Some(gamma), None, None),
            Objective::TargetReturn(target) => (None, Some(target), None),
            Objective::TargetVolatility(sigma) => (None, None, Some(sigma)),
        };
        Wire {
            risk_aversion,
            target_return,
            target_volatility,
            expected_returns: self.expected_returns.clone(),
            bounds,
            allow_short,
            covariance: self.covariance.clone(),
            current_weights: self.trading.current.clone(),
            max_turnover: self.trading.max_turnover,
            turnover_penalty: self.trading.penalty,
            transaction_costs,
            transaction_costs_per_asset,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MeanVariance {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        let objective = match (
            wire.risk_aversion,
            wire.target_return,
            wire.target_volatility,
        ) {
            (Some(gamma), None, None) => Objective::RiskAversion(gamma),
            (None, Some(target), None) => Objective::TargetReturn(target),
            (None, None, Some(sigma)) => Objective::TargetVolatility(sigma),
            _ => {
                return Err(D::Error::custom(
                    "mean-variance takes exactly one of `risk_aversion`, `target_return` and \
                     `target_volatility`",
                ));
            }
        };
        let costs = match (wire.transaction_costs, wire.transaction_costs_per_asset) {
            (None, None) => None,
            (Some(cost), None) => Some(Costs::Uniform(cost)),
            (None, Some(costs)) => Some(Costs::PerAsset(costs)),
            (Some(_), Some(_)) => {
                return Err(D::Error::custom(
                    "mean-variance takes `transaction_costs` or `transaction_costs_per_asset`, \
                     not both",
                ));
            }
        };
        Ok(MeanVariance {
            objective,
            expected_returns: wire.expected_returns,
            covariance: wire.covariance,
            bounds: bounds_of(wire.bounds, wire.allow_short)?,
            trading: Trading {
                current: wire.current_weights,
                max_turnover: wire.max_turnover,
                penalty: wire.turnover_penalty,
                costs,
            },
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PointWire {
    expected_return: f64,
    volatility: f64,
    weights: Weights,
}

impl Serialize for FrontierPoint {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        PointWire {
            expected_return: self.expected_return,
            volatility: self.volatility,
            weights: self.weights.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FrontierPoint {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = PointWire::deserialize(deserializer)?;
        Ok(FrontierPoint {
            expected_return: wire.expected_return,
            volatility: wire.volatility,
            weights: wire.weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn round_trip(method: &MeanVariance, json: serde_json::Value) {
        assert_eq!(serde_json::to_value(method).unwrap(), json);
        let text = serde_json::to_string(method).unwrap();
        assert_eq!(
            &serde_json::from_str::<MeanVariance>(&text).unwrap(),
            method
        );
    }

    #[test]
    fn each_objective_and_setting_is_its_builder() {
        round_trip(
            &MeanVariance::risk_aversion(2.5),
            json!({"risk_aversion": 2.5}),
        );
        round_trip(
            &MeanVariance::target_volatility(0.01).allow_short(),
            json!({"target_volatility": 0.01, "allow_short": true}),
        );
        round_trip(
            &MeanVariance::target_return(0.001)
                .expected_returns([("SPY", 0.0012), ("TLT", 0.0004)])
                .bounds(WeightBounds::uniform(0.0, 0.6))
                .ledoit_wolf(crate::ShrinkageTarget::ConstantVariance)
                .current_weights([0.5, 0.5])
                .max_turnover(0.3)
                .turnover_penalty(1e-4)
                .transaction_costs(0.001),
            json!({
                "target_return": 0.001,
                "expected_returns": {"SPY": 0.0012, "TLT": 0.0004},
                "bounds": {"lower": 0.0, "upper": 0.6},
                "covariance": {"ledoit_wolf": "constant_variance"},
                "current_weights": [0.5, 0.5],
                "max_turnover": 0.3,
                "turnover_penalty": 1e-4,
                "transaction_costs": 0.001
            }),
        );
        round_trip(
            &MeanVariance::risk_aversion(1.0)
                .current_weights([1.0, 0.0])
                .transaction_costs_per_asset([0.001, 0.002]),
            json!({
                "risk_aversion": 1.0,
                "current_weights": [1.0, 0.0],
                "transaction_costs_per_asset": [0.001, 0.002]
            }),
        );
    }

    #[test]
    fn one_objective_and_one_kind_of_cost_are_required() {
        for json in [
            json!({}),
            json!({"risk_aversion": 1.0, "target_return": 0.001}),
            json!({"risk_aversion": 1.0, "transaction_costs": 0.001,
                   "transaction_costs_per_asset": [0.001]}),
            json!({"risk_aversion": 1.0, "bounds": {}, "allow_short": true}),
        ] {
            assert!(
                serde_json::from_value::<MeanVariance>(json.clone()).is_err(),
                "{json}"
            );
        }
    }

    #[test]
    fn a_frontier_point_is_its_return_volatility_and_weights() {
        let point = FrontierPoint {
            expected_return: 0.001,
            volatility: 0.02,
            weights: Weights::from(vec![0.25, 0.75]),
        };
        let json = json!({
            "expected_return": 0.001,
            "volatility": 0.02,
            "weights": {"0": 0.25, "1": 0.75}
        });
        assert_eq!(serde_json::to_value(&point).unwrap(), json);
        assert_eq!(
            serde_json::from_value::<FrontierPoint>(json).unwrap(),
            point
        );
    }
}
