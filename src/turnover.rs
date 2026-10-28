//! Turnover and transaction costs against current weights, for min variance and
//! mean-variance.

use crate::{interior_start, qp, CutupError, WeightBounds, Weights};
use nalgebra::{DMatrix, DVector};

/// The trading settings of [`MinVariance`](crate::MinVariance) and
/// [`MeanVariance`](crate::MeanVariance): current weights, and what moving away from them
/// costs or is allowed.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Trading {
    pub(crate) current: Option<Vec<f64>>,
    pub(crate) penalty: Option<f64>,
    pub(crate) max_turnover: Option<f64>,
    pub(crate) costs: Option<Costs>,
}

/// Proportional transaction costs, per unit of weight traded.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Costs {
    Uniform(f64),
    PerAsset(Vec<f64>),
}

impl Trading {
    /// Whether any setting changes the optimum. Current weights alone don't.
    pub(crate) fn is_active(&self) -> bool {
        self.penalty.is_some() || self.max_turnover.is_some() || self.costs.is_some()
    }

    /// The first setting that changes the optimum, by name, for errors.
    pub(crate) fn first_setting(&self) -> Option<&'static str> {
        if self.max_turnover.is_some() {
            Some("max_turnover")
        } else if self.penalty.is_some() {
            Some("turnover_penalty")
        } else if self.costs.is_some() {
            Some("transaction_costs")
        } else {
            None
        }
    }

    /// Checks the settings for `n` assets, returning the current weights, if any.
    pub(crate) fn check(&self, n: usize) -> Result<Option<DVector<f64>>, CutupError> {
        let nonnegative = |field: &'static str, value: f64| {
            if value.is_finite() && value >= 0.0 {
                Ok(())
            } else {
                Err(CutupError::InvalidConfig { field, value })
            }
        };
        if let Some(penalty) = self.penalty {
            nonnegative("turnover_penalty", penalty)?;
        }
        if let Some(cap) = self.max_turnover {
            nonnegative("max_turnover", cap)?;
        }
        match &self.costs {
            Some(Costs::Uniform(cost)) => nonnegative("transaction_costs", *cost)?,
            Some(Costs::PerAsset(costs)) if costs.len() != n => {
                return Err(CutupError::DimensionMismatch {
                    expected: n,
                    actual: costs.len(),
                })
            }
            Some(Costs::PerAsset(costs)) => {
                for &cost in costs {
                    nonnegative("transaction_costs", cost)?;
                }
            }
            None => {}
        }
        let Some(current) = &self.current else {
            return match self.first_setting() {
                Some(_) => Err(CutupError::MissingCurrentWeights),
                None => Ok(None),
            };
        };
        if current.len() != n {
            return Err(CutupError::DimensionMismatch {
                expected: n,
                actual: current.len(),
            });
        }
        if let Some(&value) = current.iter().find(|w| !w.is_finite()) {
            return Err(CutupError::InvalidConfig {
                field: "current_weights",
                value,
            });
        }
        let total: f64 = current.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(CutupError::InvalidConfig {
                field: "current_weights",
                value: total,
            });
        }
        Ok(Some(DVector::from_column_slice(current)))
    }

    fn costs(&self, n: usize) -> DVector<f64> {
        match &self.costs {
            Some(Costs::Uniform(cost)) => DVector::from_element(n, *cost),
            Some(Costs::PerAsset(costs)) => DVector::from_column_slice(costs),
            None => DVector::zeros(n),
        }
    }
}

/// Minimizes `½ wᵀQw − cᵀw + (h/2)κ‖w − w₀‖² + Σᵢ cᵢ|wᵢ − w₀ᵢ|` over portfolios summing to 1,
/// within `bounds` or, without them, long-short, and with the turnover `Σᵢ|wᵢ − w₀ᵢ|` capped
/// if `trading` caps it. `h` is the penalty's curvature in the caller's objective: 1 for min
/// variance's `wᵀΣw + κ‖w − w₀‖²`, which is twice `½ wᵀΣw + ½κ‖w − w₀‖²`, and 2 for
/// mean-variance's `−μᵀw + (γ/2) wᵀΣw + κ‖w − w₀‖²`.
pub(crate) fn rebalance(
    q: DMatrix<f64>,
    c: DVector<f64>,
    bounds: Option<&WeightBounds>,
    trading: &Trading,
    curvature: f64,
) -> Result<Weights, CutupError> {
    let n = q.ncols();
    let current = trading.check(n)?.ok_or(CutupError::MissingCurrentWeights)?;
    let bounds = match bounds {
        Some(bounds) => bounds.resolve(n)?,
        None => qp::Bounds {
            lower: DVector::from_element(n, f64::NEG_INFINITY),
            upper: DVector::from_element(n, f64::INFINITY),
        },
    };
    let penalty = curvature * trading.penalty.unwrap_or(0.0);
    let q = q + DMatrix::identity(n, n) * penalty;
    let c = c + &current * penalty;
    let (e, b) = (
        DMatrix::from_element(n, 1, 1.0),
        DVector::from_element(1, 1.0),
    );
    let kinks = qp::Kinks {
        at: current.clone(),
        slopes: trading.costs(n),
    };
    // Long-short, the optimum without the kinks is a feasible start that, unlike the current
    // weights, isn't at every kink at once.
    let start = if bounds.lower.iter().all(|l| l.is_finite()) {
        interior_start(&bounds)
    } else {
        qp::solve_kkt(&q, &c, &e, &b)?
            .ok_or(CutupError::Unbounded)?
            .0
    };
    let weights = match trading.max_turnover {
        Some(cap) => {
            let minimum = minimum_turnover(&current, &bounds);
            if minimum > cap + 4.0 * n as f64 * f64::EPSILON {
                return Err(CutupError::InfeasibleTurnover { minimum, cap });
            }
            qp::solve_turnover_capped(&q, &c, &e, &b, &bounds, (&kinks, cap), start)?
        }
        None => qp::solve_piecewise(&q, &c, &e, &b, &bounds, &kinks, start)?,
    };
    let total = weights.sum();
    Ok(Weights(weights.iter().map(|w| w / total).collect()))
}

/// The least turnover from `current` to a portfolio summing to 1 within `bounds`: move each
/// weight into its bounds, then make up what that left over or short of 1 on assets with room.
/// Every unit of that last step is turned over once, wherever it goes.
fn minimum_turnover(current: &DVector<f64>, bounds: &qp::Bounds) -> f64 {
    let clipped = DVector::from_fn(current.len(), |i, _| {
        current[i].clamp(bounds.lower[i], bounds.upper[i])
    });
    (&clipped - current).abs().sum() + (1.0 - clipped.sum()).abs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Allocation, MeanVariance, MinVariance, PortfolioAllocator};
    use nalgebra::dmatrix;

    fn allocator() -> PortfolioAllocator {
        PortfolioAllocator::from_returns(dmatrix![
             0.010, -0.020,  0.005;
            -0.004,  0.015,  0.002;
             0.007,  0.003, -0.006;
             0.002, -0.001,  0.004;
            -0.003,  0.006,  0.001
        ])
        .unwrap()
    }

    #[test]
    fn settings_need_current_weights() {
        let allocator = allocator();
        for method in [
            MinVariance::default().max_turnover(0.1),
            MinVariance::default().turnover_penalty(1e-4),
        ] {
            assert_eq!(
                method.allocate(&allocator),
                Err(CutupError::MissingCurrentWeights)
            );
        }
        assert_eq!(
            MeanVariance::risk_aversion(10.0)
                .transaction_costs(0.001)
                .allocate(&allocator),
            Err(CutupError::MissingCurrentWeights)
        );
    }

    #[test]
    fn current_weights_are_checked_even_alone() {
        let allocator = allocator();
        let invalid = |value| {
            Err(CutupError::InvalidConfig {
                field: "current_weights",
                value,
            })
        };
        assert_eq!(
            MinVariance::default()
                .current_weights([0.5, 0.5, 0.5])
                .allocate(&allocator),
            invalid(1.5)
        );
        assert_eq!(
            MinVariance::default()
                .current_weights([0.5, 0.5])
                .max_turnover(0.1)
                .allocate(&allocator),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
        // Current weights alone change nothing.
        assert_eq!(
            MinVariance::default()
                .current_weights([0.2, 0.3, 0.5])
                .allocate(&allocator),
            MinVariance::default().allocate(&allocator)
        );
    }

    #[test]
    fn settings_are_checked() {
        let allocator = allocator();
        let from = || MeanVariance::risk_aversion(10.0).current_weights([0.2, 0.3, 0.5]);
        let invalid = |field, value| Err(CutupError::InvalidConfig { field, value });
        assert_eq!(
            from().max_turnover(-0.1).allocate(&allocator),
            invalid("max_turnover", -0.1)
        );
        assert_eq!(
            from().turnover_penalty(-1.0).allocate(&allocator),
            invalid("turnover_penalty", -1.0)
        );
        assert_eq!(
            from().transaction_costs(-0.001).allocate(&allocator),
            invalid("transaction_costs", -0.001)
        );
        assert_eq!(
            from()
                .transaction_costs_per_asset([0.001, 0.002])
                .allocate(&allocator),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn target_returns_dont_take_trading_settings_yet() {
        assert_eq!(
            MeanVariance::target_return(0.001)
                .current_weights([0.2, 0.3, 0.5])
                .max_turnover(0.1)
                .allocate(&allocator()),
            Err(CutupError::NotSupported {
                setting: "max_turnover",
                with: "a target return"
            })
        );
    }

    #[test]
    fn a_cap_too_small_to_reach_the_bounds_is_infeasible() {
        // Holding 70% of the first asset, with a cap of 50% on it: selling 20% and buying it
        // elsewhere turns over 40% at least.
        let method = MinVariance::default()
            .bounds(WeightBounds::uniform(0.0, 0.5))
            .current_weights([0.7, 0.2, 0.1]);
        let allocator = allocator();
        assert_eq!(
            method.clone().max_turnover(0.3).allocate(&allocator),
            Err(CutupError::InfeasibleTurnover {
                minimum: 0.4,
                cap: 0.3
            })
        );
        let weights = method.max_turnover(0.4).allocate(&allocator).unwrap();
        assert!(weights[0] <= 0.5 + 1e-15);
    }

    #[test]
    fn a_zero_cap_keeps_the_current_weights() {
        let current = [0.2, 0.3, 0.5];
        let weights = MinVariance::default()
            .current_weights(current)
            .max_turnover(0.0)
            .allocate(&allocator())
            .unwrap();
        for (w, c) in weights.iter().zip(current) {
            assert!((w - c).abs() < 1e-15);
        }
    }

    #[test]
    fn minimum_turnover_counts_both_sides_of_a_trade() {
        let bounds = qp::Bounds {
            lower: DVector::from_element(3, 0.1),
            upper: DVector::from_element(3, 0.5),
        };
        // Moving into the bounds sells 0.2 of the first asset and buys 0.1 of the third, 0.3 in
        // all, and leaves the weights summing to 0.9, so 0.1 more is bought.
        let current = DVector::from_vec(vec![0.7, 0.3, 0.0]);
        assert!((minimum_turnover(&current, &bounds) - 0.4).abs() < 1e-15);
    }
}
