//! Turnover and transaction costs against current weights, for min variance and
//! mean-variance.

use crate::{
    CutupError, PerAsset, ReturnRange, Returns, TargetProblem, WeightBounds, Weights,
    interior_start, lp, normalized, qp,
};
use nalgebra::{DMatrix, DVector};

/// The trading settings of [`MinVariance`](crate::MinVariance) and
/// [`MeanVariance`](crate::MeanVariance): current weights, and what moving away from them
/// costs or is allowed.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Trading {
    pub(crate) current: Option<PerAsset>,
    pub(crate) penalty: Option<f64>,
    pub(crate) max_turnover: Option<f64>,
    pub(crate) costs: Option<Costs>,
}

/// Proportional transaction costs, per unit of weight traded.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Costs {
    Uniform(f64),
    PerAsset(PerAsset),
}

impl Trading {
    /// The first setting that changes the optimum, by name, or `None` if none does. Current
    /// weights alone don't.
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

    /// The first setting that charges for trading rather than limiting it, by name: the
    /// penalty or the costs, both in the units of the returns.
    pub(crate) fn first_charge(&self) -> Option<&'static str> {
        if self.penalty.is_some() {
            Some("turnover_penalty")
        } else if self.costs.is_some() {
            Some("transaction_costs")
        } else {
            None
        }
    }

    /// Checks the settings for the assets `labels`, returning the current weights, if any,
    /// rescaled to sum to exactly 1: within `1e-9` of it counts as rounding, but a turnover cap
    /// of 0 could never be met from weights that sum to anything else. Current weights by asset
    /// are 0 for the assets they leave out.
    pub(crate) fn check(&self, labels: &[String]) -> Result<Option<DVector<f64>>, CutupError> {
        let n = labels.len();
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
            Some(Costs::PerAsset(costs)) => {
                let costs = costs.resolve(labels, None)?;
                if costs.len() != n {
                    return Err(CutupError::DimensionMismatch {
                        expected: n,
                        actual: costs.len(),
                    });
                }
                for cost in costs {
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
        let current = current.resolve(labels, Some(0.0))?;
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
        Ok(Some(DVector::from_vec(current) / total))
    }

    /// The transaction costs for the assets `labels`, once [`Trading::check`] has checked them.
    fn costs(&self, labels: &[String]) -> Result<DVector<f64>, CutupError> {
        let n = labels.len();
        Ok(match &self.costs {
            Some(Costs::Uniform(cost)) => DVector::from_element(n, *cost),
            Some(Costs::PerAsset(costs)) => DVector::from_vec(costs.resolve(labels, None)?),
            None => DVector::zeros(n),
        })
    }
}

/// Minimizes `½ wᵀQw − cᵀw + (h/2)κ‖w − w₀‖² + Σᵢ cᵢ|wᵢ − w₀ᵢ|` over portfolios summing to 1,
/// within `bounds` or, without them, long-short, and with the turnover `Σᵢ|wᵢ − w₀ᵢ|` capped
/// if `trading` caps it. `h` is the penalty's curvature in the caller's objective: 1 for min
/// variance's `wᵀΣw + κ‖w − w₀‖²`, which is twice `½ wᵀΣw + ½κ‖w − w₀‖²`, and 2 for
/// mean-variance's `−μᵀw + (γ/2) wᵀΣw + κ‖w − w₀‖²`.
pub(crate) fn rebalance(
    data: &Returns,
    q: DMatrix<f64>,
    c: DVector<f64>,
    bounds: Option<&WeightBounds>,
    trading: &Trading,
    curvature: f64,
) -> Result<Weights, CutupError> {
    let (n, labels) = (q.ncols(), data.labels());
    let current = trading
        .check(labels)?
        .ok_or(CutupError::MissingCurrentWeights)?;
    let bounds = match bounds {
        Some(bounds) => bounds.resolve(labels)?,
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
        slopes: trading.costs(labels)?,
    };
    // Long-short, the optimum without the kinks is a feasible start that, unlike the current
    // weights, isn't at every kink at once. Where there's none, as when a singular covariance
    // lets expected return grow without limit, the costs or the cap may still bound the
    // problem, and the current weights are a feasible start.
    let start = if bounds.lower.iter().all(|l| l.is_finite()) {
        interior_start(&bounds)
    } else {
        qp::solve_kkt(&q, &c, &e, &b)?.map_or_else(|| current.clone(), |(x, _)| x)
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
    Ok(data.weights(normalized(weights)))
}

/// Minimizes `wᵀΣw` over portfolios summing to 1 with expected return `μᵀw = target`, within
/// `bounds` or, without them, long-short, with the turnover `Σᵢ|wᵢ − w₀ᵢ|` capped at `cap`.
/// The caller rejects a penalty or costs, which a variance can't be traded off against.
/// `cov` builds the covariance, once an unreachable target has been rejected.
pub(crate) fn rebalance_to_target(
    data: &Returns,
    cov: impl FnOnce() -> Result<DMatrix<f64>, CutupError>,
    mu: DVector<f64>,
    target: f64,
    bounds: Option<&WeightBounds>,
    trading: &Trading,
    cap: f64,
) -> Result<Weights, CutupError> {
    let (n, labels) = (mu.len(), data.labels());
    let current = trading
        .check(labels)?
        .ok_or(CutupError::MissingCurrentWeights)?;
    let (
        cov,
        TargetProblem {
            e,
            b,
            bounds,
            start,
        },
    ) = match bounds {
        Some(bounds) => {
            let problem = ReturnRange::target_problem(&mu, bounds.resolve(labels)?, target)?;
            (cov()?, problem)
        }
        None => {
            let cov = cov()?;
            let e = DMatrix::from_columns(&[DVector::from_element(n, 1.0), mu]);
            let b = DVector::from_vec(vec![1.0, target]);
            let start = qp::solve_kkt(&cov, &DVector::zeros(n), &e, &b)?
                .ok_or(CutupError::DegenerateExpectedReturns)?
                .0;
            let bounds = qp::Bounds {
                lower: DVector::from_element(n, f64::NEG_INFINITY),
                upper: DVector::from_element(n, f64::INFINITY),
            };
            (
                cov,
                TargetProblem {
                    e,
                    b,
                    bounds,
                    start,
                },
            )
        }
    };
    // With only the budget, as at an end of the range, the closed form gives the least.
    let minimum = if e.ncols() == 1 {
        minimum_turnover(&current, &bounds)
    } else {
        least_turnover(&current, &e, &b, &bounds)?
    };
    // A cap at the least turnover leaves only the portfolios that reach it, which the solver
    // finds only to the rounding of the target's constraint, a little worse than the linear
    // program. So within rounding of the least, as for bounds that sum to 1, counts as at it.
    let rounding = 1e-12 * minimum.max(1.0);
    if minimum > cap + rounding {
        return Err(CutupError::InfeasibleTurnover { minimum, cap });
    }
    let kinks = qp::Kinks {
        at: current,
        slopes: DVector::zeros(n),
    };
    let weights = qp::solve_turnover_capped(
        &cov,
        &DVector::zeros(n),
        &e,
        &b,
        &bounds,
        (&kinks, cap.max(minimum + rounding)),
        start,
    )?;
    Ok(data.weights(normalized(weights)))
}

/// The least turnover `Σᵢ|wᵢ − w₀ᵢ|` from `current` to a portfolio with `eᵀw = b` within
/// `bounds`, from the linear program over the weights and the trades `w − w₀ = p − s`, with
/// purchases `p` and sales `s` both nonnegative. Unlike [`minimum_turnover`], it allows for
/// constraints besides the budget, such as a target return.
fn least_turnover(
    current: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &qp::Bounds,
) -> Result<f64, CutupError> {
    let (n, k) = e.shape();
    let mut a = DMatrix::zeros(n + k, 3 * n);
    let mut rhs = DVector::zeros(n + k);
    for i in 0..n {
        a[(i, i)] = 1.0;
        a[(i, n + i)] = -1.0;
        a[(i, 2 * n + i)] = 1.0;
        rhs[i] = current[i];
        for r in 0..k {
            a[(n + r, i)] = e[(i, r)];
        }
    }
    rhs.rows_mut(n, k).copy_from(b);
    let cost = DVector::from_fn(3 * n, |j, _| if j < n { 0.0 } else { 1.0 });
    let lower = DVector::from_fn(3 * n, |j, _| if j < n { bounds.lower[j] } else { 0.0 });
    let upper = DVector::from_fn(3 * n, |j, _| {
        if j < n {
            bounds.upper[j]
        } else {
            f64::INFINITY
        }
    });
    // A feasible portfolio exists, and turnover is bounded below by 0.
    match lp::solve(&cost, &a, &rhs, &lower, &upper)? {
        lp::Outcome::Optimal(solution) => Ok(solution.objective),
        _ => Err(CutupError::NotConverged),
    }
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
    use crate::{Allocation, MeanVariance, MinVariance, Returns};
    use nalgebra::dmatrix;

    fn allocator() -> Returns {
        Returns::returns_matrix(dmatrix![
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
    fn target_returns_take_a_cap_but_not_charges() {
        let method = MeanVariance::target_return(0.001).current_weights([0.2, 0.3, 0.5]);
        assert_eq!(
            method.clone().turnover_penalty(0.1).allocate(&allocator()),
            Err(CutupError::NotSupported {
                setting: "turnover_penalty",
                with: "a target return"
            })
        );
        assert_eq!(
            method
                .max_turnover(0.1)
                .transaction_costs(0.001)
                .allocate(&allocator()),
            Err(CutupError::NotSupported {
                setting: "transaction_costs",
                with: "a target return"
            })
        );
    }

    fn turnover(weights: &Weights, current: [f64; 3]) -> f64 {
        weights
            .iter()
            .zip(current)
            .map(|(w, c)| (w - c).abs())
            .sum()
    }

    #[test]
    fn a_capped_target_return_meets_its_target_within_the_cap() {
        // Expected returns 0.0024, 0.0006 and 0.0012; the current weights return 0.00126.
        let allocator = allocator();
        let mu = [0.0024, 0.0006, 0.0012];
        let current = [0.2, 0.3, 0.5];
        for method in [
            MeanVariance::target_return(0.0018),
            MeanVariance::target_return(0.0018).allow_short(),
        ] {
            let free = method.clone().allocate(&allocator).unwrap();
            assert!(turnover(&free, current) > 0.7);
            let capped = method
                .clone()
                .current_weights(current)
                .max_turnover(0.7)
                .allocate(&allocator)
                .unwrap();
            let achieved: f64 = capped.iter().zip(mu).map(|(w, m)| w * m).sum();
            assert!((achieved - 0.0018).abs() < 1e-17, "{achieved}");
            assert!((turnover(&capped, current) - 0.7).abs() < 1e-14);
            // A cap that doesn't bind changes nothing.
            let loose = method
                .current_weights(current)
                .max_turnover(10.0)
                .allocate(&allocator)
                .unwrap();
            for (w, f) in loose.iter().zip(free.iter()) {
                assert!((w - f).abs() < 1e-12, "{w} vs {f}");
            }
        }
    }

    #[test]
    fn a_target_return_needs_its_least_turnover() {
        // Raising the return from 0.00126 to 0.0018 most cheaply sells all 0.3 of the second
        // asset, the lowest, for the first, the highest: 0.6 of turnover.
        let current = [0.2, 0.3, 0.5];
        let method = MeanVariance::target_return(0.0018).current_weights(current);
        match method.clone().max_turnover(0.5).allocate(&allocator()) {
            Err(CutupError::InfeasibleTurnover { minimum, cap }) => {
                assert!((minimum - 0.6).abs() < 1e-15 && cap == 0.5, "{minimum}");
            }
            other => panic!("{other:?}"),
        }
        assert!(method.max_turnover(0.6).allocate(&allocator()).is_ok());
        // A cap of 0 keeps the current weights at their own return.
        let weights = MeanVariance::target_return(0.00126)
            .current_weights(current)
            .max_turnover(0.0)
            .allocate(&allocator())
            .unwrap();
        for (w, c) in weights.iter().zip(current) {
            assert!((w - c).abs() < 1e-15);
        }
    }

    #[test]
    fn a_cap_holds_with_equal_expected_returns() {
        // Every portfolio returns 0.004, so a cap of 0 keeps the current weights, long-only or
        // long-short.
        let current = [0.2, 0.3, 0.5];
        for method in [
            MeanVariance::target_return(0.004),
            MeanVariance::target_return(0.004).allow_short(),
        ] {
            let weights = method
                .expected_returns([0.004; 3])
                .current_weights(current)
                .max_turnover(0.0)
                .allocate(&allocator())
                .unwrap();
            assert!(turnover(&weights, current) < 1e-15, "{weights:?}");
        }
    }

    #[test]
    fn a_cap_below_the_least_turnover_isnt_raised() {
        // Expected returns 1e-14 apart: the least turnover to the target is about 0.5, and a
        // cap 1e-6 short of it is infeasible however badly the target is conditioned.
        let mu = [0.001, 0.001 + 1e-14, 0.001 + 5e-15];
        for long_short in [false, true] {
            let method = |cap: f64| {
                let method = MeanVariance::target_return(0.001 + 8e-15)
                    .expected_returns(mu)
                    .current_weights([0.2, 0.3, 0.5])
                    .max_turnover(cap);
                if long_short {
                    method.allow_short()
                } else {
                    method
                }
            };
            let Err(CutupError::InfeasibleTurnover { minimum, .. }) =
                method(0.0).allocate(&allocator())
            else {
                panic!("a cap of 0 can't reach the target");
            };
            assert!(matches!(
                method(minimum - 1e-6).allocate(&allocator()),
                Err(CutupError::InfeasibleTurnover { .. })
            ));
        }
    }

    #[test]
    fn an_unreachable_capped_target_is_rejected_before_the_covariance() {
        // The third asset never moves, so its variance is zero and the covariance can't be
        // built, but the target is out of range first, with a cap as without one.
        let allocator = Returns::returns_matrix(dmatrix![
            0.01, -0.02, 0.0;
            -0.004, 0.015, 0.0;
            0.007, 0.003, 0.0
        ])
        .unwrap();
        let method = MeanVariance::target_return(0.1).expected_returns([0.001, 0.002, 0.003]);
        let uncapped = method.clone().allocate(&allocator);
        assert!(matches!(
            uncapped,
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        assert_eq!(
            method
                .current_weights([0.2, 0.3, 0.5])
                .max_turnover(0.1)
                .allocate(&allocator),
            uncapped
        );
    }

    #[test]
    fn a_capped_target_at_the_top_of_the_bounds_is_their_face() {
        // At most half in each asset, the highest return is half each in the first and third,
        // 0.0018, which is 0.6 of turnover away.
        let method = MeanVariance::target_return(0.0018)
            .bounds(WeightBounds::uniform(0.0, 0.5))
            .current_weights([0.2, 0.3, 0.5]);
        assert!(matches!(
            method.clone().max_turnover(0.5).allocate(&allocator()),
            Err(CutupError::InfeasibleTurnover { .. })
        ));
        let weights = method.max_turnover(0.6).allocate(&allocator()).unwrap();
        for (w, want) in weights.iter().zip([0.5, 0.0, 0.5]) {
            assert!((w - want).abs() < 1e-15, "{w}");
        }
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
    fn current_weights_within_rounding_of_one_are_rescaled() {
        // Summing to 1 − 1e-10, which a cap of 0 could never be met from as given.
        let current = [0.2, 0.3, 0.4999999999];
        for method in [MinVariance::default(), MinVariance::default().allow_short()] {
            let weights = method
                .current_weights(current)
                .max_turnover(0.0)
                .allocate(&allocator())
                .unwrap();
            let total: f64 = current.iter().sum();
            for (w, c) in weights.iter().zip(current) {
                assert!((w - c / total).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn a_cap_bounds_an_unbounded_long_short_objective() {
        // Two periods of three assets: Σ is singular, so long-short mean-variance gains
        // expected return without limit, until a turnover cap stops it.
        let allocator = Returns::returns_matrix(dmatrix![
            0.01, -0.02, 0.005;
            -0.004, 0.015, 0.002
        ])
        .unwrap();
        let method = MeanVariance::risk_aversion(10.0).allow_short();
        assert_eq!(method.allocate(&allocator), Err(CutupError::Unbounded));
        let current = [0.3, 0.3, 0.4];
        let weights = method
            .current_weights(current)
            .max_turnover(0.2)
            .allocate(&allocator)
            .unwrap();
        let turnover: f64 = weights
            .iter()
            .zip(current)
            .map(|(w, c)| (w - c).abs())
            .sum();
        assert!((turnover - 0.2).abs() < 1e-15, "{turnover}");
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
        let least = least_turnover(
            &current,
            &DMatrix::from_element(3, 1, 1.0),
            &DVector::from_element(1, 1.0),
            &bounds,
        )
        .unwrap();
        assert!((least - 0.4).abs() < 1e-15, "{least}");
    }
}
