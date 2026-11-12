//! Turnover and transaction costs against current weights, for min variance and
//! mean-variance.

use crate::asset::check_values;
use crate::bounds::{ReturnRange, TargetProblem, feasible_start, interior_start, without_groups};
use crate::methods::normalized;
use crate::{CutupError, PerAsset, Returns, WeightBounds, Weights, lp, qp};
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

    /// Checks the settings for the assets `labels`, returning the current weights, if any,
    /// rescaled to sum to exactly 1: within `1e-9` of it counts as rounding, but a turnover cap
    /// of 0 could never be met from weights that sum to anything else. Current weights by asset
    /// are 0 for the assets they leave out. Returns the transaction costs per asset too.
    pub(crate) fn check(&self, labels: &[String]) -> Result<Checked, CutupError> {
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
        let costs = match &self.costs {
            Some(Costs::Uniform(cost)) => {
                nonnegative("transaction_costs", *cost)?;
                DVector::from_element(n, *cost)
            }
            Some(Costs::PerAsset(costs)) => {
                let costs = costs.resolve("transaction_costs", labels, None)?;
                check_values("transaction_costs", labels, &costs, |c| {
                    c.is_finite() && c >= 0.0
                })?;
                DVector::from_vec(costs)
            }
            None => DVector::zeros(n),
        };
        let Some(current) = &self.current else {
            return match self.first_setting() {
                Some(_) => Err(CutupError::MissingCurrentWeights),
                None => Ok(Checked {
                    current: None,
                    costs,
                }),
            };
        };
        let current = current.resolve("current_weights", labels, Some(0.0))?;
        check_values("current_weights", labels, &current, f64::is_finite)?;
        let total: f64 = current.iter().sum();
        if (total - 1.0).abs() > 1e-9 {
            return Err(CutupError::InvalidConfig {
                field: "current_weights",
                value: total,
            });
        }
        Ok(Checked {
            current: Some(DVector::from_vec(current) / total),
            costs,
        })
    }
}

/// The settings [`Trading::check`] has checked, in column order.
pub(crate) struct Checked {
    /// The current weights, if any, summing to exactly 1.
    pub(crate) current: Option<DVector<f64>>,
    /// The transaction costs per asset, 0 without any.
    pub(crate) costs: DVector<f64>,
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
    let checked = trading.check(labels)?;
    let current = checked.current.ok_or(CutupError::MissingCurrentWeights)?;
    let bounds = resolve_bounds(bounds, labels)?;
    let penalty = curvature * trading.penalty.unwrap_or(0.0);
    let q = q + DMatrix::identity(n, n) * penalty;
    let c = c + &current * penalty;
    let (e, b) = (
        DMatrix::from_element(n, 1, 1.0),
        DVector::from_element(1, 1.0),
    );
    let kinks = qp::Kinks {
        at: current.clone(),
        slopes: checked.costs,
    };
    // Long-short, the optimum without the kinks is a feasible start that, unlike the current
    // weights, isn't at every kink at once. Where there's none, as when a singular covariance
    // lets expected return grow without limit, the costs or the cap may still bound the
    // problem, and the current weights are a feasible start. So are they when the optimum is
    // too ill-conditioned to meet the budget: the solve below only needs somewhere to start.
    let start = if bounds.lower.iter().all(|l| l.is_finite()) {
        feasible_start(&bounds)?
    } else {
        match qp::solve_kkt(&q, &c, &e, &b) {
            Ok(Some((x, _))) => x,
            Ok(None) | Err(CutupError::SingularCovariance) => current.clone(),
            Err(error) => return Err(error),
        }
    };
    let weights = match trading.max_turnover {
        Some(cap) => {
            check_cap(&current, &bounds, cap)?;
            qp::solve_turnover_capped(&q, &c, &e, &b, &bounds, (&kinks, cap), start)?
        }
        None => qp::solve_piecewise(&q, &c, &e, &b, &bounds, &kinks, start)?,
    };
    Ok(data.weights(normalized(weights)))
}

/// Minimizes `wᵀΣw + κ‖w − w₀‖²` over portfolios summing to 1 with expected return
/// `μᵀw = target`, within `bounds` or, without them, long-short, with the turnover
/// `Σᵢ|wᵢ − w₀ᵢ|` capped at `cap`, `κ` being `trading`'s penalty, if any. The caller sends
/// costs to [`rebalance_to_net_target`] instead. `cov` builds the covariance, once an
/// unreachable target has been rejected.
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
        .current
        .ok_or(CutupError::MissingCurrentWeights)?;
    let penalty = trading.penalty.unwrap_or(0.0);
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
            let bounds = without_groups(bounds.resolve(labels)?, "a target return")?;
            let problem = ReturnRange::target_problem(&mu, bounds, target)?;
            (cov()?, problem)
        }
        None => {
            let cov = cov()?;
            let e = DMatrix::from_columns(&[DVector::from_element(n, 1.0), mu]);
            let b = DVector::from_vec(vec![1.0, target]);
            let start = qp::solve_kkt(&cov, &DVector::zeros(n), &e, &b)?
                .ok_or(CutupError::DegenerateExpectedReturns)?
                .0;
            let bounds = unbounded(n);
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
    let minimum = if cap.is_infinite() {
        0.0
    } else if e.ncols() == 1 {
        minimum_turnover(&current, &bounds)?
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
    // Half of wᵀΣw + κ‖w − w₀‖², less a constant: ½wᵀ(Σ + κI)w − κw₀ᵀw.
    let q = cov + DMatrix::identity(n, n) * penalty;
    let c = &current * penalty;
    let kinks = qp::Kinks {
        at: current,
        slopes: DVector::zeros(n),
    };
    let weights = qp::solve_turnover_capped(
        &q,
        &c,
        &e,
        &b,
        &bounds,
        (&kinks, cap.max(minimum + rounding)),
        start,
    )?;
    Ok(data.weights(normalized(weights)))
}

/// Minimizes `wᵀΣw + κ‖w − w₀‖²` over portfolios summing to 1 with a net return
/// `μᵀw − Σᵢ cᵢ|wᵢ − w₀ᵢ|` of at least `target`, within `bounds` or, without them, long-short,
/// with the turnover capped at `cap`, `κ` and `c` being `trading`'s penalty and costs. With
/// every cost 0, it's [`rebalance_to_target`].
///
/// The net return is concave, so this is convex. A multiplier `λ ≥ 0` on the target turns it
/// into the kinked problem `½wᵀ(Σ + κI)w − (κw₀ + λμ)ᵀw + λΣᵢ cᵢ|wᵢ − w₀ᵢ|`, whose optimum's
/// net return grows with `λ`, so [`qp::search_multiplier`] finds the `λ` that reaches the
/// target. If the optimum at `λ = 0`, the least variance within the bounds and the cap,
/// already nets the target, that's the answer.
///
/// # Errors
///
/// [`CutupError::Unbounded`], long-short, if `Σ + κI` is singular and some fully invested
/// portfolio adds net return without adding variance: the least variance is then reached
/// along a whole line of portfolios, and every `λ > 0` runs off along it.
/// [`CutupError::TargetReturnOutOfRange`] if no portfolio within the bounds and the cap nets
/// `target`, with `lowest` −∞, since every target up to `highest` is reachable, and
/// [`CutupError::InfeasibleTurnover`] if no portfolio within the bounds is within the cap,
/// both before `cov` builds the covariance.
pub(crate) fn rebalance_to_net_target(
    data: &Returns,
    cov: impl FnOnce() -> Result<DMatrix<f64>, CutupError>,
    mu: DVector<f64>,
    target: f64,
    bounds: Option<&WeightBounds>,
    trading: &Trading,
    cap: f64,
) -> Result<Weights, CutupError> {
    let (n, labels) = (mu.len(), data.labels());
    let checked = trading.check(labels)?;
    // Costs of 0 charge nothing, so they change nothing: the target is the expected return,
    // exactly, as without them.
    if checked.costs.iter().all(|&cost| cost == 0.0) {
        return rebalance_to_target(data, cov, mu, target, bounds, trading, cap);
    }
    let current = checked.current.ok_or(CutupError::MissingCurrentWeights)?;
    let costs = checked.costs;
    let bounds = without_groups(resolve_bounds(bounds, labels)?, "a target return")?;
    let cap = cap.max(check_cap(&current, &bounds, cap)?);
    let (highest, rounding) = highest_net_return(&current, (&mu, &costs), &bounds, cap)?;
    if target > highest + rounding {
        return Err(CutupError::TargetReturnOutOfRange {
            target_return: target,
            lowest: f64::NEG_INFINITY,
            highest,
        });
    }
    // A target within rounding of the highest is at it, which a large enough λ reaches.
    let target = target.min(highest);

    let penalty = trading.penalty.unwrap_or(0.0);
    let q = cov()? + DMatrix::identity(n, n) * penalty;
    let anchor = &current * penalty;
    let (e, b) = (
        DMatrix::from_element(n, 1, 1.0),
        DVector::from_element(1, 1.0),
    );
    let net = |w: &DVector<f64>| mu.dot(w) - costs.dot(&(w - &current).abs());
    let solve = |lambda: f64, start: DVector<f64>| {
        let c = &anchor + &mu * lambda;
        let kinks = qp::Kinks {
            at: current.clone(),
            slopes: &costs * lambda,
        };
        let w = if cap.is_finite() {
            qp::solve_turnover_capped(&q, &c, &e, &b, &bounds, (&kinks, cap), start)?
        } else {
            qp::solve_piecewise(&q, &c, &e, &b, &bounds, &kinks, start)?
        };
        Ok((-net(&w), Some(w)))
    };
    // The net return is a sum over the assets, which rounds by a few ulps of its terms.
    let slack = |w: &DVector<f64>| qp::sum_rounding(n, net_terms(&mu, &costs, &current, w));
    let start = if bounds.lower.iter().all(|l| l.is_finite()) {
        interior_start(&bounds)
    } else {
        current.clone()
    };
    // The search doubles λ from 1 until it nets the target, then narrows it down, so it finds
    // λ from any start.
    let weights = qp::search_multiplier(solve, (-target, slack), 1.0, start)?;
    Ok(data.weights(normalized(weights)))
}

/// The least turnover `Σᵢ|wᵢ − w₀ᵢ|` from `current` to a portfolio with `eᵀw = b` within
/// `bounds`. Unlike [`minimum_turnover`], it allows for constraints besides the budget, such as
/// a target return.
fn least_turnover(
    current: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &qp::Bounds,
) -> Result<f64, CutupError> {
    let n = current.len();
    let trades = Trades::new(current, e, b, bounds, None);
    // The purchases and sales; not the weights or the groups' totals.
    let cost = DVector::from_fn(trades.columns(), |j, _| {
        if (n..3 * n).contains(&j) { 1.0 } else { 0.0 }
    });
    // A feasible portfolio exists, and turnover is bounded below by 0.
    match trades.solve(&cost)? {
        lp::Outcome::Optimal(solution) => Ok(solution.objective),
        _ => Err(CutupError::NotConverged),
    }
}

/// The highest net return `μᵀw − Σᵢ cᵢ|wᵢ − w₀ᵢ|` of a portfolio summing to 1 within `bounds`,
/// with the turnover capped at `cap`, and the rounding of the net return there, or infinity
/// if it grows without limit. The caller has checked that some portfolio is within the cap.
fn highest_net_return(
    current: &DVector<f64>,
    (mu, costs): (&DVector<f64>, &DVector<f64>),
    bounds: &qp::Bounds,
    cap: f64,
) -> Result<(f64, f64), CutupError> {
    let n = current.len();
    let trades = Trades::new(
        current,
        &DMatrix::from_element(n, 1, 1.0),
        &DVector::from_element(1, 1.0),
        bounds,
        cap.is_finite().then_some(cap),
    );
    let cost = DVector::from_fn(trades.columns(), |j, _| match j {
        _ if j < n => -mu[j],
        _ if j < 3 * n => costs[j % n],
        _ => 0.0,
    });
    match trades.solve(&cost)? {
        lp::Outcome::Optimal(solution) => {
            let w = solution.x.rows(0, n).into_owned();
            Ok((
                -solution.objective,
                1e-12 * net_terms(mu, costs, current, &w),
            ))
        }
        lp::Outcome::Unbounded => Ok((f64::INFINITY, 0.0)),
        lp::Outcome::Infeasible => Err(CutupError::NotConverged),
    }
}

/// The terms of `w`'s net return from `current`, `Σᵢ|μᵢwᵢ| + Σᵢ cᵢ|wᵢ − w₀ᵢ|`, which its
/// rounding scales with.
fn net_terms(
    mu: &DVector<f64>,
    costs: &DVector<f64>,
    current: &DVector<f64>,
    w: &DVector<f64>,
) -> f64 {
    mu.component_mul(w).abs().sum() + costs.dot(&(w - current).abs())
}

/// The linear program over the weights `w` and the trades to them from `current`,
/// `w − w₀ = p − s`, with purchases `p` and sales `s` both nonnegative, so that `Σᵢ(pᵢ + sᵢ)`
/// is at least the turnover, and equal to it at an optimum that charges for trading. The
/// weights meet `eᵀw = b` within `bounds`, and with a cap a slack variable makes
/// `Σᵢ(pᵢ + sᵢ) ≤ cap` a row. Each of the bounds' groups gets a row too, `aⱼᵀw − tⱼ = 0`, with
/// its total `tⱼ` within its limits. The variables are `w`, `p` and `s`, then the cap's slack,
/// then the groups' totals.
struct Trades {
    a: DMatrix<f64>,
    rhs: DVector<f64>,
    lower: DVector<f64>,
    upper: DVector<f64>,
}

impl Trades {
    fn new(
        current: &DVector<f64>,
        e: &DMatrix<f64>,
        b: &DVector<f64>,
        bounds: &qp::Bounds,
        cap: Option<f64>,
    ) -> Self {
        let (n, k) = e.shape();
        let capped = usize::from(cap.is_some());
        let groups = &bounds.rows;
        let r = groups.a.ncols();
        let (rows, columns) = (n + k + capped + r, 3 * n + capped + r);
        let mut a = DMatrix::zeros(rows, columns);
        let mut rhs = DVector::zeros(rows);
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
        if let Some(cap) = cap {
            a.row_mut(n + k).columns_mut(n, 2 * n + 1).fill(1.0);
            rhs[n + k] = cap;
        }
        let (first_row, first_total) = (rows - r, columns - r);
        a.view_mut((first_row, 0), (r, n))
            .copy_from(&groups.a.transpose());
        a.view_mut((first_row, first_total), (r, r))
            .fill_diagonal(-1.0);
        Trades {
            a,
            rhs,
            lower: DVector::from_fn(columns, |j, _| match j {
                _ if j < n => bounds.lower[j],
                _ if j < first_total => 0.0,
                _ => groups.lower[j - first_total],
            }),
            upper: DVector::from_fn(columns, |j, _| match j {
                _ if j < n => bounds.upper[j],
                _ if j < first_total => f64::INFINITY,
                _ => groups.upper[j - first_total],
            }),
        }
    }

    fn columns(&self) -> usize {
        self.a.ncols()
    }

    fn solve(&self, cost: &DVector<f64>) -> Result<lp::Outcome, CutupError> {
        lp::solve(cost, &self.a, &self.rhs, &self.lower, &self.upper)
    }
}

/// `bounds` resolved for the assets `labels` or, without them, none: long-short.
fn resolve_bounds(
    bounds: Option<&WeightBounds>,
    labels: &[String],
) -> Result<qp::Bounds, CutupError> {
    match bounds {
        Some(bounds) => bounds.resolve(labels),
        None => Ok(unbounded(labels.len())),
    }
}

/// No bounds on `n` weights.
fn unbounded(n: usize) -> qp::Bounds {
    qp::Bounds::new(
        DVector::from_element(n, f64::NEG_INFINITY),
        DVector::from_element(n, f64::INFINITY),
    )
}

/// The least turnover from `current` into `bounds`, or [`CutupError::InfeasibleTurnover`] if
/// it's above `cap` by more than summing it rounds by.
fn check_cap(current: &DVector<f64>, bounds: &qp::Bounds, cap: f64) -> Result<f64, CutupError> {
    let minimum = minimum_turnover(current, bounds)?;
    if minimum > cap + qp::sum_rounding(current.len(), 1.0) {
        return Err(CutupError::InfeasibleTurnover { minimum, cap });
    }
    Ok(minimum)
}

/// The least turnover from `current` to a portfolio summing to 1 within `bounds`: move each
/// weight into its bounds, then make up what that left over or short of 1 on assets with room.
/// Every unit of that last step is turned over once, wherever it goes. With groups, which
/// that can break, from the linear program.
fn minimum_turnover(current: &DVector<f64>, bounds: &qp::Bounds) -> Result<f64, CutupError> {
    let n = current.len();
    if bounds.rows.a.ncols() != 0 {
        let budget = (
            DMatrix::from_element(n, 1, 1.0),
            DVector::from_element(1, 1.0),
        );
        return least_turnover(current, &budget.0, &budget.1, bounds);
    }
    let clipped = DVector::from_fn(n, |i, _| current[i].clamp(bounds.lower[i], bounds.upper[i]));
    Ok((&clipped - current).abs().sum() + (1.0 - clipped.sum()).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Allocation, MeanVariance, MinVariance, Returns, WeightBounds};
    use nalgebra::dmatrix;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

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
                field: "current_weights",
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
                field: "transaction_costs",
                expected: 3,
                actual: 2
            })
        );
        for bad in [-0.001, f64::INFINITY] {
            assert_eq!(
                from()
                    .transaction_costs_per_asset([0.001, bad, 0.0])
                    .allocate(&allocator),
                Err(CutupError::InvalidAssetValue {
                    field: "transaction_costs",
                    column: 1,
                    asset: "1".into(),
                    value: bad
                })
            );
        }
    }

    fn net_return(weights: &Weights, current: [f64; 3], cost: f64) -> f64 {
        let mu = [0.0024, 0.0006, 0.0012];
        let gross: f64 = weights.iter().zip(mu).map(|(w, m)| w * m).sum();
        gross - cost * turnover(weights, current)
    }

    #[test]
    fn a_target_return_with_costs_is_a_floor_on_the_net_return() {
        // Expected returns 0.0024, 0.0006 and 0.0012; the current weights return 0.00126.
        // At a cost of 0.0004, moving a unit from the second asset to the first nets 0.001,
        // and from the third 0.0004, so all in the first nets the most: 0.00176.
        let allocator = allocator();
        let current = [0.2, 0.3, 0.5];
        let method = |target| {
            MeanVariance::target_return(target)
                .current_weights(current)
                .transaction_costs(0.0004)
        };
        let weights = method(0.0015).allocate(&allocator).unwrap();
        assert!((net_return(&weights, current, 0.0004) - 0.0015).abs() < 1e-15);
        // Within rounding of the highest net return, the target is at it.
        let weights = method(0.00176 + 1e-17).allocate(&allocator).unwrap();
        assert!((weights[0] - 1.0).abs() < 1e-12, "{weights}");
        let Err(CutupError::TargetReturnOutOfRange {
            target_return,
            lowest,
            highest,
        }) = method(0.0018).allocate(&allocator)
        else {
            panic!("expected TargetReturnOutOfRange");
        };
        assert_eq!((target_return, lowest), (0.0018, f64::NEG_INFINITY));
        assert!((highest - 0.00176).abs() < 1e-15, "{highest}");
        // Costs of 0 change nothing: the target is exact, as without them, even below the
        // least variance portfolio's return, where a floor wouldn't bind.
        let free = MeanVariance::target_return(0.0007).current_weights(current);
        let zero = free.clone().transaction_costs(0.0).allocate(&allocator);
        assert_eq!(zero, free.turnover_penalty(0.0).allocate(&allocator));
        assert!((net_return(&zero.unwrap(), current, 0.0) - 0.0007).abs() < 1e-15);
        // Below the least variance portfolio's net return, the target doesn't bind.
        assert_eq!(
            method(-1.0).allocate(&allocator),
            MinVariance::default().allocate(&allocator)
        );
        // A cap below the turnover into the bounds, 0.2, fails as such, before the target.
        let capped = method(0.0018)
            .max_turnover(0.1)
            .bounds(WeightBounds::uniform(0.0, 0.4))
            .allocate(&allocator);
        assert!(
            matches!(capped, Err(CutupError::InfeasibleTurnover { minimum, cap: 0.1 })
                if (minimum - 0.2).abs() < 1e-15),
            "{capped:?}"
        );
    }

    #[test]
    fn a_target_return_with_a_penalty_stays_closer_to_the_current_weights() {
        let allocator = allocator();
        let current = [0.2, 0.3, 0.5];
        let method = MeanVariance::target_return(0.0018).current_weights(current);
        let free = method
            .clone()
            .turnover_penalty(0.0)
            .allocate(&allocator)
            .unwrap();
        assert_eq!(
            Ok(&free),
            MeanVariance::target_return(0.0018)
                .allocate(&allocator)
                .as_ref()
        );
        let penalized = method.turnover_penalty(1e-3).allocate(&allocator).unwrap();
        // Still exactly at the target.
        assert!((net_return(&penalized, current, 0.0) - 0.0018).abs() < 1e-15);
        assert!(turnover(&penalized, current) < turnover(&free, current) - 0.01);
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
        let uncapped = method.allocate(&allocator);
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
        let bounds = qp::Bounds::new(DVector::from_element(3, 0.1), DVector::from_element(3, 0.5));
        // Moving into the bounds sells 0.2 of the first asset and buys 0.1 of the third, 0.3 in
        // all, and leaves the weights summing to 0.9, so 0.1 more is bought.
        let current = DVector::from_vec(vec![0.7, 0.3, 0.0]);
        assert!((minimum_turnover(&current, &bounds).unwrap() - 0.4).abs() < 1e-15);
        let least = least_turnover(
            &current,
            &DMatrix::from_element(3, 1, 1.0),
            &DVector::from_element(1, 1.0),
            &bounds,
        )
        .unwrap();
        assert!((least - 0.4).abs() < 1e-15, "{least}");
    }

    #[test]
    fn net_terms_are_the_returns_and_costs_in_magnitude() {
        // |1·1| + |−2·0| and 0.5·|1 − 0.5| + 0.25·|0 − 0.5|.
        let terms = net_terms(
            &DVector::from_vec(vec![1.0, -2.0]),
            &DVector::from_vec(vec![0.5, 0.25]),
            &DVector::from_vec(vec![0.5, 0.5]),
            &DVector::from_vec(vec![1.0, 0.0]),
        );
        assert_eq!(terms, 1.375);
    }

    #[test]
    fn a_cap_at_the_least_turnover_into_the_bounds_counts_up_to_rounding() {
        // From all in the first asset to at most half in each: sell 0.5, buy 0.5.
        let current = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let bounds = qp::Bounds::new(DVector::zeros(3), DVector::from_element(3, 0.5));
        let rounding = qp::sum_rounding(3, 1.0);
        assert_eq!(check_cap(&current, &bounds, 1.0 - rounding), Ok(1.0));
        assert_eq!(
            check_cap(&current, &bounds, 1.0 - 2.0 * rounding),
            Err(CutupError::InfeasibleTurnover {
                minimum: 1.0,
                cap: 1.0 - 2.0 * rounding
            })
        );
    }

    #[test]
    fn a_net_target_up_to_rounding_past_the_highest_is_at_it() {
        // All in the first asset nets the most, 0.00176, with terms 0.0024 + 0.0004·1.6. The
        // sample means, as the method computes them, so the boundary is exactly its own.
        let current = DVector::from_vec(vec![0.2, 0.3, 0.5]);
        let mu = allocator().expected_returns(None).unwrap();
        let costs = DVector::from_element(3, 0.0004);
        let bounds = qp::Bounds::new(DVector::zeros(3), DVector::from_element(3, 1.0));
        let (highest, rounding) =
            highest_net_return(&current, (&mu, &costs), &bounds, f64::INFINITY).unwrap();
        assert!((highest - 0.00176).abs() < 1e-15, "{highest}");
        assert!((rounding - 1e-12 * 0.00304).abs() < 1e-24, "{rounding}");
        let method = |target| {
            MeanVariance::target_return(target)
                .current_weights([0.2, 0.3, 0.5])
                .transaction_costs(0.0004)
        };
        assert!(method(highest + rounding).allocate(&allocator()).is_ok());
        assert!(matches!(
            method(highest + 2.0 * rounding).allocate(&allocator()),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
    }
}
