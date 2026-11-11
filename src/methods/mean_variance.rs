//! [`MeanVariance`]: Markowitz mean-variance optimization.

use super::min_variance::{bounded_min_variance, min_variance};
use super::{correlation, in_null_space, long_only_weights, normalized};
use crate::bounds::{ReturnRange, interior_start};
use crate::turnover::{Costs, Trading};
use crate::{
    Allocation, CovarianceConfig, CutupError, PerAsset, Returns, WeightBounds, Weights, qp,
    turnover,
};
use nalgebra::{DMatrix, DVector};

/// Markowitz mean-variance optimization, with the budget constraint `1ᵀw = 1`. Long-only by
/// default.
///
/// Start from an objective. Unlike the other methods it has no default: a sensible risk
/// aversion depends on the period of the returns, and for per-period returns, whose variances
/// are small, it's large.
///
/// * [`MeanVariance::risk_aversion`] `γ`: maximize `μᵀw − (γ/2) wᵀΣw`.
/// * [`MeanVariance::target_return`] `ρ`: minimize `wᵀΣw` subject to `μᵀw = ρ`.
/// * [`MeanVariance::target_volatility`] `σ`: maximize `μᵀw` subject to `wᵀΣw ≤ σ²`.
///
/// The expected returns `μ` are the sample mean of the returns unless you set
/// [`MeanVariance::expected_returns`], for example from [`BlackLitterman`](crate::BlackLitterman).
///
/// # Long-only
///
/// By default `w ≥ 0`, and with [`MeanVariance::bounds`] each weight is between its bounds,
/// `lᵢ ≤ wᵢ ≤ uᵢ`. Solved exactly by the same active-set method as
/// [`MinVariance`](crate::MinVariance). When Σ is invertible and the long-short portfolio has no
/// short positions, the two agree. Unlike it, a risk aversion always has an optimum, since no
/// long-only portfolio can gain expected return without limit.
///
/// Only targets between the lowest and highest expected return of any portfolio within the
/// bounds can be reached. Those come from filling the bounds in order of expected return,
/// lowest or highest first; with the default bounds, they're the lowest and highest expected
/// return of any asset. A target within rounding of either end counts as at it.
///
/// When Σ is singular, several long-only portfolios can share the optimum, and which one is
/// returned depends on the solver's path. Add shrinkage or regularization for a unique answer.
///
/// # Long-short
///
/// With [`MeanVariance::allow_short`] there are no bounds. When Σ is invertible, with
/// `S = Σ⁻¹`, `A = 1ᵀS1`, `B = 1ᵀSμ`, `C = μᵀSμ` and `D = AC − B²`, the solutions are
/// `w = (1/γ) S(μ − η1)` with `η = (B − γ) / A`, and `w = [(C − Bρ) S1 + (Aρ − B) Sμ] / D`
/// (Merton, 1972). Both are computed through the KKT system instead, so a singular Σ gives an
/// exact optimum (as for [`MinVariance`](crate::MinVariance)) or an error when there is none.
///
/// # Rebalancing
///
/// From [current weights](MeanVariance::current_weights) `w₀`, with a risk aversion it
/// maximizes `μᵀw − (γ/2) wᵀΣw − κ‖w − w₀‖² − Σcᵢ|wᵢ − w₀ᵢ|`: the expected return net of
/// [transaction costs](MeanVariance::transaction_costs) `c` and a
/// [turnover penalty](MeanVariance::turnover_penalty) `κ`, optionally with the turnover
/// capped by [`MeanVariance::max_turnover`]. Costs and the penalty are in the units of the
/// returns.
///
/// With a target return, it minimizes `wᵀΣw + κ‖w − w₀‖²` at that return, optionally with
/// the turnover capped. With transaction costs, any of them above 0, the target is a floor on
/// the net return instead: `μᵀw − Σcᵢ|wᵢ − w₀ᵢ| ≥ ρ`. If the least variance portfolio, within
/// the bounds and the cap, already nets `ρ`, that's the answer; otherwise the net return is
/// exactly `ρ`. The net return is concave, so this is convex, and it's solved by a search on
/// the target's multiplier, each step the same kinked problem as with a risk aversion. Costs
/// of 0 charge nothing, and change nothing.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the risk aversion is not finite and positive, the
///   target return is not finite, the target volatility is not finite and nonnegative, or the
///   covariance settings are invalid.
/// * [`CutupError::InvalidAssetValue`] if an expected return is not finite, or an asset's
///   lower bound is negative or not finite, or its upper bound is below its lower bound or NaN.
///   For [uniform](WeightBounds::uniform) bounds, it's [`CutupError::InvalidConfig`].
/// * [`CutupError::DimensionMismatch`] if per-asset bounds or the expected returns don't have
///   one entry per asset.
/// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if a setting names an
///   asset that isn't in the data, or names one twice.
/// * [`CutupError::InfeasibleBounds`] if the lower bounds sum to more than 1 or the upper
///   bounds to less, by more than `1e-12`.
/// * [`CutupError::TargetReturnOutOfRange`], long-only, for a target return below the lowest
///   or above the highest reachable within the bounds, by more than `1e-12` of `Σ|μᵢwᵢ|` for
///   the portfolio that reaches that end. With transaction costs, for a target above the
///   highest net return within the bounds and the cap, by more than `1e-12` of
///   `Σ|μᵢwᵢ| + Σcᵢ|wᵢ − w₀ᵢ|` there, with `lowest` −∞, since every target up to it is
///   reachable.
/// * [`CutupError::DegenerateExpectedReturns`], long-short, for a target return when every
///   asset has the same expected return and the target differs from it.
/// * [`CutupError::TargetVolatilityOutOfRange`] for a target volatility below the minimum
///   variance portfolio's, by more than `1e-12` of it.
/// * [`CutupError::Unbounded`], long-short, for a risk aversion or a target volatility when Σ
///   is singular and some fully invested portfolio gains expected return without adding
///   variance, or for a target return with transaction costs when one gains net return that
///   way: the least variance is reached along a whole line of portfolios.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::NonFiniteCovariance`] if the adjusted covariance, or the covariance times
///   the risk aversion, isn't finite.
/// * [`CutupError::SingularCovariance`] if no finite weights can be computed, or the
///   covariance is too ill-conditioned for the weights to meet their constraints.
/// * [`CutupError::NotConverged`] if the long-only solver stops early, which only extreme
///   inputs should cause.
///
/// With rebalancing settings, also:
///
/// * [`CutupError::MissingCurrentWeights`] if one is set without current weights.
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9`, or
///   the cap, penalty or a uniform cost is negative or not finite.
/// * [`CutupError::InvalidAssetValue`] if a current weight isn't finite, or a per-asset cost
///   is negative or not finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight, or per-asset cost,
///   per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds, and to the target return if there is one without
///   transaction costs.
/// * [`CutupError::NotSupported`] for any of them with a target volatility.
///
/// # Example
///
/// ```
/// use cutup::{MeanVariance, Returns, WeightBounds};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
/// ];
/// let data = Returns::new(&returns)?;
/// let expected = [0.004, 0.006, 0.002];
///
/// // With little aversion to risk, the long-short portfolio leverages up on the highest
/// // expected returns and shorts the lowest.
/// let method = MeanVariance::risk_aversion(1.0).expected_returns(expected);
/// assert!(data.allocate(&method.clone().allow_short())?.iter().any(|&w| w < 0.0));
/// assert!(data.allocate(&method)?.iter().all(|&w| w >= 0.0));
///
/// // With at most 40% in any asset, the highest reachable return is
/// // 0.4 · 0.006 + 0.4 · 0.004 + 0.2 · 0.002 = 0.0044.
/// let target = MeanVariance::target_return(0.005).expected_returns(expected);
/// assert!(data.allocate(&target.clone().bounds(WeightBounds::uniform(0.0, 0.4))).is_err());
///
/// let weights = data.allocate(&target.bounds(WeightBounds::uniform(0.1, 0.6)))?;
/// let portfolio_return: f64 = weights.iter().zip(expected).map(|(w, m)| w * m).sum();
/// assert!((portfolio_return - 0.005).abs() < 1e-12);
///
/// // The highest expected return at a volatility of 1% a period.
/// let risk = MeanVariance::target_volatility(0.01).expected_returns(expected);
/// let weights = data.allocate(&risk)?;
/// assert!(data.stats(&weights)?.volatility() <= 0.01 + 1e-12);
///
/// // Rebalance net of 10 basis points a trade, turning over at most 20%.
/// let rebalance = MeanVariance::risk_aversion(100.0)
///     .current_weights([0.4, 0.3, 0.3])
///     .transaction_costs(0.001)
///     .max_turnover(0.2);
///
/// // Or hold the target return, turning over at most 20%.
/// let capped = MeanVariance::target_return(0.005)
///     .expected_returns(expected)
///     .current_weights([0.4, 0.3, 0.3])
///     .max_turnover(0.2);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MeanVariance {
    objective: Objective,
    expected_returns: Option<PerAsset>,
    covariance: CovarianceConfig,
    /// `None` when short positions are allowed.
    bounds: Option<WeightBounds>,
    trading: Trading,
}

impl MeanVariance {
    /// Maximizes `μᵀw − (γ/2) wᵀΣw` with risk aversion `gamma`. Larger values favor lower
    /// variance; as `γ → ∞` the result approaches the minimum variance portfolio.
    pub fn risk_aversion(gamma: f64) -> Self {
        Self::new(Objective::RiskAversion(gamma))
    }

    /// Minimizes variance subject to an expected portfolio return of `target`. With
    /// [transaction costs](MeanVariance::transaction_costs), `target` is a floor on the return
    /// net of them; see [Rebalancing](MeanVariance#rebalancing).
    pub fn target_return(target: f64) -> Self {
        Self::new(Objective::TargetReturn(target))
    }

    /// Maximizes the expected return subject to a volatility of at most `sigma`: the
    /// portfolio on the efficient frontier at that volatility, as PyPortfolioOpt's
    /// `efficient_risk`. `sigma` is a standard deviation, in the units and period of the
    /// returns.
    ///
    /// A `sigma` below the minimum variance portfolio's volatility, by more than `1e-12` of it,
    /// is [`CutupError::TargetVolatilityOutOfRange`]. Long-only, a `sigma` at or above the
    /// volatility of the highest-return portfolio within the bounds gives that portfolio: the
    /// limit doesn't bind. Long-short, the expected return grows without limit with the
    /// volatility, so it always binds, unless every asset has the same expected return, when
    /// the minimum variance portfolio is as good as any.
    ///
    /// Long-short, the solution is exact: every frontier portfolio is the minimum variance
    /// portfolio `w₀` plus a multiple of the direction `z` that adds expected return with the
    /// least variance, and `z` is uncorrelated with `w₀`, so the variance at `w₀ + tz` is
    /// `σ₀² + t²zᵀΣz`. Long-only, the variance on the frontier rises with the target return
    /// from the minimum variance portfolio's to the highest, so the target return is found by
    /// bisection until the two ends hold the same assets at their bounds. Between them the
    /// weights move in a straight line and the variance is a quadratic, solved exactly.
    ///
    /// Rebalancing settings aren't supported with a target volatility yet, and return
    /// [`CutupError::NotSupported`].
    pub fn target_volatility(sigma: f64) -> Self {
        Self::new(Objective::TargetVolatility(sigma))
    }

    fn new(objective: Objective) -> Self {
        Self {
            objective,
            expected_returns: None,
            covariance: CovarianceConfig::default(),
            bounds: Some(WeightBounds::default()),
            trading: Trading::default(),
        }
    }

    /// Uses `expected_returns`, [one per asset](PerAsset) in column order or by asset, instead
    /// of the sample mean of the returns. They're in the same units and period as the returns.
    pub fn expected_returns(mut self, expected_returns: impl Into<PerAsset>) -> Self {
        self.expected_returns = Some(expected_returns.into());
        self
    }

    covariance_setters!();
    turnover_setters!();

    /// Subtracts `κ‖w − w₀‖²` from the objective, pulling the weights toward the
    /// [current weights](MeanVariance::current_weights) `w₀`. `κ` is in the units of the
    /// returns. With a target return, `κ‖w − w₀‖²` is added to the variance instead, so `κ`
    /// is in the units of the variance there.
    pub fn turnover_penalty(mut self, kappa: f64) -> Self {
        self.trading.penalty = Some(kappa);
        self
    }

    /// Charges `cost` per unit of weight traded, `Σ cost·|wᵢ − w₀ᵢ|`, against the expected
    /// return: 0.001 is 10 basis points. Needs [current weights](MeanVariance::current_weights).
    /// With a target return, the target is on the return net of them.
    pub fn transaction_costs(mut self, cost: f64) -> Self {
        self.trading.costs = Some(Costs::Uniform(cost));
        self
    }

    /// Charges `costs[i]` per unit of asset `i`'s weight traded, [one per asset](PerAsset) in
    /// column order or by asset. See [`MeanVariance::transaction_costs`].
    pub fn transaction_costs_per_asset(mut self, costs: impl Into<PerAsset>) -> Self {
        self.trading.costs = Some(Costs::PerAsset(costs.into()));
        self
    }

    /// Keeps each weight within `bounds`, and the portfolio long-only. Replaces
    /// [`MeanVariance::allow_short`].
    pub fn bounds(mut self, bounds: WeightBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Allows short positions, with no bounds on the weights. Replaces
    /// [`MeanVariance::bounds`].
    pub fn allow_short(mut self) -> Self {
        self.bounds = None;
        self
    }
}

impl Allocation for MeanVariance {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let expected_returns = data.resolve("expected_returns", self.expected_returns.as_ref())?;
        if let (Objective::TargetVolatility(_), Some(setting)) =
            (self.objective, self.trading.first_setting())
        {
            return Err(CutupError::NotSupported {
                setting,
                with: "a target volatility",
            });
        }
        if self.trading.first_setting().is_some() {
            check_objective(self.objective)?;
            let covariance = || data.adjusted_covariance(&self.covariance);
            return match self.objective {
                Objective::RiskAversion(gamma) => {
                    let mu = data.expected_returns(expected_returns.as_ref())?;
                    let cov = covariance()? * gamma;
                    turnover::rebalance(data, cov, mu, self.bounds.as_ref(), &self.trading, 2.0)
                }
                Objective::TargetVolatility(_) => {
                    unreachable!("a target volatility with trading settings is rejected above")
                }
                // With costs the target is a net return, and at least it; otherwise the
                // expected return, exactly.
                Objective::TargetReturn(target) => {
                    let mu = data.expected_returns(expected_returns.as_ref())?;
                    let cap = self.trading.max_turnover.unwrap_or(f64::INFINITY);
                    let bounds = self.bounds.as_ref();
                    if self.trading.costs.is_some() {
                        turnover::rebalance_to_net_target(
                            data,
                            covariance,
                            mu,
                            target,
                            bounds,
                            &self.trading,
                            cap,
                        )
                    } else {
                        turnover::rebalance_to_target(
                            data,
                            covariance,
                            mu,
                            target,
                            bounds,
                            &self.trading,
                            cap,
                        )
                    }
                }
            };
        }
        self.trading.check(data.labels())?;
        let mu = expected_returns.as_ref();
        match &self.bounds {
            Some(bounds) => {
                data.bounded_mean_variance(self.objective, mu, &self.covariance, bounds)
            }
            None => data.mean_variance(self.objective, mu, &self.covariance),
        }
    }
}

impl Returns {
    /// The mean-variance portfolio with short positions allowed, for [`MeanVariance`].
    fn mean_variance(
        &self,
        objective: Objective,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        check_objective(objective)?;
        let n = self.cov_matrix.ncols();
        let mu = self.expected_returns(expected_returns)?;

        let cov = self.adjusted_covariance(covariance)?;

        let weights = match objective {
            Objective::RiskAversion(gamma) => {
                qp::solve_kkt(
                    &(cov * gamma),
                    &mu,
                    &DMatrix::from_element(n, 1, 1.0),
                    &DVector::from_element(1, 1.0),
                )?
                .ok_or(CutupError::Unbounded)?
                .0
            }
            Objective::TargetReturn(target) => target_return(&cov, &mu, target)?,
            Objective::TargetVolatility(sigma) => target_volatility(&cov, &mu, sigma)?,
        };
        Ok(self.weights(weights.iter().copied().collect()))
    }

    /// The mean-variance portfolio within `bounds`, for [`MeanVariance`].
    fn bounded_mean_variance(
        &self,
        objective: Objective,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        check_objective(objective)?;
        let mu = self.expected_returns(expected_returns)?;
        let bounds = bounds.resolve(&self.labels)?;
        let budget = |n: usize| {
            (
                DMatrix::from_element(n, 1, 1.0),
                DVector::from_element(1, 1.0),
            )
        };
        let target = match objective {
            Objective::RiskAversion(gamma) => {
                let cov = self.adjusted_covariance(covariance)?;
                let (e, b) = budget(cov.ncols());
                return long_only_weights(
                    &(cov * gamma),
                    &mu,
                    &e,
                    &b,
                    &bounds,
                    interior_start(&bounds),
                )
                .map(|w| self.weights(w));
            }
            Objective::TargetReturn(target) => target,
            Objective::TargetVolatility(sigma) => {
                let cov = self.adjusted_covariance(covariance)?;
                return bounded_target_volatility(&cov, &mu, &bounds, sigma)
                    .map(|w| self.weights(w));
            }
        };
        // An unreachable target is rejected before the covariance is built.
        let problem = ReturnRange::target_problem(&mu, bounds, target)?;
        let cov = self.adjusted_covariance(covariance)?;
        long_only_weights(
            &cov,
            &DVector::zeros(cov.ncols()),
            &problem.e,
            &problem.b,
            &problem.bounds,
            problem.start,
        )
        .map(|w| self.weights(w))
    }
}

impl Returns {
    /// The efficient frontier: `points` portfolios with the least variance for their
    /// expected return, at returns evenly spaced from the minimum variance portfolio's to the
    /// highest reachable, as PyPortfolioOpt's `efficient_return` at each.
    ///
    /// It uses `method`'s expected returns, covariance adjustments and bounds, and ignores its
    /// objective. The first point is the minimum variance portfolio, as
    /// [`MinVariance`](crate::MinVariance) with the same covariance and bounds finds it, and
    /// each after it is [`MeanVariance::target_return`] at its return:
    ///
    /// * Long-only, or within bounds, the last is the portfolio with the highest expected
    ///   return within the bounds.
    /// * Long-short, the expected return grows without limit, so the frontier stops at the
    ///   highest expected return of any one asset. If the minimum variance portfolio's return
    ///   is already at or above it, the frontier is that one point.
    ///
    /// When the range of returns is within `1e-12` of their size, as when every asset has the
    /// same expected return, the frontier is the single minimum variance portfolio.
    ///
    /// Each point's [expected return](FrontierPoint::expected_return) and
    /// [volatility](FrontierPoint::volatility) are under the method's expected returns and its
    /// adjusted covariance, which can differ from [`Returns::stats`]' sample estimates.
    ///
    /// # Errors
    ///
    /// * [`CutupError::InvalidConfig`] if `points` is less than 2.
    /// * [`CutupError::NotSupported`] if `method` has rebalancing settings.
    /// * [`CutupError::NonFiniteWeight`] if a point's weights aren't finite, as for
    ///   [`Returns::allocate`].
    /// * Otherwise, the errors [`MeanVariance`] returns for its settings, other than for its
    ///   objective.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MeanVariance, MinVariance, Returns};
    ///
    /// let returns = [
    ///      [ 0.010, -0.020,  0.005],
    ///      [-0.004,  0.015,  0.002],
    ///      [ 0.007,  0.003, -0.006],
    ///      [ 0.002, -0.001,  0.004],
    ///      [-0.003,  0.006,  0.001],
    /// ];
    /// let data = Returns::new(&returns)?;
    /// let expected = [0.004, 0.006, 0.002];
    ///
    /// // The objective is ignored; the expected returns, covariance and bounds are used.
    /// let method = MeanVariance::risk_aversion(1.0).expected_returns(expected);
    /// let frontier = data.frontier(&method, 5)?;
    /// assert_eq!(frontier.len(), 5);
    ///
    /// // From the minimum variance portfolio to all in the asset with the highest return.
    /// assert_eq!(frontier[0].weights(), &data.allocate(&MinVariance::default())?);
    /// assert!((frontier[4].expected_return() - 0.006).abs() < 1e-12);
    ///
    /// // Each step up in return costs more volatility.
    /// for pair in frontier.windows(2) {
    ///     assert!(pair[0].expected_return() < pair[1].expected_return());
    ///     assert!(pair[0].volatility() < pair[1].volatility());
    /// }
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn frontier(
        &self,
        method: &MeanVariance,
        points: usize,
    ) -> Result<Vec<FrontierPoint>, CutupError> {
        if points < 2 {
            return Err(CutupError::InvalidConfig {
                field: "points",
                value: points as f64,
            });
        }
        let expected_returns =
            self.resolve("expected_returns", method.expected_returns.as_ref())?;
        if let Some(setting) = method.trading.first_setting() {
            return Err(CutupError::NotSupported {
                setting,
                with: "a frontier",
            });
        }
        method.trading.check(self.labels())?;
        let mu = self.expected_returns(expected_returns.as_ref())?;
        let bounds = match &method.bounds {
            Some(bounds) => Some(bounds.resolve(&self.labels)?),
            None => None,
        };
        let cov = self.adjusted_covariance(&method.covariance)?;
        // Before any point is solved, so a covariance it can't measure fails first.
        let risk = Risk::new(&cov)?;
        let (minimum, highest) = match &bounds {
            Some(bounds) => (
                bounded_min_variance(&cov, bounds)?,
                ReturnRange::new(&mu, bounds).highest(),
            ),
            None => (min_variance(&cov)?, mu.max()),
        };
        let lowest = mu.dot(&minimum);
        let highest = highest.max(lowest);
        let mut portfolios = vec![minimum];
        if highest - lowest > 1e-12 * lowest.abs().max(highest.abs()) {
            for k in 1..points {
                // A weighted average, which can't overflow where the difference could.
                let t = k as f64 / (points - 1) as f64;
                let target = lowest * (1.0 - t) + highest * t;
                portfolios.push(match &bounds {
                    Some(bounds) => bounded_target_return(&cov, &mu, bounds, target)?,
                    None => target_return(&cov, &mu, target)?,
                });
            }
        }
        portfolios
            .into_iter()
            .map(|w| {
                // As `allocate` checks every method's weights.
                if let Some(column) = w.iter().position(|w| !w.is_finite()) {
                    return Err(CutupError::NonFiniteWeight {
                        column,
                        asset: self.labels[column].clone(),
                    });
                }
                Ok(FrontierPoint {
                    expected_return: mu.dot(&w),
                    volatility: risk.volatility(&w) * risk.scale,
                    weights: self.weights(w.iter().copied().collect()),
                })
            })
            .collect()
    }
}

/// A portfolio on the efficient frontier, from [`Returns::frontier`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FrontierPoint {
    expected_return: f64,
    volatility: f64,
    weights: Weights,
}

impl FrontierPoint {
    /// The portfolio's expected return, `μᵀw`, under the method's expected returns.
    pub fn expected_return(&self) -> f64 {
        self.expected_return
    }

    /// The portfolio's volatility, `sqrt(wᵀΣw)`, under the method's adjusted covariance.
    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    /// The portfolio's weights.
    pub fn weights(&self) -> &Weights {
        &self.weights
    }
}

/// What a mean-variance optimization trades off, for [`MeanVariance`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum Objective {
    /// Maximize `μᵀw − (γ/2) wᵀΣw` for a risk aversion `γ > 0`.
    RiskAversion(f64),
    /// Minimize `wᵀΣw` subject to an expected portfolio return `μᵀw` equal to this value.
    TargetReturn(f64),
    /// Maximize `μᵀw` subject to a volatility `sqrt(wᵀΣw)` of at most this value.
    TargetVolatility(f64),
}

/// Checks a mean-variance objective's parameter.
fn check_objective(objective: Objective) -> Result<(), CutupError> {
    match objective {
        Objective::RiskAversion(gamma) if !(gamma.is_finite() && gamma > 0.0) => {
            Err(CutupError::InvalidConfig {
                field: "risk_aversion",
                value: gamma,
            })
        }
        Objective::TargetReturn(target) if !target.is_finite() => Err(CutupError::InvalidConfig {
            field: "target_return",
            value: target,
        }),
        Objective::TargetVolatility(sigma) if !(sigma.is_finite() && sigma >= 0.0) => {
            Err(CutupError::InvalidConfig {
                field: "target_volatility",
                value: sigma,
            })
        }
        _ => Ok(()),
    }
}

/// Portfolio volatilities under a covariance, measured in correlation units scaled by the
/// largest volatility, so that neither the volatilities nor their squares overflow or
/// underflow on the way.
struct Risk {
    correlation: DMatrix<f64>,
    volatilities: DVector<f64>,
    /// The largest volatility, which [`Risk::volatilities`] are divided by.
    scale: f64,
}

impl Risk {
    fn new(cov: &DMatrix<f64>) -> Result<Self, CutupError> {
        let (correlation, volatilities) = correlation(cov)?;
        let scale = volatilities.max();
        Ok(Risk {
            correlation,
            volatilities: volatilities / scale,
            scale,
        })
    }

    /// `w` in scaled correlation units, `diag(σ)w / scale`.
    fn units(&self, w: &DVector<f64>) -> DVector<f64> {
        w.component_mul(&self.volatilities)
    }

    /// `xᵀCy` for `x` and `y` in scaled correlation units.
    fn product(&self, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
        x.dot(&(&self.correlation * y))
    }

    /// The volatility of `w`, divided by [`Risk::scale`]. A variance that's NaN, from weights
    /// so large that it overflows both ways, stays NaN rather than reading as 0.
    fn volatility(&self, w: &DVector<f64>) -> f64 {
        let u = self.units(w);
        self.product(&u, &u).clamp(0.0, f64::INFINITY).sqrt()
    }
}

/// The minimum variance portfolio's volatility, divided by `risk`'s scale, checked against the
/// target `sigma`, also divided by it: a target below it by more than `1e-12` of it is out of
/// range, and one just below it is moved onto it.
fn volatility_floor(risk: &Risk, w: &DVector<f64>, sigma: f64) -> Result<f64, CutupError> {
    let floor = risk.volatility(w);
    let target = sigma / risk.scale;
    if target < floor * (1.0 - 1e-12) {
        return Err(CutupError::TargetVolatilityOutOfRange {
            target_volatility: sigma,
            minimum: floor * risk.scale,
        });
    }
    Ok(target.max(floor))
}

/// The long-short portfolio with the least variance at expected return `target`.
fn target_return(
    cov: &DMatrix<f64>,
    mu: &DVector<f64>,
    target: f64,
) -> Result<DVector<f64>, CutupError> {
    let n = cov.ncols();
    Ok(qp::solve_kkt(
        cov,
        &DVector::zeros(n),
        &DMatrix::from_columns(&[DVector::from_element(n, 1.0), mu.clone()]),
        &DVector::from_vec(vec![1.0, target]),
    )?
    .ok_or(CutupError::DegenerateExpectedReturns)?
    .0)
}

/// The portfolio within `bounds` with the least variance at expected return `target`.
fn bounded_target_return(
    cov: &DMatrix<f64>,
    mu: &DVector<f64>,
    bounds: &qp::Bounds,
    target: f64,
) -> Result<DVector<f64>, CutupError> {
    let problem = ReturnRange::target_problem(mu, bounds.clone(), target)?;
    Ok(DVector::from_vec(long_only_weights(
        cov,
        &DVector::zeros(cov.ncols()),
        &problem.e,
        &problem.b,
        &problem.bounds,
        problem.start,
    )?))
}

/// The long-short portfolio with the highest expected return at a volatility of at most
/// `sigma`, for [`MeanVariance::target_volatility`].
fn target_volatility(
    cov: &DMatrix<f64>,
    mu: &DVector<f64>,
    sigma: f64,
) -> Result<DVector<f64>, CutupError> {
    let n = mu.len();
    let minimum = min_variance(cov)?;
    let risk = Risk::new(cov)?;
    let target = volatility_floor(&risk, &minimum, sigma)?;
    // The direction z adding one unit of expected return with the least variance. Σw₀ is a
    // multiple of 1 and 1ᵀz = 0, so z is uncorrelated with w₀. There's none when every
    // expected return is the same: then w₀ is as good as any portfolio.
    let Some((direction, _)) = qp::solve_kkt(
        cov,
        &DVector::zeros(n),
        &DMatrix::from_columns(&[DVector::from_element(n, 1.0), mu.clone()]),
        &DVector::from_vec(vec![0.0, 1.0]),
    )?
    else {
        return Ok(minimum);
    };
    // A direction with no variance gains expected return for free, without limit.
    let units = direction.component_mul(&risk.volatilities);
    if in_null_space(&risk.correlation, &units)? {
        return Err(CutupError::Unbounded);
    }
    let floor = risk.volatility(&minimum);
    // Two roots, so that a target far above the floor can't overflow the product.
    let step = (target - floor).sqrt() * (target + floor).sqrt() / risk.volatility(&direction);
    Ok(minimum + direction * step)
}

/// Which bound each weight is held at, if any: the active set of a long-only solution.
///
/// A weight held at a bound comes back within rounding of it, not on it: the solver clamps it
/// in scaled units, and the weights are then divided by their sum, which is 1 within `n` ulps.
/// So a weight within `4nε` of a bound's size counts as at it. Compared exactly, rounding hid
/// every lower bound of a solution on wasm, the bisection below stopped between two that held
/// different assets, and the step along the line between them was off by 1e-3.
fn held(w: &DVector<f64>, bounds: &qp::Bounds) -> Vec<Option<bool>> {
    let n = w.len();
    (0..n)
        .map(|i| {
            let (lower, upper) = (bounds.lower[i], bounds.upper[i]);
            if w[i] <= lower + qp::sum_rounding(n, lower.abs()) {
                Some(false)
            } else if w[i] >= upper - qp::sum_rounding(n, upper.abs()) {
                Some(true)
            } else {
                None
            }
        })
        .collect()
}

/// The long-only portfolio within `bounds` with the highest expected return at a volatility of
/// at most `sigma`, for [`MeanVariance::target_volatility`].
fn bounded_target_volatility(
    cov: &DMatrix<f64>,
    mu: &DVector<f64>,
    bounds: &qp::Bounds,
    sigma: f64,
) -> Result<Vec<f64>, CutupError> {
    let minimum = bounded_min_variance(cov, bounds)?;
    let risk = Risk::new(cov)?;
    let target = volatility_floor(&risk, &minimum, sigma)?;
    // At the floor, the target's variance is the minimum's own. The floor squared misses it by
    // the rounding of the square root, and the step from the minimum, the square root of that,
    // moved the weights by 2e-8.
    let units = risk.units(&minimum);
    let target_variance = if target > risk.volatility(&minimum) {
        target * target
    } else {
        risk.product(&units, &units)
    };
    let at = |rho: f64| bounded_target_return(cov, mu, bounds, rho);
    // The variance on the frontier is convex in the target return, and no higher at the
    // minimum variance portfolio's return than the target's, so it's within the target from
    // there up to some return and above it after.
    let highest = ReturnRange::new(mu, bounds).highest();
    let top = at(highest)?;
    if risk.volatility(&top) <= target {
        return Ok(normalized(top));
    }
    let (mut low, mut high) = ((mu.dot(&minimum), minimum), (highest, top));
    for _ in 0..200 {
        let width = high.0 - low.0;
        if held(&low.1, bounds) == held(&high.1, bounds)
            || width <= 4.0 * f64::EPSILON * low.0.abs().max(high.0.abs())
        {
            break;
        }
        let middle = low.0 + width / 2.0;
        let w = at(middle)?;
        if risk.volatility(&w) <= target {
            low = (middle, w);
        } else {
            high = (middle, w);
        }
    }
    // With the same assets held, the weights move in a straight line from low to high, and the
    // variance along it, at low + t·d, is at² + 2bt + c. The target's is at the larger root.
    let d = &high.1 - &low.1;
    let (u, v) = (risk.units(&low.1), risk.units(&d));
    let (a, b, c) = (
        risk.product(&v, &v),
        risk.product(&u, &v),
        risk.product(&u, &u),
    );
    // Low is within the target, but for rounding.
    let gap = (target_variance - c).max(0.0);
    #[allow(
        clippy::manual_clamp,
        reason = "f64::max drops a NaN from 0 / 0, which clamp keeps"
    )]
    let t = larger_root(a, b, gap).max(0.0).min(1.0);
    Ok(normalized(low.1 + d * t))
}

/// The larger root `t` of `at² + 2bt = gap`, for `a ≥ 0` and `gap ≥ 0`, in whichever form
/// doesn't subtract nearly equal numbers.
fn larger_root(a: f64, b: f64, gap: f64) -> f64 {
    let root = (b * b + a * gap).sqrt();
    if b >= 0.0 {
        gap / (b + root)
    } else {
        (root - b) / a
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinVariance;
    use crate::test_support::{
        assert_weights_close, factor_returns, full_rank_prices, portfolio_variance,
        rank_one_returns, sample_mean,
    };
    use nalgebra::{dmatrix, dvector};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    #[test]
    fn test_mean_variance_target_return() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let mu = sample_mean(&allocator);
        let target = mu.mean() * 1.5;
        let weights = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!((mu.dot(&w) - target).abs() < 1e-12);
    }

    #[test]
    fn test_mean_variance_target_at_min_variance_return_is_min_variance() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let min_variance = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        let mu = sample_mean(&allocator);
        let target = mu.dot(&DVector::from_column_slice(min_variance.as_slice()));

        let weights = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();
        assert_weights_close(&weights, &min_variance, 1e-9);
    }

    #[test]
    fn test_mean_variance_risk_aversion_first_order_condition() {
        // At the optimum of μᵀw − (γ/2) wᵀΣw subject to 1ᵀw = 1, μ − γΣw is a multiple of 1.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let gamma = 3.0;
        let weights = allocator
            .allocate(&MeanVariance::risk_aversion(gamma).allow_short())
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        let gradient = sample_mean(&allocator) - &allocator.cov_matrix * &w * gamma;
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!(gradient.max() - gradient.min() < 1e-12);
    }

    #[test]
    fn test_mean_variance_high_risk_aversion_approaches_min_variance() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let weights = allocator
            .allocate(&MeanVariance::risk_aversion(1e9).allow_short())
            .unwrap();
        assert_weights_close(
            &weights,
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-6,
        );
    }

    #[test]
    fn test_mean_variance_equal_expected_returns() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let flat = DVector::from_element(3, 0.01);

        // With nothing to choose between on return, risk aversion reduces to minimum variance.
        let mean_variance = MeanVariance::risk_aversion(2.0).expected_returns(flat.as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance.allow_short()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );

        // And no portfolio can target any other return.
        let mean_variance = MeanVariance::target_return(0.02).expected_returns(flat.as_slice());
        assert_eq!(
            allocator.allocate(&mean_variance.allow_short()).err(),
            Some(CutupError::DegenerateExpectedReturns)
        );
    }

    #[test]
    fn test_mean_variance_defaults_to_sample_mean() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let default = MeanVariance::risk_aversion(5.0);
        let explicit = default
            .clone()
            .expected_returns(sample_mean(&allocator).as_slice());
        assert_eq!(
            allocator.allocate(&default.allow_short()).unwrap(),
            allocator.allocate(&explicit.allow_short()).unwrap()
        );
    }

    #[test]
    fn test_mean_variance_applies_covariance_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let plain = MeanVariance::risk_aversion(5.0);
        let shrunk = plain
            .clone()
            .covariance(CovarianceConfig::default().shrinkage(0.5));
        assert_ne!(
            allocator.allocate(&plain.clone().allow_short()).unwrap(),
            allocator.allocate(&shrunk.allow_short()).unwrap()
        );

        let invalid = plain.covariance(CovarianceConfig::default().shrinkage(2.0));
        assert_eq!(
            allocator.allocate(&invalid.allow_short()).err(),
            Some(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 2.0
            })
        );
    }

    #[test]
    fn test_mean_variance_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        for gamma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            match allocator.allocate(&MeanVariance::risk_aversion(gamma).allow_short()) {
                Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "risk_aversion"),
                other => panic!("gamma {gamma} gave {other:?}"),
            }
        }
        match allocator.allocate(&MeanVariance::target_return(f64::NAN).allow_short()) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "target_return"),
            other => panic!("NaN target gave {other:?}"),
        }

        let mean_variance = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_vec(vec![0.01, f64::NAN, 0.02]).as_slice());
        match allocator.allocate(&mean_variance.allow_short()) {
            Err(CutupError::InvalidAssetValue { field, column, .. }) => {
                assert_eq!((field, column), ("expected_returns", 1))
            }
            other => panic!("NaN expected return gave {other:?}"),
        }

        let mean_variance = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_vec(vec![0.01, 0.02]).as_slice());
        assert_eq!(
            allocator.allocate(&mean_variance.allow_short()).err(),
            Some(CutupError::DimensionMismatch {
                field: "expected_returns",
                expected: 3,
                actual: 2
            })
        );
    }

    #[test]
    fn test_mean_variance_rejects_zero_variance() {
        let allocator =
            Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            allocator
                .allocate(&MeanVariance::risk_aversion(1.0).allow_short())
                .err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
    }

    #[test]
    fn test_long_only_mean_variance_matches_unconstrained_when_long() {
        // A diagonal covariance and close expected returns keep every unconstrained weight
        // positive, so the constraint is inactive.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let expected = dvector![0.001, 0.002, 0.003];
        for mean_variance in [
            MeanVariance::risk_aversion(100.0),
            MeanVariance::target_return(0.0015),
        ] {
            let mean_variance = mean_variance.expected_returns(expected.as_slice());
            let unconstrained = allocator
                .allocate(&mean_variance.clone().allow_short())
                .unwrap();
            assert!(unconstrained.iter().all(|&w| w > 0.0), "{unconstrained:?}");
            assert_weights_close(
                &allocator.allocate(&mean_variance).unwrap(),
                &unconstrained,
                1e-12,
            );
        }
    }

    #[test]
    fn test_long_only_mean_variance_at_the_extremes() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];

        // With almost no aversion to risk, everything goes to the highest expected return.
        let mean_variance = MeanVariance::risk_aversion(1e-9).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // Only the asset with the highest expected return reaches it.
        let mean_variance =
            MeanVariance::target_return(0.003).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_mean_variance_hits_the_target() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];
        for target in [0.001, 0.0012, 0.0025, 0.0029] {
            let mean_variance =
                MeanVariance::target_return(target).expected_returns(expected.as_slice());
            let weights = allocator.allocate(&mean_variance).unwrap();
            let w = DVector::from_column_slice(weights.as_slice());
            assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
            assert!((w.sum() - 1.0).abs() < 1e-12);
            assert!(
                (w.dot(&expected) - target).abs() < 1e-12 * target,
                "{weights:?}"
            );
        }
    }

    #[test]
    fn test_long_only_mean_variance_rejects_unreachable_targets() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let mean_variance = MeanVariance::target_return(0.004)
            .expected_returns(dvector![0.001, 0.003, 0.002].as_slice());
        assert_eq!(
            allocator.allocate(&mean_variance),
            Err(CutupError::TargetReturnOutOfRange {
                target_return: 0.004,
                lowest: 0.001,
                highest: 0.003
            })
        );

        // With equal expected returns, only that return can be targeted, by any portfolio.
        let flat = dvector![0.002, 0.002, 0.002];
        let mean_variance = MeanVariance::target_return(0.001).expected_returns(flat.as_slice());
        assert!(matches!(
            allocator.allocate(&mean_variance),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        let mean_variance = MeanVariance::target_return(0.002).expected_returns(flat.as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance).unwrap(),
            &allocator.allocate(&MinVariance::default()).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_mean_variance_target_edge_cases() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();

        // A target a rounding error above the highest expected return is that return.
        let expected = dvector![0.001, 0.003, 0.002];
        let mean_variance = MeanVariance::target_return(0.003 * (1.0 + 4.0 * f64::EPSILON))
            .expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // highest − lowest overflows here, which mustn't spoil the starting point.
        let expected = dvector![-1e308, 1e308, 0.0];
        let mean_variance =
            MeanVariance::target_return(5e307).expected_returns(expected.as_slice());
        let weights = allocator.allocate(&mean_variance).unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
        assert!(
            (w.dot(&expected) - 5e307).abs() < 1e-12 * 5e307,
            "{weights:?}"
        );

        // An unreachable target is reported before the covariance is checked.
        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let mean_variance = MeanVariance::target_return(1.0);
        assert!(matches!(
            allocator.allocate(&mean_variance),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
    }

    #[test]
    fn test_long_only_mean_variance_bounded_on_singular_covariance() {
        // Unconstrained, utility grows without limit (see the next test). Long-only, the
        // weights can't grow, so there is always an optimum.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let mean_variance = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator.allocate(&mean_variance.clone().allow_short()),
            Err(CutupError::Unbounded)
        );
        let weights = allocator.allocate(&mean_variance).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_long_only_mean_variance_rejects_bad_input() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        assert_eq!(
            allocator.allocate(&MeanVariance::risk_aversion(0.0)),
            Err(CutupError::InvalidConfig {
                field: "risk_aversion",
                value: 0.0
            })
        );
        assert_eq!(
            allocator.allocate(&MeanVariance::target_return(f64::INFINITY)),
            Err(CutupError::InvalidConfig {
                field: "target_return",
                value: f64::INFINITY
            })
        );
        let mean_variance =
            MeanVariance::risk_aversion(1.0).expected_returns(dvector![0.01].as_slice());
        assert_eq!(
            allocator.allocate(&mean_variance),
            Err(CutupError::DimensionMismatch {
                field: "expected_returns",
                expected: 3,
                actual: 1
            })
        );
    }

    #[test]
    fn test_mean_variance_unbounded_on_singular_covariance() {
        // With Σ singular, d = [5, -7, 2] has Σd = 0 and 1ᵀd = 0, but μᵀd ≠ 0 for the sample
        // mean, so utility grows without limit along d.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let mean_variance = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator
                .allocate(&mean_variance.clone().allow_short())
                .err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = mean_variance.covariance(CovarianceConfig::default().shrinkage(0.1));
        assert!(allocator.allocate(&shrunk.allow_short()).is_ok());
    }

    #[test]
    fn test_mean_variance_target_on_singular_covariance() {
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let target = 0.004;
        let weights = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();

        let w = DVector::from_column_slice(weights.as_slice());
        assert!((w.sum() - 1.0).abs() < 1e-12);
        assert!((sample_mean(&allocator).dot(&w) - target).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_long_short_target_is_degenerate_on_one_asset_at_any_scale() {
        // One asset can only be held in full, so only its own return can be targeted. At these
        // scales the KKT check used to overflow and return a weight of about 8.9e172. Found by
        // the `allocate` fuzz target.
        let data = Returns::new_flat(&[-5e122, 2.4064e126], 1).unwrap();
        assert_eq!(
            data.allocate(&MeanVariance::target_return(1.065197824640524e299).allow_short()),
            Err(CutupError::DegenerateExpectedReturns)
        );
    }

    /// Six assets over three periods, from the `allocate` fuzz target. Asset 3 is constant, so
    /// a tiny regularization leaves its variance vastly smaller than the others', near 1e24.
    fn near_zero_variance_data() -> Returns {
        let values: Vec<f64> = [
            5119, 247, -17, -1, -1, -1, -256, -1, 8, -1, -65, -1, -32747, 0, -1, -1, -1, 3838,
        ]
        .iter()
        .map(|&v: &i32| f64::from(v) * 1e8)
        .collect();
        Returns::new_flat(&values, 6).unwrap()
    }

    #[test]
    fn test_long_short_risk_aversion_with_a_near_zero_variance_asset() {
        // At a regularization of 3.9e-308, asset 3's variance is some 1e332 times smaller than
        // the others'. The weights summed to −7.6e75, and then, once that was checked, the
        // solve returned SingularCovariance. Its null-space basis had the entry for asset 3,
        // near 1e-62, wrong in most of its digits, and converting back multiplied that by
        // about 1e50. The answer is the near-riskless asset 3 at every regularization the
        // pseudo-inverse's cutoff ignores. Found by the `allocate` fuzz target.
        let data = near_zero_variance_data();
        for (gamma, epsilon) in [
            (2.574958403566059e207, 3.909151256095177e-308),
            (1.0, 1e-4),
            (1.0, 1e-300),
        ] {
            let mean_variance = MeanVariance::risk_aversion(gamma)
                .allow_short()
                .regularization(epsilon);
            let weights = data.allocate(&mean_variance).unwrap();
            for (i, w) in weights.iter().enumerate() {
                let expected = if i == 3 { 1.0 } else { 0.0 };
                assert!((w - expected).abs() < 1e-9, "{weights:?}");
            }
        }
    }

    #[test]
    fn test_long_short_rebalance_with_a_near_zero_variance_asset() {
        // The same problem rebalanced, with costs or a cap. The optimum without them is the
        // start, and the current weights are, where that optimum can't be computed.
        let mean_variance = MeanVariance::risk_aversion(2.574958403566059e207)
            .allow_short()
            .regularization(3.909151256095177e-308)
            .current_weights(vec![1.0 / 6.0; 6]);
        for rebalance in [
            mean_variance.clone().transaction_costs(1e-3),
            mean_variance.max_turnover(10.0),
        ] {
            let weights = near_zero_variance_data().allocate(&rebalance).unwrap();
            let sum: f64 = weights.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "{weights:?}");
        }
    }

    #[test]
    fn test_rebalance_to_target_with_a_near_zero_variance_asset() {
        // Rebalancing to a target return checked only the budget, and that only after dividing
        // by the weights' sum. Long-short, the target was missed through the same inaccurate
        // null-space basis; it's now met. Long-only, the budget and the target are parallel to
        // rounding once rescaled, so the target constraint was dropped and the weights earned
        // −1e8. That can't be solved in these units, so it's SingularCovariance.
        let data = near_zero_variance_data();
        let mu = data.expected_returns(None).unwrap();
        let rebalance = |target: f64, epsilon: f64| {
            MeanVariance::target_return(target)
                .regularization(epsilon)
                .current_weights(vec![1.0 / 6.0; 6])
                .max_turnover(10.0)
        };
        for target in [0.0, 1e9] {
            let weights = data
                .allocate(&rebalance(target, 1e-4).allow_short())
                .unwrap();
            let earned: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
            let gross: f64 = weights
                .iter()
                .zip(mu.iter())
                .map(|(w, m)| (w * m).abs())
                .sum();
            assert!(
                (earned - target).abs() <= 1e-9 * gross,
                "{weights:?} earns {earned}"
            );
            assert!(matches!(
                data.allocate(&rebalance(target, 1e-100)),
                Err(CutupError::SingularCovariance)
            ));
        }
    }

    #[test]
    fn test_mean_variance_target_equal_to_common_return() {
        // Every fully invested portfolio earns the common return, so the target is met by the
        // minimum variance portfolio.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let mean_variance = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_element(3, 0.01).as_slice());
        assert_weights_close(
            &allocator.allocate(&mean_variance.allow_short()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );
    }

    fn volatility(data: &Returns, weights: &Weights) -> f64 {
        portfolio_variance(data, weights).sqrt()
    }

    fn return_of(data: &Returns, weights: &Weights) -> f64 {
        sample_mean(data).dot(&DVector::from_column_slice(weights.as_slice()))
    }

    #[test]
    fn test_target_volatility_long_short_is_mertons_frontier() {
        // With S = Σ⁻¹, A = 1ᵀS1, B = 1ᵀSμ, C = μᵀSμ and D = AC − B², the frontier's variance
        // at return ρ is (Aρ² − 2Bρ + C) / D, so volatility σ is at ρ = B/A + sqrt(D(Aσ² − 1))/A.
        let data = Returns::returns_matrix(factor_returns(5, 40, 7)).unwrap();
        let mu = sample_mean(&data);
        let s = data.cov_matrix.clone().try_inverse().unwrap();
        let ones = DVector::from_element(5, 1.0);
        let (a, b, c) = (
            ones.dot(&(&s * &ones)),
            ones.dot(&(&s * &mu)),
            mu.dot(&(&s * &mu)),
        );
        let d = a * c - b * b;
        let floor = (1.0 / a).sqrt();
        for sigma in [floor, 1.5 * floor, 3.0 * floor] {
            let rho = b / a + (d * (a * sigma * sigma - 1.0)).max(0.0).sqrt() / a;
            let merton = ((c - b * rho) * &s * &ones + (a * rho - b) * &s * &mu) / d;
            let weights = data
                .allocate(&MeanVariance::target_volatility(sigma).allow_short())
                .unwrap();
            assert_weights_close(&weights, &Weights::from(merton.as_slice().to_vec()), 1e-9);
            let got = volatility(&data, &weights);
            assert!((got - sigma).abs() <= 1e-12 * sigma, "{got} vs {sigma}");
        }
    }

    #[test]
    fn test_target_volatility_at_a_singular_floor_has_its_highest_return() {
        // The first two assets are the same, so every split of their weight has the least
        // variance. At the floor, or within rounding below it, the one with the most return
        // holds all of it in the one with the higher mean, whichever that is. Its bisection runs
        // along that flat face, so it's found to about √ε.
        let mut returns = factor_returns(3, 40, 5);
        let first = returns.column(0).into_owned();
        returns.set_column(1, &first);
        let data = Returns::returns_matrix(returns).unwrap();
        let minimum = data.allocate(&MinVariance::default()).unwrap();
        let floor = volatility(&data, &minimum);
        let pair = minimum[0] + minimum[1];
        for (mu, higher) in [([0.002, 0.001, 0.0015], 0), ([0.001, 0.002, 0.0015], 1)] {
            for sigma in [floor, floor * (1.0 - 1e-13)] {
                let weights = data
                    .allocate(&MeanVariance::target_volatility(sigma).expected_returns(mu))
                    .unwrap();
                assert!((weights[higher] - pair).abs() < 1e-7, "{weights:?}");
                assert!((volatility(&data, &weights) - floor).abs() <= 1e-12 * floor);
            }
        }
    }

    #[test]
    fn test_target_volatility_long_only_is_on_the_frontier() {
        let data = Returns::returns_matrix(factor_returns(6, 40, 11)).unwrap();
        let mu = sample_mean(&data);
        let minimum = data.allocate(&MinVariance::default()).unwrap();
        let floor = volatility(&data, &minimum);
        let top = data
            .allocate(&MeanVariance::target_return(mu.max()))
            .unwrap();
        let ceiling = volatility(&data, &top);
        // At the floor, or within rounding below it, the minimum itself. Squaring the floor
        // back misses its variance by an ulp, and stepping by the square root of that moved
        // the weights by 2e-8.
        for sigma in [floor, floor * (1.0 - 1e-13)] {
            assert_weights_close(
                &data
                    .allocate(&MeanVariance::target_volatility(sigma))
                    .unwrap(),
                &minimum,
                1e-12,
            );
        }
        for fraction in [0.05, 0.3, 0.6, 0.95] {
            let sigma = floor + fraction * (ceiling - floor);
            let weights = data
                .allocate(&MeanVariance::target_volatility(sigma))
                .unwrap();
            assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
            let got = volatility(&data, &weights);
            assert!((got - sigma).abs() <= 1e-12 * sigma, "{got} vs {sigma}");
            // The least variance at its return, and a slightly higher return needs more.
            let rho = return_of(&data, &weights);
            let frontier = data.allocate(&MeanVariance::target_return(rho)).unwrap();
            assert_weights_close(&weights, &frontier, 1e-9);
            let above = data
                .allocate(&MeanVariance::target_return(rho + 1e-6 * (mu.max() - rho)))
                .unwrap();
            assert!(volatility(&data, &above) > sigma);
        }
        // From the top portfolio's volatility up, the limit doesn't bind.
        for sigma in [ceiling, 2.0 * ceiling] {
            let weights = data
                .allocate(&MeanVariance::target_volatility(sigma))
                .unwrap();
            assert_weights_close(&weights, &top, 1e-12);
        }
    }

    #[test]
    fn test_target_volatility_within_bounds() {
        let data = Returns::returns_matrix(factor_returns(6, 40, 11)).unwrap();
        let bounds = WeightBounds::uniform(0.05, 0.3);
        let floor = volatility(
            &data,
            &data
                .allocate(&MinVariance::default().bounds(bounds.clone()))
                .unwrap(),
        );
        let highest =
            ReturnRange::new(&sample_mean(&data), &bounds.resolve(data.labels()).unwrap())
                .highest();
        let top = data
            .allocate(&MeanVariance::target_return(highest).bounds(bounds.clone()))
            .unwrap();
        let sigma = (floor + volatility(&data, &top)) / 2.0;
        let weights = data
            .allocate(&MeanVariance::target_volatility(sigma).bounds(bounds.clone()))
            .unwrap();
        // To the rounding of rescaling them to sum to 1.
        let within = |w: f64| (0.05 - 1e-15..=0.3 + 1e-15).contains(&w);
        assert!(weights.iter().all(|&w| within(w)), "{weights:?}");
        let got = volatility(&data, &weights);
        assert!((got - sigma).abs() <= 1e-12 * sigma, "{got} vs {sigma}");
        let rho = return_of(&data, &weights);
        let frontier = data
            .allocate(&MeanVariance::target_return(rho).bounds(bounds))
            .unwrap();
        assert_weights_close(&weights, &frontier, 1e-9);
    }

    #[test]
    fn test_target_volatility_below_the_minimum_is_out_of_range() {
        let data = Returns::returns_matrix(factor_returns(4, 30, 3)).unwrap();
        for method in [
            MeanVariance::target_volatility(0.0),
            MeanVariance::target_volatility(0.0).allow_short(),
        ] {
            let minimum = match method.bounds {
                Some(_) => MinVariance::default(),
                None => MinVariance::default().allow_short(),
            };
            let floor = volatility(&data, &data.allocate(&minimum).unwrap());
            let at = |sigma: f64| {
                let mut method = method.clone();
                method.objective = Objective::TargetVolatility(sigma);
                data.allocate(&method)
            };
            for sigma in [0.0, 0.99 * floor, floor * (1.0 - 2e-12)] {
                match at(sigma) {
                    Err(CutupError::TargetVolatilityOutOfRange {
                        target_volatility,
                        minimum,
                    }) => {
                        assert_eq!(target_volatility, sigma);
                        assert!(
                            (minimum - floor).abs() <= 1e-12 * floor,
                            "{minimum} vs {floor}"
                        );
                    }
                    other => panic!("{sigma}: {other:?}"),
                }
            }
            // Within rounding of the minimum, the minimum variance portfolio.
            let weights = at(floor * (1.0 - 5e-13)).unwrap();
            assert_weights_close(&weights, &data.allocate(&minimum).unwrap(), 1e-9);
        }
    }

    #[test]
    fn test_target_volatility_rejects_bad_settings() {
        let data = Returns::returns_matrix(factor_returns(3, 20, 5)).unwrap();
        for sigma in [-0.01, f64::NAN, f64::INFINITY] {
            for method in [
                MeanVariance::target_volatility(sigma),
                MeanVariance::target_volatility(sigma).allow_short(),
            ] {
                assert!(matches!(
                    data.allocate(&method),
                    Err(CutupError::InvalidConfig {
                        field: "target_volatility",
                        ..
                    })
                ));
            }
        }
        let method = || MeanVariance::target_volatility(0.02).current_weights([0.4, 0.3, 0.3]);
        for (method, setting) in [
            (method().max_turnover(0.2), "max_turnover"),
            (method().turnover_penalty(0.1), "turnover_penalty"),
            (method().transaction_costs(0.001), "transaction_costs"),
        ] {
            assert_eq!(
                data.allocate(&method),
                Err(CutupError::NotSupported {
                    setting,
                    with: "a target volatility"
                })
            );
        }
    }

    #[test]
    fn test_target_volatility_with_equal_or_free_returns() {
        // Every portfolio earns the common return, so the minimum variance one is as good as
        // any, at any volatility above its own.
        let data = Returns::prices_matrix(full_rank_prices()).unwrap();
        let flat = MeanVariance::target_volatility(1.0).expected_returns([0.01; 3]);
        assert_weights_close(
            &data.allocate(&flat.clone().allow_short()).unwrap(),
            &data
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );
        assert_weights_close(
            &data.allocate(&flat).unwrap(),
            &data.allocate(&MinVariance::default()).unwrap(),
            1e-9,
        );
        // With Σ of rank 1, some change to the weights adds expected return with no variance.
        let rank_one = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert_eq!(
            rank_one.allocate(&MeanVariance::target_volatility(1.0).allow_short()),
            Err(CutupError::Unbounded)
        );
    }

    #[test]
    fn test_target_volatility_far_above_the_returns() {
        let data = Returns::returns_matrix(factor_returns(4, 30, 17)).unwrap();
        let mu = sample_mean(&data);
        // Long-only, the limit doesn't bind, however large.
        let top = data
            .allocate(&MeanVariance::target_return(mu.max()))
            .unwrap();
        assert_weights_close(
            &data
                .allocate(&MeanVariance::target_volatility(f64::MAX))
                .unwrap(),
            &top,
            1e-12,
        );
        // Long-short, the weights grow with it, and the volatility is still σ.
        for sigma in [1e100, 1e150] {
            let weights = data
                .allocate(&MeanVariance::target_volatility(sigma).allow_short())
                .unwrap();
            let got = volatility(&data, &weights);
            assert!((got - sigma).abs() <= 1e-9 * sigma, "{got} vs {sigma}");
        }
    }

    #[test]
    fn test_target_volatility_helpers() {
        // A floor of exactly 1, so a target exactly 1e-12 below it is the last one allowed.
        let risk = Risk::new(&DMatrix::identity(2, 2)).unwrap();
        let w = DVector::from_vec(vec![1.0, 0.0]);
        let edge = 1.0 * (1.0 - 1e-12);
        assert_eq!(volatility_floor(&risk, &w, edge), Ok(1.0));
        assert!(matches!(
            volatility_floor(&risk, &w, edge - 1e-16),
            Err(CutupError::TargetVolatilityOutOfRange { .. })
        ));

        let bounds = qp::Bounds {
            lower: DVector::from_vec(vec![0.1, 0.1, 0.1]),
            upper: DVector::from_vec(vec![0.5, 0.5, 0.5]),
        };
        let w = DVector::from_vec(vec![0.1, 0.4, 0.5]);
        assert_eq!(held(&w, &bounds), vec![Some(false), None, Some(true)]);
        // Within 4nε of a bound's size, and an ulp past that.
        let (lower, upper) = (
            0.1 + qp::sum_rounding(3, 0.1),
            0.5 - qp::sum_rounding(3, 0.5),
        );
        let w = DVector::from_vec(vec![lower, 0.4, upper]);
        assert_eq!(held(&w, &bounds), vec![Some(false), None, Some(true)]);
        let past =
            |x: f64, up: bool| f64::from_bits(if up { x.to_bits() + 1 } else { x.to_bits() - 1 });
        let w = DVector::from_vec(vec![past(lower, true), 0.4, past(upper, false)]);
        assert_eq!(held(&w, &bounds), vec![None, None, None]);

        // at² + 2bt = gap: t = 7 for a = 1, b = −3, gap = 7. With b far larger than the rest,
        // (root − b) / a would cancel to 0, but t is about gap / 2b.
        assert_eq!(larger_root(1.0, -3.0, 7.0), 7.0);
        assert_eq!(larger_root(1.0, 1.0, 3.0), 1.0);
        let t = larger_root(1.0, 1e8, 1.0);
        assert!((t - 5e-9).abs() <= 1e-12 * 5e-9, "{t}");
    }

    #[test]
    fn test_risk_keeps_a_nan_volatility() {
        let correlated = DMatrix::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.9 });
        let risk = Risk::new(&correlated).unwrap();
        // No variance is no volatility.
        assert_eq!(risk.volatility(&DVector::zeros(3)), 0.0);
        // Overflowing to inf + inf − inf is NaN, not 0.
        let huge = DVector::from_vec(vec![1e308, 1e308, -1e308]);
        assert!(risk.volatility(&huge).is_nan());
    }

    #[test]
    fn test_target_volatility_scales_with_the_returns() {
        let returns = factor_returns(5, 40, 13);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let floor = volatility(&data, &data.allocate(&MinVariance::default()).unwrap());
        for exponent in [-500, 500] {
            let scale = 2f64.powi(exponent);
            let scaled = Returns::returns_matrix(&returns * scale).unwrap();
            for method in [
                MeanVariance::target_volatility(1.3 * floor),
                MeanVariance::target_volatility(1.3 * floor).allow_short(),
            ] {
                let mut at_scale = method.clone();
                at_scale.objective = Objective::TargetVolatility(1.3 * floor * scale);
                assert_weights_close(
                    &scaled.allocate(&at_scale).unwrap(),
                    &data.allocate(&method).unwrap(),
                    1e-12,
                );
            }
        }
    }

    /// Checks `frontier` against `method`'s minimum variance and target return portfolios, and
    /// each point's return and volatility against its weights.
    fn assert_frontier(data: &Returns, method: &MeanVariance, frontier: &[FrontierPoint]) {
        let minimum = match &method.bounds {
            Some(bounds) => MinVariance::default().bounds(bounds.clone()),
            None => MinVariance::default().allow_short(),
        };
        assert_eq!(
            frontier[0].weights(),
            &data
                .allocate(&minimum.covariance(method.covariance.clone()))
                .unwrap()
        );
        // Under the method's expected returns and covariance.
        let given = data
            .resolve("expected_returns", method.expected_returns.as_ref())
            .unwrap();
        let mu = data.expected_returns(given.as_ref()).unwrap();
        let cov = data.adjusted_covariance(&method.covariance).unwrap();
        for point in frontier {
            let w = DVector::from_column_slice(point.weights().as_slice());
            assert_eq!(point.expected_return(), mu.dot(&w));
            let sigma = w.dot(&(&cov * &w)).sqrt();
            assert!((point.volatility() - sigma).abs() <= 1e-12 * sigma);
        }
        for point in &frontier[1..] {
            let mut target = method.clone();
            target.objective = Objective::TargetReturn(point.expected_return());
            assert_weights_close(point.weights(), &data.allocate(&target).unwrap(), 1e-9);
        }
    }

    #[test]
    fn test_frontier_long_only_runs_from_min_variance_to_the_highest_return() {
        let data = Returns::returns_matrix(factor_returns(6, 40, 11)).unwrap();
        let mu = sample_mean(&data);
        let method = MeanVariance::risk_aversion(1.0);
        let frontier = data.frontier(&method, 6).unwrap();
        assert_eq!(frontier.len(), 6);
        assert_frontier(&data, &method, &frontier);
        assert_eq!(frontier[5].expected_return(), mu.max());
        // Evenly spaced in return, rising in volatility.
        let lowest = frontier[0].expected_return();
        for (k, pair) in frontier.windows(2).enumerate() {
            let step = (mu.max() - lowest) / 5.0;
            let want = lowest + step * (k + 1) as f64;
            assert!((pair[1].expected_return() - want).abs() <= 1e-12 * mu.amax());
            assert!(pair[0].volatility() < pair[1].volatility());
            assert!(pair[1].weights().iter().all(|&w| w >= 0.0));
        }
        // Two points are its ends.
        let ends = data.frontier(&method, 2).unwrap();
        assert_eq!(ends[0], frontier[0]);
        assert_eq!(ends[1].expected_return(), mu.max());
    }

    #[test]
    fn test_frontier_long_short_stops_at_the_highest_asset_return() {
        let data = Returns::returns_matrix(factor_returns(6, 40, 11)).unwrap();
        let mu = sample_mean(&data);
        let method = MeanVariance::target_return(0.0)
            .allow_short()
            .shrinkage(0.2);
        let frontier = data.frontier(&method, 4).unwrap();
        assert_eq!(frontier.len(), 4);
        assert_frontier(&data, &method, &frontier);
        // Under the shrunk covariance, which the sample volatility doesn't use.
        let sample = data.frontier(&MeanVariance::target_return(0.0).allow_short(), 4);
        assert_ne!(sample.unwrap()[0], frontier[0]);
        assert!((frontier[3].expected_return() - mu.max()).abs() <= 4.0 * f64::EPSILON * mu.amax());
        assert!(frontier[3].weights().iter().any(|&w| w < 0.0));
        for pair in frontier.windows(2) {
            assert!(pair[0].volatility() < pair[1].volatility());
        }
    }

    #[test]
    fn test_frontier_within_bounds() {
        let data = Returns::returns_matrix(factor_returns(6, 40, 11)).unwrap();
        let bounds = WeightBounds::uniform(0.05, 0.3);
        let mu = sample_mean(&data);
        let method = MeanVariance::risk_aversion(1.0).bounds(bounds.clone());
        let frontier = data.frontier(&method, 5).unwrap();
        assert_frontier(&data, &method, &frontier);
        let highest = ReturnRange::new(&mu, &bounds.resolve(data.labels()).unwrap()).highest();
        assert!((frontier[4].expected_return() - highest).abs() <= 4.0 * f64::EPSILON * mu.amax());
        for point in &frontier {
            assert!(
                point
                    .weights()
                    .iter()
                    .all(|&w| (0.05 - 1e-15..=0.3 + 1e-15).contains(&w))
            );
        }
    }

    #[test]
    fn test_frontier_of_one_return_is_one_point() {
        let data = Returns::returns_matrix(factor_returns(4, 30, 5)).unwrap();
        // Every expected return the same, or all 0, so there's nothing to trade volatility for.
        for expected in [[0.01; 4], [0.0; 4]] {
            for method in [
                MeanVariance::risk_aversion(1.0),
                MeanVariance::risk_aversion(1.0).allow_short(),
            ] {
                let method = method.expected_returns(expected);
                let frontier = data.frontier(&method, 5).unwrap();
                assert_eq!(frontier.len(), 1);
                assert_frontier(&data, &method, &frontier);
            }
        }
        // A range of 1e-9 of the returns is enough for a frontier.
        let close =
            MeanVariance::risk_aversion(1.0).expected_returns([0.01, 0.01, 0.01, 0.01 + 1e-11]);
        let frontier = data.frontier(&close, 5).unwrap();
        assert_eq!(frontier.len(), 5);
        // There the target pins the weights only to about ε·max|μ| over the range, far coarser
        // than 1e-9, so it's the returns that are checked: evenly spaced, to that resolution.
        assert_eq!(
            frontier[0].weights(),
            &data.allocate(&MinVariance::default()).unwrap()
        );
        let (lowest, highest) = (frontier[0].expected_return(), 0.01 + 1e-11);
        for (k, point) in frontier.iter().enumerate() {
            let want = lowest + (highest - lowest) * k as f64 / 4.0;
            assert!(
                (point.expected_return() - want).abs() <= 1e-15,
                "{k}: {point:?}"
            );
        }
        // Long-short, the minimum variance portfolio shorts an asset with a negative return
        // to earn more than any asset.
        let data = Returns::new(&[
            [0.010, 0.021, 0.004],
            [-0.008, -0.015, -0.002],
            [0.004, 0.009, 0.006],
            [-0.002, -0.006, -0.003],
            [0.006, 0.011, 0.001],
        ])
        .unwrap();
        let method = MeanVariance::risk_aversion(1.0)
            .expected_returns([0.0, -0.01, 0.0])
            .allow_short();
        let frontier = data.frontier(&method, 3).unwrap();
        assert_eq!(frontier.len(), 1);
        assert!(frontier[0].expected_return() > 0.0);
        // Long-only, it holds none of that asset, so earns 0, as much as any portfolio.
        let long_only = MeanVariance::risk_aversion(1.0).expected_returns([0.0, -0.01, 0.0]);
        let frontier = data.frontier(&long_only, 3).unwrap();
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].expected_return(), 0.0);
    }

    #[test]
    fn test_frontier_rejects_bad_settings() {
        let data = Returns::returns_matrix(factor_returns(4, 30, 5)).unwrap();
        let method = MeanVariance::risk_aversion(1.0);
        for points in [0, 1] {
            assert_eq!(
                data.frontier(&method, points),
                Err(CutupError::InvalidConfig {
                    field: "points",
                    value: points as f64,
                })
            );
        }
        let rebalance = method.clone().current_weights([0.25; 4]).max_turnover(0.1);
        assert_eq!(
            data.frontier(&rebalance, 3),
            Err(CutupError::NotSupported {
                setting: "max_turnover",
                with: "a frontier",
            })
        );
        // Current weights alone change nothing, but are still checked.
        let current = method.clone().current_weights([0.25; 4]);
        assert_eq!(data.frontier(&current, 3), data.frontier(&method, 3));
        let unbalanced = method.clone().current_weights([0.5; 4]);
        assert!(matches!(
            data.frontier(&unbalanced, 3),
            Err(CutupError::InvalidConfig { .. })
        ));
        // The objective is ignored, even an invalid one.
        assert_eq!(
            data.frontier(&MeanVariance::risk_aversion(f64::NAN), 3),
            data.frontier(&method, 3)
        );
        assert!(matches!(
            data.frontier(&method.clone().expected_returns([0.0; 3]), 3),
            Err(CutupError::DimensionMismatch { .. })
        ));
        let infeasible = method.bounds(WeightBounds::uniform(0.0, 0.2));
        assert!(matches!(
            data.frontier(&infeasible, 3),
            Err(CutupError::InfeasibleBounds { .. })
        ));
    }

    #[test]
    fn test_frontier_scales_with_the_returns() {
        let returns = factor_returns(5, 40, 13);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        for method in [
            MeanVariance::risk_aversion(1.0),
            MeanVariance::risk_aversion(1.0).allow_short(),
        ] {
            let frontier = data.frontier(&method, 4).unwrap();
            for exponent in [-500, 500] {
                let scale = 2f64.powi(exponent);
                let scaled = Returns::returns_matrix(&returns * scale).unwrap();
                let at_scale = scaled.frontier(&method, 4).unwrap();
                for (got, want) in at_scale.iter().zip(&frontier) {
                    assert_weights_close(got.weights(), want.weights(), 1e-12);
                    let rho = got.expected_return() / scale;
                    assert!(
                        (rho - want.expected_return()).abs()
                            <= 1e-12 * want.expected_return().abs()
                    );
                    let sigma = got.volatility() / scale;
                    assert!((sigma - want.volatility()).abs() <= 1e-12 * want.volatility());
                }
            }
        }
    }
}
