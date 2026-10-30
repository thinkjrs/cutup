//! Allocation methods as values, and the [`Allocation`] trait that runs them.

use crate::turnover::{self, Costs, Trading};
use crate::{
    CovarianceConfig, CutupError, Linkage, Objective, PortfolioAllocator, WeightBounds, Weights,
    check_objective,
};
use nalgebra::DVector;

/// An allocation method: anything that turns a [`PortfolioAllocator`]'s returns into
/// [`Weights`].
///
/// Each built-in method is a value that implements it, built from `default()` and chained
/// settings, and run with [`PortfolioAllocator::allocate`]:
///
/// ```
/// use cutup::{Hrp, Linkage, MaxSharpe, MinVariance, PortfolioAllocator, WeightBounds};
/// use nalgebra::dmatrix;
///
/// let returns = dmatrix![
///      0.010, -0.020,  0.005;
///     -0.004,  0.015,  0.002;
///      0.007,  0.003, -0.006;
///      0.002, -0.001,  0.004;
///     -0.003,  0.006,  0.001
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
///
/// let capped = allocator.allocate(&MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.5)))?;
/// let tangency = allocator.allocate(&MaxSharpe::default().expected_returns([0.004, 0.006, 0.002]))?;
/// let hrp = allocator.allocate(&Hrp::default().linkage(Linkage::Ward))?;
///
/// assert!(capped.iter().all(|&w| w <= 0.5 + 1e-12));
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// Methods can be mixed in one collection:
///
/// ```
/// use cutup::{Allocation, EqualWeight, InverseVolatility, PortfolioAllocator, RiskParity};
/// use nalgebra::dmatrix;
///
/// let allocator = PortfolioAllocator::from_returns(dmatrix![
///      0.01,  0.02;
///     -0.01, -0.02;
///      0.01, -0.02
/// ])?;
/// let methods: Vec<Box<dyn Allocation>> = vec![
///     Box::new(EqualWeight),
///     Box::new(InverseVolatility),
///     Box::new(RiskParity::default()),
/// ];
/// for method in &methods {
///     assert_eq!(allocator.allocate(method)?.len(), 2);
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// Implement it to add a method of your own, returning one weight per asset that sum to 1, as
/// every method in this crate does. Build the [`Weights`] from a `Vec<f64>`.
pub trait Allocation {
    /// Computes the weights for the returns in `data`.
    ///
    /// # Errors
    ///
    /// Whatever the method can't compute weights for. See each method's type.
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError>;
}

impl<A: Allocation + ?Sized> Allocation for &A {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        (**self).allocate(data)
    }
}

impl<A: Allocation + ?Sized> Allocation for Box<A> {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        (**self).allocate(data)
    }
}

/// Equal weights, `1/n` each. Never fails.
///
/// ```
/// use cutup::{EqualWeight, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// let prices = dmatrix![
///     100.0, 200.0, 300.0;
///     110.0, 210.0, 310.0;
///     120.0, 220.0, 320.0
/// ];
///
/// let weights = PortfolioAllocator::from_prices(prices)?.allocate(&EqualWeight)?;
/// assert!(weights.iter().all(|&w| w == 1.0 / 3.0));
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EqualWeight;

impl Allocation for EqualWeight {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let n = data.returns.ncols();
        Ok(Weights(vec![1.0 / n as f64; n]))
    }
}

/// Inverse volatility weights, `wᵢ ∝ 1 / σᵢ`.
///
/// Each asset's weight is inversely proportional to the standard deviation of its returns,
/// so riskier assets get less. Correlations are ignored. Compared with [`InverseVariance`],
/// the tilt away from risky assets is milder. When assets are uncorrelated, this gives each
/// asset the same risk contribution.
///
/// The weights are positive and sum to 1.
///
/// # Errors
///
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
///
/// # Example
///
/// ```
/// use cutup::{InverseVolatility, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// // Volatilities in the ratio 1 : 2.
/// let returns = dmatrix![
///      0.01,  0.02;
///     -0.01, -0.02;
///      0.01, -0.02;
///     -0.01,  0.02
/// ];
///
/// let allocator = PortfolioAllocator::from_returns(returns)?;
/// let weights = allocator.allocate(&InverseVolatility)?;
///
/// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
/// assert!((weights[1] - 1.0 / 3.0).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct InverseVolatility;

impl Allocation for InverseVolatility {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.weight_by_variance(|variance| 1.0 / variance.sqrt())
    }
}

/// Inverse variance weights, `wᵢ ∝ 1 / σᵢ²`.
///
/// Each asset's weight is inversely proportional to the variance of its returns.
/// Correlations are ignored. When assets are uncorrelated this is the minimum variance
/// portfolio, and it is what [`Hrp`] and [`Herc`] use within each cluster.
///
/// The weights are positive and sum to 1.
///
/// # Errors
///
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
///
/// # Example
///
/// ```
/// use cutup::{InverseVariance, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// // Volatilities in the ratio 1 : 2, so variances in the ratio 1 : 4.
/// let returns = dmatrix![
///      0.01,  0.02;
///     -0.01, -0.02;
///      0.01, -0.02;
///     -0.01,  0.02
/// ];
///
/// let weights = PortfolioAllocator::from_returns(returns)?.allocate(&InverseVariance)?;
///
/// assert!((weights[0] - 0.8).abs() < 1e-12);
/// assert!((weights[1] - 0.2).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct InverseVariance;

impl Allocation for InverseVariance {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.weight_by_variance(|variance| 1.0 / variance)
    }
}

/// The minimum variance portfolio: minimize `wᵀΣw` subject to `1ᵀw = 1`. Long-only by
/// default.
///
/// It uses only the covariance matrix, not expected returns. For a portfolio that also
/// accounts for them, use [`MeanVariance`].
///
/// # Long-only
///
/// By default `w ≥ 0`, and with [`MinVariance::bounds`] each weight is between its bounds,
/// `lᵢ ≤ wᵢ ≤ uᵢ`. Where the long-short portfolio would short an asset, this leaves it out
/// instead and reweights the rest; when the long-short portfolio has no negative weights, the
/// two agree. There is no closed form, so the weights come from an active-set solver, which
/// finds the exact optimum in a finite number of steps, holding each asset at its lower
/// bound, at its upper bound, or between them.
///
/// When Σ is singular, several long-only portfolios can share the minimum variance, and which
/// one is returned depends on the solver's path. Add [shrinkage](MinVariance::shrinkage) for a
/// unique answer.
///
/// # Long-short
///
/// With [`MinVariance::allow_short`] there are no bounds, and when Σ is invertible this is the
/// global minimum variance portfolio `w = Σ⁻¹1 / 1ᵀΣ⁻¹1`.
///
/// When Σ is singular (no more rows of returns than assets, or perfectly collinear assets),
/// many portfolios share the minimum variance and the one with the smallest norm in
/// correlation units, `diag(σ)w`, is returned, so the choice doesn't depend on each asset's
/// units. That minimum is often zero, an artifact of fitting too little data. Add shrinkage
/// for a unique, better-conditioned answer.
///
/// # Rebalancing
///
/// From [current weights](MinVariance::current_weights) `w₀`, it can cap the turnover with
/// [`MinVariance::max_turnover`], or penalize it with [`MinVariance::turnover_penalty`].
/// Transaction costs are in units of return, so they only apply to [`MeanVariance`], whose
/// objective is too. For min variance with costs, use mean-variance with zero expected returns:
/// `MeanVariance::risk_aversion(γ).expected_returns(vec![0.0; n]).transaction_costs(c)`
/// minimizes `(γ/2) wᵀΣw + Σcᵢ|wᵢ − w₀ᵢ|`, where `γ` is the rate at which you trade variance
/// for cost.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if `regularization` is negative or not finite, `shrinkage`
///   is outside `[0, 1]`, or a lower bound is negative or not finite, or an upper bound is
///   below its lower bound or NaN.
/// * [`CutupError::DimensionMismatch`] if per-asset bounds don't have one entry per asset.
/// * [`CutupError::InfeasibleBounds`] if the lower bounds sum to more than 1 or the upper
///   bounds to less, by more than `1e-12`.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::NonFiniteCovariance`] if the adjusted covariance isn't finite.
/// * [`CutupError::SingularCovariance`] if no finite weights can be computed.
/// * [`CutupError::NotConverged`] if the long-only solver stops early, which only extreme
///   inputs should cause.
///
/// With rebalancing settings, also:
///
/// * [`CutupError::MissingCurrentWeights`] if one is set without current weights.
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9` or
///   aren't finite, or the cap or penalty is negative or not finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds.
///
/// # Example
///
/// ```
/// use cutup::{MinVariance, PortfolioAllocator, WeightBounds};
/// use nalgebra::dmatrix;
///
/// // The second asset moves with the first but twice as far, so the long-short minimum
/// // variance portfolio shorts it as a hedge.
/// let returns = dmatrix![
///      0.010,  0.021,  0.004;
///     -0.008, -0.015, -0.002;
///      0.004,  0.009,  0.006;
///     -0.002, -0.006, -0.003;
///      0.006,  0.011,  0.001
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
/// assert!(allocator.allocate(&MinVariance::default().allow_short())?[1] < 0.0);
///
/// let weights = allocator.allocate(&MinVariance::default())?;
/// assert_eq!(weights[1], 0.0);
/// assert!(weights[2] > 0.9);
///
/// // Capping every asset at 50% moves the rest of the weight to the next least risky.
/// let capped = allocator.allocate(&MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.5)))?;
/// assert!((capped[2] - 0.5).abs() < 1e-12);
///
/// // Full shrinkage leaves no correlations to hedge, so every asset is held equally.
/// let shrunk = allocator.allocate(&MinVariance::default().shrinkage(1.0))?;
/// assert!(shrunk.iter().all(|w| (w - 1.0 / 3.0).abs() < 1e-12));
///
/// // Rebalance from the current weights, turning over at most 10%.
/// let rebalanced = allocator.allocate(
///     &MinVariance::default().current_weights([0.5, 0.3, 0.2]).max_turnover(0.1),
/// )?;
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MinVariance {
    covariance: CovarianceConfig,
    /// `None` when short positions are allowed.
    bounds: Option<WeightBounds>,
    trading: Trading,
}

impl Default for MinVariance {
    fn default() -> Self {
        Self {
            covariance: CovarianceConfig::default(),
            bounds: Some(WeightBounds::default()),
            trading: Trading::default(),
        }
    }
}

impl MinVariance {
    covariance_setters!();
    turnover_setters!();

    /// Adds `κ‖w − w₀‖²` to the variance minimized, pulling the weights toward the
    /// [current weights](MinVariance::current_weights) `w₀`. `κ` is in the units of variance,
    /// so a meaningful value is on the scale of the assets' variances.
    pub fn turnover_penalty(mut self, kappa: f64) -> Self {
        self.trading.penalty = Some(kappa);
        self
    }

    /// Keeps each weight within `bounds`, and the portfolio long-only. Replaces
    /// [`MinVariance::allow_short`].
    pub fn bounds(mut self, bounds: WeightBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }

    /// Allows short positions, with no bounds on the weights. Replaces
    /// [`MinVariance::bounds`].
    pub fn allow_short(mut self) -> Self {
        self.bounds = None;
        self
    }
}

impl Allocation for MinVariance {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        if self.trading.first_setting().is_some() {
            let cov = data.adjusted_covariance(&self.covariance)?;
            let n = cov.ncols();
            return turnover::rebalance(
                cov,
                DVector::zeros(n),
                self.bounds.as_ref(),
                &self.trading,
                1.0,
            );
        }
        self.trading.check(data.cov_matrix.ncols())?;
        match &self.bounds {
            Some(bounds) => data.bounded_min_variance(&self.covariance, bounds),
            None => data.min_variance(&self.covariance),
        }
    }
}

/// Markowitz mean-variance optimization, with the budget constraint `1ᵀw = 1`. Long-only by
/// default.
///
/// Start from an objective. Unlike the other methods it has no default: a sensible risk
/// aversion depends on the period of the returns, and for per-period returns, whose variances
/// are small, it's large.
///
/// * [`MeanVariance::risk_aversion`] `γ`: maximize `μᵀw − (γ/2) wᵀΣw`.
/// * [`MeanVariance::target_return`] `ρ`: minimize `wᵀΣw` subject to `μᵀw = ρ`.
///
/// The expected returns `μ` are the sample mean of the returns unless you set
/// [`MeanVariance::expected_returns`], for example from [`BlackLitterman`](crate::BlackLitterman).
///
/// # Long-only
///
/// By default `w ≥ 0`, and with [`MeanVariance::bounds`] each weight is between its bounds,
/// `lᵢ ≤ wᵢ ≤ uᵢ`. Solved exactly by the same active-set method as [`MinVariance`]. When Σ is
/// invertible and the long-short portfolio has no short positions, the two agree. Unlike it, a
/// risk aversion always has an optimum, since no long-only portfolio can gain expected return
/// without limit.
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
/// exact optimum (as for [`MinVariance`]) or an error when there is none.
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
/// With a target return, it minimizes the variance at that return with the turnover capped.
/// Only the cap works there: the objective is a variance, so neither costs nor the penalty,
/// both in the units of the returns, has anything to be traded off against, and they return
/// [`CutupError::NotSupported`].
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the risk aversion is not finite and positive, the
///   target return or an expected return is not finite, a lower bound is negative or not
///   finite, an upper bound is below its lower bound or NaN, or the covariance settings are
///   invalid.
/// * [`CutupError::DimensionMismatch`] if per-asset bounds or the expected returns don't have
///   one entry per asset.
/// * [`CutupError::InfeasibleBounds`] if the lower bounds sum to more than 1 or the upper
///   bounds to less, by more than `1e-12`.
/// * [`CutupError::TargetReturnOutOfRange`], long-only, for a target return below the lowest
///   or above the highest reachable within the bounds, by more than `1e-12` of the larger in
///   magnitude.
/// * [`CutupError::DegenerateExpectedReturns`], long-short, for a target return when every
///   asset has the same expected return and the target differs from it.
/// * [`CutupError::Unbounded`], long-short, for a risk aversion when Σ is singular and some
///   fully invested portfolio gains expected return without adding variance.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::NonFiniteCovariance`] if the adjusted covariance, or the covariance times
///   the risk aversion, isn't finite.
/// * [`CutupError::SingularCovariance`] if no finite weights can be computed.
/// * [`CutupError::NotConverged`] if the long-only solver stops early, which only extreme
///   inputs should cause.
///
/// With rebalancing settings, also:
///
/// * [`CutupError::MissingCurrentWeights`] if one is set without current weights.
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9` or
///   aren't finite, or the cap, penalty or a cost is negative or not finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight, or per-asset cost,
///   per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds, and to the target return if there is one.
/// * [`CutupError::NotSupported`] for a penalty or costs with a target return.
///
/// # Example
///
/// ```
/// use cutup::{MeanVariance, PortfolioAllocator, WeightBounds};
/// use nalgebra::dmatrix;
///
/// let returns = dmatrix![
///      0.010, -0.020,  0.005;
///     -0.004,  0.015,  0.002;
///      0.007,  0.003, -0.006;
///      0.002, -0.001,  0.004;
///     -0.003,  0.006,  0.001
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
/// let expected = [0.004, 0.006, 0.002];
///
/// // With little aversion to risk, the long-short portfolio leverages up on the highest
/// // expected returns and shorts the lowest.
/// let method = MeanVariance::risk_aversion(1.0).expected_returns(expected);
/// assert!(allocator.allocate(&method.clone().allow_short())?.iter().any(|&w| w < 0.0));
/// assert!(allocator.allocate(&method)?.iter().all(|&w| w >= 0.0));
///
/// // With at most 40% in any asset, the highest reachable return is
/// // 0.4 · 0.006 + 0.4 · 0.004 + 0.2 · 0.002 = 0.0044.
/// let target = MeanVariance::target_return(0.005).expected_returns(expected);
/// assert!(allocator.allocate(&target.clone().bounds(WeightBounds::uniform(0.0, 0.4))).is_err());
///
/// let weights = allocator.allocate(&target.bounds(WeightBounds::uniform(0.1, 0.6)))?;
/// let portfolio_return: f64 = weights.iter().zip(expected).map(|(w, m)| w * m).sum();
/// assert!((portfolio_return - 0.005).abs() < 1e-12);
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
    expected_returns: Option<Vec<f64>>,
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

    /// Minimizes variance subject to an expected portfolio return of `target`.
    pub fn target_return(target: f64) -> Self {
        Self::new(Objective::TargetReturn(target))
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

    /// Uses `expected_returns`, one per asset in column order, instead of the sample mean of
    /// the returns. They're in the same units and period as the returns.
    pub fn expected_returns(mut self, expected_returns: impl Into<Vec<f64>>) -> Self {
        self.expected_returns = Some(expected_returns.into());
        self
    }

    covariance_setters!();
    turnover_setters!();

    /// Subtracts `κ‖w − w₀‖²` from the objective, pulling the weights toward the
    /// [current weights](MeanVariance::current_weights) `w₀`. `κ` is in the units of the
    /// returns.
    pub fn turnover_penalty(mut self, kappa: f64) -> Self {
        self.trading.penalty = Some(kappa);
        self
    }

    /// Charges `cost` per unit of weight traded, `Σ cost·|wᵢ − w₀ᵢ|`, against the expected
    /// return: 0.001 is 10 basis points. Needs [current weights](MeanVariance::current_weights).
    pub fn transaction_costs(mut self, cost: f64) -> Self {
        self.trading.costs = Some(Costs::Uniform(cost));
        self
    }

    /// Charges `costs[i]` per unit of asset `i`'s weight traded, one per asset in column order.
    /// See [`MeanVariance::transaction_costs`].
    pub fn transaction_costs_per_asset(mut self, costs: impl Into<Vec<f64>>) -> Self {
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
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let expected_returns = self.expected_returns.clone().map(DVector::from_vec);
        if self.trading.first_setting().is_some() {
            check_objective(self.objective)?;
            let covariance = || data.adjusted_covariance(&self.covariance);
            return match (self.objective, self.trading.first_charge()) {
                (Objective::RiskAversion(gamma), _) => {
                    let mu = data.expected_returns(expected_returns.as_ref())?;
                    let cov = covariance()? * gamma;
                    turnover::rebalance(cov, mu, self.bounds.as_ref(), &self.trading, 2.0)
                }
                (Objective::TargetReturn(_), Some(setting)) => Err(CutupError::NotSupported {
                    setting,
                    with: "a target return",
                }),
                // Neither a penalty nor costs, so the setting is the cap.
                (Objective::TargetReturn(target), None) => {
                    let mu = data.expected_returns(expected_returns.as_ref())?;
                    let cap = self.trading.max_turnover.unwrap_or(f64::INFINITY);
                    turnover::rebalance_to_target(
                        covariance,
                        mu,
                        target,
                        self.bounds.as_ref(),
                        &self.trading,
                        cap,
                    )
                }
            };
        }
        self.trading.check(data.cov_matrix.ncols())?;
        let mu = expected_returns.as_ref();
        match &self.bounds {
            Some(bounds) => {
                data.bounded_mean_variance(self.objective, mu, &self.covariance, bounds)
            }
            None => data.mean_variance(self.objective, mu, &self.covariance),
        }
    }
}

/// The maximum Sharpe ratio (tangency) portfolio: maximize `(μᵀw − r_f) / sqrt(wᵀΣw)` subject
/// to `1ᵀw = 1`. Long-only by default.
///
/// The expected returns `μ` are the sample mean of the returns unless you set
/// [`MaxSharpe::expected_returns`], and the risk-free rate `r_f` is 0 unless you set
/// [`MaxSharpe::risk_free_rate`].
///
/// # Long-only
///
/// With `y = w / (μ − r_f·1)ᵀw`, this is the convex problem of minimizing `yᵀΣy` subject to
/// `(μ − r_f·1)ᵀy = 1` and `y ≥ 0` (Cornuejols and Tütüncü, 2007, §8.2), and `w = y / 1ᵀy`. It
/// is solved exactly by the same active-set method as [`MinVariance`].
///
/// A long-only portfolio has a positive Sharpe ratio only if some asset's expected return is
/// above the risk-free rate, so this needs the rate to be below the highest expected return,
/// not below the minimum variance portfolio's return as long-short does. An expected return
/// within `1e-10 · max(|r_f|, |μᵢ|)` of the rate counts as at it, a tolerance in each asset's
/// own units. When Σ is invertible and the long-short portfolio has no short positions, the
/// two agree.
///
/// When Σ is singular, several long-only portfolios can share the maximum, and which one is
/// returned depends on the solver's path. Add shrinkage or regularization for a unique answer.
///
/// # Long-short
///
/// With [`MaxSharpe::allow_short`], the solution is `w = Σ⁻¹(μ − r_f·1) / 1ᵀΣ⁻¹(μ − r_f·1)`.
///
/// The maximum exists only when the risk-free rate is below the expected return of the minimum
/// variance portfolio, `r_mvp = 1ᵀΣ⁻¹μ / 1ᵀΣ⁻¹1`. At or above it, the formula would give the
/// portfolio with the *lowest* Sharpe ratio, so an error is returned instead. A rate within
/// `1e-10 · Σ |wᵢμᵢ|` of `r_mvp`, for the minimum variance weights `w`, counts as at it: the
/// weights there are pure leverage on rounding noise.
///
/// When Σ is singular, its null space (portfolio changes that add no variance) decides the
/// outcome:
///
/// * If some fully invested change adds no variance but does add expected return, or a fully
///   invested zero-variance portfolio beats the risk-free rate, the Sharpe ratio is unbounded:
///   [`CutupError::Unbounded`].
/// * If a fully invested zero-variance portfolio exists but doesn't beat the risk-free rate,
///   it is the minimum variance portfolio and the rate isn't below its return:
///   [`CutupError::RiskFreeRateTooHigh`].
/// * Otherwise the weights above are computed with the pseudo-inverse, and are the
///   minimum-norm optimum in correlation units, as for [`MinVariance`].
///
/// The null space is that of the correlation matrix `C`, with `Σ = diag(σ) C diag(σ)`, so the
/// outcome doesn't depend on each asset's units.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the risk-free rate or an expected return is not finite,
///   or the covariance settings are invalid.
/// * [`CutupError::DimensionMismatch`] if the expected returns don't have one entry per asset.
/// * [`CutupError::NoPositiveExcessReturn`], long-only, if the risk-free rate is at or above
///   the highest expected return.
/// * [`CutupError::RiskFreeRateTooHigh`], long-short, if the risk-free rate is at or above the
///   minimum variance portfolio's expected return.
/// * [`CutupError::Unbounded`] when Σ is singular and the Sharpe ratio has no upper bound:
///   long-only, when some long-only portfolio with a positive excess return lies in the
///   correlation matrix's null space, with the long-short cutoff.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::NonFiniteCovariance`] if the adjusted covariance isn't finite.
/// * [`CutupError::SingularCovariance`] if the covariance can't be decomposed or no finite
///   weights can be computed.
/// * [`CutupError::NotConverged`] if the long-only solver stops early, which only extreme
///   inputs should cause.
///
/// # Example
///
/// ```
/// use cutup::{MaxSharpe, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// // Uncorrelated assets: the tangency weights are proportional to (μᵢ − r_f) / σᵢ².
/// let returns = dmatrix![
///      0.01,  0.02;
///     -0.01, -0.02;
///      0.01, -0.02;
///     -0.01,  0.02
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
///
/// // σ² is in the ratio 1 : 4 and μ in the ratio 1 : 2, so the weights are 2 : 1.
/// let weights = allocator.allocate(&MaxSharpe::default().expected_returns([0.002, 0.004]))?;
/// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
/// assert!((weights[1] - 1.0 / 3.0).abs() < 1e-12);
///
/// // With a negative expected return, the tangency portfolio shorts the second asset, which
/// // long-only can't.
/// let method = MaxSharpe::default().expected_returns([0.002, -0.001]);
/// assert!(allocator.allocate(&method.clone().allow_short())?[1] < 0.0);
/// let weights = allocator.allocate(&method)?;
/// assert!((weights[0] - 1.0).abs() < 1e-12 && weights[1] == 0.0);
///
/// // 4% a year, for daily returns.
/// let method = MaxSharpe::default().risk_free_rate(0.04 / 252.0).shrinkage(0.1);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MaxSharpe {
    risk_free_rate: f64,
    expected_returns: Option<Vec<f64>>,
    covariance: CovarianceConfig,
    long_only: bool,
}

impl Default for MaxSharpe {
    fn default() -> Self {
        Self {
            risk_free_rate: 0.0,
            expected_returns: None,
            covariance: CovarianceConfig::default(),
            long_only: true,
        }
    }
}

impl MaxSharpe {
    /// Sets the risk-free rate per period, in the same units as the returns (a daily rate for
    /// daily returns). Defaults to 0.
    pub fn risk_free_rate(mut self, risk_free_rate: f64) -> Self {
        self.risk_free_rate = risk_free_rate;
        self
    }

    /// Uses `expected_returns`, one per asset in column order, instead of the sample mean of
    /// the returns. They're in the same units and period as the returns.
    pub fn expected_returns(mut self, expected_returns: impl Into<Vec<f64>>) -> Self {
        self.expected_returns = Some(expected_returns.into());
        self
    }

    covariance_setters!();

    /// Allows short positions.
    pub fn allow_short(mut self) -> Self {
        self.long_only = false;
        self
    }
}

impl Allocation for MaxSharpe {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let mu = self.expected_returns.clone().map(DVector::from_vec);
        if self.long_only {
            data.long_only_max_sharpe(self.risk_free_rate, mu.as_ref(), &self.covariance)
        } else {
            data.max_sharpe(self.risk_free_rate, mu.as_ref(), &self.covariance)
        }
    }
}

/// The maximum diversification portfolio (Choueifaty and Coignard, 2008): maximize the
/// diversification ratio `σᵀw / sqrt(wᵀΣw)` subject to `1ᵀw = 1` and `w ≥ 0`, where `σ` holds
/// the assets' volatilities.
///
/// The ratio is the weighted average volatility over the portfolio's volatility, which is at
/// least 1, and is 1 only when the held assets are perfectly correlated. It is long-only by
/// definition. This is the long-only [`MaxSharpe`] problem with `σ` in place of the excess
/// returns, solved the same way: with `y = w / σᵀw`, minimize `yᵀΣy` subject to `σᵀy = 1` and
/// `y ≥ 0`, then `w = y / 1ᵀy`. The solver works in correlation units `diag(σ)·y`, where that
/// is minimizing `vᵀCv` for the correlation matrix `C` subject to `1ᵀv = 1`, so assets whose
/// volatilities differ by many orders of magnitude stay accurate. For uncorrelated assets it
/// is inverse volatility weighting. With shrinkage or regularization, the volatilities are
/// those of the adjusted covariance.
///
/// When Σ is singular, several portfolios can share the maximum, and which one is returned
/// depends on the solver's path. Add shrinkage for a unique answer.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if `regularization` is negative or not finite, or
///   `shrinkage` is outside `[0, 1]`.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::NonFiniteCovariance`] if the adjusted covariance isn't finite.
/// * [`CutupError::SingularCovariance`] if some long-only portfolio has no variance (in the
///   correlation matrix's null space, with the cutoff of long-short [`MaxSharpe`]), so the
///   ratio has no upper bound, or no finite weights can be computed.
/// * [`CutupError::NotConverged`] if the solver stops early, which only extreme inputs should
///   cause.
///
/// # Example
///
/// ```
/// use cutup::{MaxDiversification, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// // Uncorrelated assets, with volatilities in the ratio 1 : 2 : 4.
/// let returns = dmatrix![
///      0.01,  0.02,  0.04;
///     -0.01,  0.02, -0.04;
///      0.01, -0.02, -0.04;
///     -0.01, -0.02,  0.04
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
/// let weights = allocator.allocate(&MaxDiversification::default())?;
///
/// // Inverse volatility: 4 : 2 : 1.
/// for (w, expected) in weights.iter().zip([4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]) {
///     assert!((w - expected).abs() < 1e-12);
/// }
///
/// // Fewer observations than assets: some long-only portfolio has no variance.
/// let allocator = PortfolioAllocator::from_returns(dmatrix![
///      0.01, -0.02,  0.03;
///     -0.01,  0.02, -0.03
/// ])?;
/// assert!(allocator.allocate(&MaxDiversification::default()).is_err());
/// assert!(allocator.allocate(&MaxDiversification::default().shrinkage(0.3)).is_ok());
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct MaxDiversification {
    covariance: CovarianceConfig,
}

impl MaxDiversification {
    covariance_setters!();
}

impl Allocation for MaxDiversification {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.max_diversification(&self.covariance)
    }
}

/// The risk parity (equal risk contribution) portfolio, or risk budgeting with
/// [`RiskParity::risk_budgets`]: long-only weights where asset `i` contributes the share `bᵢ`
/// of the portfolio's variance, `wᵢ (Σw)ᵢ / wᵀΣw = bᵢ / Σⱼ bⱼ`. With equal budgets, the default,
/// every asset contributes the same share.
///
/// Unlike [`InverseVolatility`], which gives equal risk contributions only when assets are
/// uncorrelated, this accounts for every correlation. The weights are the unique solution,
/// found by minimizing `½ yᵀΣy − Σᵢ bᵢ ln yᵢ` over `y > 0` with Newton's method and a
/// backtracking line search (Spinu, 2013), then normalizing `y` to sum to 1. The result
/// doesn't depend on the scale of the returns or of the budgets.
///
/// The solution exists whenever every long-only portfolio has positive variance, which holds
/// when Σ is invertible and also for most singular Σ. When some long-only portfolio has zero
/// variance, a [`CutupError::SingularCovariance`] is returned; shrinkage or regularization
/// removes it. Risk budgets can span many orders of magnitude, up to about 10¹⁴ between the
/// largest and the smallest, beyond which the solver may stop early with
/// [`CutupError::NotConverged`].
///
/// The weights are positive and sum to 1.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if a risk budget isn't positive and finite, or the
///   covariance settings are invalid.
/// * [`CutupError::DimensionMismatch`] if the risk budgets don't have one entry per asset.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::SingularCovariance`] if some long-only portfolio has zero variance.
/// * [`CutupError::NotConverged`] if the solver stops early, which takes risk budgets more
///   than about 10¹⁴ apart.
///
/// # Example
///
/// ```
/// use cutup::{PortfolioAllocator, RiskParity};
/// use nalgebra::dmatrix;
///
/// // Uncorrelated assets with volatilities in the ratio 1 : 2.
/// let returns = dmatrix![
///      0.01,  0.02;
///     -0.01, -0.02;
///      0.01, -0.02;
///     -0.01,  0.02
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
///
/// // Equal budgets: weights in inverse proportion to volatility, 2 : 1.
/// let weights = allocator.allocate(&RiskParity::default())?;
/// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
///
/// // Budgets 1 : 4: wᵢ ∝ sqrt(bᵢ) / σᵢ, so 1 : 1.
/// let weights = allocator.allocate(&RiskParity::default().risk_budgets([1.0, 4.0]))?;
/// assert!((weights[0] - 0.5).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RiskParity {
    risk_budgets: Option<Vec<f64>>,
    covariance: CovarianceConfig,
}

impl RiskParity {
    /// Sets each asset's share of the portfolio's risk, in column order. Only the ratios
    /// matter, so they needn't sum to 1. Defaults to the same share for every asset.
    pub fn risk_budgets(mut self, risk_budgets: impl Into<Vec<f64>>) -> Self {
        self.risk_budgets = Some(risk_budgets.into());
        self
    }

    covariance_setters!();
}

impl Allocation for RiskParity {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let budgets = self.risk_budgets.clone().map(DVector::from_vec);
        data.risk_parity(budgets.as_ref(), &self.covariance)
    }
}

/// Hierarchical Risk Parity (López de Prado, 2016).
///
/// 1. **Cluster.** Assets are clustered on the correlation distance `sqrt((1 − ρ) / 2)` with
///    the [linkage](Hrp::linkage), and ordered so that correlated assets sit next to each
///    other (see [`Hrp::cluster_order`]).
/// 2. **Recursive bisection.** The ordered list is split in half, repeatedly. At each split,
///    the two halves share their parent's weight in inverse proportion to their risk: each
///    half's variance is that of its own inverse-variance portfolio, and the first half gets
///    `α = 1 − V₁ / (V₁ + V₂)`.
///
/// The clustering follows SciPy and PyPortfolioOpt's `HRPOpt` step for step, including the
/// tie-breaking rule, so results match `HRPOpt(returns).optimize(linkage_method)` up to
/// floating-point rounding. (The correlations are computed here rather than by pandas, so two
/// distances that tie in one may differ by a rounding error in the other and break the tie the
/// other way.)
///
/// HRP never inverts the covariance matrix, so it works when Σ is singular. Weights are never
/// negative, and sum to 1. A half with zero variance, such as a perfectly hedged pair, takes
/// its parent's whole weight and leaves the other half at zero, as in `HRPOpt`.
///
/// Like the original algorithm and `HRPOpt`, the result can depend on the order of the input
/// columns. The hierarchy puts the lower-indexed subtree on the left, and the bisection splits
/// the ordered list at its midpoint rather than at the tree's branches, so reordering the
/// columns can move a split. Correlated assets still end up next to each other; only where the
/// halves are cut changes. [`Herc`] splits at the branches instead.
///
/// # Errors
///
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
/// * [`CutupError::NonFiniteCovariance`] if correlations can't be computed because the
///   variances are too small.
///
/// # Example
///
/// ```
/// use cutup::{Hrp, Linkage, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// let returns = dmatrix![
///      0.010,  0.020,  0.011,  0.019;
///     -0.010,  0.010, -0.012,  0.012;
///      0.020, -0.020,  0.018, -0.021;
///     -0.015, -0.010, -0.014, -0.009
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
///
/// let weights = allocator.allocate(&Hrp::default().linkage(Linkage::Ward))?;
/// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
/// assert!(weights.iter().all(|&w| w >= 0.0));
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Hrp {
    linkage: Linkage,
}

impl Hrp {
    /// Sets how the distance between clusters is measured when building the hierarchy.
    /// Defaults to [`Linkage::Single`].
    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Returns the order HRP puts the assets in: the leaves of the clustering hierarchy, left
    /// to right, so that correlated assets sit next to each other.
    ///
    /// Assets are clustered on the correlation distance `sqrt((1 − ρ) / 2)` with the
    /// [linkage](Hrp::linkage). The clustering reproduces SciPy's `linkage` and
    /// `to_tree().pre_order()` step for step, including the tie-breaking rule, so the order
    /// matches SciPy and PyPortfolioOpt's `HRPOpt` up to floating-point rounding.
    ///
    /// Every asset's column index appears exactly once.
    ///
    /// # Errors
    ///
    /// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
    /// * [`CutupError::NonFiniteCovariance`] if correlations can't be computed because the
    ///   variances are too small.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{Hrp, PortfolioAllocator};
    /// use nalgebra::dmatrix;
    ///
    /// // Assets 0 and 2 move together, as do 1 and 3.
    /// let returns = dmatrix![
    ///      0.010,  0.020,  0.011,  0.019;
    ///     -0.010,  0.010, -0.012,  0.012;
    ///      0.020, -0.020,  0.018, -0.021;
    ///     -0.015, -0.010, -0.014, -0.009
    /// ];
    ///
    /// let allocator = PortfolioAllocator::from_returns(returns)?;
    /// let order = Hrp::default().cluster_order(&allocator)?;
    ///
    /// let position = |asset| order.iter().position(|&a| a == asset).unwrap();
    /// assert_eq!(position(0).abs_diff(position(2)), 1);
    /// assert_eq!(position(1).abs_diff(position(3)), 1);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn cluster_order(&self, data: &PortfolioAllocator) -> Result<Vec<usize>, CutupError> {
        data.cluster_order(self.linkage)
    }
}

impl Allocation for Hrp {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.hrp(self.linkage)
    }
}

/// The most clusters [`Herc`]'s gap statistic may choose, unless set.
const HERC_MAX_CLUSTERS: usize = 10;

/// Hierarchical Equal Risk Contribution (Raffinot, 2018), as riskfolio-lib implements it with
/// the variance as the risk measure.
///
/// 1. **Cluster.** Assets are clustered on the correlation distance `sqrt((1 − ρ) / 2)` with
///    the [linkage](Herc::linkage), as for [`Hrp::cluster_order`].
/// 2. **Cut.** The hierarchy is cut into `k` clusters: [`Herc::clusters`] if set, otherwise
///    the number chosen by the two-difference gap statistic, as in riskfolio-lib.
/// 3. **Split down the tree.** Each of the `k − 1` merges above the cut splits its weight
///    between its two branches in inverse proportion to their risk. A branch's risk is the
///    sum of the variances of the clusters under it, each cluster's being that of its own
///    inverse-variance portfolio. The left branch gets `α = 1 − R₁ / (R₁ + R₂)`.
/// 4. **Within clusters.** Each cluster's weight is shared by inverse variance.
///
/// The defaults match riskfolio-lib's `HCPortfolio.optimization(model="HERC")`: single
/// linkage, and the number of clusters chosen by the gap statistic from at most 10.
///
/// Unlike [`Hrp`], which bisects the ordered list of assets at its midpoint, HERC splits where
/// the tree branches, so similar assets are never separated and the result doesn't depend on
/// the order of the input columns. Results match riskfolio-lib's
/// `HCPortfolio.optimization(model="HERC", rm="MV")` up to floating-point rounding.
///
/// Both hold except when merges tie in distance at the cut, for example when assets are
/// duplicated. The cut is then made after exactly `n − k` merges, taking tied merges in the
/// order the clustering found them, which depends on column order. riskfolio-lib instead takes
/// the `k − 1` highest merges in the order of an unstable sort, and builds the clusters with
/// SciPy's `fcluster`, which can return fewer than `k` clusters at a tie, so its weights can
/// differ there.
///
/// HERC never inverts the covariance matrix, so it works when Σ is singular, and weights are
/// never negative, and sum to 1. Two branches that both have zero risk split their weight
/// evenly.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the number of clusters is 0 or more than the number of
///   assets, or, when the number isn't set, the most clusters is 0.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance.
/// * [`CutupError::NonFiniteCovariance`] if correlations can't be computed because the
///   variances are too small.
///
/// # Example
///
/// ```
/// use cutup::{Herc, InverseVariance, PortfolioAllocator};
/// use nalgebra::dmatrix;
///
/// // Assets 0 and 2 move together, as do 1 and 3.
/// let returns = dmatrix![
///      0.010,  0.020,  0.011,  0.019;
///     -0.010,  0.010, -0.012,  0.012;
///      0.020, -0.020,  0.018, -0.021;
///     -0.015, -0.010, -0.014, -0.009
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
/// let weights = allocator.allocate(&Herc::default().clusters(2))?;
///
/// // With two clusters, {0, 2} and {1, 3}, each pair is weighted by inverse variance.
/// let inverse_variance = allocator.allocate(&InverseVariance)?;
/// let ratio = |a: usize, b: usize| weights[a] / weights[b];
/// assert!((ratio(0, 2) - inverse_variance[0] / inverse_variance[2]).abs() < 1e-12);
/// assert!((ratio(1, 3) - inverse_variance[1] / inverse_variance[3]).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Herc {
    linkage: Linkage,
    clusters: Option<usize>,
    max_clusters: usize,
}

impl Default for Herc {
    fn default() -> Self {
        Self {
            linkage: Linkage::default(),
            clusters: None,
            max_clusters: HERC_MAX_CLUSTERS,
        }
    }
}

impl Herc {
    /// Sets how the distance between clusters is measured when building the hierarchy.
    /// Defaults to [`Linkage::Single`].
    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Cuts the hierarchy into `clusters` clusters, from 1 to the number of assets, instead
    /// of choosing the number.
    pub fn clusters(mut self, clusters: usize) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// Sets the most clusters the gap statistic may choose, at least 1. Defaults to 10. It
    /// also never chooses more than `√n` for `n` assets.
    pub fn max_clusters(mut self, max_clusters: usize) -> Self {
        self.max_clusters = max_clusters;
        self
    }
}

impl Allocation for Herc {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.herc(self.linkage, self.clusters, self.max_clusters)
    }
}
