//! Allocation methods as values, and the [`Allocation`] trait that runs them.

use crate::turnover::{self, Costs, Trading};
use crate::{
    check_objective, CovarianceConfig, CutupError, HercConfig, HrpConfig, Linkage, MaxSharpeConfig,
    MeanVarianceConfig, MeanVarianceObjective, PortfolioAllocator, RiskParityConfig, WeightBounds,
    Weights,
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
///     Box::new(InverseVolatility::default()),
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
/// Same as [`PortfolioAllocator::ew_allocation`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EqualWeight;

impl Allocation for EqualWeight {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        Ok(data.ew_allocation())
    }
}

/// Inverse volatility weights, `wᵢ ∝ 1 / σᵢ`.
///
/// Same as [`PortfolioAllocator::inverse_volatility_allocation`], which lists the errors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct InverseVolatility;

impl Allocation for InverseVolatility {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.inverse_volatility_allocation()
    }
}

/// Inverse variance weights, `wᵢ ∝ 1 / σᵢ²`.
///
/// Same as [`PortfolioAllocator::inverse_variance_allocation`], which lists the errors.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct InverseVariance;

impl Allocation for InverseVariance {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.inverse_variance_allocation()
    }
}

/// The minimum variance portfolio. Long-only by default.
///
/// * Long-only, optionally within [`MinVariance::bounds`]: same as
///   [`PortfolioAllocator::long_only_min_variance_allocation_with_bounds`].
/// * With [`MinVariance::allow_short`]: same as
///   [`PortfolioAllocator::min_variance_allocation_with_config`].
///
/// Those methods list the errors.
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
/// Besides those methods' errors, these settings return:
///
/// * [`CutupError::MissingCurrentWeights`] if one is set without current weights.
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9` or
///   aren't finite, or the cap or penalty is negative or not finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds.
///
/// ```
/// use cutup::{MinVariance, WeightBounds};
///
/// let capped = MinVariance::default().shrinkage(0.1).bounds(WeightBounds::uniform(0.0, 0.3));
/// let long_short = MinVariance::default().allow_short();
/// let rebalance = MinVariance::default().current_weights([0.5, 0.3, 0.2]).max_turnover(0.1);
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
            Some(bounds) => {
                data.long_only_min_variance_allocation_with_bounds(&self.covariance, bounds)
            }
            None => data.min_variance_allocation_with_config(&self.covariance),
        }
    }
}

/// Mean-variance optimization. Long-only by default.
///
/// Start from an objective, [`MeanVariance::risk_aversion`] or [`MeanVariance::target_return`].
/// Unlike the other methods it has no default: a sensible risk aversion depends on the period
/// of the returns, and for per-period returns, whose variances are small, it's large.
///
/// * Long-only, optionally within [`MeanVariance::bounds`]: same as
///   [`PortfolioAllocator::long_only_mean_variance_allocation_with_bounds`].
/// * With [`MeanVariance::allow_short`]: same as
///   [`PortfolioAllocator::mean_variance_allocation`].
///
/// Those methods list the errors.
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
/// Besides those methods' errors, these settings return:
///
/// * [`CutupError::MissingCurrentWeights`] if one is set without current weights.
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9` or
///   aren't finite, or the cap, penalty or a cost is negative or not finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight, or per-asset cost,
///   per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds, and to the target return if there is one.
///
/// ```
/// use cutup::MeanVariance;
///
/// let method = MeanVariance::target_return(0.005)
///     .expected_returns([0.004, 0.006, 0.002])
///     .shrinkage(0.1);
///
/// // Rebalance net of 10 basis points a trade, turning over at most 20%.
/// let rebalance = MeanVariance::risk_aversion(100.0)
///     .current_weights([0.4, 0.3, 0.3])
///     .transaction_costs(0.001)
///     .max_turnover(0.2);
///
/// // Or hold the target return, turning over at most 20%.
/// let capped = MeanVariance::target_return(0.005)
///     .expected_returns([0.004, 0.006, 0.002])
///     .current_weights([0.4, 0.3, 0.3])
///     .max_turnover(0.2);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MeanVariance {
    objective: MeanVarianceObjective,
    expected_returns: Option<Vec<f64>>,
    covariance: CovarianceConfig,
    /// `None` when short positions are allowed.
    bounds: Option<WeightBounds>,
    trading: Trading,
}

impl MeanVariance {
    /// Maximizes `μᵀw − (γ/2) wᵀΣw` with risk aversion `gamma`. Larger values favor lower
    /// variance.
    pub fn risk_aversion(gamma: f64) -> Self {
        Self::new(MeanVarianceObjective::RiskAversion(gamma))
    }

    /// Minimizes variance subject to an expected portfolio return of `target`.
    pub fn target_return(target: f64) -> Self {
        Self::new(MeanVarianceObjective::TargetReturn(target))
    }

    fn new(objective: MeanVarianceObjective) -> Self {
        Self {
            objective,
            expected_returns: None,
            covariance: CovarianceConfig::default(),
            bounds: Some(WeightBounds::default()),
            trading: Trading::default(),
        }
    }

    /// Uses `expected_returns`, one per asset in column order, instead of the sample mean of
    /// the returns.
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
        let config = MeanVarianceConfig {
            objective: self.objective,
            expected_returns: self.expected_returns.clone().map(DVector::from_vec),
            covariance: self.covariance.clone(),
        };
        if self.trading.first_setting().is_some() {
            check_objective(config.objective)?;
            let covariance = || data.adjusted_covariance(&config.covariance);
            return match (config.objective, self.trading.first_charge()) {
                (MeanVarianceObjective::RiskAversion(gamma), _) => {
                    let mu = data.expected_returns(config.expected_returns.as_ref())?;
                    let cov = covariance()? * gamma;
                    turnover::rebalance(cov, mu, self.bounds.as_ref(), &self.trading, 2.0)
                }
                (MeanVarianceObjective::TargetReturn(_), Some(setting)) => {
                    Err(CutupError::NotSupported {
                        setting,
                        with: "a target return",
                    })
                }
                // Neither a penalty nor costs, so the setting is the cap.
                (MeanVarianceObjective::TargetReturn(target), None) => {
                    let mu = data.expected_returns(config.expected_returns.as_ref())?;
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
        match &self.bounds {
            Some(bounds) => data.long_only_mean_variance_allocation_with_bounds(&config, bounds),
            None => data.mean_variance_allocation(&config),
        }
    }
}

/// The maximum Sharpe ratio (tangency) portfolio. Long-only by default.
///
/// * Long-only: same as [`PortfolioAllocator::long_only_max_sharpe_allocation`].
/// * With [`MaxSharpe::allow_short`]: same as [`PortfolioAllocator::max_sharpe_allocation`].
///
/// Those methods list the errors.
///
/// ```
/// use cutup::MaxSharpe;
///
/// // 4% a year, for daily returns.
/// let method = MaxSharpe::default().risk_free_rate(0.04 / 252.0).shrinkage(0.1);
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
    /// Sets the risk-free rate per period, in the same units as the returns. Defaults to 0.
    pub fn risk_free_rate(mut self, risk_free_rate: f64) -> Self {
        self.risk_free_rate = risk_free_rate;
        self
    }

    /// Uses `expected_returns`, one per asset in column order, instead of the sample mean of
    /// the returns.
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
        let config = MaxSharpeConfig {
            risk_free_rate: self.risk_free_rate,
            expected_returns: self.expected_returns.clone().map(DVector::from_vec),
            covariance: self.covariance.clone(),
        };
        if self.long_only {
            data.long_only_max_sharpe_allocation(&config)
        } else {
            data.max_sharpe_allocation(&config)
        }
    }
}

/// The maximum diversification portfolio, which is always long-only.
///
/// Same as [`PortfolioAllocator::max_diversification_allocation_with_config`], which lists
/// the errors.
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
        data.max_diversification_allocation_with_config(&self.covariance)
    }
}

/// Risk parity, or risk budgeting with [`RiskParity::risk_budgets`]. Always long-only.
///
/// Same as [`PortfolioAllocator::risk_parity_allocation_with_config`], which lists the
/// errors.
///
/// ```
/// use cutup::RiskParity;
///
/// // Asset 0 carries half the risk, assets 1 and 2 a quarter each.
/// let method = RiskParity::default().risk_budgets([2.0, 1.0, 1.0]);
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RiskParity {
    risk_budgets: Option<Vec<f64>>,
    covariance: CovarianceConfig,
}

impl RiskParity {
    /// Sets each asset's share of the portfolio's risk, in column order. Only the ratios
    /// matter. Defaults to the same share for every asset.
    pub fn risk_budgets(mut self, risk_budgets: impl Into<Vec<f64>>) -> Self {
        self.risk_budgets = Some(risk_budgets.into());
        self
    }

    covariance_setters!();
}

impl Allocation for RiskParity {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let config = RiskParityConfig {
            risk_budgets: self.risk_budgets.clone().map(DVector::from_vec),
            covariance: self.covariance.clone(),
        };
        data.risk_parity_allocation_with_config(&config)
    }
}

/// Hierarchical Risk Parity (López de Prado, 2016).
///
/// Same as [`PortfolioAllocator::hrp_allocation_with_config`], which lists the errors.
///
/// ```
/// use cutup::{Hrp, Linkage};
///
/// let method = Hrp::default().linkage(Linkage::Average);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Hrp {
    linkage: Linkage,
}

impl Hrp {
    /// Sets the linkage used to build the clustering hierarchy. Defaults to
    /// [`Linkage::Single`].
    pub fn linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

impl Allocation for Hrp {
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        data.hrp_allocation_with_config(&HrpConfig {
            linkage: self.linkage,
        })
    }
}

/// Hierarchical Equal Risk Contribution (Raffinot, 2018).
///
/// The defaults match riskfolio-lib's `HCPortfolio.optimization(model="HERC")`: single
/// linkage, and the number of clusters chosen by the two-difference gap statistic from at
/// most 10.
///
/// Same as [`PortfolioAllocator::herc_allocation_with_config`], which lists the errors.
///
/// ```
/// use cutup::{Herc, Linkage};
///
/// let method = Herc::default().linkage(Linkage::Ward).clusters(3);
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
        let config = HercConfig::default();
        Self {
            linkage: config.linkage,
            clusters: config.clusters,
            max_clusters: config.max_clusters,
        }
    }
}

impl Herc {
    /// Sets the linkage used to build the clustering hierarchy. Defaults to
    /// [`Linkage::Single`].
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
        data.herc_allocation_with_config(&HercConfig {
            linkage: self.linkage,
            clusters: self.clusters,
            max_clusters: self.max_clusters,
        })
    }
}
