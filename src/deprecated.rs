//! The API from before [`Allocation`](crate::Allocation): a method on [`PortfolioAllocator`]
//! per allocation method, and a config struct for each method's settings. Deprecated since
//! 0.2.8, and removed in 0.3.0. Each method returns what [`PortfolioAllocator::allocate`]
//! returns for the method value its deprecation note names.
//!
//! Each one warns where it's used:
//!
//! ```compile_fail
//! #![deny(deprecated)]
//! let allocator = cutup::PortfolioAllocator::from_returns(nalgebra::dmatrix![0.01; -0.01])?;
//! let _ = allocator.min_variance_allocation();
//! # Ok::<(), cutup::CutupError>(())
//! ```
//!
//! ```compile_fail
//! #![deny(deprecated)]
//! let _ = cutup::run_portfolio_allocation(nalgebra::dmatrix![1.0; 1.1; 1.2]);
//! ```
//!
//! ```compile_fail
//! #![deny(deprecated)]
//! let _ = cutup::HrpConfig::default();
//! ```
//!
//! Without the deprecated items, the same code compiles:
//!
//! ```
//! #![deny(deprecated)]
//! use cutup::{Hrp, MinVariance, PortfolioAllocator};
//!
//! let allocator = PortfolioAllocator::from_returns(nalgebra::dmatrix![0.01; -0.01])?;
//! let _ = allocator.allocate(&MinVariance::default().allow_short());
//! let _ = PortfolioAllocator::from_prices(nalgebra::dmatrix![1.0; 1.1; 1.2])?
//!     .allocate(&MinVariance::default().allow_short());
//! let _ = Hrp::default();
//! # Ok::<(), cutup::CutupError>(())
//! ```

#![allow(deprecated)]

use crate::{
    CovarianceConfig, CutupError, Herc, Hrp, InverseVariance, InverseVolatility, Linkage,
    MaxDiversification, MaxSharpe, MeanVariance, MinVariance, PortfolioAllocator, RiskParity,
    WeightBounds, Weights,
};
use nalgebra::{DMatrix, DVector};

/// What a mean-variance optimization trades off.
#[deprecated(
    since = "0.2.8",
    note = "use `MeanVariance::risk_aversion` or `MeanVariance::target_return`"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MeanVarianceObjective {
    /// Maximize `μᵀw − (γ/2) wᵀΣw` for a risk aversion `γ > 0`. Larger values favor lower
    /// variance; as `γ → ∞` the result approaches the minimum variance portfolio.
    RiskAversion(f64),
    /// Minimize `wᵀΣw` subject to an expected portfolio return `μᵀw` equal to this value.
    TargetReturn(f64),
}

/// Configuration for [`PortfolioAllocator::mean_variance_allocation`] and
/// [`PortfolioAllocator::long_only_mean_variance_allocation`].
#[deprecated(since = "0.2.8", note = "use `MeanVariance`")]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MeanVarianceConfig {
    /// The objective to optimize.
    pub objective: MeanVarianceObjective,
    /// Expected return per asset, in the same units and period as the returns. When `None`, the
    /// sample mean of the returns is used.
    pub expected_returns: Option<DVector<f64>>,
    /// Adjustments applied to the covariance matrix.
    pub covariance: CovarianceConfig,
}

impl MeanVarianceConfig {
    /// Maximizes `μᵀw − (γ/2) wᵀΣw` with risk aversion `gamma`.
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
        }
    }

    /// Uses `expected_returns` instead of the sample mean of the returns.
    pub fn with_expected_returns(mut self, expected_returns: DVector<f64>) -> Self {
        self.expected_returns = Some(expected_returns);
        self
    }

    /// Applies `covariance` adjustments to the covariance matrix.
    pub fn with_covariance(mut self, covariance: CovarianceConfig) -> Self {
        self.covariance = covariance;
        self
    }

    /// The same settings as a [`MeanVariance`], long-only.
    fn method(&self) -> MeanVariance {
        let method = match self.objective {
            MeanVarianceObjective::RiskAversion(gamma) => MeanVariance::risk_aversion(gamma),
            MeanVarianceObjective::TargetReturn(target) => MeanVariance::target_return(target),
        }
        .covariance(self.covariance.clone());
        match &self.expected_returns {
            Some(mu) => method.expected_returns(mu.as_slice()),
            None => method,
        }
    }
}

/// Configuration for [`PortfolioAllocator::max_sharpe_allocation`] and
/// [`PortfolioAllocator::long_only_max_sharpe_allocation`].
#[deprecated(since = "0.2.8", note = "use `MaxSharpe`")]
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct MaxSharpeConfig {
    /// The risk-free rate per period, in the same units as the returns (a daily rate for
    /// daily returns).
    pub risk_free_rate: f64,
    /// Expected return per asset, in the same units and period as the returns. When `None`, the
    /// sample mean of the returns is used.
    pub expected_returns: Option<DVector<f64>>,
    /// Adjustments applied to the covariance matrix.
    pub covariance: CovarianceConfig,
}

impl MaxSharpeConfig {
    /// Sets the risk-free rate per period.
    pub fn with_risk_free_rate(mut self, risk_free_rate: f64) -> Self {
        self.risk_free_rate = risk_free_rate;
        self
    }

    /// Uses `expected_returns` instead of the sample mean of the returns.
    pub fn with_expected_returns(mut self, expected_returns: DVector<f64>) -> Self {
        self.expected_returns = Some(expected_returns);
        self
    }

    /// Applies `covariance` adjustments to the covariance matrix.
    pub fn with_covariance(mut self, covariance: CovarianceConfig) -> Self {
        self.covariance = covariance;
        self
    }

    /// The same settings as a [`MaxSharpe`], long-only.
    fn method(&self) -> MaxSharpe {
        let method = MaxSharpe::default()
            .risk_free_rate(self.risk_free_rate)
            .covariance(self.covariance.clone());
        match &self.expected_returns {
            Some(mu) => method.expected_returns(mu.as_slice()),
            None => method,
        }
    }
}

/// Configuration for [`PortfolioAllocator::hrp_allocation_with_config`] and
/// [`PortfolioAllocator::hrp_cluster_order`].
#[deprecated(since = "0.2.8", note = "use `Hrp`")]
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct HrpConfig {
    /// How the distance between clusters is measured. Defaults to [`Linkage::Single`].
    pub linkage: Linkage,
}

impl HrpConfig {
    /// Sets the linkage used to build the clustering hierarchy.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }
}

/// Configuration for [`PortfolioAllocator::risk_parity_allocation_with_config`].
#[deprecated(since = "0.2.8", note = "use `RiskParity`")]
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RiskParityConfig {
    /// Each asset's share of the portfolio's risk. Must be positive and finite, one per asset;
    /// only the ratios matter, so they needn't sum to 1. When `None`, every asset gets the same
    /// share.
    pub risk_budgets: Option<DVector<f64>>,
    /// Adjustments applied to the covariance matrix.
    pub covariance: CovarianceConfig,
}

impl RiskParityConfig {
    /// Sets each asset's risk budget.
    pub fn with_risk_budgets(mut self, risk_budgets: DVector<f64>) -> Self {
        self.risk_budgets = Some(risk_budgets);
        self
    }

    /// Applies `covariance` adjustments to the covariance matrix.
    pub fn with_covariance(mut self, covariance: CovarianceConfig) -> Self {
        self.covariance = covariance;
        self
    }

    /// The same settings as a [`RiskParity`].
    fn method(&self) -> RiskParity {
        let method = RiskParity::default().covariance(self.covariance.clone());
        match &self.risk_budgets {
            Some(budgets) => method.risk_budgets(budgets.as_slice()),
            None => method,
        }
    }
}

/// Configuration for [`PortfolioAllocator::herc_allocation_with_config`].
#[deprecated(since = "0.2.8", note = "use `Herc`")]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HercConfig {
    /// How the distance between clusters is measured. Defaults to [`Linkage::Single`].
    pub linkage: Linkage,
    /// How many clusters to cut the hierarchy into, from 1 to the number of assets. When
    /// `None`, the two-difference gap statistic chooses.
    pub clusters: Option<usize>,
    /// The most clusters the gap statistic may choose, at least 1. Defaults to 10. It also
    /// never chooses more than `√n` for `n` assets.
    pub max_clusters: usize,
}

impl Default for HercConfig {
    fn default() -> Self {
        Self {
            linkage: Linkage::default(),
            clusters: None,
            max_clusters: 10,
        }
    }
}

impl HercConfig {
    /// Sets the linkage used to build the clustering hierarchy.
    pub fn with_linkage(mut self, linkage: Linkage) -> Self {
        self.linkage = linkage;
        self
    }

    /// Cuts the hierarchy into `clusters` clusters instead of choosing the number.
    pub fn with_clusters(mut self, clusters: usize) -> Self {
        self.clusters = Some(clusters);
        self
    }

    /// Sets the most clusters the gap statistic may choose.
    pub fn with_max_clusters(mut self, max_clusters: usize) -> Self {
        self.max_clusters = max_clusters;
        self
    }

    /// The same settings as a [`Herc`].
    fn method(&self) -> Herc {
        let method = Herc::default()
            .linkage(self.linkage)
            .max_clusters(self.max_clusters);
        match self.clusters {
            Some(clusters) => method.clusters(clusters),
            None => method,
        }
    }
}

impl PortfolioAllocator {
    /// The global minimum variance portfolio, with short positions allowed. See
    /// [`MinVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MinVariance::default().allow_short())`"
    )]
    pub fn min_variance_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&MinVariance::default().allow_short())
    }

    /// The global minimum variance portfolio of the adjusted covariance, with short positions
    /// allowed. See [`MinVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MinVariance::default().allow_short().covariance(config))`"
    )]
    pub fn min_variance_allocation_with_config(
        &self,
        config: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(
            &MinVariance::default()
                .allow_short()
                .covariance(config.clone()),
        )
    }

    /// The long-only minimum variance portfolio. See [`MinVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MinVariance::default())`"
    )]
    pub fn long_only_min_variance_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&MinVariance::default())
    }

    /// The long-only minimum variance portfolio of the adjusted covariance. See
    /// [`MinVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MinVariance::default().covariance(config))`"
    )]
    pub fn long_only_min_variance_allocation_with_config(
        &self,
        config: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(&MinVariance::default().covariance(config.clone()))
    }

    /// The minimum variance portfolio within `bounds`. See [`MinVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MinVariance::default().covariance(config).bounds(bounds))`"
    )]
    pub fn long_only_min_variance_allocation_with_bounds(
        &self,
        config: &CovarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        self.allocate(
            &MinVariance::default()
                .covariance(config.clone())
                .bounds(bounds.clone()),
        )
    }

    /// A mean-variance portfolio, with short positions allowed. See [`MeanVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MeanVariance::risk_aversion(γ).allow_short())`, or \
                `MeanVariance::target_return(ρ)`, with `.expected_returns(..)` and \
                `.covariance(..)`"
    )]
    pub fn mean_variance_allocation(
        &self,
        config: &MeanVarianceConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(&config.method().allow_short())
    }

    /// A long-only mean-variance portfolio. See [`MeanVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MeanVariance::risk_aversion(γ))`, or \
                `MeanVariance::target_return(ρ)`, with `.expected_returns(..)` and \
                `.covariance(..)`"
    )]
    pub fn long_only_mean_variance_allocation(
        &self,
        config: &MeanVarianceConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(&config.method())
    }

    /// A mean-variance portfolio within `bounds`. See [`MeanVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MeanVariance::risk_aversion(γ).bounds(bounds))`, or \
                `MeanVariance::target_return(ρ)`, with `.expected_returns(..)` and \
                `.covariance(..)`"
    )]
    pub fn long_only_mean_variance_allocation_with_bounds(
        &self,
        config: &MeanVarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        self.allocate(&config.method().bounds(bounds.clone()))
    }

    /// The maximum Sharpe ratio portfolio, with short positions allowed. See [`MaxSharpe`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MaxSharpe::default().allow_short())`, with \
                `.risk_free_rate(..)`, `.expected_returns(..)` and `.covariance(..)`"
    )]
    pub fn max_sharpe_allocation(&self, config: &MaxSharpeConfig) -> Result<Weights, CutupError> {
        self.allocate(&config.method().allow_short())
    }

    /// The long-only maximum Sharpe ratio portfolio. See [`MaxSharpe`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MaxSharpe::default())`, with `.risk_free_rate(..)`, \
                `.expected_returns(..)` and `.covariance(..)`"
    )]
    pub fn long_only_max_sharpe_allocation(
        &self,
        config: &MaxSharpeConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(&config.method())
    }

    /// The maximum diversification portfolio. See [`MaxDiversification`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MaxDiversification::default())`"
    )]
    pub fn max_diversification_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&MaxDiversification::default())
    }

    /// The maximum diversification portfolio of the adjusted covariance. See
    /// [`MaxDiversification`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&MaxDiversification::default().covariance(config))`"
    )]
    pub fn max_diversification_allocation_with_config(
        &self,
        config: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(&MaxDiversification::default().covariance(config.clone()))
    }

    /// Equal weights, `1/n` each. See [`EqualWeight`](crate::EqualWeight).
    #[deprecated(since = "0.2.8", note = "use `allocator.allocate(&EqualWeight)`")]
    pub fn ew_allocation(&self) -> Weights {
        self.equal_weight()
    }

    /// Inverse volatility weights. See [`InverseVolatility`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&InverseVolatility::default())`"
    )]
    pub fn inverse_volatility_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&InverseVolatility::default())
    }

    /// Inverse variance weights. See [`InverseVariance`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&InverseVariance::default())`"
    )]
    pub fn inverse_variance_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&InverseVariance::default())
    }

    /// The risk parity portfolio. See [`RiskParity`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&RiskParity::default())`"
    )]
    pub fn risk_parity_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&RiskParity::default())
    }

    /// The risk budgeting portfolio. See [`RiskParity`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&RiskParity::default())`, with `.risk_budgets(..)` and \
                `.covariance(..)`"
    )]
    pub fn risk_parity_allocation_with_config(
        &self,
        config: &RiskParityConfig,
    ) -> Result<Weights, CutupError> {
        self.allocate(&config.method())
    }

    /// The order HRP puts the assets in. See [`Hrp::cluster_order`].
    #[deprecated(
        since = "0.2.8",
        note = "use `Hrp::default().linkage(..).cluster_order(&allocator)`"
    )]
    pub fn hrp_cluster_order(&self, config: &HrpConfig) -> Result<Vec<usize>, CutupError> {
        Hrp::default().linkage(config.linkage).cluster_order(self)
    }

    /// Hierarchical Risk Parity, with single linkage. See [`Hrp`].
    #[deprecated(since = "0.2.8", note = "use `allocator.allocate(&Hrp::default())`")]
    pub fn hrp_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&Hrp::default())
    }

    /// Hierarchical Risk Parity. See [`Hrp`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&Hrp::default().linkage(..))`"
    )]
    pub fn hrp_allocation_with_config(&self, config: &HrpConfig) -> Result<Weights, CutupError> {
        self.allocate(&Hrp::default().linkage(config.linkage))
    }

    /// Hierarchical Equal Risk Contribution, with riskfolio-lib's defaults. See [`Herc`].
    #[deprecated(since = "0.2.8", note = "use `allocator.allocate(&Herc::default())`")]
    pub fn herc_allocation(&self) -> Result<Weights, CutupError> {
        self.allocate(&Herc::default())
    }

    /// Hierarchical Equal Risk Contribution. See [`Herc`].
    #[deprecated(
        since = "0.2.8",
        note = "use `allocator.allocate(&Herc::default())`, with `.linkage(..)`, \
                `.clusters(..)` and `.max_clusters(..)`"
    )]
    pub fn herc_allocation_with_config(&self, config: &HercConfig) -> Result<Weights, CutupError> {
        self.allocate(&config.method())
    }
}

/// Runs a global minimum variance allocation, with short positions allowed, on a matrix of
/// prices.
///
/// # Errors
///
/// * Any error from [`PortfolioAllocator::from_prices`] or [`MinVariance`].
#[deprecated(
    since = "0.2.8",
    note = "use `PortfolioAllocator::from_prices(prices)?\
            .allocate(&MinVariance::default().allow_short())`"
)]
pub fn run_portfolio_allocation(prices: DMatrix<f64>) -> Result<Weights, CutupError> {
    PortfolioAllocator::from_prices(prices)?.allocate(&MinVariance::default().allow_short())
}
