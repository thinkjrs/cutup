//! [`MeanVariance`]: Markowitz mean-variance optimization.

use super::long_only_weights;
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
/// With a target return, it minimizes the variance at that return with the turnover capped.
/// Only the cap works there: the objective is a variance, so neither costs nor the penalty,
/// both in the units of the returns, has anything to be traded off against, and they return
/// [`CutupError::NotSupported`].
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the risk aversion is not finite and positive, the
///   target return is not finite, or the covariance settings are invalid.
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
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9`, or
///   the cap, penalty or a uniform cost is negative or not finite.
/// * [`CutupError::InvalidAssetValue`] if a current weight isn't finite, or a per-asset cost
///   is negative or not finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight, or per-asset cost,
///   per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds, and to the target return if there is one.
/// * [`CutupError::NotSupported`] for a penalty or costs with a target return.
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
        if self.trading.first_setting().is_some() {
            check_objective(self.objective)?;
            let covariance = || data.adjusted_covariance(&self.covariance);
            return match (self.objective, self.trading.first_charge()) {
                (Objective::RiskAversion(gamma), _) => {
                    let mu = data.expected_returns(expected_returns.as_ref())?;
                    let cov = covariance()? * gamma;
                    turnover::rebalance(data, cov, mu, self.bounds.as_ref(), &self.trading, 2.0)
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
                        data,
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
        let ones = DVector::from_element(n, 1.0);

        let weights = match objective {
            Objective::RiskAversion(gamma) => {
                qp::solve_kkt(
                    &(cov * gamma),
                    &mu,
                    &DMatrix::from_column_slice(n, 1, ones.as_slice()),
                    &DVector::from_element(1, 1.0),
                )?
                .ok_or(CutupError::Unbounded)?
                .0
            }
            Objective::TargetReturn(target) => {
                qp::solve_kkt(
                    &cov,
                    &DVector::zeros(n),
                    &DMatrix::from_columns(&[ones, mu]),
                    &DVector::from_vec(vec![1.0, target]),
                )?
                .ok_or(CutupError::DegenerateExpectedReturns)?
                .0
            }
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

/// What a mean-variance optimization trades off, for [`MeanVariance`].
#[derive(Debug, Clone, Copy, PartialEq)]
enum Objective {
    /// Maximize `μᵀw − (γ/2) wᵀΣw` for a risk aversion `γ > 0`.
    RiskAversion(f64),
    /// Minimize `wᵀΣw` subject to an expected portfolio return `μᵀw` equal to this value.
    TargetReturn(f64),
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
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinVariance;
    use crate::test_support::{
        assert_weights_close, full_rank_prices, portfolio_variance, rank_one_returns, sample_mean,
    };
    use nalgebra::{dmatrix, dvector};

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
        let config = MeanVariance::risk_aversion(2.0).expected_returns(flat.as_slice());
        assert_weights_close(
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );

        // And no portfolio can target any other return.
        let config = MeanVariance::target_return(0.02).expected_returns(flat.as_slice());
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
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
            allocator.allocate(&default.clone().allow_short()).unwrap(),
            allocator.allocate(&explicit.clone().allow_short()).unwrap()
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
            allocator.allocate(&shrunk.clone().allow_short()).unwrap()
        );

        let invalid = plain.covariance(CovarianceConfig::default().shrinkage(2.0));
        assert_eq!(
            allocator.allocate(&invalid.clone().allow_short()).err(),
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

        let config = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_vec(vec![0.01, f64::NAN, 0.02]).as_slice());
        match allocator.allocate(&config.clone().allow_short()) {
            Err(CutupError::InvalidAssetValue { field, column, .. }) => {
                assert_eq!((field, column), ("expected_returns", 1))
            }
            other => panic!("NaN expected return gave {other:?}"),
        }

        let config = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_vec(vec![0.01, 0.02]).as_slice());
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
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
        for config in [
            MeanVariance::risk_aversion(100.0),
            MeanVariance::target_return(0.0015),
        ] {
            let config = config.expected_returns(expected.as_slice());
            let unconstrained = allocator.allocate(&config.clone().allow_short()).unwrap();
            assert!(unconstrained.iter().all(|&w| w > 0.0), "{unconstrained:?}");
            assert_weights_close(&allocator.allocate(&config).unwrap(), &unconstrained, 1e-12);
        }
    }

    #[test]
    fn test_long_only_mean_variance_at_the_extremes() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];

        // With almost no aversion to risk, everything goes to the highest expected return.
        let config = MeanVariance::risk_aversion(1e-9).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // Only the asset with the highest expected return reaches it.
        let config = MeanVariance::target_return(0.003).expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_mean_variance_hits_the_target() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];
        for target in [0.001, 0.0012, 0.0025, 0.0029] {
            let config = MeanVariance::target_return(target).expected_returns(expected.as_slice());
            let weights = allocator.allocate(&config).unwrap();
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
        let config = MeanVariance::target_return(0.004)
            .expected_returns(dvector![0.001, 0.003, 0.002].as_slice());
        assert_eq!(
            allocator.allocate(&config),
            Err(CutupError::TargetReturnOutOfRange {
                target_return: 0.004,
                lowest: 0.001,
                highest: 0.003
            })
        );

        // With equal expected returns, only that return can be targeted, by any portfolio.
        let flat = dvector![0.002, 0.002, 0.002];
        let config = MeanVariance::target_return(0.001).expected_returns(flat.as_slice());
        assert!(matches!(
            allocator.allocate(&config),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        let config = MeanVariance::target_return(0.002).expected_returns(flat.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &allocator.allocate(&MinVariance::default()).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_mean_variance_target_edge_cases() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();

        // A target a rounding error above the highest expected return is that return.
        let expected = dvector![0.001, 0.003, 0.002];
        let config = MeanVariance::target_return(0.003 * (1.0 + 4.0 * f64::EPSILON))
            .expected_returns(expected.as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-12,
        );

        // highest − lowest overflows here, which mustn't spoil the starting point.
        let expected = dvector![-1e308, 1e308, 0.0];
        let config = MeanVariance::target_return(5e307).expected_returns(expected.as_slice());
        let weights = allocator.allocate(&config).unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        assert!(w.iter().all(|&x| x >= 0.0), "{weights:?}");
        assert!(
            (w.dot(&expected) - 5e307).abs() < 1e-12 * 5e307,
            "{weights:?}"
        );

        // An unreachable target is reported before the covariance is checked.
        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = MeanVariance::target_return(1.0);
        assert!(matches!(
            allocator.allocate(&config),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
    }

    #[test]
    fn test_long_only_mean_variance_bounded_on_singular_covariance() {
        // Unconstrained, utility grows without limit (see the next test). Long-only, the
        // weights can't grow, so there is always an optimum.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let config = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()),
            Err(CutupError::Unbounded)
        );
        let weights = allocator.allocate(&config).unwrap();
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
        let config = MeanVariance::risk_aversion(1.0).expected_returns(dvector![0.01].as_slice());
        assert_eq!(
            allocator.allocate(&config),
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
        let config = MeanVariance::risk_aversion(2.0);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = config.covariance(CovarianceConfig::default().shrinkage(0.1));
        assert!(allocator.allocate(&shrunk.clone().allow_short()).is_ok());
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
    fn test_mean_variance_target_equal_to_common_return() {
        // Every fully invested portfolio earns the common return, so the target is met by the
        // minimum variance portfolio.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = MeanVariance::target_return(0.01)
            .expected_returns(DVector::from_element(3, 0.01).as_slice());
        assert_weights_close(
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-9,
        );
    }
}
