//! [`MinVariance`]: the minimum variance portfolio.

use crate::turnover::Trading;
use crate::{
    Allocation, CovarianceConfig, CutupError, PerAsset, Returns, WeightBounds, Weights,
    interior_start, long_only_weights, qp, turnover,
};
use nalgebra::{DMatrix, DVector};

/// The minimum variance portfolio: minimize `wᵀΣw` subject to `1ᵀw = 1`. Long-only by
/// default.
///
/// It uses only the covariance matrix, not expected returns. For a portfolio that also
/// accounts for them, use [`MeanVariance`](crate::MeanVariance).
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
/// Transaction costs are in units of return, so they only apply to
/// [`MeanVariance`](crate::MeanVariance), whose objective is too. For min variance with costs, use
/// mean-variance with zero expected returns:
/// `MeanVariance::risk_aversion(γ).expected_returns(vec![0.0; n]).transaction_costs(c)` minimizes
/// `(γ/2) wᵀΣw + Σcᵢ|wᵢ − w₀ᵢ|`, where `γ` is the rate at which you trade variance for cost.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if `regularization` is negative or not finite, or
///   `shrinkage` is outside `[0, 1]`.
/// * [`CutupError::InvalidAssetValue`] if an asset's lower bound is negative or not finite, or
///   its upper bound is below its lower bound or NaN. For [uniform](WeightBounds::uniform)
///   bounds, it's [`CutupError::InvalidConfig`].
/// * [`CutupError::DimensionMismatch`] if per-asset bounds don't have one entry per asset.
/// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if a setting names an
///   asset that isn't in the data, or names one twice.
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
/// * [`CutupError::InvalidConfig`] if the current weights don't sum to 1 within `1e-9`, or
///   the cap or penalty is negative or not finite.
/// * [`CutupError::InvalidAssetValue`] if a current weight isn't finite.
/// * [`CutupError::DimensionMismatch`] if there isn't one current weight per asset.
/// * [`CutupError::InfeasibleTurnover`] if the cap is below the turnover needed to move the
///   current weights into the bounds.
///
/// # Example
///
/// ```
/// use cutup::{MinVariance, Returns, WeightBounds};
///
/// // The second asset moves with the first but twice as far, so the long-short minimum
/// // variance portfolio shorts it as a hedge.
/// let returns = [
///      [ 0.010,  0.021,  0.004],
///      [-0.008, -0.015, -0.002],
///      [ 0.004,  0.009,  0.006],
///      [-0.002, -0.006, -0.003],
///      [ 0.006,  0.011,  0.001],
/// ];
/// let data = Returns::new(&returns)?;
/// assert!(data.allocate(&MinVariance::default().allow_short())?[1] < 0.0);
///
/// let weights = data.allocate(&MinVariance::default())?;
/// assert_eq!(weights[1], 0.0);
/// assert!(weights[2] > 0.9);
///
/// // Capping every asset at 50% moves the rest of the weight to the next least risky.
/// let capped = data.allocate(&MinVariance::default().bounds(WeightBounds::uniform(0.0, 0.5)))?;
/// assert!((capped[2] - 0.5).abs() < 1e-12);
///
/// // Full shrinkage leaves no correlations to hedge, so every asset is held equally.
/// let shrunk = data.allocate(&MinVariance::default().shrinkage(1.0))?;
/// assert!(shrunk.iter().all(|w| (w - 1.0 / 3.0).abs() < 1e-12));
///
/// // Rebalance from the current weights, turning over at most 10%.
/// let rebalanced = data.allocate(
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
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        if self.trading.first_setting().is_some() {
            let cov = data.adjusted_covariance(&self.covariance)?;
            let n = cov.ncols();
            return turnover::rebalance(
                data,
                cov,
                DVector::zeros(n),
                self.bounds.as_ref(),
                &self.trading,
                1.0,
            );
        }
        self.trading.check(data.labels())?;
        match &self.bounds {
            Some(bounds) => data.bounded_min_variance(&self.covariance, bounds),
            None => data.min_variance(&self.covariance),
        }
    }
}

impl Returns {
    /// The minimum variance portfolio with short positions allowed, for [`MinVariance`].
    fn min_variance(&self, covariance: &CovarianceConfig) -> Result<Weights, CutupError> {
        let cov = self.adjusted_covariance(covariance)?;
        let n = cov.ncols();
        let ones = DMatrix::from_element(n, 1, 1.0);

        // Variance is bounded below and full investment is always feasible, so an optimum
        // exists; failing to find one is numerical.
        let weights = qp::solve_kkt(
            &cov,
            &DVector::zeros(n),
            &ones,
            &DVector::from_element(1, 1.0),
        )?
        .ok_or(CutupError::SingularCovariance)?
        .0;
        Ok(self.weights(weights.iter().copied().collect()))
    }

    /// The minimum variance portfolio within `bounds`, for [`MinVariance`].
    fn bounded_min_variance(
        &self,
        covariance: &CovarianceConfig,
        bounds: &WeightBounds,
    ) -> Result<Weights, CutupError> {
        let bounds = bounds.resolve(&self.labels)?;
        let cov = self.adjusted_covariance(covariance)?;
        let n = cov.ncols();
        // From a point strictly inside the bounds, the solver's first step is to the
        // unconstrained minimum variance portfolio. With the default bounds it's equal weights.
        let start = interior_start(&bounds);
        long_only_weights(
            &cov,
            &DVector::zeros(n),
            &DMatrix::from_element(n, 1, 1.0),
            &DVector::from_element(1, 1.0),
            &bounds,
            start,
        )
        .map(|w| self.weights(w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EqualWeight;
    use crate::test_support::{
        assert_weights_close, collinear_returns, full_rank_prices, portfolio_variance,
        rank_one_returns,
    };
    use nalgebra::dmatrix;

    #[test]
    fn test_long_short_min_variance() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert_eq!(weights.len(), 3);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_variance_errors_on_non_finite_covariance() {
        let mut allocator = Returns::prices_matrix(full_rank_prices()).unwrap();

        // Unreachable through the constructors, which reject non-finite input; this guards the
        // pseudo-inverse, whose SVD would otherwise never converge.
        allocator.cov_matrix = dmatrix![
            f64::NAN, 0.0, 0.0;
            0.0, 1.0, 0.0;
            0.0, 0.0, 1.0
        ];
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::NonFiniteCovariance)
        );

        allocator.cov_matrix = DMatrix::from_element(3, 3, f64::NAN);
        assert_eq!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .err(),
            Some(CutupError::NonFiniteCovariance)
        );
    }

    #[test]
    fn test_min_variance_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let cases = [
            (Some(-1e-3), None, "regularization"),
            (Some(f64::NAN), None, "regularization"),
            (Some(f64::INFINITY), None, "regularization"),
            (None, Some(-0.1), "shrinkage"),
            (None, Some(1.5), "shrinkage"),
            (None, Some(f64::NAN), "shrinkage"),
        ];
        for (regularization, shrinkage, field) in cases {
            let config = CovarianceConfig {
                regularization,
                shrinkage,
            };
            match allocator.allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(config.clone()),
            ) {
                Err(CutupError::InvalidConfig { field: f, .. }) => assert_eq!(f, field),
                other => panic!("{config:?} gave {other:?}"),
            }
        }

        let edges = CovarianceConfig {
            regularization: Some(0.0),
            shrinkage: Some(1.0),
        };
        assert!(
            allocator
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(edges.clone())
                )
                .is_ok()
        );
    }

    #[test]
    fn test_min_variance_diagonal_closed_form() {
        // Mutually orthogonal, zero-mean columns give a diagonal covariance, so the minimum
        // variance weights are proportional to 1 / σᵢ².
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        let inverse = [1.0 / (s1 * s1), 1.0 / (s2 * s2), 1.0 / (s3 * s3)];
        let total: f64 = inverse.iter().sum();
        let expected = Weights::from(inverse.iter().map(|x| x / total).collect::<Vec<_>>());
        assert_weights_close(&weights, &expected, 1e-12);
    }

    #[test]
    fn test_min_variance_two_asset_closed_form() {
        let prices = full_rank_prices().columns(0, 2).into_owned();
        let allocator = Returns::prices_matrix(prices).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        let cov = &allocator.cov_matrix;
        let (s11, s22, s12) = (cov[(0, 0)], cov[(1, 1)], cov[(0, 1)]);
        let w1 = (s22 - s12) / (s11 + s22 - 2.0 * s12);
        assert_weights_close(&weights, &Weights::from(vec![w1, 1.0 - w1]), 1e-12);
    }

    #[test]
    fn test_min_variance_beats_equal_weight() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let min_variance = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        let equal = allocator.allocate(&EqualWeight).unwrap();
        assert!(
            portfolio_variance(&allocator, &min_variance) <= portfolio_variance(&allocator, &equal)
        );
    }

    #[test]
    fn test_min_variance_rank_deficient_covariance() {
        // Three rows of returns for four assets: Σ has rank at most 2. The pseudo-inverse must
        // still give finite weights that sum to 1 and do no worse than equal weight.
        let prices = dmatrix![
            125.0, 1500.0, 210.0, 600.0;
            123.0, 1520.0, 215.0, 620.0;
            130.0, 1510.0, 220.0, 610.0;
            128.0, 1530.0, 225.0, 630.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();

        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(weights.iter().all(|w| w.is_finite() && w.abs() < 10.0));
        assert!(
            portfolio_variance(&allocator, &weights)
                <= portfolio_variance(&allocator, &allocator.allocate(&EqualWeight).unwrap())
                    + 1e-15
        );
    }

    #[test]
    fn test_min_variance_finds_zero_variance_portfolio_on_singular_covariance() {
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_min_variance_anti_correlated_pair() {
        // 1 lies in Σ's null space, so 1ᵀΣ⁺1 is rounding noise; the hedged pair has zero variance.
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let weights = allocator
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.5]), 1e-12);
    }

    #[test]
    fn test_min_variance_survives_inaccurate_svd() {
        // Heavily shrunk, this covariance is close to a multiple of I. nalgebra's SVD with a
        // convergence tolerance of exactly ε reconstructed its KKT matrix only to about 1e-2,
        // and the solve was rejected as inconsistent.
        let returns = DMatrix::from_vec(
            10,
            5,
            vec![
                0.014883239895260967,
                -0.01493530508682466,
                0.016430855321270307,
                -0.013476652725590775,
                0.048376573241446175,
                -0.01454922384002048,
                0.013863889151550382,
                0.03060490436889639,
                0.005947862449558874,
                0.007242902806749439,
                -0.049791777589274035,
                -0.010276099234288459,
                0.03787642939756265,
                -0.01009095345402214,
                0.039051035567899794,
                0.04379567627495853,
                -0.014218203192808818,
                -0.0009414262734083997,
                0.04410238107402796,
                -0.0023072042163211695,
                -0.036553907215322874,
                -0.0496177694619739,
                0.033432410527783465,
                -0.04522593873754994,
                -0.013996807386067985,
                0.04641859793989761,
                -0.019811327585646278,
                0.014370795975672165,
                -0.044680407876113194,
                -0.03280219001859793,
                -0.03639028312023631,
                0.03263436706874201,
                0.03115846887684503,
                -0.0029601214724074795,
                0.0035616215605822,
                0.02380399736864574,
                -0.02372734365258026,
                0.012992821815841767,
                0.012965514435532502,
                0.042600843138559705,
                -0.024029906325403177,
                0.0,
                0.04784068605333592,
                0.005896517391334338,
                -0.01686648883472212,
                0.048269384986887494,
                0.03695469685973938,
                0.015311227622385607,
                -0.03242117667024552,
                0.034392821320402465,
            ],
        );
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = CovarianceConfig::default().shrinkage(0.9832865013278819);
        let weights = allocator
            .allocate(
                &MinVariance::default()
                    .allow_short()
                    .covariance(config.clone()),
            )
            .unwrap();
        let cov = allocator.adjusted_covariance(&config).unwrap();
        let marginal = &cov * DVector::from_column_slice(weights.as_slice());
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-14);
        assert!(marginal.max() - marginal.min() < 1e-14 * marginal.amax());
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().covariance(config.clone()))
                .unwrap(),
            &weights,
            1e-14,
        );
    }

    #[test]
    fn test_min_variance_on_rank_deficient_kkt() {
        // Three rows for four assets give Σ rank 2, and its KKT matrix a zero singular value.
        // nalgebra's SVD left a residual of about 4e-7 here, so the solve was rejected as
        // inconsistent; NumPy's solves it to rounding.
        let returns = DMatrix::from_vec(
            3,
            4,
            vec![
                0.03923072979353582,
                -0.027034431663761255,
                -0.02017662174798146,
                -0.030105448299475653,
                0.02262405031858968,
                0.01653614746213769,
                -0.009248234151376263,
                0.02095233950501841,
                -0.049975234461268725,
                -0.010174824524215684,
                -0.043910804533156975,
                0.04613336884944048,
            ],
        );
        let allocator = Returns::returns_matrix(returns).unwrap();
        // numpy.linalg.pinv's minimum-norm solution of the same KKT system, in correlation
        // units u = diag(σ)w.
        let expected = Weights::from(vec![0.275093649, 0.331571165, 0.219203116, 0.174132070]);
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            &expected,
            1e-9,
        );
        let weights = allocator.allocate(&MinVariance::default()).unwrap();
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);
    }

    #[test]
    fn test_long_only_min_variance_matches_unconstrained_when_long() {
        // A diagonal covariance gives all-positive minimum variance weights, so the
        // constraint is inactive.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&MinVariance::default()).unwrap(),
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_min_variance_drops_the_hedge() {
        // Unconstrained, (2, −1) hedges the collinear pair to zero variance. Long-only, any mix
        // of the two has more variance than the first asset alone.
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();
        assert!(
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap()[1]
                < 0.0
        );
        assert_eq!(
            allocator.allocate(&MinVariance::default()).unwrap(),
            Weights::from(vec![1.0, 0.0])
        );
    }

    #[test]
    fn test_long_only_min_variance_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, so the minimum is zero.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let weights = allocator.allocate(&MinVariance::default()).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(portfolio_variance(&allocator, &weights) < 1e-18);

        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.03, -0.03];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_weights_close(
            &allocator.allocate(&MinVariance::default()).unwrap(),
            &Weights::from(vec![0.5, 0.5]),
            1e-12,
        );
    }

    #[test]
    fn test_long_only_min_variance_is_optimal_on_short_histories() {
        // Fewer rows than assets make Σ singular, so the free problems have many minima. The
        // solver must still reach a point where no asset's marginal variance (Σw)ᵢ is below
        // the portfolio's variance wᵀΣw.
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 0.1 - 0.05
        };
        for case in 0..300 {
            let assets = 4 + case % 9;
            let rows = 2 + case % (assets - 1);
            let returns = DMatrix::from_fn(rows, assets, |_, _| next());
            let allocator = Returns::returns_matrix(returns).unwrap();
            let weights = allocator.allocate(&MinVariance::default()).unwrap();
            let w = DVector::from_column_slice(weights.as_slice());
            let marginal = &allocator.cov_matrix * &w;
            let v = w.dot(&marginal);
            let tolerance = 1e-9 * allocator.cov_matrix.amax();
            assert!(w.iter().all(|&x| x >= 0.0), "case {case}: {weights:?}");
            assert!(
                marginal.iter().all(|&m| m >= v - tolerance),
                "case {case}: {marginal:?} vs {v}"
            );
        }
    }

    #[test]
    fn test_long_only_min_variance_rejects_bad_input() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = CovarianceConfig::default().shrinkage(1.5);
        assert_eq!(
            allocator.allocate(&MinVariance::default().covariance(config.clone())),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 1.5
            })
        );

        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&MinVariance::default()),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
    }

    #[test]
    fn test_min_variance_with_covariance_variants() {
        let prices = dmatrix![
            100.0, 200.0, 300.0;
            110.0, 210.0, 310.0;
            120.0, 220.0, 320.0
        ];
        let allocator = Returns::prices_matrix(prices).unwrap();

        let configs = [
            // 1. No regularization or shrinkage
            CovarianceConfig::default(),
            // 2. Regularization only
            CovarianceConfig {
                regularization: Some(1e-6),
                shrinkage: None,
            },
            // 3. Shrinkage only
            CovarianceConfig {
                regularization: None,
                shrinkage: Some(0.1),
            },
            // 4. Both regularization and shrinkage
            CovarianceConfig {
                regularization: Some(1e-6),
                shrinkage: Some(0.2),
            },
        ];

        for config in &configs {
            let weights = allocator
                .allocate(
                    &MinVariance::default()
                        .allow_short()
                        .covariance(config.clone()),
                )
                .unwrap();
            assert_eq!(weights.len(), 3, "{config:?}");
            assert!(
                (weights.iter().sum::<f64>() - 1.0).abs() < 1e-6,
                "{config:?}"
            );
        }
    }
}
