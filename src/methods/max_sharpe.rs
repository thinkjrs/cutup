//! [`MaxSharpe`]: the maximum Sharpe ratio portfolio.

use super::{correlation, long_only_max_ratio};
use crate::eigen::SymmetricEigen;
use crate::{Allocation, CovarianceConfig, CutupError, PerAsset, Returns, Weights};
use nalgebra::DVector;

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
/// is solved exactly by the same active-set method as [`MinVariance`](crate::MinVariance).
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
///   minimum-norm optimum in correlation units, as for [`MinVariance`](crate::MinVariance).
///
/// The null space is that of the correlation matrix `C`, with `Σ = diag(σ) C diag(σ)`, so the
/// outcome doesn't depend on each asset's units.
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the risk-free rate is not finite, or the covariance
///   settings are invalid.
/// * [`CutupError::InvalidAssetValue`] if an expected return is not finite.
/// * [`CutupError::DimensionMismatch`] if the expected returns don't have one entry per asset.
/// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if a setting names an
///   asset that isn't in the data, or names one twice.
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
/// use cutup::{MaxSharpe, Returns};
///
/// // Uncorrelated assets: the tangency weights are proportional to (μᵢ − r_f) / σᵢ².
/// let returns = [
///      [ 0.01,  0.02],
///      [-0.01, -0.02],
///      [ 0.01, -0.02],
///      [-0.01,  0.02],
/// ];
/// let data = Returns::new(&returns)?;
///
/// // σ² is in the ratio 1 : 4 and μ in the ratio 1 : 2, so the weights are 2 : 1.
/// let weights = data.allocate(&MaxSharpe::default().expected_returns([0.002, 0.004]))?;
/// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
/// assert!((weights[1] - 1.0 / 3.0).abs() < 1e-12);
///
/// // With a negative expected return, the tangency portfolio shorts the second asset, which
/// // long-only can't.
/// let method = MaxSharpe::default().expected_returns([0.002, -0.001]);
/// assert!(data.allocate(&method.clone().allow_short())?[1] < 0.0);
/// let weights = data.allocate(&method)?;
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
    expected_returns: Option<PerAsset>,
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

    /// Uses `expected_returns`, [one per asset](PerAsset) in column order or by asset, instead
    /// of the sample mean of the returns. They're in the same units and period as the returns.
    pub fn expected_returns(mut self, expected_returns: impl Into<PerAsset>) -> Self {
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
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let mu = data.resolve("expected_returns", self.expected_returns.as_ref())?;
        if self.long_only {
            data.long_only_max_sharpe(self.risk_free_rate, mu.as_ref(), &self.covariance)
        } else {
            data.max_sharpe(self.risk_free_rate, mu.as_ref(), &self.covariance)
        }
    }
}

impl Returns {
    /// The maximum Sharpe ratio portfolio with short positions allowed, for [`MaxSharpe`].
    fn max_sharpe(
        &self,
        risk_free_rate: f64,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        if !risk_free_rate.is_finite() {
            return Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: risk_free_rate,
            });
        }
        let mu = self.expected_returns(expected_returns)?;
        let cov = self.adjusted_covariance(covariance)?;
        let n = cov.ncols();
        let ones = DVector::from_element(n, 1.0);
        let excess = mu.add_scalar(-risk_free_rate);
        let too_high = |min_variance_return: f64| CutupError::RiskFreeRateTooHigh {
            risk_free_rate,
            min_variance_return,
        };

        // Decompose the correlation matrix C into its range and its null space, with
        // numpy.linalg.pinv's cutoff. In units u = Dw, with D = diag(σ), the variance is uᵀCu,
        // the budget (D⁻¹1)ᵀu and the excess return (D⁻¹(μ − r_f))ᵀu.
        let (correlation, volatilities) = correlation(&cov)?;
        // C is positive semidefinite, so its eigenvalues are its singular values.
        let eigen = SymmetricEigen::new(&correlation).ok_or(CutupError::SingularCovariance)?;
        let (range, null) = eigen.split(eigen.values.amax() * n as f64 * f64::EPSILON);
        let budget = ones.component_div(&volatilities);
        let earned = excess.component_div(&volatilities);

        if !null.is_empty() {
            // Coordinates of the budget and the excess returns in the null space.
            let a = DVector::from_iterator(null.len(), null.iter().map(|v| v.dot(&budget)));
            let b = DVector::from_iterator(null.len(), null.iter().map(|v| v.dot(&earned)));
            let noise = 1e-9 * earned.norm().max(f64::MIN_POSITIVE);
            if a.norm() > 1e-9 * budget.norm() {
                // A fully invested zero-variance portfolio exists, w0 ∝ D⁻¹ Σ aₖvₖ, earning
                // excess return b·a / a·a. Any part of b not along a is a budget-neutral,
                // variance-free change that adds return.
                let along = b.dot(&a) / a.dot(&a);
                if (&b - &a * along).norm() > noise || along * a.norm() > noise {
                    return Err(CutupError::Unbounded);
                }
                return Err(too_high(risk_free_rate + along));
            }
            if b.norm() > noise {
                return Err(CutupError::Unbounded);
            }
        }

        // Σ⁺x = D⁻¹C⁺D⁻¹x restricted to the range, where every remaining solution lives.
        let pinv = |x: &DVector<f64>| {
            let x = x.component_div(&volatilities);
            range
                .iter()
                .fold(DVector::zeros(n), |acc, (sigma, v)| {
                    acc + v * (v.dot(&x) / sigma)
                })
                .component_div(&volatilities)
        };
        let inv_ones = pinv(&ones);
        let min_variance_return = inv_ones.dot(&mu) / inv_ones.sum();
        // The rounding in r_mvp comes from its terms wᵢμᵢ, which don't change when an asset is
        // rescaled.
        let terms = inv_ones.component_mul(&mu).abs().sum() / inv_ones.sum().abs();
        let tolerance = 1e-10 * terms;
        if risk_free_rate >= min_variance_return - tolerance {
            return Err(too_high(min_variance_return));
        }

        let z = pinv(&excess);
        let weights = &z / z.sum();
        if weights.iter().any(|w| !w.is_finite()) {
            return Err(CutupError::SingularCovariance);
        }
        Ok(self.weights(weights.iter().copied().collect()))
    }

    /// The long-only maximum Sharpe ratio portfolio, for [`MaxSharpe`].
    fn long_only_max_sharpe(
        &self,
        risk_free_rate: f64,
        expected_returns: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        if !risk_free_rate.is_finite() {
            return Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: risk_free_rate,
            });
        }
        let mu = self.expected_returns(expected_returns)?;
        // Excess returns relative to the largest magnitude, so that neither they nor the
        // tolerance can overflow.
        let (best, highest_return) = mu.argmax();
        let scale = risk_free_rate.abs().max(mu.amax());
        let excess = mu.map(|m| m / scale - risk_free_rate / scale);
        // An excess return counts only beyond the rounding of its own terms, so that rescaling
        // one asset's returns doesn't change whether it has one. A zero scale means every
        // return and the rate are zero.
        let positive = |i: usize| excess[i] > 1e-10 * risk_free_rate.abs().max(mu[i].abs()) / scale;
        if scale == 0.0 || !(0..mu.len()).any(positive) {
            return Err(CutupError::NoPositiveExcessReturn {
                risk_free_rate,
                highest_return,
            });
        }
        let cov = self.adjusted_covariance(covariance)?;
        // Scaled so that the best asset's excess return is 1.
        long_only_max_ratio(&cov, &(&excess / excess[best]), best)?
            .map(|w| self.weights(w))
            .ok_or(CutupError::Unbounded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        assert_weights_close, collinear_returns, full_rank_prices, min_variance_return,
        rank_one_returns, sample_mean, sharpe,
    };
    use crate::{EqualWeight, Hrp, InverseVariance, InverseVolatility, MeanVariance, MinVariance};
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_long_only_max_sharpe_matches_unconstrained_when_long() {
        // Uncorrelated assets with positive excess returns: the tangency weights,
        // proportional to (μᵢ − r_f) / σᵢ², are all positive.
        let (s1, s2, s3) = (0.01, 0.02, 0.04);
        let returns = dmatrix![
            s1, s2, s3;
            -s1, s2, -s3;
            s1, -s2, -s3;
            -s1, -s2, s3
        ];
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = MaxSharpe::default()
            .risk_free_rate(0.0005)
            .expected_returns(dvector![0.001, 0.002, 0.003].as_slice());
        let unconstrained = allocator.allocate(&config.clone().allow_short()).unwrap();
        assert!(unconstrained.iter().all(|&w| w > 0.0), "{unconstrained:?}");
        assert_weights_close(&allocator.allocate(&config).unwrap(), &unconstrained, 1e-12);
    }

    #[test]
    fn test_long_only_max_sharpe_risk_free_rate_limits() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let expected = dvector![0.001, 0.003, 0.002];
        let at = |rate: f64| {
            let config = MaxSharpe::default()
                .risk_free_rate(rate)
                .expected_returns(expected.as_slice());
            (
                allocator.allocate(&config.clone().allow_short()),
                allocator.allocate(&config),
            )
        };

        // Just below the highest expected return, the unconstrained maximum doesn't exist,
        // but the long-only one does, and holds only that asset.
        let (unconstrained, long_only) = at(0.0029);
        assert!(matches!(
            unconstrained,
            Err(CutupError::RiskFreeRateTooHigh { .. })
        ));
        assert_weights_close(
            &long_only.unwrap(),
            &Weights::from(vec![0.0, 1.0, 0.0]),
            1e-9,
        );

        for rate in [0.003, 0.003 * (1.0 + 1e-12), 0.01] {
            assert!(matches!(
                at(rate).1,
                Err(CutupError::NoPositiveExcessReturn {
                    highest_return: 0.003,
                    ..
                })
            ));
        }
    }

    #[test]
    fn test_long_only_max_sharpe_extreme_returns() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let at = |rate: f64, expected: DVector<f64>| {
            let config = MaxSharpe::default()
                .risk_free_rate(rate)
                .expected_returns(expected.as_slice());
            allocator.allocate(&config)
        };

        // A subnormal excess return mustn't make the starting point infinite. With a zero
        // rate the weights don't depend on the scale of μ.
        assert_weights_close(
            &at(0.0, dvector![1e-310, 0.0, 0.0]).unwrap(),
            &at(0.0, dvector![1e-3, 0.0, 0.0]).unwrap(),
            1e-12,
        );

        // Neither the tolerance nor the excess returns may overflow.
        let weights = at(-1.7e308, dvector![1.7e308, 1.0, 1.0]).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        // Every return and the rate zero: no excess return at all.
        assert!(matches!(
            at(0.0, DVector::zeros(3)),
            Err(CutupError::NoPositiveExcessReturn { .. })
        ));
    }

    #[test]
    fn test_long_only_max_sharpe_unbounded_on_singular_covariance() {
        // [4/7, 0, 3/7] is long-only with zero variance, and with these expected returns its
        // excess return is positive, so the Sharpe ratio grows without limit.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        let config = MaxSharpe::default().expected_returns(dvector![0.01, 0.0, 0.01].as_slice());
        assert_eq!(allocator.allocate(&config), Err(CutupError::Unbounded));

        // Shrinkage gives every portfolio some variance.
        let config = config.covariance(CovarianceConfig::default().shrinkage(0.3));
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights.iter().all(|&w| w >= 0.0), "{weights:?}");
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_long_only_max_sharpe_rejects_bad_input() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let config = MaxSharpe::default().risk_free_rate(f64::NAN);
        assert!(matches!(
            allocator.allocate(&config),
            Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                ..
            })
        ));
        let config = MaxSharpe::default().expected_returns(dvector![0.01].as_slice());
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
    fn test_max_sharpe_closed_form() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let weights = allocator.allocate(&config.clone().allow_short()).unwrap();

        let excess = sample_mean(&allocator).add_scalar(-risk_free_rate);
        let z = allocator.cov_matrix.clone().try_inverse().unwrap() * excess;
        let expected = Weights::from((&z / z.sum()).iter().copied().collect::<Vec<_>>());
        assert_weights_close(&weights, &expected, 1e-10);
    }

    #[test]
    fn test_max_sharpe_beats_every_other_method() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
        let best = sharpe(
            &allocator,
            &allocator.allocate(&config.clone().allow_short()).unwrap(),
            risk_free_rate,
        );

        let others = [
            allocator.allocate(&EqualWeight).unwrap(),
            allocator.allocate(&InverseVolatility).unwrap(),
            allocator.allocate(&InverseVariance).unwrap(),
            allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            allocator.allocate(&Hrp::default()).unwrap(),
            allocator
                .allocate(&MeanVariance::risk_aversion(50.0).allow_short())
                .unwrap(),
        ];
        for weights in &others {
            assert!(sharpe(&allocator, weights, risk_free_rate) <= best + 1e-12);
        }
    }

    #[test]
    fn test_max_sharpe_lies_on_the_efficient_frontier() {
        // The tangency portfolio is the minimum variance portfolio for its own return.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let risk_free_rate = min_variance_return(&allocator) - 0.002;
        let tangency = allocator
            .allocate(
                &MaxSharpe::default()
                    .risk_free_rate(risk_free_rate)
                    .allow_short(),
            )
            .unwrap();
        let target = sample_mean(&allocator).dot(&DVector::from_column_slice(tangency.as_slice()));
        let frontier = allocator
            .allocate(&MeanVariance::target_return(target).allow_short())
            .unwrap();
        assert_weights_close(&tangency, &frontier, 1e-9);
    }

    #[test]
    fn test_max_sharpe_rejects_risk_free_rate_at_or_above_min_variance_return() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let threshold = min_variance_return(&allocator);
        for risk_free_rate in [threshold, threshold + 0.001] {
            let config = MaxSharpe::default().risk_free_rate(risk_free_rate);
            match allocator.allocate(&config.clone().allow_short()) {
                Err(CutupError::RiskFreeRateTooHigh {
                    risk_free_rate: got,
                    min_variance_return,
                }) => {
                    assert_eq!(got, risk_free_rate);
                    assert!((min_variance_return - threshold).abs() < 1e-12);
                }
                other => panic!("risk-free rate {risk_free_rate} gave {other:?}"),
            }
        }
    }

    #[test]
    fn test_max_sharpe_singular_covariance() {
        // The sample mean isn't in Σ's range, so some zero-variance portfolio earns excess
        // return and the Sharpe ratio has no maximum. Shrinkage restores one.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert_eq!(
            allocator
                .allocate(&MaxSharpe::default().allow_short())
                .err(),
            Some(CutupError::Unbounded)
        );

        let shrunk = MaxSharpe::default()
            .risk_free_rate(-0.01)
            .covariance(CovarianceConfig::default().shrinkage(0.5));
        let weights = allocator.allocate(&shrunk.clone().allow_short()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_max_sharpe_collinear_assets() {
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();

        // The zero-variance portfolio is the minimum variance portfolio.
        assert_weights_close(
            &allocator
                .allocate(&MinVariance::default().allow_short())
                .unwrap(),
            &Weights::from(vec![2.0, -1.0]),
            1e-9,
        );

        // It returns −0.001, below a risk-free rate of 0: no maximum, and the threshold reported
        // is its return.
        match allocator.allocate(&MaxSharpe::default().allow_short()) {
            Err(CutupError::RiskFreeRateTooHigh {
                min_variance_return,
                ..
            }) => assert!((min_variance_return + 0.001).abs() < 1e-12),
            other => panic!("expected RiskFreeRateTooHigh, got {other:?}"),
        }

        // Below its return, the zero-variance portfolio has excess return and no risk, so the
        // Sharpe ratio is unbounded.
        let config = MaxSharpe::default().risk_free_rate(-0.01);
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::Unbounded)
        );
    }

    #[test]
    fn test_max_sharpe_just_below_the_threshold() {
        // A rate clearly below the threshold (by far more than the rounding tolerance) has a
        // maximum, however leveraged.
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let threshold = min_variance_return(&allocator);
        let config = MaxSharpe::default().risk_free_rate(threshold - 1e-6);
        let weights = allocator.allocate(&config.clone().allow_short()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_sharpe_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        match allocator.allocate(&MaxSharpe::default().risk_free_rate(f64::NAN).allow_short()) {
            Err(CutupError::InvalidConfig { field, .. }) => assert_eq!(field, "risk_free_rate"),
            other => panic!("NaN risk-free rate gave {other:?}"),
        }
        let config =
            MaxSharpe::default().expected_returns(DVector::from_vec(vec![0.01]).as_slice());
        assert_eq!(
            allocator.allocate(&config.clone().allow_short()).err(),
            Some(CutupError::DimensionMismatch {
                field: "expected_returns",
                expected: 3,
                actual: 1
            })
        );

        let constant = Returns::returns_matrix(dmatrix![0.1, 0.01; 0.1, -0.02; 0.1, 0.03]).unwrap();
        assert_eq!(
            constant.allocate(&MaxSharpe::default().allow_short()).err(),
            Some(CutupError::ZeroVariance {
                column: 0,
                asset: "0".into()
            })
        );
    }
}
