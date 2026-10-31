//! [`RiskParity`]: equal or budgeted risk contributions.

use super::{correlation, in_null_space};
use crate::{Allocation, CovarianceConfig, CutupError, PerAsset, Returns, Weights, asset};
use nalgebra::{DMatrix, DVector};

/// The risk parity (equal risk contribution) portfolio, or risk budgeting with
/// [`RiskParity::risk_budgets`]: long-only weights where asset `i` contributes the share `bᵢ`
/// of the portfolio's variance, `wᵢ (Σw)ᵢ / wᵀΣw = bᵢ / Σⱼ bⱼ`. With equal budgets, the default,
/// every asset contributes the same share.
///
/// Unlike [`InverseVolatility`](crate::InverseVolatility), which gives equal risk contributions
/// only when assets are uncorrelated, this accounts for every correlation. The weights are the
/// unique solution, found by minimizing `½ yᵀΣy − Σᵢ bᵢ ln yᵢ` over `y > 0` with Newton's method
/// and a backtracking line search (Spinu, 2013), then normalizing `y` to sum to 1. The result
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
/// * [`CutupError::InvalidConfig`] if the covariance settings are invalid.
/// * [`CutupError::InvalidAssetValue`] if a risk budget isn't positive and finite.
/// * [`CutupError::DimensionMismatch`] if the risk budgets don't have one entry per asset.
/// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if a setting names an
///   asset that isn't in the data, or names one twice.
/// * [`CutupError::ZeroVariance`] if any asset's returns have zero variance and neither
///   shrinkage nor regularization is applied.
/// * [`CutupError::SingularCovariance`] if some long-only portfolio has zero variance.
/// * [`CutupError::NotConverged`] if the solver stops early, which takes risk budgets more
///   than about 10¹⁴ apart.
///
/// # Example
///
/// ```
/// use cutup::{Returns, RiskParity};
///
/// // Uncorrelated assets with volatilities in the ratio 1 : 2.
/// let returns = [
///      [ 0.01,  0.02],
///      [-0.01, -0.02],
///      [ 0.01, -0.02],
///      [-0.01,  0.02],
/// ];
/// let data = Returns::new(&returns)?;
///
/// // Equal budgets: weights in inverse proportion to volatility, 2 : 1.
/// let weights = data.allocate(&RiskParity::default())?;
/// assert!((weights[0] - 2.0 / 3.0).abs() < 1e-12);
///
/// // Budgets 1 : 4: wᵢ ∝ sqrt(bᵢ) / σᵢ, so 1 : 1.
/// let weights = data.allocate(&RiskParity::default().risk_budgets([1.0, 4.0]))?;
/// assert!((weights[0] - 0.5).abs() < 1e-12);
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct RiskParity {
    risk_budgets: Option<PerAsset>,
    covariance: CovarianceConfig,
}

impl RiskParity {
    /// Sets each asset's share of the portfolio's risk, [one per asset](PerAsset) in column
    /// order or by asset. Only the ratios matter, so they needn't sum to 1. Defaults to the same
    /// share for every asset.
    pub fn risk_budgets(mut self, risk_budgets: impl Into<PerAsset>) -> Self {
        self.risk_budgets = Some(risk_budgets.into());
        self
    }

    covariance_setters!();
}

impl Allocation for RiskParity {
    fn allocate(&self, data: &Returns) -> Result<Weights, CutupError> {
        let budgets = data.resolve("risk_budgets", self.risk_budgets.as_ref())?;
        data.risk_parity(budgets.as_ref(), &self.covariance)
    }
}

impl Returns {
    /// The risk budgeting portfolio, for [`RiskParity`].
    fn risk_parity(
        &self,
        risk_budgets: Option<&DVector<f64>>,
        covariance: &CovarianceConfig,
    ) -> Result<Weights, CutupError> {
        let n = self.cov_matrix.ncols();
        let budgets = match risk_budgets {
            Some(b) if b.len() != n => {
                return Err(CutupError::DimensionMismatch {
                    field: "risk_budgets",
                    expected: n,
                    actual: b.len(),
                });
            }
            Some(b) => {
                asset::check_values("risk_budgets", &self.labels, b.as_slice(), |b| {
                    b.is_finite() && b > 0.0
                })?;
                b.clone()
            }
            None => DVector::from_element(n, 1.0),
        };
        let cov = self.adjusted_covariance(covariance)?;
        Returns::risk_budget_weights(&cov, &budgets)
            .map(|w| self.weights(w.iter().copied().collect()))
    }

    /// Minimizes `f(y) = ½ yᵀΣy − bᵀ ln y` over `y > 0` and returns `y / 1ᵀy`.
    ///
    /// At the minimum, `yᵢ (Σy)ᵢ = bᵢ`, so the normalized `y` has the budgeted risk
    /// contributions. `f` is strictly convex, so Newton's method with a backtracking line
    /// search converges to it from any start. `f` is unbounded below exactly when a
    /// long-only `d` has `dᵀΣd = 0`: then `y` grows along `d`, and the normalized `y`
    /// approaches that zero-variance portfolio until the iteration breaks down. A failed
    /// iteration is classified by the variance of its last normalized `y`.
    ///
    /// The iteration runs on the correlation matrix `C`, in units `u = diag(σ)y`, where the
    /// risk contributions `uᵢ(Cu)ᵢ = yᵢ(Σy)ᵢ` are the same. The weights are then `u / σ`,
    /// normalized. That makes the tolerances independent of each asset's units.
    fn risk_budget_weights(
        cov: &DMatrix<f64>,
        budgets: &DVector<f64>,
    ) -> Result<DVector<f64>, CutupError> {
        let n = cov.ncols();
        let (sigma, volatilities) = correlation(cov)?;
        // Scaling b to a sum of 1 makes the tolerances below independent of its scale, and
        // doesn't change the normalized solution. Dividing b by its largest entry first keeps
        // the sum from overflowing.
        let b = budgets / budgets.max();
        let b = &b / b.sum();
        let smallest_budget = b.min();
        let objective = |y: &DVector<f64>| {
            0.5 * y.dot(&(&sigma * y))
                - b.iter().zip(y.iter()).map(|(b, y)| b * y.ln()).sum::<f64>()
        };
        let failed = |y: &DVector<f64>| {
            let w = y / y.sum();
            if w.dot(&(&sigma * &w)) <= 1e-12 {
                CutupError::SingularCovariance
            } else {
                CutupError::NotConverged
            }
        };
        // When a long-only portfolio has zero variance, rounding can stall the iteration on
        // that direction, or let it converge on the rounding left in a singular C. So y is
        // checked with max diversification's null-space cutoff before it's returned.
        let finish = |u: &DVector<f64>| {
            if in_null_space(&sigma, u)? {
                return Err(CutupError::SingularCovariance);
            }
            let w = u.component_div(&volatilities);
            Ok(&w / w.sum())
        };

        // Start from inverse volatility, scaled so that yᵀΣy = 1 as at the solution.
        let mut y = sigma.diagonal().map(|v| 1.0 / v.sqrt());
        let start_variance = y.dot(&(&sigma * &y));
        if start_variance.is_nan() || start_variance <= 1e-12 * y.norm_squared() {
            return Err(CutupError::SingularCovariance);
        }
        y /= start_variance.sqrt();

        let mut previous = f64::INFINITY;
        for _ in 0..500 {
            let gradient = &sigma * &y - b.component_div(&y);
            let mut hessian = sigma.clone();
            for i in 0..n {
                hessian[(i, i)] += b[i] / (y[i] * y[i]);
            }
            let Some(cholesky) = hessian.cholesky() else {
                return Err(failed(&y));
            };
            let step = cholesky.solve(&gradient);
            // The squared Newton decrement, which is never negative unless rounding has broken
            // the Hessian. Divided by the smallest budget, it bounds (Δᵢ / yᵢ)² for every i.
            let decrement = gradient.dot(&step);
            if decrement.is_nan() || decrement < 0.0 {
                return Err(failed(&y));
            }
            let scaled = decrement / smallest_budget;

            if scaled < 1e-2 {
                // Close enough for full steps, which stay positive and converge quadratically.
                // A line search would only see rounding in f from here on.
                if scaled >= previous {
                    // Rounding noise: y is as close as it gets.
                    return finish(&y);
                }
                y -= step;
                if scaled < 1e-20 {
                    return finish(&y);
                }
                previous = scaled;
                continue;
            }
            previous = f64::INFINITY;

            let current = objective(&y);
            let mut t = 1.0;
            loop {
                let candidate = &y - t * &step;
                if candidate.iter().all(|v| v.is_finite() && *v > 0.0)
                    && objective(&candidate) <= current - 0.25 * t * decrement
                {
                    y = candidate;
                    break;
                }
                t *= 0.5;
                if t < 1e-10 {
                    return Err(failed(&y));
                }
            }
        }
        Err(failed(&y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InverseVolatility;
    use crate::test_support::{
        assert_weights_close, collinear_returns, full_rank_prices, rank_one_returns, risk_shares,
    };
    use nalgebra::dmatrix;

    #[test]
    fn test_risk_parity_equalizes_risk_contributions() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
        for share in risk_shares(&allocator, &weights) {
            assert!((share - 1.0 / 3.0).abs() < 1e-12, "{share}");
        }
    }

    #[test]
    fn test_risk_parity_meets_risk_budgets() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        // Spanning eight orders of magnitude, and not summing to 1.
        let budgets = [1e-8, 3.0, 1.0];
        let config =
            RiskParity::default().risk_budgets(DVector::from_row_slice(&budgets).as_slice());
        let weights = allocator.allocate(&config).unwrap();
        let total: f64 = budgets.iter().sum();
        for (share, budget) in risk_shares(&allocator, &weights).iter().zip(budgets) {
            assert!(
                (share - budget / total).abs() < 1e-12,
                "{share} vs {}",
                budget / total
            );
        }
    }

    #[test]
    fn test_risk_parity_converges_below_rounding_noise() {
        // Found by the property tests: near the solution the decrease a line search checks
        // for is below the rounding in the objective, so only full Newton steps finish.
        let returns = DMatrix::from_column_slice(
            5,
            2,
            &[
                -0.011054998419255687,
                0.02268592417867769,
                0.028346158812172808,
                -0.04316318369020081,
                0.04868903549835187,
                -0.01337776138331863,
                0.012639084624998789,
                -0.04497829673506405,
                -0.030576298169914445,
                -0.04666742643312797,
            ],
        );
        let allocator = Returns::returns_matrix(returns).unwrap();
        let config = RiskParity::default()
            .risk_budgets(
                DVector::from_row_slice(&[0.8571666963481355, 0.45606979884293564]).as_slice(),
            )
            .covariance(CovarianceConfig::default().shrinkage(0.29613802865778266));
        assert!(allocator.allocate(&config).is_ok());
    }

    #[test]
    fn test_risk_parity_extreme_budgets() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let budgets =
            |b: &[f64]| RiskParity::default().risk_budgets(DVector::from_row_slice(b).as_slice());
        // Budgets whose sum overflows give the same weights as equal budgets.
        assert_weights_close(
            &allocator
                .allocate(&budgets(&[1e308, 1e308, 1e308]))
                .unwrap(),
            &allocator.allocate(&RiskParity::default()).unwrap(),
            1e-15,
        );
        // Budgets 10¹⁴ apart still converge.
        let weights = allocator.allocate(&budgets(&[1.0, 1e-14, 1e-14])).unwrap();
        let shares = risk_shares(&allocator, &weights);
        // Rounding in (Σw)ᵢ limits a tiny share's relative accuracy, so compare absolutely.
        for (share, budget) in shares.iter().zip([1.0, 1e-14, 1e-14]) {
            assert!((share - budget).abs() < 1e-12, "{share} vs {budget}");
        }
        // Far beyond that the solver can stop early, which isn't a singular covariance.
        assert_eq!(
            allocator.allocate(&budgets(&[1.0, 1e-200, 1.0])),
            Err(CutupError::NotConverged)
        );
    }

    #[test]
    fn test_risk_parity_two_assets_is_inverse_volatility() {
        // With two assets, equal contributions mean w₀σ₀ = w₁σ₁ whatever the correlation.
        let allocator =
            Returns::prices_matrix(full_rank_prices().columns(0, 2).into_owned()).unwrap();
        assert_weights_close(
            &allocator.allocate(&RiskParity::default()).unwrap(),
            &allocator.allocate(&InverseVolatility).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_risk_parity_ignores_return_and_budget_scale() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let scaled = Returns::returns_matrix(&allocator.returns * 1e-3).unwrap();
        let budgets = DVector::from_row_slice(&[1.0, 2.0, 3.0]);
        let config = RiskParity::default().risk_budgets(budgets.as_slice());
        let scaled_config = RiskParity::default().risk_budgets((budgets * 7.0).as_slice());
        assert_weights_close(
            &allocator.allocate(&config).unwrap(),
            &scaled.allocate(&scaled_config).unwrap(),
            1e-12,
        );
    }

    #[test]
    fn test_risk_parity_on_singular_covariance() {
        // Positively collinear assets have a singular Σ, but every long-only portfolio is risky.
        let allocator = Returns::returns_matrix(collinear_returns()).unwrap();
        let weights = allocator.allocate(&RiskParity::default()).unwrap();
        assert_weights_close(&weights, &Weights::from(vec![2.0 / 3.0, 1.0 / 3.0]), 1e-12);

        // Here the long-only portfolio (1, 1, 1) / 3 has zero variance.
        let allocator = Returns::returns_matrix(rank_one_returns()).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::SingularCovariance)
        );
        let config = RiskParity::default().covariance(CovarianceConfig::default().shrinkage(0.3));
        let weights = allocator.allocate(&config).unwrap();
        assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(weights.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_risk_parity_anti_correlated_pair() {
        let returns = dmatrix![0.01, -0.01; -0.02, 0.02; 0.015, -0.015];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::SingularCovariance)
        );
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().regularization(1e-6));
        let weights = allocator.allocate(&config).unwrap();
        assert_weights_close(&weights, &Weights::from(vec![0.5, 0.5]), 1e-12);
    }

    #[test]
    fn test_risk_parity_rejects_invalid_config() {
        let allocator = Returns::prices_matrix(full_rank_prices()).unwrap();
        let budgets =
            |b: &[f64]| RiskParity::default().risk_budgets(DVector::from_row_slice(b).as_slice());
        assert_eq!(
            allocator.allocate(&budgets(&[1.0, 1.0])),
            Err(CutupError::DimensionMismatch {
                field: "risk_budgets",
                expected: 3,
                actual: 2
            })
        );
        for bad in [0.0, -1.0, f64::INFINITY] {
            assert_eq!(
                allocator.allocate(&budgets(&[1.0, bad, 1.0])),
                Err(CutupError::InvalidAssetValue {
                    field: "risk_budgets",
                    column: 1,
                    asset: "1".into(),
                    value: bad
                })
            );
        }
        assert!(matches!(
            allocator.allocate(&budgets(&[1.0, 1.0, f64::NAN])),
            Err(CutupError::InvalidAssetValue {
                field: "risk_budgets",
                column: 2,
                ..
            })
        ));
        let config = RiskParity::default().covariance(CovarianceConfig::default().shrinkage(2.0));
        assert_eq!(
            allocator.allocate(&config),
            Err(CutupError::InvalidConfig {
                field: "shrinkage",
                value: 2.0
            })
        );
    }

    #[test]
    fn test_risk_parity_rejects_zero_variance() {
        let returns = dmatrix![0.01, 0.0; -0.02, 0.0; 0.03, 0.0];
        let allocator = Returns::returns_matrix(returns).unwrap();
        assert_eq!(
            allocator.allocate(&RiskParity::default()),
            Err(CutupError::ZeroVariance {
                column: 1,
                asset: "1".into()
            })
        );
        // Regularization gives the constant asset a variance, and nearly all the weight.
        let config =
            RiskParity::default().covariance(CovarianceConfig::default().regularization(1e-8));
        let weights = allocator.allocate(&config).unwrap();
        assert!(weights[1] > 0.99);
    }
}
