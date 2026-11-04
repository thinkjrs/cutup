//! [`PortfolioStats`]: the expected return, risk and risk split of a portfolio, from
//! [`Returns::stats`].

use crate::cvar::{tail_loss, tail_weight};
use crate::{CutupError, Returns, Weights};
use nalgebra::{DMatrix, DVector};

/// Statistics of one portfolio over the returns in a [`Returns`], from [`Returns::stats`].
///
/// Each is per period, in the units of the returns, with the sample mean and covariance:
/// with daily returns, the volatility is daily and the Sharpe ratio is daily too. Nothing is
/// annualized. They're computed from the portfolio's return in each period, scaled by powers of
/// two, so they don't overflow at any scale of returns or weights, and the variance can't round
/// below 0. Only a value that's itself past `f64::MAX`, such as the expected return of weights
/// several times the size of returns near it, is infinite; the ratios and shares stay finite.
///
/// The weights needn't sum to 1, but every ratio here assumes they're the whole portfolio.
///
/// # Example
///
/// ```
/// use cutup::{MaxSharpe, MinVariance, Returns};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
/// ];
/// let data = Returns::new(&returns)?;
/// let weights = data.allocate(&MinVariance::default())?;
/// let stats = data.stats(&weights)?;
///
/// println!("{:.4} ± {:.4} a day", stats.expected_return(), stats.volatility());
///
/// // Each asset's share of the variance.
/// let contributions = stats.risk_contributions().unwrap();
/// assert!((contributions.iter().sum::<f64>() - 1.0).abs() < 1e-12);
///
/// // The same CVaR as `Returns::cvar`.
/// assert_eq!(stats.cvar(0.8)?, data.cvar(&weights, 0.8)?);
///
/// // No long-only portfolio has a higher Sharpe ratio than MaxSharpe's.
/// let best = data.stats(&data.allocate(&MaxSharpe::default())?)?;
/// assert!(best.sharpe_ratio(0.0) >= stats.sharpe_ratio(0.0));
/// # Ok::<(), cutup::CutupError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioStats {
    expected_return: f64,
    volatility: f64,
    diversification_ratio: Option<f64>,
    risk_contributions: Option<Vec<f64>>,
    /// The portfolio's returns, divided by `2^exponent`, all smaller than 2 in size.
    scaled: Vec<f64>,
    exponent: i32,
    /// The mean and standard deviation of `scaled`, the standard deviation 0 if it counts as 0.
    mean: f64,
    deviation: f64,
}

impl PortfolioStats {
    /// The expected return `μᵀw`, with the sample mean of each asset's returns as `μ`.
    pub fn expected_return(&self) -> f64 {
        self.expected_return
    }

    /// The volatility `sqrt(wᵀΣw)`, with the sample covariance as `Σ`.
    ///
    /// A volatility below `1e-10` of the sizes of the positions' largest returns,
    /// `Σᵢ|wᵢ| maxₜ|rₜᵢ|`, counts as 0: the rule for a constant asset, applied to each position.
    /// Weights only on constant assets, such as cash, have none, though rounding leaves them a
    /// trace, and neither do positions that cancel to within rounding. So does a volatility too
    /// small to represent, below about `5e-324`.
    pub fn volatility(&self) -> f64 {
        self.volatility
    }

    /// The Sharpe ratio `(μᵀw − r) / sqrt(wᵀΣw)` for the risk-free rate `r` per period, in the
    /// units of the returns: a daily rate for daily returns.
    ///
    /// `None` if the [volatility](Self::volatility) is 0, or the rate isn't finite.
    pub fn sharpe_ratio(&self, risk_free_rate: f64) -> Option<f64> {
        if self.deviation == 0.0 || !risk_free_rate.is_finite() {
            return None;
        }
        Some((self.mean - pow2_scaled(risk_free_rate, -self.exponent)) / self.deviation)
    }

    /// The diversification ratio `σᵀw / sqrt(wᵀΣw)`, where `σ` holds the assets' volatilities:
    /// what [`MaxDiversification`](crate::MaxDiversification) maximizes. Long-only, it's at
    /// least 1, and 1 only when the assets held are perfectly correlated.
    ///
    /// `None` if the [volatility](Self::volatility) is 0.
    pub fn diversification_ratio(&self) -> Option<f64> {
        self.diversification_ratio
    }

    /// Each asset's share of the portfolio's variance, `wᵢ(Σw)ᵢ / wᵀΣw`, in column order: the
    /// shares [`RiskParity`](crate::RiskParity) sets equal, or to its risk budgets. They sum to
    /// 1. A short position, or a long one that hedges others, can have a negative share.
    ///
    /// `None` if the [volatility](Self::volatility) is 0.
    pub fn risk_contributions(&self) -> Option<&[f64]> {
        self.risk_contributions.as_deref()
    }

    /// The conditional value at risk at confidence `beta`, the same as [`Returns::cvar`]: the
    /// average loss over the worst `1 − β` of the periods.
    ///
    /// # Errors
    ///
    /// [`CutupError::InvalidConfig`] if `beta` isn't in `(0, 1)`.
    pub fn cvar(&self, beta: f64) -> Result<f64, CutupError> {
        let cap = tail_weight(beta, self.scaled.len())?;
        let losses = self.scaled.iter().map(|r| -r).collect();
        Ok(pow2_scaled(tail_loss(losses, cap), self.exponent))
    }
}

impl Returns {
    /// The expected return, volatility, Sharpe ratio, diversification ratio, risk contributions
    /// and CVaR of the portfolio `weights`, over these returns. See [`PortfolioStats`].
    ///
    /// They use the sample mean and covariance, even for weights from a method given other
    /// expected returns or covariance adjustments, so methods are compared on the same data.
    ///
    /// # Errors
    ///
    /// * [`CutupError::DimensionMismatch`] if there isn't one weight per asset.
    /// * [`CutupError::LabelMismatch`] if the weights are labeled with other assets than the
    ///   data, or in another order. Weights labeled by column, such as `Weights::from(vec)`,
    ///   are taken in column order.
    /// * [`CutupError::NonFiniteWeight`] if a weight is NaN or infinite.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{Returns, Weights};
    ///
    /// // A risky asset, and cash earning 0.1% a period.
    /// let returns = [[0.02, 0.001], [-0.01, 0.001], [0.03, 0.001]];
    /// let data = Returns::new(&returns)?;
    ///
    /// let cash = data.stats(&Weights::from(vec![0.0, 1.0]))?;
    /// assert!((cash.expected_return() - 0.001).abs() < 1e-15);
    /// assert_eq!(cash.volatility(), 0.0);
    /// assert_eq!(cash.sharpe_ratio(0.0), None);
    ///
    /// let half = data.stats(&Weights::from(vec![0.5, 0.5]))?;
    /// // Cash adds no risk: the risky asset carries all of it.
    /// let contributions = half.risk_contributions().unwrap();
    /// assert!((contributions[0] - 1.0).abs() < 1e-15 && contributions[1] == 0.0);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn stats(&self, weights: &Weights) -> Result<PortfolioStats, CutupError> {
        self.check_weights(weights)?;
        if let Some(column) = weights.iter().position(|w| !w.is_finite()) {
            return Err(CutupError::NonFiniteWeight {
                column,
                asset: self.labels[column].clone(),
            });
        }
        let returns = &self.returns;
        let t = returns.nrows();

        // The weights divided by a power of two near the largest, then the portfolio's returns
        // by one near theirs. Powers of two scale exactly, so the CVaR matches `Returns::cvar`.
        let weight_exponent = binary_exponent(largest(weights.iter().copied()));
        let w = DVector::from_iterator(
            weights.len(),
            weights.iter().map(|&w| pow2_scaled(w, -weight_exponent)),
        );
        let mut portfolio = returns * &w;
        // The sizes of the positions' largest returns, Σᵢ|wᵢ| maxₜ|rₜᵢ|.
        let positions = |returns: &DMatrix<f64>| -> f64 {
            returns
                .column_iter()
                .zip(w.iter())
                .map(|(column, w)| w.abs() * column.amax())
                .sum()
        };
        let mut sizes = positions(returns);
        let mut exponent_sum = weight_exponent;
        if portfolio.iter().any(|r| !r.is_finite()) {
            // Returns near f64::MAX: scale them too.
            let returns_exponent = binary_exponent(returns.amax());
            let scaled_returns = returns.map(|r| pow2_scaled(r, -returns_exponent));
            portfolio = &scaled_returns * &w;
            sizes = positions(&scaled_returns);
            exponent_sum += returns_exponent;
        }
        let portfolio_exponent = binary_exponent(portfolio.amax());
        let scaled: Vec<f64> = portfolio
            .iter()
            .map(|&r| pow2_scaled(r, -portfolio_exponent))
            .collect();
        let exponent = exponent_sum + portfolio_exponent;

        let mean = scaled.iter().sum::<f64>() / t as f64;
        let deviation = standard_deviation(&scaled, mean);
        // The rule for a constant asset, applied to each position: if every asset held counts
        // as constant, σ ≤ Σᵢ|wᵢ|σᵢ is within it. It's far above the rounding in the returns'
        // sums, about nε of the positions, so it covers positions that cancel to within rounding
        // too; and a deviation that underflows to 0 once unscaled counts as none, so the ratios
        // are `None` whenever `volatility()` is 0.
        let deviation = if deviation <= 1e-10 * pow2_scaled(sizes, -portfolio_exponent)
            || pow2_scaled(deviation, exponent) == 0.0
        {
            0.0
        } else {
            deviation
        };

        let (diversification_ratio, risk_contributions) = if deviation == 0.0 {
            (None, None)
        } else {
            // Each asset's returns are scaled by a power of two of their own, so its volatility
            // and its covariance with the portfolio stay finite.
            let mut weighted_volatility = 0.0;
            let mut contributions = Vec::with_capacity(w.len());
            let mut asset = Vec::with_capacity(t);
            for (i, column) in returns.column_iter().enumerate() {
                let asset_exponent = binary_exponent(column.amax());
                asset.clear();
                asset.extend(column.iter().map(|&r| pow2_scaled(r, -asset_exponent)));
                let asset_mean = asset.iter().sum::<f64>() / t as f64;
                let asset_deviation = standard_deviation(&asset, asset_mean);
                // A constant asset, such as cash, adds no risk, though rounding leaves it a trace.
                if asset_deviation <= 1e-10 * largest(asset.iter().copied()) {
                    contributions.push(0.0);
                    continue;
                }
                // wᵢ cov(rᵢ, r), with wᵢ and rᵢ in their own scales, over the portfolio's: each
                // asset's part of the variance, times T − 1. The weight is rescaled first, exactly
                // even when it's subnormal; a portfolio that isn't riskless keeps wᵢσᵢ/σ within
                // about 1/(100nε), so it can't overflow.
                let weight = pow2_scaled(w[i], weight_exponent + asset_exponent - exponent);
                let comovement: f64 = asset
                    .iter()
                    .zip(&scaled)
                    .map(|(a, p)| (a - asset_mean) * (p - mean))
                    .sum();
                contributions.push(weight * comovement);
                weighted_volatility += weight * asset_deviation / deviation;
            }
            // Their sum is the variance, times T − 1, but for the rounding in the portfolio's
            // returns, which subnormal weights can leave in every bit: shares of their sum
            // keep the promise that they sum to 1.
            let total: f64 = contributions.iter().sum();
            contributions.iter_mut().for_each(|c| *c /= total);
            (Some(weighted_volatility), Some(contributions))
        };

        Ok(PortfolioStats {
            expected_return: pow2_scaled(mean, exponent),
            volatility: pow2_scaled(deviation, exponent),
            diversification_ratio,
            risk_contributions,
            scaled,
            exponent,
            mean,
            deviation,
        })
    }
}

/// The sample standard deviation of `values` around their `mean`.
fn standard_deviation(values: &[f64], mean: f64) -> f64 {
    let squares: f64 = values.iter().map(|v| (v - mean) * (v - mean)).sum();
    (squares / (values.len() - 1) as f64).sqrt()
}

/// The largest of `values` in size, or 0 if there are none.
fn largest(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0, |m, v| m.max(v.abs()))
}

/// The exponent `k` of the power of two `2^k ≤ x < 2^(k+1)`, for a finite `x ≥ 0`, or 0 for 0.
pub(crate) fn binary_exponent(x: f64) -> i32 {
    if x == 0.0 {
        return 0;
    }
    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale it into the normal range first.
        binary_exponent(x * pow2(64)) - 64
    } else {
        biased - 1023
    }
}

/// `2^k`, for `k` in the normal range, `-1022..=1023`.
fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// `x · 2^k`, exactly unless the result is subnormal or overflows, in steps that stay in the
/// normal range.
pub(crate) fn pow2_scaled(x: f64, k: i32) -> f64 {
    let step = k.clamp(-1022, 1023);
    let x = x * pow2(step);
    if step == k {
        x
    } else {
        pow2_scaled(x, k - step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{
        collinear_returns, factor_returns, labeled, portfolio_variance, risk_shares, sample_mean,
        sharpe, uncorrelated_returns,
    };
    use crate::{MaxDiversification, MinVariance, RiskParity};
    use nalgebra::dmatrix;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * b.abs().max(1e-300)
    }

    /// The closed forms from the sample mean and covariance.
    fn assert_closed_forms(data: &Returns, weights: &Weights) {
        let stats = data.stats(weights).unwrap();
        let w = DVector::from_column_slice(weights.as_slice());
        let expected = sample_mean(data).dot(&w);
        let volatility = portfolio_variance(data, weights).sqrt();
        assert!(close(stats.expected_return(), expected, 1e-12), "{stats:?}");
        assert!(close(stats.volatility(), volatility, 1e-12), "{stats:?}");
        for rate in [0.0, 1e-4, -2e-3] {
            let ratio = stats.sharpe_ratio(rate).unwrap();
            assert!(close(ratio, sharpe(data, weights, rate), 1e-11), "{ratio}");
        }
        let volatilities = data.cov_matrix.diagonal().map(f64::sqrt);
        let ratio = stats.diversification_ratio().unwrap();
        assert!(
            close(ratio, volatilities.dot(&w) / volatility, 1e-12),
            "{ratio}"
        );
        let shares = stats.risk_contributions().unwrap();
        for (share, expected) in shares.iter().zip(risk_shares(data, weights)) {
            assert!((share - expected).abs() <= 1e-12, "{shares:?}");
        }
        assert!((shares.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn stats_match_their_closed_forms() {
        let data = Returns::returns_matrix(factor_returns(6, 40, 7)).unwrap();
        let long_only = data.allocate(&MinVariance::default()).unwrap();
        let long_short = data
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        assert!(long_short.iter().any(|&w| w < 0.0));
        let unnormalized = Weights::from(vec![0.3, -0.7, 1.4, 0.0, 2.5, -0.1]);
        for weights in [&long_only, &long_short, &unnormalized] {
            assert_closed_forms(&data, weights);
        }
    }

    #[test]
    fn stats_of_one_asset() {
        let data = Returns::returns_matrix(factor_returns(3, 20, 3)).unwrap();
        let stats = data.stats(&Weights::from(vec![0.0, 1.0, 0.0])).unwrap();
        // Its volatility, computed the same way, over itself.
        assert_eq!(stats.diversification_ratio(), Some(1.0));
        assert_eq!(stats.risk_contributions(), Some(&[0.0, 1.0, 0.0][..]));
        let volatility = data.cov_matrix[(1, 1)].sqrt();
        assert!(close(stats.volatility(), volatility, 1e-14));
    }

    #[test]
    fn stats_of_uncorrelated_assets() {
        // Variances in the ratio 1 : 2 : 4. Uncorrelated, each asset's share is wᵢ²σᵢ² over
        // their sum: 16 : 8 : 4 for inverse variance weights, and equal for inverse volatility.
        let data = Returns::returns_matrix(uncorrelated_returns()).unwrap();
        let weights = Weights::from(vec![4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]);
        let stats = data.stats(&weights).unwrap();
        let shares = stats.risk_contributions().unwrap();
        for (share, expected) in shares.iter().zip([4.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0]) {
            assert!((share - expected).abs() < 1e-15, "{shares:?}");
        }
        let inverse_volatility = Weights::from(vec![1.0, 0.5f64.sqrt(), 0.5]);
        for share in data
            .stats(&inverse_volatility)
            .unwrap()
            .risk_contributions()
            .unwrap()
        {
            assert!((share - 1.0 / 3.0).abs() < 1e-15, "{share}");
        }
        // σᵀw = (4 + 2√2 + 2)/7 σ₀ over σ_p = sqrt(16 + 8 + 4)/7 σ₀.
        let ratio = (4.0 + 2.0 * 2f64.sqrt() + 2.0) / 28f64.sqrt();
        assert!(close(stats.diversification_ratio().unwrap(), ratio, 1e-14));
        // The returns average 0, to rounding of √2.
        assert!(stats.expected_return().abs() < 1e-18);
        assert!(stats.sharpe_ratio(0.0).unwrap().abs() < 1e-15);
    }

    #[test]
    fn stats_match_the_methods_that_optimize_them() {
        let data = Returns::returns_matrix(factor_returns(5, 30, 11)).unwrap();
        let parity = data.allocate(&RiskParity::default()).unwrap();
        for share in data.stats(&parity).unwrap().risk_contributions().unwrap() {
            assert!((share - 0.2).abs() < 1e-9, "{share}");
        }
        let best = data.allocate(&MaxDiversification::default()).unwrap();
        let highest = data.stats(&best).unwrap().diversification_ratio().unwrap();
        for weights in [&parity, &data.allocate(&MinVariance::default()).unwrap()] {
            let ratio = data
                .stats(weights)
                .unwrap()
                .diversification_ratio()
                .unwrap();
            assert!(ratio <= highest, "{ratio} > {highest}");
        }
    }

    #[test]
    fn stats_cvar_matches_returns_cvar() {
        let data = Returns::returns_matrix(factor_returns(4, 37, 5)).unwrap();
        for weights in [
            data.allocate(&MinVariance::default().allow_short())
                .unwrap(),
            Weights::from(vec![3.0, -1.5, 0.25, -0.75]),
            Weights::from(vec![1e-200, 3e-201, 0.0, -2e-200]),
        ] {
            let stats = data.stats(&weights).unwrap();
            for beta in [0.5, 0.9, 0.95, 0.99, 0.999] {
                assert_eq!(stats.cvar(beta), data.cvar(&weights, beta));
            }
        }
        let stats = data.stats(&Weights::from(vec![0.25; 4])).unwrap();
        for beta in [0.0, 1.0, f64::NAN] {
            assert!(matches!(
                stats.cvar(beta),
                Err(CutupError::InvalidConfig {
                    field: "confidence",
                    ..
                })
            ));
        }
    }

    #[test]
    fn stats_scale_with_the_returns_and_weights() {
        let returns = factor_returns(4, 25, 9);
        let data = Returns::returns_matrix(returns.clone()).unwrap();
        let weights = Weights::from(vec![0.4, -0.2, 0.5, 0.3]);
        let base = data.stats(&weights).unwrap();
        // Powers of two scale exactly, other factors to rounding.
        for (factor, tolerance) in [(2f64.powi(-60), 0.0), (2f64.powi(500), 0.0), (1e140, 1e-14)] {
            let scaled = Returns::returns_matrix(&returns * factor).unwrap();
            let stats = scaled.stats(&weights).unwrap();
            assert!(close(
                stats.expected_return(),
                base.expected_return() * factor,
                tolerance
            ));
            assert!(close(
                stats.volatility(),
                base.volatility() * factor,
                tolerance
            ));
            let ratio = stats.diversification_ratio().unwrap();
            assert!(close(ratio, base.diversification_ratio().unwrap(), 1e-14));
            for (a, b) in stats
                .risk_contributions()
                .unwrap()
                .iter()
                .zip(base.risk_contributions().unwrap())
            {
                assert!((a - b).abs() <= 1e-14, "{a} vs {b}");
            }
        }
        for factor in [1e-250, 1e250] {
            let scaled: Vec<f64> = weights.iter().map(|w| w * factor).collect();
            let stats = data.stats(&Weights::from(scaled)).unwrap();
            assert!(close(
                stats.expected_return(),
                base.expected_return() * factor,
                1e-14
            ));
            assert!(close(stats.volatility(), base.volatility() * factor, 1e-14));
            assert!(close(
                stats.cvar(0.9).unwrap(),
                base.cvar(0.9).unwrap() * factor,
                1e-14
            ));
            // The excess return over a rate of 0 scales too.
            let ratio = stats.sharpe_ratio(0.0).unwrap();
            assert!(close(ratio, base.sharpe_ratio(0.0).unwrap(), 1e-14));
        }
    }

    #[test]
    fn stats_of_weights_whose_returns_overflow() {
        // The returns are finite, but 1.5 times their sum isn't, so they're scaled too.
        // Divided by 2^-1, the weights are 1.98, and their returns overflow.
        let data = Returns::returns_matrix(dmatrix![8e307, 8e307; 8e307, 8e307]).unwrap();
        let stats = data.stats(&Weights::from(vec![0.99, 0.99])).unwrap();
        assert!(
            close(stats.expected_return(), 1.584e308, 1e-15),
            "{stats:?}"
        );
        assert_eq!(stats.volatility(), 0.0);
        assert!(close(stats.cvar(0.5).unwrap(), -1.584e308, 1e-15));
    }

    #[test]
    fn stats_of_riskless_portfolios() {
        // Cash alone, and two assets that offset exactly but for a constant.
        let cash =
            Returns::returns_matrix(dmatrix![0.02, 0.001; -0.01, 0.001; 0.03, 0.001]).unwrap();
        let collinear = Returns::returns_matrix(collinear_returns()).unwrap();
        for (data, weights, expected) in [
            (&cash, vec![0.0, 1.0], 0.001),
            (&collinear, vec![2.0, -1.0], -0.001),
        ] {
            let stats = data.stats(&Weights::from(weights)).unwrap();
            assert!(close(stats.expected_return(), expected, 1e-12), "{stats:?}");
            assert_eq!(stats.volatility(), 0.0);
            assert_eq!(stats.sharpe_ratio(0.0), None);
            assert_eq!(stats.diversification_ratio(), None);
            assert_eq!(stats.risk_contributions(), None);
            assert!(close(stats.cvar(0.5).unwrap(), -expected, 1e-12));
        }
        let stats = cash.stats(&Weights::from(vec![0.0, 0.0])).unwrap();
        assert_eq!((stats.expected_return(), stats.volatility()), (0.0, 0.0));
        assert_eq!(stats.risk_contributions(), None);
    }

    #[test]
    fn a_constant_asset_contributes_no_risk() {
        let data = Returns::returns_matrix(dmatrix![
            0.02, 0.001, -0.01;
            -0.01, 0.001, 0.015;
            0.03, 0.001, 0.002;
            0.00, 0.001, -0.004
        ])
        .unwrap();
        let weights = Weights::from(vec![0.3, 0.5, 0.2]);
        let stats = data.stats(&weights).unwrap();
        let shares = stats.risk_contributions().unwrap();
        assert_eq!(shares[1], 0.0);
        assert!((shares.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        let w = DVector::from_column_slice(weights.as_slice());
        let volatilities = DVector::from_vec(vec![
            data.cov_matrix[(0, 0)].sqrt(),
            0.0,
            data.cov_matrix[(2, 2)].sqrt(),
        ]);
        let ratio = volatilities.dot(&w) / portfolio_variance(&data, &weights).sqrt();
        assert!(close(stats.diversification_ratio().unwrap(), ratio, 1e-14));
    }

    #[test]
    fn stats_of_returns_far_from_zero() {
        // Centering both the asset's and the portfolio's returns keeps the covariance accurate
        // when their means dwarf their spread.
        let data = Returns::returns_matrix(factor_returns(4, 30, 13).add_scalar(1000.0)).unwrap();
        let weights = Weights::from(vec![0.4, 0.1, 0.3, 0.2]);
        let shares = data.stats(&weights).unwrap();
        let shares = shares.risk_contributions().unwrap();
        for (share, expected) in shares.iter().zip(risk_shares(&data, &weights)) {
            assert!((share - expected).abs() <= 1e-9, "{shares:?}");
        }
    }

    #[test]
    fn a_spread_within_1e_10_of_the_largest_return_counts_as_none() {
        // Scaled by 2, the returns are about 1.9, and their spread about 1.5e-10: under 1e-10
        // of 1.9, but over 1e-10 and 1e-10 / 1.9.
        let e = 0.8e-10 / 2f64.sqrt();
        let data = Returns::returns_matrix(dmatrix![
            0.02, 0.95 * (1.0 + e);
            -0.01, 0.95 * (1.0 - e)
        ])
        .unwrap();
        let stats = data.stats(&Weights::from(vec![0.0, 1.0])).unwrap();
        assert_eq!(stats.volatility(), 0.0);
        let stats = data.stats(&Weights::from(vec![0.5, 0.5])).unwrap();
        assert_eq!(stats.risk_contributions().unwrap()[1], 0.0);
    }

    #[test]
    fn a_volatility_that_underflows_counts_as_none() {
        // Half the smallest subnormal's spread rounds to 0 once unscaled.
        let tiny = f64::from_bits(1);
        let data = Returns::returns_matrix(dmatrix![0.0, 0.0; tiny, 0.0; 0.0, 0.0]).unwrap();
        let stats = data.stats(&Weights::from(vec![0.5, 0.5])).unwrap();
        assert_eq!(stats.volatility(), 0.0);
        assert_eq!(stats.sharpe_ratio(0.0), None);
        assert_eq!(stats.diversification_ratio(), None);
        assert_eq!(stats.risk_contributions(), None);
    }

    #[test]
    fn risk_contributions_sum_to_1_when_subnormal_weights_lose_their_precision() {
        // Found by fuzzing MinCvar: the weight of 1 is on an asset whose returns are 0, so the
        // portfolio's returns come from weights of 3 and 45 times the smallest subnormal,
        // whose products with the returns keep only a few bits.
        let data = Returns::returns_matrix(dmatrix![
            -1.0, -5.0, 0.0, 1280.0, -1.0, -1.0, -1.0, 694.0;
            -20047.0, 177.0, 0.0, -20047.0, -20047.0, -20047.0, -20047.0, -20047.0
        ])
        .unwrap();
        let tiny = f64::from_bits(1);
        let weights = Weights::from(vec![0.0, 3.0 * tiny, 1.0, 45.0 * tiny, 0.0, 0.0, 0.0, 0.0]);
        let stats = data.stats(&weights).unwrap();
        let shares = stats.risk_contributions().unwrap();
        assert!(
            (shares.iter().sum::<f64>() - 1.0).abs() <= 1e-15,
            "{shares:?}"
        );
        // With all of the risk on a weight of 3 times the smallest subnormal, its share,
        // multiplied by it before being rescaled, underflowed to 0, and the shares summed to 0.
        let data = Returns::returns_matrix(dmatrix![
            -1.0, -5.0, 0.0, 1280.0, -1.0, -1.0, 11007.0, 23543.0;
            -20047.0, 176.0, 0.0, -20047.0, -20047.0, -20047.0, -20047.0, 16049.0
        ])
        .unwrap();
        let weights = Weights::from(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 3.0 * tiny]);
        let stats = data.stats(&weights).unwrap();
        let shares = stats.risk_contributions().unwrap();
        assert_eq!(shares, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(stats.diversification_ratio(), Some(1.0));
    }

    #[test]
    fn a_volatility_rounding_could_account_for_counts_as_none() {
        // Found by fuzzing MinCvar: positions of about 1e132 cancel to returns near 1e116, below
        // the rounding in their sum.
        let returns = dmatrix![9984.0, -115.0, -1.0, -29299.0; -214.0, -115.0, 2047.0, 1792.0];
        let data = Returns::returns_matrix(returns * 1e130).unwrap();
        let weights = Weights::from(vec![
            0.010781254606248916,
            0.9355335723104862,
            0.05368517308326488,
            0.0,
        ]);
        let stats = data.stats(&weights).unwrap();
        assert_eq!(stats.volatility(), 0.0, "{stats:?}");
        assert_eq!(stats.risk_contributions(), None);
        // Long 1 ± δ and short a constant 1, the portfolio's returns are ±δ, but its positions'
        // largest returns sum to about 2: a spread of δ√2 under 2e-10 is within the rule, and
        // there the long asset counts as constant too.
        for (spread, riskless) in [(1.9e-10, true), (2.1e-10, false)] {
            let delta = spread / 2f64.sqrt();
            let data =
                Returns::returns_matrix(dmatrix![1.0 + delta, 1.0; 1.0 - delta, 1.0]).unwrap();
            let stats = data.stats(&Weights::from(vec![1.0, -1.0])).unwrap();
            assert_eq!(stats.volatility() == 0.0, riskless, "{spread}: {stats:?}");
            if !riskless {
                assert_eq!(stats.risk_contributions(), Some(&[1.0, 0.0][..]));
            }
        }
    }

    #[test]
    fn sharpe_ratio_needs_a_finite_rate() {
        let data = Returns::returns_matrix(factor_returns(3, 12, 1)).unwrap();
        let stats = data.stats(&Weights::from(vec![0.5, 0.25, 0.25])).unwrap();
        for rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(stats.sharpe_ratio(rate), None);
        }
        assert!(stats.sharpe_ratio(f64::MAX).unwrap() < -1e300);
    }

    #[test]
    fn stats_rejects_weights_that_dont_fit() {
        let data = labeled();
        assert_eq!(
            data.stats(&Weights::from(vec![0.5, 0.5])),
            Err(CutupError::DimensionMismatch {
                field: "weights",
                expected: 3,
                actual: 2
            })
        );
        let other = Returns::returns_matrix(data.returns.clone())
            .unwrap()
            .labeled(["SPY", "IEF", "GLD"])
            .unwrap();
        let weights = other.allocate(&crate::EqualWeight).unwrap();
        assert_eq!(
            data.stats(&weights),
            Err(CutupError::LabelMismatch {
                column: 1,
                expected: "TLT".into(),
                actual: "IEF".into()
            })
        );
        assert_eq!(
            data.stats(&Weights::from(vec![0.5, f64::NAN, 0.5])),
            Err(CutupError::NonFiniteWeight {
                column: 1,
                asset: "TLT".into()
            })
        );
        assert!(
            data.stats(&Weights::from(vec![0.5, 0.5, f64::INFINITY]))
                .is_err()
        );
    }

    #[test]
    fn binary_exponents() {
        for (x, k) in [
            (0.0, 0),
            (1.0, 0),
            (1.5, 0),
            (2.0, 1),
            (0.75, -1),
            (f64::MAX, 1023),
            (f64::MIN_POSITIVE, -1022),
            (f64::MIN_POSITIVE / 2.0, -1023),
            (5e-324, -1074),
        ] {
            assert_eq!(binary_exponent(x), k, "{x}");
        }
    }

    #[test]
    fn powers_of_two_scale_in_range() {
        assert_eq!(pow2(0), 1.0);
        assert_eq!(pow2(1023), 2f64.powi(1023));
        assert_eq!(pow2(-1022), f64::MIN_POSITIVE);
        assert_eq!(pow2_scaled(1.0, -1074), 5e-324);
        assert_eq!(pow2_scaled(1.0, -1075), 0.0);
        assert_eq!(pow2_scaled(1.0, 1024), f64::INFINITY);
        let half = 2f64.powi(-1023);
        assert_eq!(pow2_scaled(f64::MAX, -2046), f64::MAX * half * half);
        assert_eq!(pow2_scaled(5e-324, 2097), 2f64.powi(1023));
        assert_eq!(pow2_scaled(-3.0, 2), -12.0);
    }
}
