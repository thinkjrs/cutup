//! Expected-return estimates from the returns, in place of the sample mean, to pass to a
//! method's `expected_returns` setter.

use crate::black_litterman::market_portfolio;
use crate::covariance::Decay;
use crate::eigen::SymmetricEigen;
use crate::methods::correlation;
use crate::stats::{binary_exponent, pow2_scaled};
use crate::{CutupError, PerAsset, Returns};
use nalgebra::{DMatrix, DVector};

impl Returns {
    /// The exponentially weighted mean of each asset's returns, with weights that halve every
    /// `half_life` periods back from the last row: an expected return that follows recent
    /// periods more than the sample mean does, to pass to a method's `expected_returns`
    /// setter.
    ///
    /// Period `t` of `T` weighs `λ^(T−1−t)` with `λ = 2^(−1/half_life)`, and each mean is
    /// `Σₜ λ^(T−1−t) rₜ / Σₜ λ^(T−1−t)`: pandas' `ewm(halflife=half_life).mean()`, the mean
    /// [`CovarianceConfig::exponentially_weighted`](crate::CovarianceConfig::exponentially_weighted)
    /// uses, and riskfolio-lib's `"ewma1"`. A long half-life gives the sample mean, and a short
    /// one the last period's returns. Each asset's returns are scaled by a power of two first,
    /// so returns scaled by a power of two give a mean scaled by exactly the same.
    ///
    /// The half-life is in periods, the unit of the rows: with daily returns, a half-life of
    /// 20 is about a month.
    ///
    /// # Errors
    ///
    /// [`CutupError::InvalidConfig`] with field `"half_life"` if `half_life` isn't finite and
    /// positive, or is so small that `λ` is below `e^−708`, about 3.3e-308.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MaxSharpe, Returns};
    ///
    /// // The first asset did well early on, the second lately.
    /// let data = Returns::new(&[
    ///     [0.030, -0.010],
    ///     [0.025,  0.000],
    ///     [0.000,  0.020],
    ///     [-0.010, 0.025],
    /// ])?;
    /// let recent = data.exponentially_weighted_mean(1.0)?;
    /// assert!(recent[1] > recent[0]);
    ///
    /// let weights = data.allocate(&MaxSharpe::default().expected_returns(recent))?;
    /// assert!(weights[1] > weights[0]);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    ///
    /// <https://en.wikipedia.org/wiki/Exponential_smoothing>
    pub fn exponentially_weighted_mean(&self, half_life: f64) -> Result<Vec<f64>, CutupError> {
        self.weighted_mean(Decay::HalfLife(half_life))
    }

    /// The exponentially weighted mean of each asset's returns with decay `λ`, in `(0, 1)`:
    /// each period back from the last row weighs `λ` times the one after it, as riskfolio-lib's
    /// `d`. Otherwise it's the same as [`Returns::exponentially_weighted_mean`], with a
    /// half-life of `−ln 2 / ln λ`.
    ///
    /// # Errors
    ///
    /// [`CutupError::InvalidConfig`] with field `"decay"` if `λ` isn't in `(0, 1)`, or is
    /// below `e^−708`, about 3.3e-308.
    pub fn exponentially_weighted_mean_decay(&self, lambda: f64) -> Result<Vec<f64>, CutupError> {
        self.weighted_mean(Decay::Lambda(lambda))
    }

    /// The expected returns the capital asset pricing model gives each asset against the
    /// market portfolio `market_weights`: `μᵢ = r_f + βᵢ(E[r_m] − r_f)`, where `r_m` is the
    /// market portfolio's return each period, `E[r_m]` its sample mean, and
    /// `βᵢ = Cov(rᵢ, r_m) / Var(r_m)` each asset's beta to it, from the sample covariance.
    ///
    /// The market weights are one per asset, in column order or by asset, and are divided by
    /// their sum, so capitalizations work too, as for [`BlackLitterman`](crate::BlackLitterman).
    /// Equal weights give PyPortfolioOpt's `capm_return` with its default market, per period
    /// and without compounding. The risk-free rate is per period, in the units of the returns.
    ///
    /// Every asset's excess return is its beta times the market's, so the market weights of
    /// these returns give back the market's mean. They're computed in scales of powers of two, so
    /// only a return that's itself past `f64::MAX`, such as a rate far beyond the returns times
    /// a beta far from 1, is infinite. For the returns the market weights imply
    /// through a risk aversion instead, `δΣw`, see
    /// [`BlackLitterman::prior_returns`](crate::BlackLitterman::prior_returns).
    ///
    /// # Errors
    ///
    /// * [`CutupError::DimensionMismatch`] if there isn't one market weight per asset.
    /// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if the market weights
    ///   name an asset that isn't in the data, or name one twice.
    /// * [`CutupError::InvalidAssetValue`] if a market weight is negative or not finite.
    /// * [`CutupError::InvalidConfig`] if the market weights sum to 0, or the risk-free rate
    ///   isn't finite.
    /// * [`CutupError::RisklessMarket`] if the market portfolio has no variance, by the rule of
    ///   [`Returns::stats`]: no asset has a beta to it.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MaxSharpe, Returns, Weights};
    ///
    /// let data = Returns::new(&[
    ///     [ 0.010, -0.020,  0.005],
    ///     [-0.004,  0.015,  0.002],
    ///     [ 0.007,  0.003, -0.006],
    ///     [ 0.002, -0.001,  0.004],
    ///     [-0.006,  0.008, -0.003],
    /// ])?;
    /// // Market capitalizations, and a risk-free rate of 0.01% a period.
    /// let capm = data.capm_returns([600.0, 300.0, 100.0], 0.0001)?;
    ///
    /// // The market's mean, back from its weights.
    /// let market = data.stats(&Weights::from(vec![0.6, 0.3, 0.1]))?.expected_return();
    /// let weighted = 0.6 * capm[0] + 0.3 * capm[1] + 0.1 * capm[2];
    /// assert!((weighted - market).abs() < 1e-15);
    ///
    /// let weights = data.allocate(&MaxSharpe::default().expected_returns(capm))?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    ///
    /// <https://en.wikipedia.org/wiki/Capital_asset_pricing_model>
    pub fn capm_returns(
        &self,
        market_weights: impl Into<PerAsset>,
        risk_free_rate: f64,
    ) -> Result<Vec<f64>, CutupError> {
        let w = market_portfolio(&market_weights.into(), &self.labels)?;
        if !risk_free_rate.is_finite() {
            return Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: risk_free_rate,
            });
        }
        // The market's return each period, in a power-of-two scale of its own, with its mean
        // and volatility, as `stats` gives them.
        let market = self.portfolio_stats(&self.weights(w.as_slice().to_vec()), false)?;
        if market.volatility() == 0.0 {
            return Err(CutupError::RisklessMarket);
        }
        let t = self.returns.nrows();
        // The premium over σ_m, divided by a power of two near it: σ_m is `deviation` in units
        // of 2^exponent, nonzero and at least 1e-10 of the scaled returns.
        let premium = market.expected_return() - risk_free_rate;
        let premium_exponent = binary_exponent(premium.abs());
        let ratio = pow2_scaled(premium, -premium_exponent) / market.deviation;
        let rate_exponent = binary_exponent(risk_free_rate.abs());
        let mut asset = Vec::with_capacity(t);
        Ok(self
            .returns
            .column_iter()
            .map(|column| {
                // Cov(rᵢ, r_m) / σ_m from each period's market return, not from Σw, which
                // underflows for small returns and cancels for a hedged market. Each asset is
                // in a scale of its own, so it's at most about σᵢ in that scale.
                let asset_exponent = binary_exponent(column.amax());
                asset.clear();
                asset.extend(column.iter().map(|&r| pow2_scaled(r, -asset_exponent)));
                let asset_mean = asset.iter().sum::<f64>() / t as f64;
                let comovement: f64 = asset
                    .iter()
                    .zip(&market.scaled)
                    .map(|(a, m)| (a - asset_mean) * (m - market.mean))
                    .sum::<f64>()
                    / (t - 1) as f64;
                let excess = comovement / market.deviation * ratio;
                if excess == 0.0 {
                    return risk_free_rate;
                }
                // r_f + βᵢ(E[r_m] − r_f) in the larger of the two terms' scales, so it's
                // infinite only if the return itself is past f64::MAX. The excess return is
                // `excess` times 2^shift, and `excess` is within a few dozen powers of two of
                // 1, so 2^shift is near enough its scale.
                let shift = asset_exponent + premium_exponent - market.exponent;
                let k = rate_exponent.max(shift);
                pow2_scaled(
                    pow2_scaled(risk_free_rate, -k) + pow2_scaled(excess, shift - k),
                    k,
                )
            })
            .collect())
    }

    /// The James-Stein estimate of the expected returns: the sample means shrunk toward their
    /// average, `μ̂ = (1 − α)μ + α·μ̄1`, by an intensity `α` estimated from the data.
    ///
    /// `α = (tr Σ − 2λ_max) / (T‖μ − μ̄1‖²)`, with `Σ` the sample covariance, `λ_max` its
    /// largest eigenvalue and `T` the number of periods, as in riskfolio-lib's `"JS"` with its
    /// default target: the further apart the means are next to their noise, the less they're
    /// shrunk. It's clamped to `[0, 1]`, the positive-part estimator, so no mean moves past
    /// the average or away from it. With fewer than three assets, or one variance dominating
    /// the rest, `tr Σ ≤ 2λ_max` and these are the sample means; with every mean equal, too.
    ///
    /// Shrinking toward the average keeps the cross-sectional mean, so the portfolio tilts
    /// that come from differences between the means are smaller, and so are their errors.
    ///
    /// # Errors
    ///
    /// [`CutupError::SingularCovariance`] if the covariance's eigenvalues can't be computed,
    /// which takes NaN or a failure to converge.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MaxSharpe, Returns};
    ///
    /// let data = Returns::new(&[
    ///     [ 0.010, -0.020,  0.005,  0.003],
    ///     [-0.004,  0.015,  0.002, -0.001],
    ///     [ 0.007,  0.003, -0.006,  0.004],
    ///     [ 0.002, -0.001,  0.004,  0.000],
    ///     [-0.006,  0.008, -0.003,  0.002],
    /// ])?;
    /// let shrunk = data.james_stein_mean()?;
    /// let weights = data.allocate(&MaxSharpe::default().expected_returns(shrunk))?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    ///
    /// <https://en.wikipedia.org/wiki/James%E2%80%93Stein_estimator>
    pub fn james_stein_mean(&self) -> Result<Vec<f64>, CutupError> {
        let exponent = binary_exponent(self.returns.amax());
        let moments = Moments::new(self, vec![exponent; self.returns.ncols()]);
        let target = moments.mean.mean();
        let deviations = moments.mean.add_scalar(-target);
        let largest = deviations.amax();
        let alpha = if largest == 0.0 {
            0.0
        } else {
            let eigen = SymmetricEigen::new(&moments.cov).ok_or(CutupError::SingularCovariance)?;
            let excess = moments.cov.trace() - 2.0 * eigen.values.max();
            // The deviations divided by a power of two near the largest, exactly, so a spread
            // far below the means can't underflow to 0: ‖μ − μ̄1‖² = spread · 2^(2·shift). An
            // intensity past f64::MAX is clamped to 1.
            let shift = binary_exponent(largest);
            let spread = deviations.map(|d| pow2_scaled(d, -shift)).norm_squared();
            pow2_scaled(excess / (moments.periods * spread), -2 * shift).clamp(0.0, 1.0)
        };
        Ok(moments.shrunk(target, alpha))
    }

    /// The Bayes-Stein estimate of the expected returns (Jorion, 1986): the sample means shrunk
    /// toward the mean of the minimum variance portfolio, `μ̂ = (1 − α)μ + α·μ₀1`, with
    /// `μ₀ = 1ᵀΣ⁻¹μ / 1ᵀΣ⁻¹1` and `α = (n + 2) / (n + 2 + T(μ − μ₀1)ᵀΣ⁻¹(μ − μ₀1))`.
    ///
    /// `Σ` is the sample covariance, `n` the number of assets and `T` of periods: riskfolio-lib's
    /// `"BS"` with target `"b2"`. `α` is in `(0, 1]`: the more the means differ from `μ₀`
    /// next to their noise, the less they're shrunk. Shrinking toward the minimum variance
    /// portfolio's mean, the return estimated most precisely, keeps that portfolio's return.
    ///
    /// It works in correlation units, as the allocation methods do, so it doesn't depend on
    /// each asset's units.
    ///
    /// # Errors
    ///
    /// * [`CutupError::ZeroVariance`] for an asset whose variance is 0 to rounding.
    /// * [`CutupError::SingularCovariance`] if the correlation matrix is singular to rounding:
    ///   an eigenvalue at most `nε` of the largest, as for more assets than periods, or an
    ///   asset that's a combination of others. `Σ⁻¹` doesn't exist then.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{MeanVariance, Returns};
    ///
    /// let data = Returns::new(&[
    ///     [ 0.010, -0.020,  0.005],
    ///     [-0.004,  0.015,  0.002],
    ///     [ 0.007,  0.003, -0.006],
    ///     [ 0.002, -0.001,  0.004],
    ///     [-0.006,  0.008, -0.003],
    /// ])?;
    /// let shrunk = data.bayes_stein_mean()?;
    /// let weights = data.allocate(&MeanVariance::risk_aversion(3.0).expected_returns(shrunk))?;
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn bayes_stein_mean(&self) -> Result<Vec<f64>, CutupError> {
        let n = self.returns.ncols();
        // Each asset's returns divided by a power of two near their largest, exactly, so no
        // asset's variance underflows next to another's: Σ = SDCDS for S the powers of two and
        // D the volatilities in these units.
        let exponents: Vec<i32> = self
            .returns
            .column_iter()
            .map(|column| binary_exponent(column.amax()))
            .collect();
        let moments = Moments::new(self, exponents);
        let exponents = &moments.exponents;
        // The rule of `Returns::find_zero_variance`, which doesn't depend on the scale.
        if let Some(asset) = (0..n).find(|&i| {
            let largest = pow2_scaled(self.returns.column(i).amax(), -exponents[i]);
            moments.cov[(i, i)].sqrt() <= 1e-10 * largest
        }) {
            return Err(self.zero_variance(asset));
        }
        let (correlation, volatilities) = correlation(&moments.cov)?;
        let eigen = SymmetricEigen::new(&correlation).ok_or(CutupError::SingularCovariance)?;
        if eigen.values.min() <= n as f64 * f64::EPSILON * eigen.values.max() {
            return Err(CutupError::SingularCovariance);
        }
        // xᵀΣ⁻¹y = (D⁻¹S⁻¹x)ᵀC⁻¹(D⁻¹S⁻¹y) = Σₑ (vₑ·x')(vₑ·y')/λₑ, for x' = D⁻¹S⁻¹x and
        // y' = D⁻¹S⁻¹y in correlation units, from their projections onto the eigenvectors.
        let project = |x: &DVector<f64>| DVector::from_fn(n, |e, _| eigen.vectors.column(e).dot(x));
        let form = |x: &DVector<f64>, y: &DVector<f64>| -> f64 {
            (0..n).map(|e| x[e] * y[e] / eigen.values[e]).sum()
        };
        // D⁻¹S⁻¹1 and D⁻¹S⁻¹μ, each as a vector and a power of two.
        let ones: Vec<(f64, i32)> = (0..n)
            .map(|i| (1.0 / volatilities[i], -exponents[i]))
            .collect();
        let ones = gather(&ones);
        let means: Vec<(f64, i32)> = (0..n)
            .map(|i| (moments.mean[i] / volatilities[i], 0))
            .collect();
        let means = gather(&means);
        let projected_ones = project(&ones.0);
        // μ₀ = ratio · 2^(means − ones). 1ᵀΣ⁻¹1 > 0, Σ⁻¹ being positive definite.
        let ratio =
            form(&projected_ones, &project(&means.0)) / form(&projected_ones, &projected_ones);
        // D⁻¹S⁻¹(μ − μ₀1), in the means' scale.
        let deviations = project(&DVector::from_fn(n, |i, _| means.0[i] - ratio * ones.0[i]));
        let distance = form(&deviations, &deviations).max(0.0);
        let distance = moments.periods * pow2_scaled(distance, 2 * means.1);
        let alpha = (n as f64 + 2.0) / (n as f64 + 2.0 + distance);
        let target = pow2_scaled(ratio, means.1 - ones.1);
        Ok((0..n)
            .map(|i| pow2_scaled((1.0 - alpha) * moments.mean[i], exponents[i]) + alpha * target)
            .collect())
    }

    /// The exponentially weighted mean of each asset, the last row weighted most.
    fn weighted_mean(&self, decay: Decay) -> Result<Vec<f64>, CutupError> {
        let weights = exponential_weights(self.returns.nrows(), decay.log()?);
        Ok(self.weighted_means(&weights))
    }

    /// Each asset's mean under `weights`, one per row, as from [`exponential_weights`]. The
    /// exponentially weighted covariance centers on these too.
    pub(crate) fn weighted_means(&self, weights: &[f64]) -> Vec<f64> {
        // At least 1, the last period's weight.
        let total: f64 = weights.iter().sum();
        self.returns
            .column_iter()
            .map(|column| {
                // Divided by a power of two near the largest return, exactly, so subnormal
                // returns don't lose their bits to the weights, and the mean scales exactly
                // with the returns.
                let exponent = binary_exponent(column.amax());
                let sum: f64 = weights
                    .iter()
                    .zip(column.iter())
                    .map(|(w, &r)| w * pow2_scaled(r, -exponent))
                    .sum();
                pow2_scaled(sum / total, exponent)
            })
            .collect()
    }
}

/// The sample mean and covariance of the returns, each asset's divided by `2^exponents[i]`,
/// exactly, a power of two near the largest return so that their squares can't overflow or
/// underflow: shrinkage intensities are ratios of the two that don't depend on the scale.
struct Moments {
    exponents: Vec<i32>,
    mean: DVector<f64>,
    cov: DMatrix<f64>,
    periods: f64,
}

impl Moments {
    fn new(data: &Returns, exponents: Vec<i32>) -> Self {
        let (rows, n) = data.returns.shape();
        let mut centered = DMatrix::from_fn(rows, n, |t, i| {
            pow2_scaled(data.returns[(t, i)], -exponents[i])
        });
        let mean = centered.row_mean().transpose();
        for (mut column, m) in centered.column_iter_mut().zip(mean.iter()) {
            column.add_scalar_mut(-m);
        }
        let periods = rows as f64;
        let cov = centered.tr_mul(&centered) / (periods - 1.0);
        Moments {
            exponents,
            mean,
            cov,
            periods,
        }
    }

    /// `(1 − α)μ + α·target`, in the returns' scale, for exponents that are all the same. At
    /// `α = 0` it's the sample mean exactly.
    fn shrunk(&self, target: f64, alpha: f64) -> Vec<f64> {
        self.mean
            .iter()
            .zip(&self.exponents)
            .map(|(&m, &exponent)| pow2_scaled(m + alpha * (target - m), exponent))
            .collect()
    }
}

/// Entries `mᵢ·2^kᵢ` as a vector and one power of two, the vector's largest entry near 1, so
/// none overflows or underflows on the way unless it's negligible next to the largest.
fn gather(entries: &[(f64, i32)]) -> (DVector<f64>, i32) {
    let exponent = entries
        .iter()
        .filter(|(m, _)| *m != 0.0)
        .map(|&(m, k)| binary_exponent(m.abs()) + k)
        .max()
        .unwrap_or(0);
    let vector = DVector::from_iterator(
        entries.len(),
        entries.iter().map(|&(m, k)| pow2_scaled(m, k - exponent)),
    );
    (vector, exponent)
}

/// The weights `λ^(T−1−t)` of `periods` rows for `λ = e^log_decay`, the last row's 1.
pub(crate) fn exponential_weights(periods: usize, log_decay: f64) -> Vec<f64> {
    (0..periods)
        .map(|t| ((periods - 1 - t) as f64 * log_decay).exp())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::test_support::{factor_returns, labeled};
    use crate::{Asset, CutupError, MaxSharpe, MeanVariance, MinVariance, Returns, Weights};
    use nalgebra::{DMatrix, DVector, dmatrix};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * a.abs().max(b.abs())
    }

    fn data(returns: DMatrix<f64>) -> Returns {
        Returns::returns_matrix(returns).unwrap()
    }

    #[test]
    fn exponentially_weighted_mean_by_hand() {
        // λ = 1/2: weights 1/4, 1/2 and 1 from the first row, summing to 7/4.
        let data = data(dmatrix![
            4.0, -8.0;
            2.0, 1.0;
            1.0, 0.0;
        ]);
        let want = vec![(1.0 + 1.0 + 1.0) / 1.75, (-2.0 + 0.5) / 1.75];
        assert_eq!(data.exponentially_weighted_mean(1.0).unwrap(), want);
        assert_eq!(data.exponentially_weighted_mean_decay(0.5).unwrap(), want);
    }

    #[test]
    fn exponentially_weighted_mean_of_long_half_lives_is_the_sample_mean() {
        let returns = factor_returns(4, 30, 3);
        let scale = returns.amax();
        let data = data(returns);
        let sample: Vec<f64> = data.returns.row_mean().iter().copied().collect();
        let means = [1e15, 1e300, f64::MAX]
            .iter()
            .map(|&h| data.exponentially_weighted_mean(h).unwrap())
            .chain([data
                .exponentially_weighted_mean_decay(1.0 - f64::EPSILON)
                .unwrap()]);
        for mean in means {
            // The weights differ from 1 by at most 30·ln 2 / 1e15, about 2e-14.
            for (a, b) in mean.iter().zip(&sample) {
                assert!((a - b).abs() <= 1e-13 * scale, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn exponentially_weighted_mean_of_short_half_lives_is_the_last_return() {
        let returns = factor_returns(3, 25, 5);
        let last: Vec<f64> = returns.row(24).iter().copied().collect();
        let data = data(returns);
        // λ = e^−708: the period before weighs 3.3e-308 against 1, too little to change it.
        let mean = data
            .exponentially_weighted_mean_decay((-708f64).exp())
            .unwrap();
        assert_eq!(mean, last);
        let mean = data
            .exponentially_weighted_mean(std::f64::consts::LN_2 / 707.0)
            .unwrap();
        assert_eq!(mean, last);
    }

    #[test]
    fn exponentially_weighted_mean_weights_recent_periods_more() {
        // Returns that rise each period: the more the recent ones weigh, the higher the mean.
        let data = data(DMatrix::from_fn(10, 1, |t, _| t as f64 * 0.01));
        let means: Vec<f64> = [100.0, 10.0, 3.0, 1.0, 0.3]
            .iter()
            .map(|&h| data.exponentially_weighted_mean(h).unwrap()[0])
            .collect();
        assert!(means.windows(2).all(|m| m[0] < m[1]), "{means:?}");
        assert!(means[0] > 0.045 && means[4] < 0.09, "{means:?}");
    }

    #[test]
    fn exponentially_weighted_mean_rejects_invalid_decays() {
        let data = labeled();
        for half_life in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e-4] {
            assert!(
                matches!(
                    data.exponentially_weighted_mean(half_life),
                    Err(CutupError::InvalidConfig {
                        field: "half_life",
                        ..
                    })
                ),
                "{half_life}"
            );
        }
        for lambda in [0.0, 1.0, -0.5, 1.5, f64::NAN, 1e-308] {
            assert!(
                matches!(
                    data.exponentially_weighted_mean_decay(lambda),
                    Err(CutupError::InvalidConfig { field: "decay", .. })
                ),
                "{lambda}"
            );
        }
    }

    #[test]
    fn exponentially_weighted_mean_scales_exactly_by_powers_of_two() {
        let returns = factor_returns(3, 20, 7);
        let mean = data(returns.clone())
            .exponentially_weighted_mean(4.0)
            .unwrap();
        for k in [-900, -3, 3, 400] {
            let scale = 2f64.powi(k);
            let scaled = data(&returns * scale)
                .exponentially_weighted_mean(4.0)
                .unwrap();
            let want: Vec<f64> = mean.iter().map(|m| m * scale).collect();
            assert_eq!(scaled, want, "{k}");
        }
    }

    #[test]
    fn exponentially_weighted_mean_of_extreme_returns() {
        // Returns as large as the covariance allows, and subnormal ones, which keep their bits.
        let big = data(dmatrix![1e154; 1e154; 5e153]);
        let mean = big.exponentially_weighted_mean(1.0).unwrap()[0];
        assert!(close(mean, 1e154 * 1.25 / 1.75, 1e-15), "{mean}");
        let tiny = data(dmatrix![5e-324; 1e-323; 2e-323]);
        let mean = tiny.exponentially_weighted_mean_decay(0.5).unwrap()[0];
        // (1/4 + 1 + 4) / 1.75 = 3 times 5e-324.
        assert_eq!(mean, 1.5e-323);
    }

    #[test]
    fn exponentially_weighted_mean_feeds_a_method() {
        let data = data(factor_returns(4, 30, 3));
        let mean = data.exponentially_weighted_mean(2.0).unwrap();
        assert_eq!(mean.len(), 4);
        let method = MeanVariance::risk_aversion(100.0);
        let weights = data
            .allocate(&method.clone().expected_returns(mean))
            .unwrap();
        let sample = data.allocate(&method).unwrap();
        assert_ne!(weights, sample);
    }

    /// Sample-covariance CAPM returns written out: betas from the covariance with the market's
    /// return each period.
    fn capm_by_hand(returns: &DMatrix<f64>, market: &[f64], rate: f64) -> Vec<f64> {
        let (t, n) = returns.shape();
        let total: f64 = market.iter().sum();
        let m: Vec<f64> = (0..t)
            .map(|s| (0..n).map(|i| market[i] / total * returns[(s, i)]).sum())
            .collect();
        let mean_m = m.iter().sum::<f64>() / t as f64;
        let cov = |x: &dyn Fn(usize) -> f64| {
            let mean_x = (0..t).map(x).sum::<f64>() / t as f64;
            (0..t)
                .map(|s| (x(s) - mean_x) * (m[s] - mean_m))
                .sum::<f64>()
                / (t - 1) as f64
        };
        let variance = cov(&|s| m[s]);
        (0..n)
            .map(|i| rate + cov(&|s| returns[(s, i)]) / variance * (mean_m - rate))
            .collect()
    }

    #[test]
    fn capm_returns_match_their_definition() {
        let returns = factor_returns(4, 30, 11);
        let data = data(returns.clone());
        for (market, rate) in [
            (vec![1.0, 1.0, 1.0, 1.0], 0.0),
            (vec![400.0, 300.0, 200.0, 100.0], 0.0002),
            (vec![0.0, 1.0, 0.0, 3.0], -0.001),
        ] {
            let got = data.capm_returns(market.clone(), rate).unwrap();
            let want = capm_by_hand(&returns, &market, rate);
            for (a, b) in got.iter().zip(&want) {
                assert!(close(*a, *b, 1e-12), "{market:?}: {got:?} vs {want:?}");
            }
        }
    }

    #[test]
    fn capm_returns_give_the_market_its_mean() {
        // The market weights of the returns are the market's mean, whatever the rate; an asset
        // that is the market gets the market's mean; and at a rate of the market's mean, so
        // does every asset.
        let data = data(factor_returns(3, 25, 13));
        let market = [0.5, 0.3, 0.2];
        let mean = data
            .stats(&Weights::from(market.to_vec()))
            .unwrap()
            .expected_return();
        for rate in [0.0, 0.001, -0.002] {
            let capm = data.capm_returns(market, rate).unwrap();
            let weighted: f64 = capm.iter().zip(market).map(|(m, w)| m * w).sum();
            assert!(close(weighted, mean, 1e-12), "{rate}: {weighted} vs {mean}");
        }
        let capm = data.capm_returns([0.0, 1.0, 0.0], 0.001).unwrap();
        let sample = data.returns.column(1).mean();
        assert!(close(capm[1], sample, 1e-13), "{capm:?} vs {sample}");
        let capm = data.capm_returns(market, mean).unwrap();
        assert!(capm.iter().all(|&m| close(m, mean, 1e-15)), "{capm:?}");
    }

    #[test]
    fn capm_returns_take_market_weights_by_asset() {
        let data = labeled();
        let in_order = data.capm_returns([0.0, 2.0, 1.0], 0.0).unwrap();
        // By asset, every asset is named, as for Black-Litterman.
        let by_asset = data
            .capm_returns(
                vec![
                    (Asset::from("GLD"), 1.0),
                    (Asset::from("SPY"), 0.0),
                    (Asset::from("TLT"), 2.0),
                ],
                0.0,
            )
            .unwrap();
        assert_eq!(in_order, by_asset);
        // Capitalizations are divided by their sum.
        assert_eq!(data.capm_returns([0.0, 0.5, 0.25], 0.0).unwrap(), in_order);
    }

    #[test]
    fn capm_returns_reject_invalid_inputs() {
        let data = labeled();
        let invalid = |field| move |e: CutupError| matches!(e, CutupError::InvalidConfig { field: f, .. } if f == field);
        assert!(invalid("market_weights")(
            data.capm_returns([0.0, 0.0, 0.0], 0.0).unwrap_err()
        ));
        assert!(invalid("risk_free_rate")(
            data.capm_returns([1.0, 1.0, 1.0], f64::NAN).unwrap_err()
        ));
        assert!(invalid("risk_free_rate")(
            data.capm_returns([1.0, 1.0, 1.0], f64::INFINITY)
                .unwrap_err()
        ));
        assert!(matches!(
            data.capm_returns([1.0, -1.0, 1.0], 0.0),
            Err(CutupError::InvalidAssetValue {
                field: "market_weights",
                column: 1,
                ..
            })
        ));
        assert!(matches!(
            data.capm_returns([1.0, f64::NAN, 1.0], 0.0),
            Err(CutupError::InvalidAssetValue {
                field: "market_weights",
                column: 1,
                ..
            })
        ));
        assert!(matches!(
            data.capm_returns([1.0, 1.0], 0.0),
            Err(CutupError::DimensionMismatch {
                field: "market_weights",
                ..
            })
        ));
        assert!(matches!(
            data.capm_returns(vec![(Asset::from("QQQ"), 1.0)], 0.0),
            Err(CutupError::UnknownAsset { .. })
        ));
    }

    #[test]
    fn capm_returns_need_a_risky_market() {
        // All of the market in cash, or in two assets that hedge each other exactly.
        let cash = data(dmatrix![
            0.01, 0.001;
            -0.02, 0.001;
            0.03, 0.001;
        ]);
        assert_eq!(
            cash.capm_returns([0.0, 1.0], 0.0),
            Err(CutupError::RisklessMarket)
        );
        assert!(cash.capm_returns([1.0, 1.0], 0.0).is_ok());
        let mirrored = data(dmatrix![
            0.01, -0.01;
            -0.02, 0.02;
            0.03, -0.03;
        ]);
        assert_eq!(
            mirrored.capm_returns([1.0, 1.0], 0.0),
            Err(CutupError::RisklessMarket)
        );
    }

    #[test]
    fn capm_returns_feed_a_method() {
        let data = data(factor_returns(4, 30, 3));
        let capm = data.capm_returns([1.0, 1.0, 1.0, 1.0], 0.0).unwrap();
        let weights = data
            .allocate(&MaxSharpe::default().allow_short().expected_returns(capm))
            .unwrap();
        // With every excess return proportional to its covariance with the market, the
        // tangency portfolio is the market.
        for w in weights.iter() {
            assert!(close(*w, 0.25, 1e-9), "{weights:?}");
        }
    }

    #[test]
    fn capm_returns_at_extreme_rates() {
        // A rate far beyond the returns: the excess return comes from the market's Sharpe
        // ratio, so it doesn't overflow on the way, and the returns are all about the rate.
        let data = data(factor_returns(3, 20, 17));
        let market = [1.0, 2.0, 3.0];
        let base = data.capm_returns(market, 0.0).unwrap();
        let mean = data
            .stats(&Weights::from(vec![1.0 / 6.0, 2.0 / 6.0, 3.0 / 6.0]))
            .unwrap()
            .expected_return();
        for rate in [1e300, -1e300, f64::MAX, -f64::MAX] {
            let capm = data.capm_returns(market, rate).unwrap();
            for (got, b) in capm.iter().zip(&base) {
                // r_f(1 − β) for β = b / mean, the market's mean negligible next to the rate.
                let beta = b / mean;
                let want = rate * (1.0 - beta);
                assert!(!got.is_nan(), "{rate}: {capm:?}");
                if want.is_finite() {
                    assert!(got.is_finite(), "{rate}: {capm:?} vs {want}");
                    assert!(close(*got, want, 1e-12), "{rate}: {capm:?} vs {want}");
                }
            }
        }
        // A beta of 2 at a rate past f64::MAX / 2: β·(E[r_m] − r_f) is past f64::MAX, but
        // r_f + β(E[r_m] − r_f), about −r_f, isn't.
        let leveraged = Returns::returns_matrix(dmatrix![
            0.01, 0.02;
            -0.01, -0.02;
            0.02, 0.04;
            -0.015, -0.03;
        ])
        .unwrap();
        for rate in [1e308, f64::MAX, -1e308, -f64::MAX] {
            let capm = leveraged.capm_returns([1.0, 0.0], rate).unwrap();
            assert!(capm[1].is_finite(), "{rate}: {capm:?}");
            assert!(close(capm[1], -rate, 1e-12), "{rate}: {capm:?}");
        }
        // Against a market far smaller in scale than the rate, an asset with a beta of 1 still
        // gets about the market's mean, and one uncorrelated with it the rate, though the
        // market's Sharpe ratio is past f64::MAX.
        let data = data_with_scales();
        let capm = data.capm_returns([1.0, 0.0], 1e300).unwrap();
        let mean = data.returns.column(0).mean();
        assert!(capm[0].abs() <= 1e-12 * 1e300, "{capm:?} vs {mean}");
        assert_eq!(capm[1], 1e300);
    }

    #[test]
    fn capm_returns_scale_with_the_returns() {
        // Returns divided by 2^600, near 1e-182, where their covariances underflow: the CAPM
        // returns are divided by exactly the same, and the market's weights give its mean.
        let returns = factor_returns(4, 30, 11);
        let market = [400.0, 300.0, 200.0, 100.0];
        let base = data(returns.clone()).capm_returns(market, 0.0).unwrap();
        let scale = 2f64.powi(-600);
        let small = data(returns.map(|r| r * scale));
        let capm = small.capm_returns(market, 0.0).unwrap();
        for (got, b) in capm.iter().zip(&base) {
            assert_eq!(*got, b * scale, "{capm:?} vs {base:?}");
        }
        let mean = small
            .stats(&Weights::from(vec![0.4, 0.3, 0.2, 0.1]))
            .unwrap()
            .expected_return();
        let weighted: f64 = capm
            .iter()
            .zip([0.4, 0.3, 0.2, 0.1])
            .map(|(m, w)| m * w)
            .sum();
        assert!(close(weighted, mean, 1e-12), "{weighted} vs {mean}");
    }

    #[test]
    fn capm_returns_against_a_hedged_market() {
        // Two assets that hedge each other to about 1e-9, far less than their returns but over
        // the riskless rule: the betas are large, and the market's weights still give its mean.
        let data = data(dmatrix![
            0.01, -0.01 + 2e-9, 0.003;
            -0.02, 0.02 + 1e-9, -0.001;
            0.03, -0.03 - 1e-9, 0.002;
            0.005, -0.005 + 3e-9, 0.0;
        ]);
        let market = [1.0, 1.0, 0.0];
        let capm = data.capm_returns(market, 0.0).unwrap();
        let mean = data
            .stats(&Weights::from(vec![0.5, 0.5, 0.0]))
            .unwrap()
            .expected_return();
        let weighted = 0.5 * capm[0] + 0.5 * capm[1];
        assert!(close(weighted, mean, 1e-6), "{weighted} vs {mean}");
        let want = capm_by_hand(&data.returns, &market, 0.0);
        for (a, b) in capm.iter().zip(&want) {
            assert!(close(*a, *b, 1e-6), "{capm:?} vs {want:?}");
        }
    }

    #[test]
    fn capm_returns_of_assets_with_large_means() {
        // Returns near 1, with spreads near 1e-6: each asset's mean is far from 0 next to its
        // spread, and so is the market's, so the comovement must be of the centered returns.
        let noise = factor_returns(3, 30, 23);
        let returns = noise.map(|r| 1.0 + r * 1e-4);
        let market = [3.0, 2.0, 1.0];
        let data = data(returns.clone());
        let capm = data.capm_returns(market, 0.5).unwrap();
        let want = capm_by_hand(&returns, &market, 0.5);
        for (a, b) in capm.iter().zip(&want) {
            assert!(close(*a, *b, 1e-9), "{capm:?} vs {want:?}");
        }
    }

    /// An asset with returns near 1e-150 and another near 1e150, exactly uncorrelated.
    fn data_with_scales() -> Returns {
        data(dmatrix![
            1e-150, 1e150;
            -1e-150, 1e150;
            1e-150, -1e150;
            -1e-150, -1e150;
        ])
    }

    /// The sample mean and covariance, written out.
    fn sample_moments(returns: &DMatrix<f64>) -> (DVector<f64>, DMatrix<f64>) {
        let mean = returns.row_mean().transpose();
        let t = returns.nrows() as f64;
        let centered = DMatrix::from_fn(returns.nrows(), returns.ncols(), |s, i| {
            returns[(s, i)] - mean[i]
        });
        let cov = centered.transpose() * &centered / (t - 1.0);
        (mean, cov)
    }

    fn assert_close_all(got: &[f64], want: &[f64], tolerance: f64) {
        let scale = want.iter().fold(0.0f64, |m, w| m.max(w.abs()));
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() <= tolerance * scale, "{got:?} vs {want:?}");
        }
    }

    /// Independent returns of about 1% a period, from a fixed linear congruential generator.
    fn independent_returns(assets: usize, periods: usize, seed: u64) -> DMatrix<f64> {
        let mut state = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        DMatrix::from_fn(periods, assets, |_, _| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.04
        })
    }

    /// Five independent assets whose means are close next to their noise, so James-Stein
    /// shrinks them partway.
    fn close_means() -> DMatrix<f64> {
        let noise = independent_returns(5, 12, 29);
        DMatrix::from_fn(12, 5, |t, i| noise[(t, i)] + 0.0002 * i as f64)
    }

    #[test]
    fn james_stein_mean_matches_its_definition() {
        let returns = close_means();
        let (mean, cov) = sample_moments(&returns);
        let target = mean.mean();
        let spread = mean.add_scalar(-target).norm_squared();
        let largest = cov.clone().symmetric_eigen().eigenvalues.max();
        let alpha = (cov.trace() - 2.0 * largest) / (12.0 * spread);
        assert!(0.0 < alpha && alpha < 1.0, "{alpha}");
        let want: Vec<f64> = mean
            .iter()
            .map(|m| (1.0 - alpha) * m + alpha * target)
            .collect();
        let got = data(returns).james_stein_mean().unwrap();
        assert_close_all(&got, &want, 1e-12);
        // The average of the means is kept.
        let average = got.iter().sum::<f64>() / 5.0;
        assert!(close(average, target, 1e-13), "{average} vs {target}");
    }

    #[test]
    fn james_stein_mean_clamps_its_intensity() {
        // Two assets: tr Σ ≤ 2λ_max, so α ≤ 0 and these are the sample means exactly.
        let returns = factor_returns(2, 20, 31);
        let sample: Vec<f64> = returns.row_mean().iter().copied().collect();
        assert_eq!(data(returns).james_stein_mean().unwrap(), sample);
        // Means far closer together than their noise: α would be past 1, so every mean is
        // the average, not past it.
        let noise = independent_returns(5, 12, 37);
        let mean = noise.row_mean();
        let returns = DMatrix::from_fn(12, 5, |t, i| noise[(t, i)] - mean[i] + 1e-9 * i as f64);
        let got = data(returns).james_stein_mean().unwrap();
        let average = got.iter().sum::<f64>() / 5.0;
        for m in &got {
            assert!(close(*m, average, 1e-6), "{got:?}");
        }
        // Every mean equal: nothing to shrink.
        let equal = data(dmatrix![
            0.01, 0.03, -0.01;
            0.03, 0.01, 0.05;
        ]);
        assert_eq!(equal.james_stein_mean().unwrap(), vec![0.02; 3]);
    }

    #[test]
    fn bayes_stein_mean_matches_its_definition() {
        let returns = close_means();
        let (mean, cov) = sample_moments(&returns);
        let inverse = cov.try_inverse().unwrap();
        let ones = DVector::from_element(5, 1.0);
        let target =
            (ones.transpose() * &inverse * &mean)[0] / (ones.transpose() * &inverse * &ones)[0];
        let deviation = mean.add_scalar(-target);
        let distance = (deviation.transpose() * &inverse * &deviation)[0];
        let alpha = 7.0 / (7.0 + 12.0 * distance);
        let want: Vec<f64> = mean
            .iter()
            .map(|m| (1.0 - alpha) * m + alpha * target)
            .collect();
        let got = data(returns).bayes_stein_mean().unwrap();
        assert_close_all(&got, &want, 1e-11);
    }

    #[test]
    fn bayes_stein_mean_keeps_the_minimum_variance_return() {
        // The minimum variance portfolio's return is the target, under the sample means and the
        // shrunk ones alike.
        let returns = factor_returns(4, 30, 41);
        let data = data(returns);
        let weights = data
            .allocate(&MinVariance::default().allow_short())
            .unwrap();
        let shrunk = data.bayes_stein_mean().unwrap();
        let sample = data.returns.row_mean();
        let before: f64 = weights.iter().zip(sample.iter()).map(|(w, m)| w * m).sum();
        let after: f64 = weights.iter().zip(&shrunk).map(|(w, m)| w * m).sum();
        assert!(close(before, after, 1e-10), "{before} vs {after}");
        // And every mean moved toward it, none past it.
        for (s, m) in shrunk.iter().zip(sample.iter()) {
            assert!(
                (s - before).abs() <= (m - before).abs(),
                "{shrunk:?} vs {sample}"
            );
            assert!((s - before) * (m - before) >= 0.0, "{shrunk:?} vs {sample}");
        }
    }

    #[test]
    fn bayes_stein_mean_needs_an_invertible_covariance() {
        let constant = data(dmatrix![
            0.01, 0.002, -0.01;
            -0.02, 0.002, 0.03;
            0.03, 0.002, 0.01;
            0.01, 0.002, -0.02;
        ]);
        assert!(matches!(
            constant.bayes_stein_mean(),
            Err(CutupError::ZeroVariance { column: 1, .. })
        ));
        // More assets than periods, and an asset that's the sum of two others.
        let wide = data(factor_returns(6, 4, 43));
        assert_eq!(wide.bayes_stein_mean(), Err(CutupError::SingularCovariance));
        let base = factor_returns(2, 10, 47);
        let collinear = DMatrix::from_fn(10, 3, |t, i| match i {
            2 => base[(t, 0)] + base[(t, 1)],
            _ => base[(t, i)],
        });
        assert_eq!(
            data(collinear).bayes_stein_mean(),
            Err(CutupError::SingularCovariance)
        );
        // James-Stein needs no inverse.
        assert!(constant.james_stein_mean().is_ok() && wide.james_stein_mean().is_ok());
    }

    #[test]
    fn stein_means_scale_exactly_by_powers_of_two() {
        // Near 1e-182 and 1e118, where the unscaled covariances would underflow or come close
        // to overflowing.
        let returns = close_means();
        let james = data(returns.clone()).james_stein_mean().unwrap();
        let bayes = data(returns.clone()).bayes_stein_mean().unwrap();
        for k in [-600, -3, 3, 400] {
            let scale = 2f64.powi(k);
            let scaled = data(returns.map(|r| r * scale));
            let want = |m: &[f64]| m.iter().map(|x| x * scale).collect::<Vec<_>>();
            assert_eq!(scaled.james_stein_mean().unwrap(), want(&james), "{k}");
            assert_eq!(scaled.bayes_stein_mean().unwrap(), want(&bayes), "{k}");
        }
    }

    #[test]
    fn bayes_stein_mean_of_assets_far_apart_in_scale() {
        // One asset's returns 2^−660, about 2e-199, times the others': its variance would
        // underflow unscaled. With Σ = CΣ₁C and μ = Cμ₁ for C = diag(c), and u = c⁻¹ / 2^660,
        // μ₀ = 2^−660 · uᵀΣ₁⁻¹μ₁ / uᵀΣ₁⁻¹u, and (μ − μ₀1)ᵀΣ⁻¹(μ − μ₀1) is the same form in
        // Σ₁ with μ₁ − μ₀c⁻¹, all of it finite.
        let base = close_means();
        let small = 2f64.powi(-660);
        let returns = DMatrix::from_fn(12, 5, |t, i| {
            base[(t, i)] * if i == 0 { small } else { 1.0 }
        });
        let got = data(returns).bayes_stein_mean().unwrap();

        let (mean, cov) = sample_moments(&base);
        let inverse = cov.try_inverse().unwrap();
        let u = DVector::from_fn(5, |i, _| if i == 0 { 1.0 } else { small });
        let ratio = (u.transpose() * &inverse * &mean)[0] / (u.transpose() * &inverse * &u)[0];
        let target = ratio * small;
        // μ₀c⁻¹: the first entry is μ₀ · 2^660, the rest μ₀.
        let shifted = DVector::from_fn(5, |i, _| mean[i] - if i == 0 { ratio } else { target });
        let distance = (shifted.transpose() * &inverse * &shifted)[0];
        let alpha = 7.0 / (7.0 + 12.0 * distance);
        for i in 0..5 {
            let m = mean[i] * if i == 0 { small } else { 1.0 };
            let want = (1.0 - alpha) * m + alpha * target;
            let scale = if i == 0 { small } else { 1.0 };
            assert!(
                (got[i] - want).abs() <= 1e-11 * 0.02 * scale,
                "{i}: {got:?} vs {want}"
            );
        }
    }

    #[test]
    fn stein_means_feed_a_method() {
        let data = data(close_means());
        for mu in [
            data.james_stein_mean().unwrap(),
            data.bayes_stein_mean().unwrap(),
        ] {
            let method = MeanVariance::risk_aversion(100.0).allow_short();
            let weights = data.allocate(&method.clone().expected_returns(mu)).unwrap();
            assert_ne!(weights, data.allocate(&method).unwrap());
        }
    }

    #[test]
    fn bayes_stein_mean_draws_the_lines_where_its_rules_do() {
        // A second asset that differs from the first by η in four periods: their correlation
        // is 1 − O(η²). At η = 5e-10 the smaller eigenvalue of the correlation matrix, 1 − ρ,
        // is rounding, about 4.7e-16, under nε times the larger, 8.9e-16; at 1.4e-9 it's
        // 3.4e-15, over it.
        let x = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01];
        let step = [1.0, -1.0, -1.0, 1.0, 0.0, 0.0];
        let pair = |eta: f64| {
            data(DMatrix::from_fn(6, 2, |t, i| {
                x[t] + if i == 1 { eta * step[t] } else { 0.0 }
            }))
        };
        assert_eq!(
            pair(5e-10).bayes_stein_mean(),
            Err(CutupError::SingularCovariance)
        );
        assert!(pair(1.4e-9).bayes_stein_mean().is_ok());
        // An asset at 1.9 but for one period 3e-10 above: a volatility of 1.5e-10, under
        // 1e-10 of its largest return, so it's constant, as for an allocation method. At 5e-10
        // above, a volatility of 2.5e-10, it isn't.
        let nearly_constant = |delta: f64| {
            let other = independent_returns(1, 4, 53);
            data(DMatrix::from_fn(4, 2, |t, i| match i {
                0 => other[(t, 0)],
                _ => 1.9 + if t == 3 { delta } else { 0.0 },
            }))
        };
        let constant = nearly_constant(3e-10);
        assert!(matches!(
            constant.bayes_stein_mean(),
            Err(CutupError::ZeroVariance { column: 1, .. })
        ));
        assert_eq!(constant.find_zero_variance(), Some(1));
        assert!(nearly_constant(5e-10).bayes_stein_mean().is_ok());
    }
}
