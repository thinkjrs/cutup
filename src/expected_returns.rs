//! Expected-return estimates from the returns, in place of the sample mean, to pass to a
//! method's `expected_returns` setter.

use crate::black_litterman::market_portfolio;
use crate::covariance::Decay;
use crate::stats::{binary_exponent, pow2_scaled};
use crate::{CutupError, PerAsset, Returns};

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
        // The market's mean and volatility from its return each period, as `stats` gives them.
        let market = self.stats(&self.weights(w.as_slice().to_vec()))?;
        let volatility = market.volatility();
        if volatility == 0.0 {
            return Err(CutupError::RisklessMarket);
        }
        // βᵢ(E[r_m] − r_f) as Cov(rᵢ, r_m) / σ_m, at most σᵢ in size, times the premium over
        // σ_m, each of those two divided by a power of two near it: exactly, and so that a rate
        // far from the returns doesn't overflow on the way to an excess return that doesn't.
        let premium = market.expected_return() - risk_free_rate;
        let (premium_exponent, volatility_exponent) =
            (binary_exponent(premium.abs()), binary_exponent(volatility));
        let ratio =
            pow2_scaled(premium, -premium_exponent) / pow2_scaled(volatility, -volatility_exponent);
        Ok((&self.cov_matrix * &w)
            .iter()
            .map(|&c| {
                let excess = c / volatility * ratio;
                risk_free_rate + pow2_scaled(excess, premium_exponent - volatility_exponent)
            })
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

/// The weights `λ^(T−1−t)` of `periods` rows for `λ = e^log_decay`, the last row's 1.
pub(crate) fn exponential_weights(periods: usize, log_decay: f64) -> Vec<f64> {
    (0..periods)
        .map(|t| ((periods - 1 - t) as f64 * log_decay).exp())
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::test_support::{factor_returns, labeled};
    use crate::{Asset, CutupError, MaxSharpe, MeanVariance, Returns, Weights};
    use nalgebra::{DMatrix, dmatrix};
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
                let want = rate - beta * rate;
                assert!(!got.is_nan(), "{rate}: {capm:?}");
                if want.is_finite() {
                    assert!(close(*got, want, 1e-12), "{rate}: {capm:?} vs {want}");
                }
            }
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

    /// An asset with returns near 1e-150 and another near 1e150, exactly uncorrelated.
    fn data_with_scales() -> Returns {
        data(dmatrix![
            1e-150, 1e150;
            -1e-150, 1e150;
            1e-150, -1e150;
            -1e-150, -1e150;
        ])
    }
}
