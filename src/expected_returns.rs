//! Expected-return estimates from the returns, in place of the sample mean, to pass to a
//! method's `expected_returns` setter.

use crate::covariance::Decay;
use crate::stats::{binary_exponent, pow2_scaled};
use crate::{CutupError, Returns};

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
    use crate::{CutupError, MeanVariance, Returns};
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
}
