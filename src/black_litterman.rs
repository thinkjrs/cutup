//! Black-Litterman expected returns: the market's implied returns, blended with views.

use crate::asset::check_values;
use crate::{Asset, CovarianceConfig, CutupError, PerAsset, Returns};
use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

/// Expected returns from the Black-Litterman model (Black and Litterman, 1992): the returns
/// the market portfolio implies, blended with your views on some assets.
///
/// It estimates expected returns rather than allocating. Pass what
/// [`BlackLitterman::expected_returns`] returns to [`MeanVariance`](crate::MeanVariance) or
/// [`MaxSharpe`](crate::MaxSharpe):
///
/// ```
/// use cutup::{BlackLitterman, MeanVariance, Returns, View};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
/// ];
/// let data = Returns::new(&returns)?;
///
/// let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
///     // Asset 0 returns 0.1% a period, and you're 60% sure of it.
///     .view(View::absolute(0, 0.001).confidence(0.6))
///     // Asset 1 beats asset 2 by 0.05% a period.
///     .view(View::relative(1, 2, 0.0005));
/// let mu = model.expected_returns(&data)?;
///
/// let weights = data.allocate(&MeanVariance::risk_aversion(2.5).expected_returns(mu))?;
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// # The model
///
/// * **Prior:** the market-implied returns `π = δΣw_mkt + r_f`, the returns under which the
///   market portfolio `w_mkt` is mean-variance optimal for a risk aversion `δ`.
/// * **Views:** each [`View`] says a portfolio `p` of the assets returns `q`, with a variance
///   `ω`. Stacking them gives `Pμ = Q + ε` with `ε ~ N(0, Ω)` and `Ω = diag(ω)`.
/// * **Posterior:** `π + τΣPᵀ(τPΣPᵀ + Ω)⁻¹(Q − Pπ)`, where `τ` scales the uncertainty in the
///   prior, 0.05 by default.
///
/// Σ is the covariance of the [`Returns`], after the same adjustments as the allocation methods:
/// use the same shrinkage here as in the method you pass the returns to, unless you pass it the
/// [posterior covariance](BlackLitterman::posterior_covariance), which already has it. Expected
/// returns are in the units and period of the data, so the views and the risk-free rate must be
/// too. `δ` is the same whatever the period, since the mean and the variance of returns both
/// scale with it: 2.5 is a common choice.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BlackLitterman {
    market_weights: PerAsset,
    risk_aversion: f64,
    risk_free_rate: f64,
    tau: f64,
    views: Vec<View>,
    covariance: CovarianceConfig,
}

impl BlackLitterman {
    /// A model with the market weights `market_weights`, in column order or by asset, and the
    /// market's risk aversion `delta`. Market capitalizations work too: they're divided by
    /// their sum.
    pub fn new(market_weights: impl Into<PerAsset>, delta: f64) -> Self {
        Self {
            market_weights: market_weights.into(),
            risk_aversion: delta,
            risk_free_rate: 0.0,
            tau: 0.05,
            views: Vec::new(),
            covariance: CovarianceConfig::default(),
        }
    }

    /// Adds a view.
    pub fn view(mut self, view: View) -> Self {
        self.views.push(view);
        self
    }

    /// Sets the risk-free rate per period, added to the implied returns. Defaults to 0.
    pub fn risk_free_rate(mut self, risk_free_rate: f64) -> Self {
        self.risk_free_rate = risk_free_rate;
        self
    }

    /// Sets `τ`, the uncertainty in the prior relative to the covariance. Defaults to 0.05.
    /// Larger values move the posterior further toward the views.
    pub fn tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    covariance_setters!();

    /// The market-implied returns `π = δΣw_mkt + r_f`, before any views.
    ///
    /// # Errors
    ///
    /// * [`CutupError::DimensionMismatch`] if there isn't one market weight per asset.
    /// * [`CutupError::UnknownAsset`] or [`CutupError::DuplicateAsset`] if the market weights
    ///   name an asset that isn't in the data, or name one twice.
    /// * [`CutupError::InvalidAssetValue`] if a market weight is negative or not finite.
    /// * [`CutupError::InvalidConfig`] if the market weights sum to 0, `δ` isn't positive and
    ///   finite, the risk-free rate isn't finite, or the covariance config is invalid.
    pub fn prior_returns(&self, data: &Returns) -> Result<Vec<f64>, CutupError> {
        let cov = data.adjusted_covariance(&self.covariance)?;
        Ok(self.prior(&cov, data.labels())?.as_slice().to_vec())
    }

    /// The posterior expected returns, one per asset in column order. With no views, these are
    /// the [prior returns](BlackLitterman::prior_returns).
    ///
    /// # Errors
    ///
    /// * Any error from [`BlackLitterman::prior_returns`].
    /// * [`CutupError::InvalidConfig`] if `τ` isn't positive and finite, or a view is invalid:
    ///   a portfolio with no nonzero weight, a return that isn't finite, a variance that is
    ///   negative or not finite, or a confidence outside `[0, 1]`.
    /// * [`CutupError::InvalidAssetValue`] if a view portfolio's weight isn't finite.
    /// * [`CutupError::UnknownAsset`] if a view names an asset that isn't in the data.
    /// * [`CutupError::DuplicateAsset`] for a relative view of an asset against itself, or a
    ///   portfolio that names an asset twice.
    /// * [`CutupError::DimensionMismatch`] if a [`View::portfolio`] in column order doesn't
    ///   have one weight per asset.
    /// * [`CutupError::DegenerateViews`] if `τPΣPᵀ + Ω` is singular, or within rounding of it:
    ///   certain views that repeat or contradict each other, or a view with the default
    ///   variance or a confidence on a portfolio with no variance, which would make it
    ///   certain. Give such a view an explicit [`View::variance`].
    pub fn expected_returns(&self, data: &Returns) -> Result<Vec<f64>, CutupError> {
        let solved = self.solve(data)?;
        let Some(views) = &solved.views else {
            return Ok(solved.pi.as_slice().to_vec());
        };
        let x = views.factor.solve(&views.surprise);
        let posterior = &solved.pi + &views.tau_sigma_pt * x;
        if posterior.iter().any(|m| !m.is_finite()) {
            return Err(CutupError::DegenerateViews);
        }
        Ok(posterior.as_slice().to_vec())
    }

    /// The posterior covariance of the returns, `Σ + τΣ − τΣPᵀ(τPΣPᵀ + Ω)⁻¹PτΣ` (He and
    /// Litterman, 1999): the covariance `Σ`, plus the uncertainty `τΣ` in the prior's expected
    /// returns, less what the views resolve of it. It's PyPortfolioOpt's `bl_cov` and the
    /// covariance riskfolio-lib's `black_litterman` returns.
    ///
    /// It comes as a [`CovarianceConfig`] holding it as a [given
    /// covariance](CovarianceConfig::matrix), to pass with the [posterior
    /// returns](BlackLitterman::expected_returns) to a method's `covariance` setter. A fixed
    /// shrinkage and regularization can still be set on it, on top of whatever `Σ` already had,
    /// but not [Ledoit-Wolf shrinkage](CovarianceConfig::ledoit_wolf), which is for the sample
    /// covariance: set that on the model instead.
    ///
    /// With no views, or only views of confidence 0, it's `(1 + τ)Σ`. A certain view leaves the
    /// variance of its portfolio at `pᵀΣp`: the uncertainty in its expected return is gone.
    /// The result is positive semidefinite whenever `Σ` is, and between `Σ` and `(1 + τ)Σ`.
    ///
    /// # Errors
    ///
    /// The same as [`BlackLitterman::expected_returns`], except for the
    /// [`CutupError::DegenerateViews`] it returns when the posterior returns overflow, which
    /// this doesn't compute, and [`CutupError::NonFiniteCovariance`] if the result overflows.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::{BlackLitterman, MeanVariance, Returns, View};
    ///
    /// let data = Returns::new(&[
    ///      [ 0.010, -0.020,  0.005],
    ///      [-0.004,  0.015,  0.002],
    ///      [ 0.007,  0.003, -0.006],
    ///      [ 0.002, -0.001,  0.004],
    ///      [-0.003,  0.006,  0.001],
    /// ])?;
    /// let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
    ///     .view(View::relative(1, 2, 0.0005).confidence(0.7));
    ///
    /// let method = MeanVariance::risk_aversion(2.5)
    ///     .expected_returns(model.expected_returns(&data)?)
    ///     .covariance(model.posterior_covariance(&data)?);
    /// let weights = data.allocate(&method)?;
    /// assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn posterior_covariance(&self, data: &Returns) -> Result<CovarianceConfig, CutupError> {
        let solved = self.solve(data)?;
        let mut posterior = &solved.cov * (1.0 + self.tau);
        if let Some(views) = &solved.views {
            // τΣPᵀ(τPΣPᵀ + Ω)⁻¹PτΣ = YᵀY for Y = L⁻¹PτΣ, with L the views' Cholesky factor, so
            // it's symmetric and positive semidefinite by construction.
            let y = views
                .factor
                .l_dirty()
                .solve_lower_triangular(&views.tau_sigma_pt.transpose())
                .ok_or(CutupError::DegenerateViews)?;
            posterior -= y.tr_mul(&y);
        }
        if posterior.iter().any(|x| !x.is_finite()) {
            return Err(CutupError::NonFiniteCovariance);
        }
        Ok(CovarianceConfig::computed(posterior))
    }

    /// The covariance, the prior and the views, checked and factored: what the posterior
    /// returns and covariance share.
    fn solve(&self, data: &Returns) -> Result<Solved, CutupError> {
        let cov = data.adjusted_covariance(&self.covariance)?;
        let pi = self.prior(&cov, data.labels())?;
        if !(self.tau.is_finite() && self.tau > 0.0) {
            return Err(CutupError::InvalidConfig {
                field: "tau",
                value: self.tau,
            });
        }
        let n = cov.ncols();
        // Each view's ω is `scale · τpᵀΣp + given`: a scale of 1 by default, (1 − c)/c for a
        // confidence c, or 0 with a given variance.
        let mut views = Vec::with_capacity(self.views.len());
        for view in &self.views {
            let p = view.weights(data.labels())?;
            if !view.q.is_finite() {
                return Err(CutupError::InvalidConfig {
                    field: "view_return",
                    value: view.q,
                });
            }
            let (scale, given) = match view.uncertainty {
                Uncertainty::Default => (1.0, 0.0),
                Uncertainty::Variance(omega) if omega.is_finite() && omega >= 0.0 => (0.0, omega),
                Uncertainty::Variance(omega) => {
                    return Err(CutupError::InvalidConfig {
                        field: "view_variance",
                        value: omega,
                    });
                }
                // Idzorek's confidence, in the closed form of Walters (2014). At 0, or so close
                // that (1 − c)/c overflows, the view's variance is infinite and it has no effect.
                Uncertainty::Confidence(c) if (0.0..=1.0).contains(&c) => {
                    let scale = (1.0 - c) / c;
                    if !scale.is_finite() {
                        continue;
                    }
                    (scale, 0.0)
                }
                Uncertainty::Confidence(c) => {
                    return Err(CutupError::InvalidConfig {
                        field: "view_confidence",
                        value: c,
                    });
                }
            };
            views.push((p, view.q, scale, given));
        }
        if views.is_empty() {
            return Ok(Solved {
                cov,
                pi,
                views: None,
            });
        }

        let k = views.len();
        let p = DMatrix::from_fn(k, n, |i, j| views[i].0[j]);
        let q = DVector::from_fn(k, |i, _| views[i].1);
        let tau_sigma_pt = &cov * p.transpose() * self.tau;
        let mut a = &p * &tau_sigma_pt;
        let volatilities = cov.diagonal().map(f64::sqrt);
        for (i, (p, _, scale, given)) in views.iter().enumerate() {
            let spread = a[(i, i)];
            // A view whose variance follows its portfolio's can't be on a portfolio with none:
            // it would be certain. Within rounding of a perfectly hedged portfolio, whose
            // variance would otherwise be (Σ|pⱼ|σⱼ)², counts as none.
            let unhedged = p.abs().dot(&volatilities).powi(2) * self.tau;
            if *scale > 0.0 && spread <= DEGENERATE * unhedged {
                return Err(CutupError::DegenerateViews);
            }
            a[(i, i)] += scale * spread + given;
        }
        let factor = factor_views(a)?;
        let surprise = q - &p * &pi;
        Ok(Solved {
            cov,
            pi,
            views: Some(SolvedViews {
                tau_sigma_pt,
                factor,
                surprise,
            }),
        })
    }

    fn prior(&self, cov: &DMatrix<f64>, labels: &[String]) -> Result<DVector<f64>, CutupError> {
        let n = cov.ncols();
        let market_weights = self
            .market_weights
            .resolve("market_weights", labels, None)?;
        check_values("market_weights", labels, &market_weights, |w| {
            w.is_finite() && w >= 0.0
        })?;
        let total: f64 = market_weights.iter().sum();
        if !(total.is_finite() && total > 0.0) {
            return Err(CutupError::InvalidConfig {
                field: "market_weights",
                value: total,
            });
        }
        if !(self.risk_aversion.is_finite() && self.risk_aversion > 0.0) {
            return Err(CutupError::InvalidConfig {
                field: "risk_aversion",
                value: self.risk_aversion,
            });
        }
        if !self.risk_free_rate.is_finite() {
            return Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: self.risk_free_rate,
            });
        }
        let w = DVector::from_iterator(n, market_weights.iter().map(|w| w / total));
        Ok((cov * w) * self.risk_aversion + DVector::from_element(n, self.risk_free_rate))
    }
}

/// The covariance, the prior, and the views if any are left, factored.
struct Solved {
    cov: DMatrix<f64>,
    pi: DVector<f64>,
    views: Option<SolvedViews>,
}

/// The views: `τΣPᵀ`, the Cholesky factor of `τPΣPᵀ + Ω`, and `Q − Pπ`.
struct SolvedViews {
    tau_sigma_pt: DMatrix<f64>,
    factor: Cholesky<f64, Dyn>,
    surprise: DVector<f64>,
}

/// How close to 0, relative to its scale, a view's variance or the part of it no other view
/// explains can be before the views are degenerate.
const DEGENERATE: f64 = 1e-12;

/// The Cholesky factor of the views' covariance `A = τPΣPᵀ + Ω`, rejecting views that repeat
/// or contradict each other: each pivot is the part of a view's variance the earlier views
/// don't explain, which must not vanish relative to the variance itself.
fn factor_views(a: DMatrix<f64>) -> Result<Cholesky<f64, Dyn>, CutupError> {
    let diagonal = a.diagonal();
    let cholesky = a.cholesky().ok_or(CutupError::DegenerateViews)?;
    let l = cholesky.l_dirty();
    if (0..diagonal.len()).any(|i| l[(i, i)].powi(2) <= DEGENERATE * diagonal[i]) {
        return Err(CutupError::DegenerateViews);
    }
    Ok(cholesky)
}

/// A view for [`BlackLitterman`]: a portfolio of the assets has an expected return `q`.
///
/// How sure you are sets how far it moves the expected returns. By default the view's
/// variance is `τpᵀΣp`, in proportion to the portfolio's own variance (He and Litterman,
/// 1999). Set it with [`View::variance`], or as a confidence with [`View::confidence`].
///
/// ```
/// use cutup::View;
///
/// let absolute = View::absolute(0, 0.001).confidence(0.6);
/// let relative = View::relative(1, 2, 0.0005).variance(1e-7);
/// let basket = View::portfolio([0.5, 0.5, -1.0], 0.0002);
///
/// // By label, once the data is labeled.
/// let labeled = View::relative("SPY", "TLT", 0.0005);
/// let labeled_basket = View::portfolio([("SPY", 0.5), ("QQQ", 0.5), ("TLT", -1.0)], 0.0002);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct View {
    assets: Assets,
    q: f64,
    uncertainty: Uncertainty,
}

#[derive(Debug, Clone, PartialEq)]
enum Assets {
    Absolute(Asset),
    Relative(Asset, Asset),
    Portfolio(PerAsset),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Uncertainty {
    Default,
    Variance(f64),
    Confidence(f64),
}

impl View {
    /// `asset`, by label or by column, has an expected return of `q`.
    pub fn absolute(asset: impl Into<Asset>, q: f64) -> Self {
        Self::new(Assets::Absolute(asset.into()), q)
    }

    /// `long` has an expected return `q` above `short`'s, each by label or by column.
    pub fn relative(long: impl Into<Asset>, short: impl Into<Asset>, q: f64) -> Self {
        Self::new(Assets::Relative(long.into(), short.into()), q)
    }

    /// The portfolio with `weights`, in column order or by asset, has an expected return of
    /// `q`. The weights needn't sum to anything: `[0.5, 0.5, -1.0]` says the first two assets
    /// together beat the third by `q`. By asset, the assets left out have a weight of 0.
    pub fn portfolio(weights: impl Into<PerAsset>, q: f64) -> Self {
        Self::new(Assets::Portfolio(weights.into()), q)
    }

    fn new(assets: Assets, q: f64) -> Self {
        Self {
            assets,
            q,
            uncertainty: Uncertainty::Default,
        }
    }

    /// Sets the view's variance `ω`. Zero makes the view certain, so the expected returns
    /// meet it exactly. Replaces [`View::confidence`].
    pub fn variance(mut self, omega: f64) -> Self {
        self.uncertainty = Uncertainty::Variance(omega);
        self
    }

    /// Sets how sure you are of the view, from 0 to 1 (Idzorek, 2005). At 1 the view is
    /// certain and at 0 it has no effect; at 0.5 its variance is the default, `τpᵀΣp`. In
    /// general the variance is `τ(1 − c)/c · pᵀΣp`, the closed form of Walters (2014), as in
    /// PyPortfolioOpt. At 0 the view is dropped, where PyPortfolioOpt gives it a variance of
    /// 1e6, whose effect depends on the units of the returns. Replaces [`View::variance`].
    pub fn confidence(mut self, confidence: f64) -> Self {
        self.uncertainty = Uncertainty::Confidence(confidence);
        self
    }

    /// The view's portfolio as a vector over the assets `labels`.
    fn weights(&self, labels: &[String]) -> Result<DVector<f64>, CutupError> {
        let n = labels.len();
        match &self.assets {
            Assets::Absolute(asset) => {
                let asset = asset.resolve(labels)?;
                Ok(DVector::from_fn(
                    n,
                    |i, _| if i == asset { 1.0 } else { 0.0 },
                ))
            }
            Assets::Relative(long, short) => {
                let (long, short) = (long.resolve(labels)?, short.resolve(labels)?);
                if long == short {
                    return Err(CutupError::DuplicateAsset {
                        column: long,
                        asset: labels[long].clone(),
                    });
                }
                Ok(DVector::from_fn(n, |i, _| match i {
                    i if i == long => 1.0,
                    i if i == short => -1.0,
                    _ => 0.0,
                }))
            }
            Assets::Portfolio(weights) => {
                let weights = weights.resolve("view_weights", labels, Some(0.0))?;
                check_values("view_weights", labels, &weights, f64::is_finite)?;
                if weights.iter().all(|&w| w == 0.0) {
                    return Err(CutupError::InvalidConfig {
                        field: "view_weights",
                        value: 0.0,
                    });
                }
                Ok(DVector::from_vec(weights))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{assert_reference, datasets, factor_returns};
    use crate::{MaxSharpe, MeanVariance};
    use nalgebra::dmatrix;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn allocator() -> Returns {
        Returns::returns_matrix(sample()).unwrap()
    }

    fn sample() -> DMatrix<f64> {
        dmatrix![
             0.010, -0.020,  0.005;
            -0.004,  0.015,  0.002;
             0.007,  0.003, -0.006;
             0.002, -0.001,  0.004;
            -0.003,  0.006,  0.001;
             0.004,  0.002, -0.003
        ]
    }

    fn close(a: &[f64], b: &[f64], tolerance: f64) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tolerance)
    }

    #[test]
    fn no_views_give_the_prior() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5).risk_free_rate(1e-4);
        let prior = model.prior_returns(&allocator).unwrap();
        assert_eq!(model.expected_returns(&allocator).unwrap(), prior);

        // δΣw + r_f, by hand.
        let cov = allocator
            .adjusted_covariance(&CovarianceConfig::default())
            .unwrap();
        let w = [0.5, 0.3, 0.2];
        for (i, &pi) in prior.iter().enumerate() {
            let expected = 2.5 * (0..3).map(|j| cov[(i, j)] * w[j]).sum::<f64>() + 1e-4;
            assert!((pi - expected).abs() < 1e-18);
        }
    }

    #[test]
    fn market_caps_are_normalized() {
        let allocator = allocator();
        assert_eq!(
            BlackLitterman::new([500.0, 300.0, 200.0], 2.5).prior_returns(&allocator),
            BlackLitterman::new([0.5, 0.3, 0.2], 2.5).prior_returns(&allocator)
        );
    }

    #[test]
    fn certain_views_are_met_exactly() {
        let allocator = allocator();
        let mu = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
            .view(View::absolute(0, 0.002).variance(0.0))
            .view(View::relative(1, 2, 0.001).confidence(1.0))
            .expected_returns(&allocator)
            .unwrap();
        assert!((mu[0] - 0.002).abs() < 1e-15);
        assert!((mu[1] - mu[2] - 0.001).abs() < 1e-15);
    }

    #[test]
    fn views_without_confidence_change_nothing() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let prior = model.prior_returns(&allocator).unwrap();
        let ignored = model
            .clone()
            .view(View::absolute(0, 0.05).confidence(0.0))
            .expected_returns(&allocator)
            .unwrap();
        assert_eq!(ignored, prior);
        let vague = model
            .view(View::absolute(0, 0.05).variance(1e12))
            .expected_returns(&allocator)
            .unwrap();
        assert!(close(&vague, &prior, 1e-15));
    }

    #[test]
    fn half_confidence_is_the_default_variance() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5).tau(0.1);
        let default = model
            .clone()
            .view(View::relative(0, 1, 0.003))
            .expected_returns(&allocator)
            .unwrap();
        let half = model
            .view(View::relative(0, 1, 0.003).confidence(0.5))
            .expected_returns(&allocator)
            .unwrap();
        assert!(close(&default, &half, 1e-18));
    }

    #[test]
    fn rescaling_a_default_view_changes_nothing() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let view = model
            .clone()
            .view(View::portfolio([1.0, 0.0, -1.0], 0.001))
            .expected_returns(&allocator)
            .unwrap();
        let scaled = model
            .clone()
            .view(View::portfolio([4.0, 0.0, -4.0], 0.004))
            .expected_returns(&allocator)
            .unwrap();
        let relative = model
            .view(View::relative(0, 2, 0.001))
            .expected_returns(&allocator)
            .unwrap();
        assert!(close(&view, &scaled, 1e-16));
        assert_eq!(view, relative);
    }

    #[test]
    fn more_confidence_moves_closer_to_the_view() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let prior = model.prior_returns(&allocator).unwrap()[0];
        let mut last = prior;
        for confidence in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let mu = model
                .clone()
                .view(View::absolute(0, 0.01).confidence(confidence))
                .expected_returns(&allocator)
                .unwrap()[0];
            assert!(mu > last && mu <= 0.01 + 1e-15, "{confidence}: {mu}");
            last = mu;
        }
    }

    #[test]
    fn invalid_input_is_rejected() {
        let allocator = allocator();
        let model = || BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let invalid =
            |field: &'static str, value: f64| Err(CutupError::InvalidConfig { field, value });
        let check = |model: BlackLitterman| model.expected_returns(&allocator);

        assert_eq!(
            check(BlackLitterman::new([0.5, 0.5], 2.5)),
            Err(CutupError::DimensionMismatch {
                field: "market_weights",
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            check(BlackLitterman::new([0.5, -0.3, 0.2], 2.5)),
            Err(CutupError::InvalidAssetValue {
                field: "market_weights",
                column: 1,
                asset: "1".into(),
                value: -0.3
            })
        );
        assert_eq!(
            check(BlackLitterman::new([0.0; 3], 2.5)),
            invalid("market_weights", 0.0)
        );
        assert_eq!(
            check(BlackLitterman::new([0.5, 0.3, 0.2], 0.0)),
            invalid("risk_aversion", 0.0)
        );
        assert_eq!(
            check(model().risk_free_rate(f64::NAN)).map_err(|e| e.to_string()),
            Err(CutupError::InvalidConfig {
                field: "risk_free_rate",
                value: f64::NAN
            }
            .to_string())
        );
        assert_eq!(check(model().tau(-1.0)), invalid("tau", -1.0));
        assert_eq!(
            check(model().view(View::absolute(3, 0.01))),
            Err(CutupError::UnknownAsset {
                asset: Asset::from(3)
            })
        );
        assert_eq!(
            check(model().view(View::relative(1, 1, 0.01))),
            Err(CutupError::DuplicateAsset {
                column: 1,
                asset: "1".into()
            })
        );
        assert_eq!(
            check(model().view(View::portfolio([0.0; 3], 0.01))),
            invalid("view_weights", 0.0)
        );
        assert_eq!(
            check(model().view(View::portfolio([1.0, 0.0], 0.01))),
            Err(CutupError::DimensionMismatch {
                field: "view_weights",
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            check(model().view(View::absolute(0, f64::INFINITY))),
            invalid("view_return", f64::INFINITY)
        );
        assert_eq!(
            check(model().view(View::absolute(0, 0.01).variance(-1.0))),
            invalid("view_variance", -1.0)
        );
        assert_eq!(
            check(model().view(View::absolute(0, 0.01).confidence(1.5))),
            invalid("view_confidence", 1.5)
        );
        assert_eq!(check(model().shrinkage(2.0)), invalid("shrinkage", 2.0));
    }

    #[test]
    fn conflicting_certain_views_are_degenerate() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
            .view(View::absolute(0, 0.01).variance(0.0))
            .view(View::portfolio([2.0, 0.0, 0.0], 0.03).variance(0.0));
        assert_eq!(
            model.expected_returns(&allocator),
            Err(CutupError::DegenerateViews)
        );
        // The same views, nearly but not exactly repeated, are as unreliable.
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
            .view(View::absolute(0, 0.01).variance(0.0))
            .view(View::portfolio([1.0, 1e-9, 0.0], 0.01).variance(0.0));
        assert_eq!(
            model.expected_returns(&allocator),
            Err(CutupError::DegenerateViews)
        );
    }

    #[test]
    fn views_on_riskless_portfolios_need_a_variance() {
        // Assets 0 and 1 move together, so going long one and short the other has no variance.
        let allocator = Returns::returns_matrix(dmatrix![
             0.01,  0.01,  0.004;
            -0.02, -0.02,  0.001;
             0.005, 0.005, -0.003
        ])
        .unwrap();
        let model = BlackLitterman::new([0.4, 0.4, 0.2], 2.5);
        for view in [
            View::relative(0, 1, 0.001),
            View::relative(0, 1, 0.001).confidence(0.5),
        ] {
            assert_eq!(
                model.clone().view(view).expected_returns(&allocator),
                Err(CutupError::DegenerateViews)
            );
        }
        let mu = model
            .view(View::relative(0, 1, 0.001).variance(1e-6))
            .expected_returns(&allocator)
            .unwrap();
        assert!(mu.iter().all(|m| m.is_finite()));
    }

    #[test]
    fn nearly_riskless_views_are_degenerate_relative_to_their_assets() {
        // Asset 1 is asset 0 plus `gap` times a noise, so going long 0 and short 1 has a
        // variance about gap² times the assets'. Below 1e-12 of it, the view is degenerate.
        let base = sample();
        let returns = |gap: f64| {
            Returns::returns_matrix(DMatrix::from_fn(6, 3, |i, j| match j {
                1 => base[(i, 0)] + gap * base[(i, 2)],
                _ => base[(i, j)],
            }))
            .unwrap()
        };
        let model = BlackLitterman::new([0.4, 0.4, 0.2], 2.5).view(View::relative(0, 1, 0.001));
        assert!(model.expected_returns(&returns(1e-5)).is_ok());
        assert_eq!(
            model.expected_returns(&returns(1e-7)),
            Err(CutupError::DegenerateViews)
        );
    }

    #[test]
    fn views_on_assets_with_tiny_variances_are_not_degenerate() {
        let scaled = Returns::returns_matrix(sample() * 1e-7).unwrap();
        let mu = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
            .view(View::absolute(0, 1e-9))
            .expected_returns(&scaled)
            .unwrap();
        assert!(mu.iter().all(|m| m.is_finite()));
    }

    #[test]
    fn a_zero_tau_is_rejected() {
        assert_eq!(
            BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
                .tau(0.0)
                .expected_returns(&allocator()),
            Err(CutupError::InvalidConfig {
                field: "tau",
                value: 0.0
            })
        );
    }

    #[test]
    fn vanishing_confidence_drops_the_view() {
        let allocator = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        let prior = model.prior_returns(&allocator).unwrap();
        for confidence in [0.0, 1e-310, f64::MIN_POSITIVE] {
            let mu = model
                .clone()
                .view(View::absolute(0, 0.05).confidence(confidence))
                .expected_returns(&allocator)
                .unwrap();
            assert!(close(&mu, &prior, 1e-15), "{confidence}: {mu:?}");
        }
    }

    /// The posterior covariance `model` gives on `data`, as a matrix.
    fn posterior(model: &BlackLitterman, data: &Returns) -> DMatrix<f64> {
        data.adjusted_covariance(&model.posterior_covariance(data).unwrap())
            .unwrap()
    }

    /// The smallest eigenvalue of `a − b`, relative to the largest of `b`.
    fn smallest_eigenvalue_of_difference(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        let largest = b.clone().symmetric_eigen().eigenvalues.amax();
        (a - b).symmetric_eigen().eigenvalues.min() / largest
    }

    /// Against the Black-Litterman cases in `tests/fixtures/covariance.json`, from
    /// `reference/covariance.py`: the formula, cross-checked against PyPortfolioOpt's `bl_cov`
    /// and riskfolio-lib's `black_litterman`.
    #[test]
    fn posterior_covariance_matches_the_python_reference() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/covariance.json")).unwrap();
        let cases = fixture["black_litterman"].as_array().unwrap();
        assert_eq!(cases.len(), 12);
        let datasets = datasets();
        let floats = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        for case in cases {
            let name = case["dataset"].as_str().unwrap();
            let (_, data) = datasets.iter().find(|(n, _)| n == name).unwrap();
            let n = data.labels().len();
            let mut model = BlackLitterman::new(vec![1.0 / n as f64; n], 2.5)
                .tau(case["tau"].as_f64().unwrap());
            for view in case["views"].as_array().unwrap() {
                let q = view["q"].as_f64().unwrap();
                let asset = |i: usize| view["assets"][i].as_u64().unwrap() as usize;
                let mut built = match view["kind"].as_str().unwrap() {
                    "absolute" => View::absolute(asset(0), q),
                    "relative" => View::relative(asset(0), asset(1), q),
                    _ => View::portfolio(floats(&view["weights"]), q),
                };
                if let Some(omega) = view["variance"].as_f64() {
                    built = built.variance(omega);
                }
                if let Some(confidence) = view["confidence"].as_f64() {
                    built = built.confidence(confidence);
                }
                model = model.view(built);
            }
            let config = model.posterior_covariance(data).unwrap();
            let what = format!("{name} {}", case["views_name"]);
            assert_reference(data, &config, case, &what);
        }
    }

    #[test]
    fn no_views_give_the_prior_uncertainty_on_top() {
        let data = allocator();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5).tau(0.1);
        assert_eq!(posterior(&model, &data), &data.cov_matrix * 1.1);
        // Views of confidence 0 are dropped.
        let dropped = model.clone().view(View::absolute(0, 0.01).confidence(0.0));
        assert_eq!(posterior(&dropped, &data), &data.cov_matrix * 1.1);
        // The covariance config applies first.
        let shrunk = model.shrinkage(0.4);
        let sigma = data
            .adjusted_covariance(&CovarianceConfig::default().shrinkage(0.4))
            .unwrap();
        assert_eq!(posterior(&shrunk, &data), sigma * 1.1);
    }

    #[test]
    fn views_resolve_the_uncertainty_in_their_portfolios() {
        let data = allocator();
        let cov = &data.cov_matrix;
        let tau = 0.05;
        let p = DVector::from_vec(vec![0.0, 1.0, -1.0]);
        let variance = |m: &DMatrix<f64>| p.dot(&(m * &p));
        let base = BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        // A certain view leaves its portfolio's variance at pᵀΣp, and a default one halfway:
        // with ω = τpᵀΣp, the uncertainty τpᵀΣp is halved.
        let certain = posterior(
            &base.clone().view(View::relative(1, 2, 0.001).variance(0.0)),
            &data,
        );
        assert!((variance(&certain) - variance(cov)).abs() <= 1e-15 * variance(cov));
        let default = posterior(&base.clone().view(View::relative(1, 2, 0.001)), &data);
        let want = variance(cov) * (1.0 + tau / 2.0);
        assert!((variance(&default) - want).abs() <= 1e-15 * want);
        // Exactly symmetric, and between Σ and (1 + τ)Σ.
        for m in [&certain, &default] {
            assert_eq!(m, &m.transpose());
            assert!(smallest_eigenvalue_of_difference(m, cov) >= -1e-14);
            assert!(smallest_eigenvalue_of_difference(&(cov * (1.0 + tau)), m) >= -1e-14);
        }
    }

    #[test]
    fn posterior_covariance_errors_match_expected_returns() {
        let data = allocator();
        let base = BlackLitterman::new([0.5, 0.3, 0.2], 2.5);
        for model in [
            base.clone().tau(0.0),
            base.clone().tau(-1.0),
            BlackLitterman::new([0.5, 0.5], 2.5),
            BlackLitterman::new([0.0, 0.0, 0.0], 2.5),
            base.clone().view(View::absolute(5, 0.01)),
            base.clone().view(View::absolute(0, f64::INFINITY)),
            base.clone()
                .view(View::absolute(0, 0.01).variance(0.0))
                .view(View::absolute(0, 0.02).variance(0.0)),
            base.clone().shrinkage(2.0),
        ] {
            let error = model.expected_returns(&data).unwrap_err();
            assert_eq!(model.posterior_covariance(&data).err(), Some(error));
        }
    }

    #[test]
    fn a_posterior_covariance_that_overflows_is_rejected() {
        // Returns near 1e151 have variances near 1e301, so (1 + τ)Σ overflows at τ = 1e10.
        let data = Returns::returns_matrix(sample() * 1e153).unwrap();
        let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5).tau(1e10);
        assert_eq!(
            model.posterior_covariance(&data),
            Err(CutupError::NonFiniteCovariance)
        );
    }

    #[test]
    fn the_posterior_goes_to_any_method() {
        let data = Returns::returns_matrix(factor_returns(4, 30, 21)).unwrap();
        let model = BlackLitterman::new([0.4, 0.3, 0.2, 0.1], 2.5)
            .view(View::absolute(0, 0.004).confidence(0.8))
            .view(View::relative(1, 3, 0.002));
        let mu = model.expected_returns(&data).unwrap();
        let cov = model.posterior_covariance(&data).unwrap();
        let methods: Vec<Box<dyn crate::Allocation>> = vec![
            Box::new(
                MeanVariance::risk_aversion(2.5)
                    .expected_returns(mu.clone())
                    .covariance(cov.clone()),
            ),
            Box::new(
                MaxSharpe::default()
                    .expected_returns(mu)
                    .covariance(cov.clone()),
            ),
            Box::new(crate::MinVariance::default().covariance(cov.clone().shrinkage(0.2))),
        ];
        for method in methods {
            let weights = data.allocate(method.as_ref()).unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        // Like any given covariance, it must fit the data.
        let other = Returns::returns_matrix(factor_returns(3, 30, 21)).unwrap();
        assert!(matches!(
            other.allocate(&crate::MinVariance::default().covariance(cov)),
            Err(CutupError::DimensionMismatch { .. })
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest::proptest! {
        #[test]
        fn posteriors_are_ordered_by_confidence(
            seed in 0u64..1000,
            assets in 2usize..6,
            long in 0usize..6,
            offset in 0usize..5,
            low in 0.01f64..0.98,
            gap in 0.01f64..1.0,
            tau in 0.01f64..1.0,
        ) {
            // Σ ⪯ Σ_BL(more confident) ⪯ Σ_BL(less confident) ⪯ (1 + τ)Σ.
            let long = long % assets;
            let short = (long + 1 + offset % (assets - 1)) % assets;
            let data =
                Returns::returns_matrix(factor_returns(assets, 3 * assets + 10, seed)).unwrap();
            let cov = &data.cov_matrix;
            let high = (low + gap).min(1.0);
            let model = |c: f64| {
                BlackLitterman::new(vec![1.0; assets], 2.5)
                    .tau(tau)
                    .view(View::relative(long, short, 0.001).confidence(c))
                    .view(View::absolute(long, 0.002))
            };
            let confident = posterior(&model(high), &data);
            let unsure = posterior(&model(low), &data);
            let inflated = cov * (1.0 + tau);
            for (upper, lower) in [(&unsure, &confident), (&confident, cov), (&inflated, &unsure)] {
                let smallest = smallest_eigenvalue_of_difference(upper, lower);
                proptest::prop_assert!(smallest >= -1e-12, "{smallest}");
            }
        }
    }
}
