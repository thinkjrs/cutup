//! Black-Litterman expected returns: the market's implied returns, blended with views.

use crate::{CovarianceConfig, CutupError, PortfolioAllocator};
use nalgebra::{DMatrix, DVector};

/// Expected returns from the Black-Litterman model (Black and Litterman, 1992): the returns
/// the market portfolio implies, blended with your views on some assets.
///
/// It estimates expected returns rather than allocating. Pass what
/// [`BlackLitterman::expected_returns`] returns to [`MeanVariance`](crate::MeanVariance) or
/// [`MaxSharpe`](crate::MaxSharpe):
///
/// ```
/// use cutup::{BlackLitterman, MeanVariance, PortfolioAllocator, View};
///
/// let returns = [
///      [ 0.010, -0.020,  0.005],
///      [-0.004,  0.015,  0.002],
///      [ 0.007,  0.003, -0.006],
///      [ 0.002, -0.001,  0.004],
///      [-0.003,  0.006,  0.001],
/// ];
/// let allocator = PortfolioAllocator::from_returns(&returns)?;
///
/// let model = BlackLitterman::new([0.5, 0.3, 0.2], 2.5)
///     // Asset 0 returns 0.1% a period, and you're 60% sure of it.
///     .view(View::absolute(0, 0.001).confidence(0.6))
///     // Asset 1 beats asset 2 by 0.05% a period.
///     .view(View::relative(1, 2, 0.0005));
/// let mu = model.expected_returns(&allocator)?;
///
/// let weights = allocator.allocate(&MeanVariance::risk_aversion(2.5).expected_returns(mu))?;
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
/// Σ is the allocator's covariance, after the same adjustments as the allocation methods:
/// use the same shrinkage here as in the method you pass the returns to. Returns are in the
/// allocator's units and period, so the views and the risk-free rate must be too. `δ` is
/// the same whatever the period, since the mean and the variance of returns both scale with
/// it: 2.5 is a common choice.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BlackLitterman {
    market_weights: Vec<f64>,
    risk_aversion: f64,
    risk_free_rate: f64,
    tau: f64,
    views: Vec<View>,
    covariance: CovarianceConfig,
}

impl BlackLitterman {
    /// A model with the market weights `market_weights`, in column order, and the market's
    /// risk aversion `delta`. Market capitalizations work too: they're divided by their sum.
    pub fn new(market_weights: impl Into<Vec<f64>>, delta: f64) -> Self {
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
    /// * [`CutupError::InvalidConfig`] if a market weight is negative or not finite, the
    ///   weights sum to 0, `δ` isn't positive and finite, the risk-free rate isn't finite, or
    ///   the covariance config is invalid.
    pub fn prior_returns(&self, data: &PortfolioAllocator) -> Result<Vec<f64>, CutupError> {
        let cov = data.adjusted_covariance(&self.covariance)?;
        Ok(self.prior(&cov)?.as_slice().to_vec())
    }

    /// The posterior expected returns, one per asset in column order. With no views, these are
    /// the [prior returns](BlackLitterman::prior_returns).
    ///
    /// # Errors
    ///
    /// * Any error from [`BlackLitterman::prior_returns`].
    /// * [`CutupError::InvalidConfig`] if `τ` isn't positive and finite, or a view is invalid:
    ///   an asset index out of range, a relative view of an asset against itself, a portfolio
    ///   with no nonzero weight, a return that isn't finite, a variance that is negative or
    ///   not finite, or a confidence outside `[0, 1]`.
    /// * [`CutupError::DimensionMismatch`] if a [`View::portfolio`] doesn't have one weight
    ///   per asset.
    /// * [`CutupError::DegenerateViews`] if `τPΣPᵀ + Ω` is singular, or within rounding of it:
    ///   certain views that repeat or contradict each other, or a view with the default
    ///   variance or a confidence on a portfolio with no variance, which would make it
    ///   certain. Give such a view an explicit [`View::variance`].
    pub fn expected_returns(&self, data: &PortfolioAllocator) -> Result<Vec<f64>, CutupError> {
        let cov = data.adjusted_covariance(&self.covariance)?;
        let pi = self.prior(&cov)?;
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
            let p = view.weights(n)?;
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
            return Ok(pi.as_slice().to_vec());
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
        let x = solve_views(a, q - &p * &pi)?;
        let posterior = pi + tau_sigma_pt * x;
        if posterior.iter().any(|m| !m.is_finite()) {
            return Err(CutupError::DegenerateViews);
        }
        Ok(posterior.as_slice().to_vec())
    }

    fn prior(&self, cov: &DMatrix<f64>) -> Result<DVector<f64>, CutupError> {
        let n = cov.ncols();
        if self.market_weights.len() != n {
            return Err(CutupError::DimensionMismatch {
                expected: n,
                actual: self.market_weights.len(),
            });
        }
        if let Some(&value) = self
            .market_weights
            .iter()
            .find(|w| !(w.is_finite() && **w >= 0.0))
        {
            return Err(CutupError::InvalidConfig {
                field: "market_weights",
                value,
            });
        }
        let total: f64 = self.market_weights.iter().sum();
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
        let w = DVector::from_iterator(n, self.market_weights.iter().map(|w| w / total));
        Ok((cov * w) * self.risk_aversion + DVector::from_element(n, self.risk_free_rate))
    }
}

/// How close to 0, relative to its scale, a view's variance or the part of it no other view
/// explains can be before the views are degenerate.
const DEGENERATE: f64 = 1e-12;

/// Solves `Ax = b` for the views' covariance `A = τPΣPᵀ + Ω`, rejecting views that repeat
/// or contradict each other: each Cholesky pivot is the part of a view's variance the earlier
/// views don't explain, which must not vanish relative to the variance itself.
fn solve_views(a: DMatrix<f64>, b: DVector<f64>) -> Result<DVector<f64>, CutupError> {
    let diagonal = a.diagonal();
    let cholesky = a.cholesky().ok_or(CutupError::DegenerateViews)?;
    let l = cholesky.l_dirty();
    if (0..diagonal.len()).any(|i| l[(i, i)].powi(2) <= DEGENERATE * diagonal[i]) {
        return Err(CutupError::DegenerateViews);
    }
    Ok(cholesky.solve(&b))
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
    Absolute(usize),
    Relative(usize, usize),
    Portfolio(Vec<f64>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Uncertainty {
    Default,
    Variance(f64),
    Confidence(f64),
}

impl View {
    /// Asset `asset` has an expected return of `q`.
    pub fn absolute(asset: usize, q: f64) -> Self {
        Self::new(Assets::Absolute(asset), q)
    }

    /// Asset `long` has an expected return `q` above asset `short`'s.
    pub fn relative(long: usize, short: usize, q: f64) -> Self {
        Self::new(Assets::Relative(long, short), q)
    }

    /// The portfolio with `weights`, one per asset in column order, has an expected return
    /// of `q`. The weights needn't sum to anything: `[0.5, 0.5, -1.0]` says the first two
    /// assets together beat the third by `q`.
    pub fn portfolio(weights: impl Into<Vec<f64>>, q: f64) -> Self {
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

    /// The view's portfolio as a vector over `n` assets.
    fn weights(&self, n: usize) -> Result<DVector<f64>, CutupError> {
        let index = |asset: usize| {
            if asset < n {
                Ok(asset)
            } else {
                Err(CutupError::InvalidConfig {
                    field: "view_asset",
                    value: asset as f64,
                })
            }
        };
        match &self.assets {
            Assets::Absolute(asset) => {
                let asset = index(*asset)?;
                Ok(DVector::from_fn(
                    n,
                    |i, _| if i == asset { 1.0 } else { 0.0 },
                ))
            }
            Assets::Relative(long, short) if long == short => Err(CutupError::InvalidConfig {
                field: "view_asset",
                value: *long as f64,
            }),
            Assets::Relative(long, short) => {
                let (long, short) = (index(*long)?, index(*short)?);
                Ok(DVector::from_fn(n, |i, _| match i {
                    i if i == long => 1.0,
                    i if i == short => -1.0,
                    _ => 0.0,
                }))
            }
            Assets::Portfolio(weights) if weights.len() != n => {
                Err(CutupError::DimensionMismatch {
                    expected: n,
                    actual: weights.len(),
                })
            }
            Assets::Portfolio(weights) => {
                if let Some(&value) = weights.iter().find(|w| !w.is_finite()) {
                    return Err(CutupError::InvalidConfig {
                        field: "view_weights",
                        value,
                    });
                }
                if weights.iter().all(|&w| w == 0.0) {
                    return Err(CutupError::InvalidConfig {
                        field: "view_weights",
                        value: 0.0,
                    });
                }
                Ok(DVector::from_column_slice(weights))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    fn allocator() -> PortfolioAllocator {
        PortfolioAllocator::returns_matrix(dmatrix![
             0.010, -0.020,  0.005;
            -0.004,  0.015,  0.002;
             0.007,  0.003, -0.006;
             0.002, -0.001,  0.004;
            -0.003,  0.006,  0.001;
             0.004,  0.002, -0.003
        ])
        .unwrap()
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
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            check(BlackLitterman::new([0.5, -0.3, 0.2], 2.5)),
            invalid("market_weights", -0.3)
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
            invalid("view_asset", 3.0)
        );
        assert_eq!(
            check(model().view(View::relative(1, 1, 0.01))),
            invalid("view_asset", 1.0)
        );
        assert_eq!(
            check(model().view(View::portfolio([0.0; 3], 0.01))),
            invalid("view_weights", 0.0)
        );
        assert_eq!(
            check(model().view(View::portfolio([1.0, 0.0], 0.01))),
            Err(CutupError::DimensionMismatch {
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
        let allocator = PortfolioAllocator::returns_matrix(dmatrix![
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
}
