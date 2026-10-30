//! Conditional value at risk: measuring it, and the long-only portfolio that minimizes it.

use crate::{
    Allocation, CutupError, PortfolioAllocator, ReturnRange, WeightBounds, Weights, lp, normalized,
};
use nalgebra::{DMatrix, DVector};

/// The long-only portfolio with the lowest conditional value at risk (CVaR, or expected
/// shortfall): the average loss over the worst `1 − β` of the return scenarios, for a
/// confidence `β`, 0.95 by default (Rockafellar and Uryasev, 2000).
///
/// The scenarios are the allocator's returns, one per row, each equally likely, so CVaR is in
/// their units and period: with daily returns, it's daily CVaR. When `(1 − β)T` isn't a whole
/// number of the `T` scenarios, the last one in the tail counts in part; when it's under one,
/// CVaR is the worst loss.
///
/// Unlike the variance-based methods it looks only at the scenarios' losses, not at a
/// covariance, so it can spread weight differently when returns are skewed or fat-tailed. It
/// needs enough scenarios to have a tail worth averaging: at 95% that's at least 20.
///
/// ```
/// use cutup::{MinCvar, PortfolioAllocator, WeightBounds};
/// use nalgebra::dmatrix;
///
/// let returns = dmatrix![
///      0.010, -0.020,  0.005;
///     -0.004,  0.015,  0.002;
///      0.007,  0.003, -0.006;
///      0.002, -0.001,  0.004;
///     -0.003,  0.006,  0.001
/// ];
/// let allocator = PortfolioAllocator::from_returns(returns)?;
///
/// let method = MinCvar::default().confidence(0.8).bounds(WeightBounds::uniform(0.0, 0.6));
/// let weights = allocator.allocate(&method)?;
/// let cvar = allocator.cvar(&weights, 0.8)?;
///
/// // No asset alone does better.
/// for asset in 0..3 {
///     let mut alone = vec![0.0; 3];
///     alone[asset] = 1.0;
///     assert!(cvar <= allocator.cvar(&alone.into(), 0.8)? + 1e-12);
/// }
/// # Ok::<(), cutup::CutupError>(())
/// ```
///
/// # Errors
///
/// * [`CutupError::InvalidConfig`] if the confidence isn't in `(0, 1)`, the target return
///   isn't finite, or the bounds are invalid.
/// * [`CutupError::DimensionMismatch`] if per-asset bounds don't have one entry per asset.
/// * [`CutupError::InfeasibleBounds`] if the bounds can't sum to 1.
/// * [`CutupError::TargetReturnOutOfRange`] if no portfolio within the bounds has the target
///   return, by the sample mean of the returns.
/// * [`CutupError::NotConverged`] if the linear program stops early, which only extreme inputs
///   should cause.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MinCvar {
    confidence: f64,
    bounds: WeightBounds,
    target_return: Option<f64>,
}

impl Default for MinCvar {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            bounds: WeightBounds::default(),
            target_return: None,
        }
    }
}

impl MinCvar {
    /// Sets the confidence `β`, in `(0, 1)`: CVaR averages the worst `1 − β` of the
    /// scenarios. Defaults to 0.95.
    pub fn confidence(mut self, beta: f64) -> Self {
        self.confidence = beta;
        self
    }

    /// Keeps each weight within `bounds`. Defaults to long-only with no caps.
    pub fn bounds(mut self, bounds: WeightBounds) -> Self {
        self.bounds = bounds;
        self
    }

    /// Requires an expected portfolio return, by the sample mean of the returns, of
    /// `target`.
    pub fn target_return(mut self, target: f64) -> Self {
        self.target_return = Some(target);
        self
    }
}

impl Allocation for MinCvar {
    /// Solves the linear program's dual, which has a row per asset rather than per scenario.
    ///
    /// CVaR is the most a probability weighting `z` of the scenarios can make of the loss,
    /// `max zᵀ(−Rw)` over `0 ≤ zₜ ≤ 1/((1 − β)T)` with `Σzₜ = 1`, so minimizing it is a
    /// min-max. Swapping the two and writing the minimum over the portfolio as its own dual
    /// gives a maximization over `z` and that dual's multipliers: `ν` for the budget, `η` for
    /// the target return, and `λ⁺`, `λ⁻` for the bounds, subject to
    /// `−Rᵀz = ν1 + ημ + λ⁺ − λ⁻`, one row per asset. The weights are those rows'
    /// multipliers, and CVaR is its optimum.
    fn allocate(&self, data: &PortfolioAllocator) -> Result<Weights, CutupError> {
        let returns = &data.returns;
        let (t, n) = returns.shape();
        let cap = tail_weight(self.confidence, t)?;
        let bounds = self.bounds.resolve(n)?;
        let target = match self.target_return {
            Some(target) if !target.is_finite() => {
                return Err(CutupError::InvalidConfig {
                    field: "target_return",
                    value: target,
                });
            }
            Some(target) => {
                let mu = returns.row_mean().transpose();
                Some((ReturnRange::new(&mu, &bounds).reachable(target)?, mu))
            }
            None => None,
        };

        // Variables: z (t), ν, then η with a target, λ⁺ (n) and λ⁻ (n). Resolved bounds are
        // all finite, so every asset has a λ⁻.
        let eta = t + 1;
        let floors = eta + usize::from(target.is_some());
        let ceilings = floors + n;
        let columns = ceilings + n;
        let mut a = DMatrix::zeros(n + 1, columns);
        for i in 0..n {
            for s in 0..t {
                a[(i, s)] = -returns[(s, i)];
            }
            a[(i, t)] = -1.0;
            if let Some((_, mu)) = &target {
                a[(i, eta)] = -mu[i];
            }
            a[(i, floors + i)] = -1.0;
            a[(i, ceilings + i)] = 1.0;
        }
        for s in 0..t {
            a[(n, s)] = 1.0;
        }
        let mut b = DVector::zeros(n + 1);
        b[n] = 1.0;

        // Maximizing ν + ρη + lᵀλ⁺ − uᵀλ⁻ is minimizing its negative.
        let mut c = DVector::zeros(columns);
        c[t] = -1.0;
        if let Some((rho, _)) = target {
            c[eta] = -rho;
        }
        for i in 0..n {
            c[floors + i] = -bounds.lower[i];
            c[ceilings + i] = bounds.upper[i];
        }
        let lower = DVector::from_fn(columns, |j, _| {
            if j == t || (target.is_some() && j == eta) {
                f64::NEG_INFINITY
            } else {
                0.0
            }
        });
        let upper = DVector::from_fn(columns, |j, _| if j < t { cap } else { f64::INFINITY });

        // The portfolio is feasible and CVaR is bounded below on it, so the dual has an optimum.
        let lp::Outcome::Optimal(solution) = lp::solve(&c, &a, &b, &lower, &upper)? else {
            return Err(CutupError::NotConverged);
        };
        let weights = DVector::from_fn(n, |i, _| {
            solution.y[i].clamp(bounds.lower[i], bounds.upper[i])
        });
        Ok(normalized(weights))
    }
}

impl PortfolioAllocator {
    /// The conditional value at risk of `weights` at confidence `beta`: the average loss over
    /// the worst `1 − β` of the return scenarios, each row of returns one equally likely
    /// scenario. When `(1 − β)T` isn't a whole number of the `T` scenarios, the last one in
    /// the tail counts in part; when it's under one, this is the worst loss. A gain counts as
    /// a negative loss, so CVaR can be negative.
    ///
    /// This is what [`MinCvar`] minimizes.
    ///
    /// # Errors
    ///
    /// * [`CutupError::InvalidConfig`] if `beta` isn't in `(0, 1)`.
    /// * [`CutupError::DimensionMismatch`] if there isn't one weight per asset.
    ///
    /// # Example
    ///
    /// ```
    /// use cutup::PortfolioAllocator;
    /// use nalgebra::dmatrix;
    ///
    /// // Losses of 4%, 2%, 1% and a 1% gain: the worst half averages 3%.
    /// let returns = dmatrix![-0.04; -0.02; -0.01; 0.01];
    /// let allocator = PortfolioAllocator::from_returns(returns)?;
    /// let cvar = allocator.cvar(&vec![1.0].into(), 0.5)?;
    /// assert!((cvar - 0.03).abs() < 1e-15);
    /// # Ok::<(), cutup::CutupError>(())
    /// ```
    pub fn cvar(&self, weights: &Weights, beta: f64) -> Result<f64, CutupError> {
        let (t, n) = self.returns.shape();
        let cap = tail_weight(beta, t)?;
        if weights.len() != n {
            return Err(CutupError::DimensionMismatch {
                expected: n,
                actual: weights.len(),
            });
        }
        let w = DVector::from_column_slice(weights.as_slice());
        let mut losses: Vec<f64> = (-(&self.returns * w)).iter().copied().collect();
        // Only the worst ⌈1/cap⌉ losses are in the tail, one more for rounding: those go
        // first, and only they're sorted.
        let tail = ((1.0 / cap).ceil() as usize + 1).min(t);
        if tail < t {
            losses.select_nth_unstable_by(tail, |a, b| b.total_cmp(a));
        }
        losses[..tail].sort_by(|a, b| b.total_cmp(a));
        // Each scenario in the tail weighs cap, up to a total of 1.
        let mut left = 1.0f64;
        let mut total = 0.0;
        for &loss in &losses[..tail] {
            let share = cap.min(left);
            total += share * loss;
            left -= share;
            if left <= 0.0 {
                break;
            }
        }
        Ok(total)
    }
}

/// The weight `1/((1 − β)T)` of each of `t` scenarios in the tail at confidence `beta`.
fn tail_weight(beta: f64, t: usize) -> Result<f64, CutupError> {
    if !(beta > 0.0 && beta < 1.0) {
        return Err(CutupError::InvalidConfig {
            field: "confidence",
            value: beta,
        });
    }
    Ok(1.0 / ((1.0 - beta) * t as f64))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;

    fn allocator() -> PortfolioAllocator {
        PortfolioAllocator::from_returns(dmatrix![
             0.010, -0.020,  0.005;
            -0.004,  0.015,  0.002;
             0.007,  0.003, -0.006;
             0.002, -0.001,  0.004;
            -0.003,  0.006,  0.001;
             0.004, -0.008,  0.003;
            -0.006,  0.011, -0.002;
             0.001,  0.002, -0.004
        ])
        .unwrap()
    }

    #[test]
    fn cvar_averages_the_tail() {
        let allocator =
            PortfolioAllocator::from_returns(dmatrix![-0.04; -0.02; -0.01; 0.01]).unwrap();
        let one = Weights::from(vec![1.0]);
        // Half of four scenarios: the two worst, 4% and 2%.
        assert!((allocator.cvar(&one, 0.5).unwrap() - 0.03).abs() < 1e-15);
        // 30% of four is 1.2 scenarios: all of the 4% and a fifth of the 2%, over 1.2.
        let expected = (0.04 + 0.2 * 0.02) / 1.2;
        assert!((allocator.cvar(&one, 0.7).unwrap() - expected).abs() < 1e-15);
        // Under one scenario, the worst loss.
        assert!((allocator.cvar(&one, 0.9).unwrap() - 0.04).abs() < 1e-15);
    }

    #[test]
    fn the_optimum_matches_its_own_cvar_and_beats_every_vertex() {
        let allocator = allocator();
        for beta in [0.5, 0.75, 0.9] {
            let weights = allocator
                .allocate(&MinCvar::default().confidence(beta))
                .unwrap();
            assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(weights.iter().all(|&w| w >= 0.0));
            let cvar = allocator.cvar(&weights, beta).unwrap();
            for asset in 0..3 {
                let mut alone = vec![0.0; 3];
                alone[asset] = 1.0;
                let single = allocator.cvar(&alone.into(), beta).unwrap();
                assert!(cvar <= single + 1e-15, "{beta}: {cvar} > {single}");
            }
        }
    }

    #[test]
    fn bounds_and_targets_hold() {
        let allocator = allocator();
        let mu = allocator.returns.row_mean();
        let target = (mu[0] + mu[1] + mu[2]) / 3.0 + 1e-4;
        let weights = allocator
            .allocate(
                &MinCvar::default()
                    .bounds(WeightBounds::uniform(0.1, 0.5))
                    .target_return(target),
            )
            .unwrap();
        assert!(
            weights
                .iter()
                .all(|&w| (0.1 - 1e-15..=0.5 + 1e-15).contains(&w))
        );
        let achieved: f64 = weights.iter().zip(mu.iter()).map(|(w, m)| w * m).sum();
        assert!((achieved - target).abs() < 1e-15, "{achieved} vs {target}");
    }

    #[test]
    fn lower_bounds_summing_to_one_fix_the_weights() {
        // 1 − 0.9 rounds to just under 0.1, the first asset's room above its floor.
        let allocator = PortfolioAllocator::from_returns(dmatrix![
             0.010, -0.020;
            -0.004,  0.015;
             0.007,  0.003
        ])
        .unwrap();
        let weights = allocator
            .allocate(
                &MinCvar::default().bounds(WeightBounds::per_asset(vec![0.1, 0.9], vec![1.0, 1.0])),
            )
            .unwrap();
        assert!((weights[0] - 0.1).abs() < 1e-15 && (weights[1] - 0.9).abs() < 1e-15);
    }

    #[test]
    fn invalid_settings_are_rejected() {
        let allocator = allocator();
        for beta in [0.0, 1.0, f64::NAN] {
            assert!(matches!(
                allocator.allocate(&MinCvar::default().confidence(beta)),
                Err(CutupError::InvalidConfig {
                    field: "confidence",
                    ..
                })
            ));
        }
        assert!(matches!(
            allocator.allocate(&MinCvar::default().target_return(1.0)),
            Err(CutupError::TargetReturnOutOfRange { .. })
        ));
        assert_eq!(
            allocator.allocate(&MinCvar::default().bounds(WeightBounds::uniform(0.0, 0.2))),
            Err(CutupError::InfeasibleBounds {
                lower_sum: 0.0,
                upper_sum: 0.6000000000000001
            })
        );
        assert_eq!(
            allocator.cvar(&vec![1.0].into(), 0.5),
            Err(CutupError::DimensionMismatch {
                expected: 3,
                actual: 1
            })
        );
    }

    #[test]
    fn degenerate_scenarios_converge() {
        // A property test's shrunk input: most returns exactly zero, so the program is highly
        // degenerate. A reduced cost of pure rounding once entered here, moved nothing, and
        // left the basis singular.
        #[rustfmt::skip]
        let data = vec![
            0.049046206985969405, 0.0, 0.0, 0.018918236666771126, 0.0, 0.0, 0.0,
            0.02331762851401267, 0.0, -0.026903675628981525, 0.04253505321714205, 0.0,
            0.0361087756727277, 0.0, 0.034043311064955735, 0.0, 0.022568977423950948, 0.0,
            0.013270785074508322, 0.0, 0.0, -0.022042427374501777, 0.036451221073154974,
            0.0473428014604323, 0.0, -0.007958233019630202, -0.008144504154924737,
            -0.024854928967824912, 0.0, -0.035089569764353054, 0.049698982240741575, 0.0,
            0.02755697787157803, 0.0, 0.02358022931946681, -0.004670622996365131, 0.0, 0.0,
            0.0, 0.02242619203429917, 0.0, 0.0, 0.0, 0.039747808049717774,
            -0.01926202201906651, 0.001966764898914833, 0.0, 0.0, -0.025971216114690443, 0.0,
            0.0, -0.02382851848035964, 0.0013794705644694543, -0.004727636380476252,
        ];
        let allocator = PortfolioAllocator::from_returns(DMatrix::from_vec(9, 6, data)).unwrap();
        let beta = 0.8549681742164362;
        let weights = allocator
            .allocate(&MinCvar::default().confidence(beta))
            .unwrap();
        let cvar = allocator.cvar(&weights, beta).unwrap();
        let even = allocator.cvar(&vec![1.0 / 6.0; 6].into(), beta).unwrap();
        assert!(cvar <= even + 1e-15, "{cvar} > {even}");
    }
}
