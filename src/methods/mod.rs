//! The allocation methods, one module each, and the solver code they share.

mod equal_weight;
mod herc;
mod hrp;
mod inverse;
mod max_diversification;
mod max_sharpe;
mod mean_variance;
mod min_variance;
mod risk_parity;

pub use equal_weight::EqualWeight;
pub use herc::Herc;
pub use hrp::Hrp;
pub use inverse::{InverseVariance, InverseVolatility};
pub use max_diversification::MaxDiversification;
pub use max_sharpe::MaxSharpe;
pub use mean_variance::MeanVariance;
pub use min_variance::MinVariance;
pub use risk_parity::RiskParity;

use crate::eigen::SymmetricEigen;
use crate::{CutupError, Returns, qp};
use nalgebra::{DMatrix, DVector};

/// Splits a covariance into the correlation matrix `C` and the volatilities `σ`, with
/// `Σ = diag(σ) C diag(σ)`. Working on `C` in units `u = diag(σ)w` keeps assets whose
/// volatilities are many orders of magnitude apart accurate, and makes cutoffs independent of
/// each asset's units.
///
/// Returns [`CutupError::SingularCovariance`] if some variance isn't positive or a
/// correlation isn't finite.
pub(crate) fn correlation(cov: &DMatrix<f64>) -> Result<(DMatrix<f64>, DVector<f64>), CutupError> {
    let volatilities = cov.diagonal().map(f64::sqrt);
    // Dividing by each volatility in turn, so that tiny volatilities' product can't underflow.
    let correlation = DMatrix::from_fn(cov.nrows(), cov.ncols(), |i, j| {
        cov[(i, j)] / volatilities[i] / volatilities[j]
    });
    if volatilities.iter().any(|&s| s.is_nan() || s <= 0.0)
        || correlation.iter().any(|c| !c.is_finite())
    {
        return Err(CutupError::SingularCovariance);
    }
    Ok((correlation, volatilities))
}

/// Solves `minimize ½ wᵀQw − cᵀw subject to Eᵀw = b and l ≤ w ≤ u` from the feasible
/// `start`, and rescales the solution to sum to 1. With a budget constraint `1ᵀw = 1`, that
/// only removes the rounding left by clamping to the bounds; for max Sharpe it maps `y` back
/// to `w`.
pub(crate) fn long_only_weights(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &qp::Bounds,
    start: DVector<f64>,
) -> Result<Vec<f64>, CutupError> {
    Ok(normalized(qp::solve_bounded(q, c, e, b, bounds, start)?))
}

/// `x` rescaled to sum to 1. With a budget constraint `1ᵀx = 1`, that only removes the
/// rounding left by clamping to the bounds; for max Sharpe it maps `y` back to `w`.
pub(crate) fn normalized(x: DVector<f64>) -> Vec<f64> {
    let total = x.sum();
    x.iter().map(|w| w / total).collect()
}

/// Maximizes `aᵀw / sqrt(wᵀΣw)` over long-only portfolios summing to 1, for `a` with its
/// largest entry `a[best] = 1`. With `y = w / aᵀw` this is minimizing `yᵀΣy` subject to
/// `aᵀy = 1` and `y ≥ 0`, from one unit of `best`, then `w = y / 1ᵀy`.
///
/// Returns `None` if the optimum, in correlation units `u = diag(σ)w`, lies in the
/// correlation matrix's null space (see [`in_null_space`]), where the ratio has no upper
/// bound, so each caller can name its own error.
pub(crate) fn long_only_max_ratio(
    cov: &DMatrix<f64>,
    a: &DVector<f64>,
    best: usize,
) -> Result<Option<Vec<f64>>, CutupError> {
    let n = cov.ncols();
    // Before the solve, so that correlations that can't be computed are reported as such.
    let (correlation, volatilities) = correlation(cov)?;
    let mut start = DVector::zeros(n);
    start[best] = 1.0;
    let weights = long_only_weights(
        cov,
        &DVector::zeros(n),
        &DMatrix::from_column_slice(n, 1, a.as_slice()),
        &DVector::from_element(1, 1.0),
        &qp::Bounds::nonnegative(n),
        start,
    )?;

    let u = DVector::from_column_slice(weights.as_slice()).component_mul(&volatilities);
    if in_null_space(&correlation, &u)? {
        return Ok(None);
    }
    Ok(Some(weights))
}

/// Whether `u` lies in the null space of the correlation matrix `c`, with the cutoff
/// long-short max Sharpe uses: a Rayleigh quotient `uᵀCu / uᵀu` within n·ε of the largest
/// eigenvalue. The trace, n, bounds that eigenvalue, so the decomposition is only needed near
/// the cutoff.
pub(crate) fn in_null_space(c: &DMatrix<f64>, u: &DVector<f64>) -> Result<bool, CutupError> {
    let n = c.ncols() as f64;
    let quotient = u.dot(&(c * u)) / u.norm_squared();
    let cutoff = |largest: f64| quotient <= n * f64::EPSILON * largest;
    if !cutoff(n) {
        return Ok(false);
    }
    let eigen = SymmetricEigen::new(c).ok_or(CutupError::SingularCovariance)?;
    Ok(cutoff(eigen.values.amax()))
}

/// The smallest variance HRP and HERC accept, `f64::MIN_POSITIVE / f64::EPSILON`, about
/// 1e-292. Above it, a covariance `ρσ²` with `|ρ|` above machine epsilon is a normal number,
/// so correlations keep full precision, and inverse variances can't overflow when summed.
/// Below it, covariances and then variances become subnormal and lose bits.
const SMALLEST_HIERARCHICAL_VARIANCE: f64 = f64::MIN_POSITIVE / f64::EPSILON;

impl Returns {
    /// Returns the correlation distance `dᵢⱼ = sqrt((1 − ρᵢⱼ) / 2)` between every pair of
    /// assets.
    ///
    /// Returns [`CutupError::NonFiniteCovariance`] if a variance is below
    /// [`SMALLEST_HIERARCHICAL_VARIANCE`].
    pub(crate) fn correlation_distance(&self) -> Result<DMatrix<f64>, CutupError> {
        let cov = &self.cov_matrix;
        if cov
            .diagonal()
            .iter()
            .any(|&v| v < SMALLEST_HIERARCHICAL_VARIANCE)
        {
            return Err(CutupError::NonFiniteCovariance);
        }
        // Dividing by each volatility in turn, so that tiny volatilities' product can't
        // underflow, as in `correlation`.
        let volatilities = cov.diagonal().map(f64::sqrt);
        let n = cov.ncols();
        let mut distance = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i + 1..n {
                let rho = cov[(i, j)] / volatilities[i] / volatilities[j];
                let d = ((1.0 - rho) / 2.0).clamp(0.0, 1.0).sqrt();
                distance[(i, j)] = d;
                distance[(j, i)] = d;
            }
        }
        Ok(distance)
    }
}

/// Returns the variance under `cov` of the inverse-variance portfolio over `cluster`.
pub(crate) fn cluster_variance(cov: &DMatrix<f64>, cluster: &[usize]) -> f64 {
    let weights = inverse_variance_weights(cov, cluster);
    let mut variance = 0.0;
    for (a, &i) in cluster.iter().enumerate() {
        for (b, &j) in cluster.iter().enumerate() {
            variance += weights[a] * weights[b] * cov[(i, j)];
        }
    }
    variance
}

/// Returns the inverse-variance weights under `cov` of the assets in `cluster`, summing
/// to 1.
pub(crate) fn inverse_variance_weights(cov: &DMatrix<f64>, cluster: &[usize]) -> Vec<f64> {
    let inverse: Vec<f64> = cluster.iter().map(|&i| 1.0 / cov[(i, i)]).collect();
    let total: f64 = inverse.iter().sum();
    inverse.iter().map(|w| w / total).collect()
}
