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
pub use mean_variance::{FrontierPoint, MeanVariance};
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
    // A variance that isn't positive leaves a volatility of 0 or NaN, and with it a diagonal
    // entry of 0/0 or NaN.
    if correlation.iter().any(|c| !c.is_finite()) {
        return Err(CutupError::SingularCovariance);
    }
    Ok((correlation, volatilities))
}

/// Solves `minimize ½ wᵀQw − cᵀw subject to Eᵀw = b and l ≤ w ≤ u` from the feasible
/// `start`, where `E`'s first column is the budget `1ᵀw = 1`, and meets the budget
/// [within the bounds](within_bounds), which only moves the rounding the solver leaves.
pub(crate) fn long_only_weights(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &qp::Bounds,
    start: DVector<f64>,
) -> Result<Vec<f64>, CutupError> {
    let x = qp::solve_bounded(q, c, e, b, bounds, start)?;
    Ok(within_bounds(x.as_slice(), bounds))
}

/// `x` rescaled to sum to 1. For max Sharpe it maps `y` back to `w`; for long-short weights
/// with a budget `1ᵀx = 1`, it only removes the solver's rounding. Weights within bounds go
/// through [`within_bounds`] instead, since scaling could move one at a bound past it.
///
/// If the sum overflows, `x` is first divided by its largest magnitude: dividing finite
/// entries by an infinite sum would make every weight 0.
pub(crate) fn normalized(x: DVector<f64>) -> Vec<f64> {
    let total = x.sum();
    let largest = x.amax();
    let x = if !total.is_finite() && largest.is_finite() && largest > 0.0 {
        x / largest
    } else {
        x
    };
    let total = x.sum();
    x.iter().map(|w| w / total).collect()
}

/// How far the sum of `n` weights can round from 1: a few ulps of 1 for each. The budget is
/// only met to this, so a weight this near a bound can't be told from one on it.
pub(crate) fn budget_rounding(n: usize) -> f64 {
    qp::sum_rounding(n, 1.0)
}

/// A solver's weights `x`, within `bounds` and summing to 1. A solver meets the budget's row
/// only to its tolerance, so `x` can sum to a little more or less than 1, and scaling it to 1,
/// as [`normalized`] does, could move a weight at its cap past it, or one at its floor below.
/// Instead:
///
/// * Each weight is clamped to its bounds, and one within [the budget's rounding](budget_rounding)
///   of a bound is put on it: a solver can leave an asset it doesn't hold at `1e-17` rather
///   than 0, as the multipliers of [`MinCvar`](crate::MinCvar)'s dual do.
/// * What the weights then miss of 1 goes to those with the most room toward it, each only as
///   far as its bounds: first those between their bounds, and only then those on the bound
///   they'd move away from, so that a weight of 0 stays 0. Bounds that a portfolio meets have
///   room for all of it.
///
/// Each weight then holds its own bounds exactly. Groups and a target hold to the rounding of
/// their sums, as they did when the weights were scaled: what moves is rounding, about `1e-15`.
pub(crate) fn within_bounds(x: &[f64], bounds: &qp::Bounds) -> Vec<f64> {
    let (lower, upper) = (&bounds.lower, &bounds.upper);
    let rounding = budget_rounding(x.len());
    let mut weights: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let w = w.clamp(lower[i], upper[i]);
            if w - lower[i] <= rounding {
                lower[i]
            } else if upper[i] - w <= rounding {
                upper[i]
            } else {
                w
            }
        })
        .collect();
    let mut missing = 1.0 - weights.iter().sum::<f64>();
    let up = missing > 0.0;
    let room = |i: usize, w: f64| if up { upper[i] - w } else { w - lower[i] };
    // A weight at the bound it would move away from, such as one the program left at 0, takes
    // only what the weights between their bounds have no room for, so that rounding doesn't
    // move it off that bound.
    let leaves = |i: usize, w: f64| if up { w <= lower[i] } else { w >= upper[i] };
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&i, &j| {
        leaves(i, weights[i])
            .cmp(&leaves(j, weights[j]))
            .then(room(j, weights[j]).total_cmp(&room(i, weights[i])))
    });
    for i in order {
        // Less than the half ulp of 1 that a sum short of it misses 1 by is no shortfall, only
        // what adding the shortfall to a weight between its bounds rounded off: not for a weight
        // at the bound it would leave to take.
        if missing == 0.0 || (leaves(i, weights[i]) && missing.abs() < f64::EPSILON / 2.0) {
            break;
        }
        let moved = (weights[i] + missing).clamp(lower[i], upper[i]);
        missing -= moved - weights[i];
        weights[i] = moved;
    }
    // A weight of -0.0, as one put on a floor of -0.0 is, is 0.0, which prints without a sign:
    // adding 0.0 makes it so and leaves every other weight as it is.
    weights.iter().map(|w| w + 0.0).collect()
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
    // Not a budget but aᵀy = 1, so the weights are y scaled to sum to 1.
    let weights = normalized(qp::solve_bounded(
        cov,
        &DVector::zeros(n),
        &DMatrix::from_column_slice(n, 1, a.as_slice()),
        &DVector::from_element(1, 1.0),
        &qp::Bounds::nonnegative(n),
        start,
    )?);

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

/// `cov` for HRP and HERC to split weight by, which only the ratios of its entries matter to.
///
/// A branch's risk is a sum of cluster variances, none larger than `cov`'s largest entry. If as
/// many of those as there are assets could sum past `f64::MAX`, `cov` is divided by a power of
/// two at least their number, which is exact and keeps every such sum finite: summed as they
/// were, two risks near `f64::MAX` are infinite, one branch of the split takes all of the weight,
/// and two infinite ones split it into NaN.
pub(crate) fn summable(cov: DMatrix<f64>) -> DMatrix<f64> {
    let n = cov.ncols();
    if cov.amax() * n as f64 <= f64::MAX {
        return cov;
    }
    cov / n.next_power_of_two() as f64
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
///
/// Each is the smallest variance over the asset's own, the same once normalized as `1 / σ²`,
/// but in `(0, 1]`: inverses of variances near the smallest normal number would overflow, or
/// sum past `f64::MAX` and leave every weight 0. If the smallest is 0, as when an adjustment's
/// variance underflows, the assets with none split the weight, the limit of `1 / σ²`, in place
/// of `0 / 0`.
pub(crate) fn inverse_variance_weights(cov: &DMatrix<f64>, cluster: &[usize]) -> Vec<f64> {
    let smallest = cluster
        .iter()
        .map(|&i| cov[(i, i)])
        .fold(f64::INFINITY, f64::min);
    let inverse: Vec<f64> = cluster
        .iter()
        .map(|&i| match cov[(i, i)] {
            v if smallest == 0.0 => f64::from(v == 0.0),
            v => smallest / v,
        })
        .collect();
    let total: f64 = inverse.iter().sum();
    inverse.iter().map(|w| w / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeightBounds;
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    /// Bounds `lower ≤ w ≤ upper` with no groups, for [`within_bounds`].
    fn plain(lower: &[f64], upper: &[f64]) -> qp::Bounds {
        let labels: Vec<String> = (0..lower.len()).map(|i| i.to_string()).collect();
        WeightBounds::per_asset(lower.to_vec(), upper.to_vec())
            .resolve(&labels)
            .unwrap()
    }

    #[test]
    fn weights_a_little_short_of_one_keep_a_weight_at_its_cap() {
        let bounds = plain(&[0.0; 3], &[0.5, 1.0, 1.0]);
        let x = [0.5, 0.3, 0.2 - 3e-15];
        // Scaling them to 1 moves the first past its cap.
        assert!(normalized(DVector::from_column_slice(&x))[0] > 0.5);
        let weights = within_bounds(&x, &bounds);
        assert_eq!(weights[0], 0.5);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 2.0 * f64::EPSILON);
        // The last, with the most room below its cap, takes what's missing.
        assert!(weights[1] == 0.3 && weights[2] > 0.2 - 3e-15);
    }

    #[test]
    fn weights_a_little_over_one_keep_a_weight_at_its_floor() {
        let bounds = plain(&[0.2, 0.0, 0.0], &[f64::INFINITY; 3]);
        let x = [0.2, 0.5, 0.3 + 3e-15];
        assert!(normalized(DVector::from_column_slice(&x))[0] < 0.2);
        let weights = within_bounds(&x, &bounds);
        assert_eq!(weights[0], 0.2);
        assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 2.0 * f64::EPSILON);
        // The middle, with the most room above its floor, gives up what's over.
        assert!(weights[1] < 0.5 && weights[2] == 0.3 + 3e-15);
    }

    #[test]
    fn what_one_weight_has_no_room_for_goes_to_the_next() {
        // 0.375 short: the last has room for 0.3125 and the first for the other 0.0625, taken
        // in that order. Each is a binary fraction, so the sums are exact.
        let bounds = plain(&[0.0; 3], &[0.5, 0.25, 0.4375]);
        let weights = within_bounds(&[0.25, 0.25, 0.125], &bounds);
        assert_eq!(weights, [0.3125, 0.25, 0.4375]);
        // A weight past its bounds comes back to them first, and then the rest makes up 1.
        let weights = within_bounds(&[0.625, 0.25, 0.125], &bounds);
        assert_eq!(weights, [0.5, 0.25, 0.25]);
    }

    #[test]
    fn room_is_the_distance_to_the_bound() {
        // Short of 1, the first has 0.5 below its cap and the second 0.375, so the first takes
        // the 0.375: by the ratio to the cap, 2 against 4, the second would.
        let bounds = plain(&[0.0; 2], &[1.0, 0.5]);
        assert_eq!(within_bounds(&[0.5, 0.125], &bounds), [0.875, 0.125]);
        // Over 1, the first has 0.5 above its floor and the second 0.375, so the first gives up
        // the 0.125: by the sum with the floor, 0.5 against 0.875, the second would.
        let bounds = plain(&[0.0, 0.25], &[f64::INFINITY; 2]);
        assert_eq!(within_bounds(&[0.5, 0.625], &bounds), [0.375, 0.625]);
        // And by the ratio to the floor, 3 against 6, the second would here.
        let bounds = plain(&[0.25, 0.0625], &[f64::INFINITY; 2]);
        assert_eq!(within_bounds(&[0.75, 0.375], &bounds), [0.625, 0.375]);
    }

    #[test]
    fn a_weight_within_rounding_of_a_bound_is_on_it() {
        // A dual program's multipliers leave an asset it doesn't hold at rounding of 0, and a
        // weight held at a floor or a cap can come back an ulp inside it. Each is on its bound.
        let bounds = plain(&[0.0, 0.25, 0.0, 0.0], &[1.0, 1.0, 0.125, 1.0]);
        let x = [
            4e-17,
            0.25 + f64::EPSILON / 4.0,
            0.125 - f64::EPSILON / 16.0,
            0.625,
        ];
        assert!(x[1] > 0.25 && x[2] < 0.125);
        assert_eq!(within_bounds(&x, &bounds), [0.0, 0.25, 0.125, 0.625]);
        // Further from a bound than the weights' sum can round, a weight is its own: of four,
        // 16 ulps of 1.
        assert_eq!(budget_rounding(4), 16.0 * f64::EPSILON);
        let x = [1e-13, 0.25 + 1e-13, 0.125 - 1e-13, 0.625];
        let weights = within_bounds(&x, &bounds);
        assert!(weights[0] > 0.0 && weights[1] > 0.25 && weights[2] < 0.125);
    }

    #[test]
    fn a_weight_at_the_bound_it_would_leave_keeps_it() {
        // Short of 1, the weight at 0 has the most room below its cap, but the one between its
        // bounds takes what's missing, so the program's 0 stays 0.
        let bounds = plain(&[0.0; 3], &[1.0; 3]);
        assert_eq!(within_bounds(&[0.5, 0.375, 0.0], &bounds), [0.5, 0.5, 0.0]);
        // Over 1, the weight at its cap has the most room above its floor, but the one between
        // its bounds gives up what's over.
        let bounds = plain(&[0.0; 2], &[0.75, 1.0]);
        assert_eq!(within_bounds(&[0.75, 0.375], &bounds), [0.75, 0.25]);
        // Only what the weights between their bounds have no room for goes to one at a bound.
        let bounds = plain(&[0.0; 3], &[0.375, 0.5, 1.0]);
        assert_eq!(
            within_bounds(&[0.25, 0.375, 0.0], &bounds),
            [0.375, 0.5, 0.125]
        );
    }

    #[test]
    fn what_adding_the_shortfall_rounds_off_isnt_for_a_weight_at_its_bound() {
        // An ulp of 1 short. The first weight, with the most room, takes it, but 0.25 + 5·2⁻⁵⁵
        // is a tie that rounds to even, 2⁻⁵⁵ short of it, and the second, in [½, 1), can't hold
        // a step that small. The third, which the program left at 0, took it.
        let bounds = plain(&[0.0; 3], &[1.0; 3]);
        let ulp = f64::EPSILON;
        let x = [0.25 - 3.0 * ulp / 8.0, 0.75 - ulp / 2.0, 0.0];
        assert_eq!(1.0 - x.iter().sum::<f64>(), ulp);
        let weights = within_bounds(&x, &bounds);
        assert_eq!(weights, [0.25 + ulp / 2.0, 0.75 - ulp / 2.0, 0.0]);
        assert_eq!(weights.iter().sum::<f64>(), 1.0);
        // Over 1 by 97 ulps, of which the first two have room above their floors for 96.75. A
        // quarter of an ulp is nothing the sum can show, so the third stays at its cap.
        let bounds = plain(&[0.25, 0.5, 0.0], &[1.0, 1.0, 0.25]);
        let x = [0.25 + 64.75 * ulp, 0.5 + 32.0 * ulp, 0.25];
        assert_eq!(x.iter().sum::<f64>() - 1.0, 97.0 * ulp);
        let weights = within_bounds(&x, &bounds);
        assert_eq!(weights, [0.25, 0.5, 0.25]);
    }

    #[test]
    fn weights_that_sum_to_one_within_their_bounds_are_left_as_they_are() {
        let bounds = plain(&[0.0; 3], &[1.0; 3]);
        assert_eq!(
            within_bounds(&[0.5, 0.25, 0.25], &bounds),
            [0.5, 0.25, 0.25]
        );
        // But a weight of -0.0 comes back as 0.0.
        let weights = within_bounds(&[1.0, -0.0, 0.0], &bounds);
        assert!(weights.iter().all(|w| w.is_sign_positive()), "{weights:?}");
        // So does one on a floor of -0.0, which the bounds accept as 0.
        let signed = plain(&[-0.0; 3], &[1.0; 3]);
        assert!(signed.lower.iter().all(|l| l.is_sign_negative()));
        let weights = within_bounds(&[1.0, 0.0, -0.0], &signed);
        assert!(weights.iter().all(|w| w.is_sign_positive()), "{weights:?}");
    }

    #[test]
    fn capped_qp_weights_stay_within_their_caps() {
        // On 0.3.12, which scaled them to sum to 1, 48 of these 2000 weights ended a few units
        // in the last place past their caps, from min variance and mean-variance alike: the
        // first, in case 8, at 0.23582384252457536 for a cap of 0.23582384252457533.
        let mut state: u64 = 0x5eed;
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        for case in 0..200 {
            let n = 3 + case % 5;
            let rows: Vec<Vec<f64>> = (0..30)
                .map(|_| (0..n).map(|_| uniform() * 0.04 - 0.02).collect())
                .collect();
            let cap = (0.1f64.max(1.0 / n as f64 + 0.05) + uniform() * 0.5).min(0.9);
            let data = Returns::new(&rows).unwrap();
            let bounds = WeightBounds::uniform(0.0, cap);
            let min_variance = crate::MinVariance::default().bounds(bounds.clone());
            let mean_variance = crate::MeanVariance::risk_aversion(10.0).bounds(bounds);
            for weights in [data.allocate(&min_variance), data.allocate(&mean_variance)] {
                let weights = weights.unwrap();
                assert!(
                    weights.iter().all(|&w| (0.0..=cap).contains(&w)),
                    "case {case}, cap {cap}: {weights:?}"
                );
                assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 1e-15);
            }
        }
    }
}
