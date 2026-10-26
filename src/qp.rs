//! Convex quadratic programs: equality-constrained ones solved through their KKT system, and
//! ones with nonnegative variables solved by a primal active-set method on top of it.

use crate::CutupError;
use nalgebra::{DMatrix, DVector};

/// Convergence tolerance for nalgebra's SVD: its own default. With exactly `ε` the iteration
/// can stop with singular vectors that reconstruct the matrix only to about 1e-3, and even at
/// this tolerance rare matrices reconstruct only to about 1e-10.
pub(crate) const SVD_EPSILON: f64 = 5.0 * f64::EPSILON;

/// A solution `x` and its constraint multipliers `λ`.
type Solution = (DVector<f64>, DVector<f64>);

/// Solves `minimize ½ xᵀKx − cᵀx subject to Eᵀx = b` through its KKT system
/// `[K E; Eᵀ 0] [x; λ] = [c; b]`, returning `x` and the multipliers `λ`.
///
/// The system is solved with the pseudo-inverse (singular values below `size · ε · σ_max`
/// count as zero, matching `numpy.linalg.pinv`), so a singular `K` still gives an exact
/// optimum: the one with the smallest norm. Two rounds of iterative refinement follow. The
/// solution is then checked against the system. If it doesn't satisfy it, the problem has no
/// optimum, either because the objective is unbounded or because the constraints are
/// infeasible, and `Ok(None)` is returned.
pub(crate) fn solve_kkt(
    k: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<Option<Solution>, CutupError> {
    solve_noisy_kkt(k, c, e, b, 0.0)
}

/// [`solve_kkt`] for a `c` computed from terms of size up to `noise`, which may have cancelled
/// to rounding error. A residual at that rounding level isn't taken as inconsistency.
fn solve_noisy_kkt(
    k: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    noise: f64,
) -> Result<Option<Solution>, CutupError> {
    // nalgebra's SVD never converges on NaN, so check first and cap the iterations below.
    if k.iter().chain(c.iter()).any(|x| !x.is_finite()) {
        return Err(CutupError::NonFiniteCovariance);
    }
    let (n, m) = (k.nrows(), e.ncols());

    // Put every block on a comparable scale so the pseudo-inverse cutoff treats them alike.
    // Scaling K and c together, or a constraint column of E with its entry of b, leaves the
    // solution unchanged and scales the multipliers, which are scaled back at the end.
    let k_scale = k_scale(k)?;
    let column_scales: Vec<f64> = (0..m).map(|j| column_scale(e, j)).collect();
    let mut kkt = DMatrix::zeros(n + m, n + m);
    let mut rhs = DVector::zeros(n + m);
    kkt.view_mut((0, 0), (n, n)).copy_from(&(k / k_scale));
    rhs.rows_mut(0, n).copy_from(&(c / k_scale));
    for (j, &scale) in column_scales.iter().enumerate() {
        let column = e.column(j) / scale;
        kkt.view_mut((0, n + j), (n, 1)).copy_from(&column);
        kkt.view_mut((n + j, 0), (1, n))
            .copy_from(&column.transpose());
        rhs[n + j] = b[j] / scale;
    }

    let size = n + m;
    let svd = kkt
        .clone()
        .try_svd(true, true, SVD_EPSILON, 10_000)
        .ok_or(CutupError::SingularCovariance)?;
    let tolerance = svd.singular_values.max() * size as f64 * f64::EPSILON;
    let pinv = svd
        .pseudo_inverse(tolerance)
        .map_err(|_| CutupError::SingularCovariance)?;
    // Iterative refinement against the exact system removes the error of an inaccurate SVD.
    let mut solution = &pinv * &rhs;
    for _ in 0..2 {
        solution += &pinv * (&rhs - &kkt * &solution);
    }

    let residual = (&kkt * &solution - &rhs).norm();
    if residual > 1e-8 * (kkt.norm() * solution.norm() + rhs.norm() + noise / k_scale) {
        return Ok(None);
    }
    let x = solution.rows(0, n).into_owned();
    if x.iter().any(|w| !w.is_finite()) {
        return Err(CutupError::SingularCovariance);
    }
    let multipliers = DVector::from_fn(m, |j, _| solution[n + j] * k_scale / column_scales[j]);
    Ok(Some((x, multipliers)))
}

/// The average diagonal entry of `k`, or 1 when `k` is zero.
fn k_scale(k: &DMatrix<f64>) -> Result<f64, CutupError> {
    let scale = k.trace() / k.nrows() as f64;
    if scale == 0.0 {
        Ok(1.0)
    } else if scale.is_finite() && scale > 0.0 {
        Ok(scale)
    } else {
        Err(CutupError::SingularCovariance)
    }
}

/// The largest absolute entry of column `j` of `e`, or 1 when the column is zero.
fn column_scale(e: &DMatrix<f64>, j: usize) -> f64 {
    match e.column(j).amax() {
        s if s > 0.0 => s,
        _ => 1.0,
    }
}

/// Solves `minimize ½ xᵀQx − cᵀx subject to Eᵀx = b and x ≥ 0` for a positive semidefinite
/// `Q`, starting from `start`, which must be feasible.
///
/// This is a primal active-set method (Nocedal and Wright, 2006, Algorithm 16.3). Variables
/// in the active set are held at zero, and each iteration solves the equality-constrained
/// problem in the others with [`solve_kkt`]. If that solution is nonnegative the method moves
/// to it, and otherwise moves toward it until a variable reaches zero, which joins the
/// active set. At an equality-constrained optimum, a variable leaves the active set if its
/// multiplier `ν = Qx − c + Eλ` is negative, since increasing it lowers the objective. When
/// none is, `x` satisfies the KKT conditions and is optimal. When `Q` is singular and the
/// equality-constrained problem has no minimum, the method follows a direction of zero
/// curvature that lowers the objective instead.
///
/// The returned `x` is clamped at zero, so rounding never leaves a slightly negative entry.
///
/// # Errors
///
/// * [`CutupError::Unbounded`] if the objective decreases without limit.
/// * [`CutupError::SingularCovariance`] if the linear algebra fails.
/// * [`CutupError::NotConverged`] if the iteration cap is reached, or a variable that would
///   lower the objective can't enter without the solver cycling. Only rounding trouble on
///   extreme inputs should cause either.
pub(crate) fn solve_nonnegative(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    start: DVector<f64>,
) -> Result<DVector<f64>, CutupError> {
    let n = q.nrows();
    let mut x = start;
    let mut free: Vec<bool> = x.iter().map(|&v| v > 0.0).collect();
    // Variables that were released and then blocked at once without x moving. Releasing them
    // again would loop, so they stay in the active set until x moves.
    let mut refused = vec![false; n];
    let mut released = None;

    for _ in 0..100 + 20 * n {
        let indices: Vec<usize> = (0..n).filter(|&i| free[i]).collect();
        let q_free = q.select_rows(&indices).select_columns(&indices);
        let c_free = c.select_rows(&indices);
        let e_free = e.select_rows(&indices);
        let x_free = x.select_rows(&indices);

        // Solve for the step from x rather than for the new point, so that when Q is singular
        // and the free problem has many minima, the target is the one closest to x. The
        // minimum-norm one could put a just-released variable below zero, blocking it at once.
        let gradient_free = &c_free - &q_free * &x_free;
        let residual = b - e_free.transpose() * &x_free;
        let noise = q_free.amax() * x_free.amax() + c_free.amax();
        let Some((delta, multipliers)) =
            solve_noisy_kkt(&q_free, &gradient_free, &e_free, &residual, noise)?
        else {
            // Without a linear term the objective is bounded below and the free problem has a
            // minimum, so the solve failed numerically.
            if c_free.iter().all(|&v| v == 0.0) {
                return Err(CutupError::SingularCovariance);
            }
            // No minimum on the free variables: move along a direction of zero curvature
            // until a variable reaches zero.
            let direction = descent_ray(&q_free, &c_free, &e_free)?;
            let (step, blocking) = ratio_test(&x_free, &direction).ok_or(CutupError::Unbounded)?;
            for (k, &i) in indices.iter().enumerate() {
                x[i] = (x_free[k] + step * direction[k]).max(0.0);
            }
            block(
                &mut x,
                &mut free,
                &mut refused,
                indices[blocking],
                step,
                released,
            );
            released = None;
            continue;
        };

        let target = &x_free + delta;

        // Step toward the target, stopping where the first variable reaches zero. Entries
        // within rounding of zero don't block, and are clamped to zero instead.
        let tolerance = 1e-12 * target.amax().max(x_free.amax());
        let blocking = (0..indices.len())
            .filter(|&k| target[k] < -tolerance)
            .map(|k| (x_free[k] / (x_free[k] - target[k]), k))
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((step, k)) = blocking {
            for (j, &i) in indices.iter().enumerate() {
                x[i] = (x_free[j] + step * (target[j] - x_free[j])).max(0.0);
            }
            block(&mut x, &mut free, &mut refused, indices[k], step, released);
            released = None;
            continue;
        }
        let mut moved = false;
        for (k, &i) in indices.iter().enumerate() {
            let next = target[k].max(0.0);
            moved |= (next - x[i]).abs() > tolerance;
            x[i] = next;
        }
        if moved {
            refused.fill(false);
        }

        // Release the active variable with the most negative multiplier, if any. A refused
        // variable with a negative multiplier means x isn't optimal, but the solver can't
        // make progress on it.
        let gradient = q * &x - c;
        let pushback = e * &multipliers;
        // Like the step, the multipliers are differences of terms up to this size.
        let scale = (q.amax() * x.amax() + c.amax()).max(pushback.amax());
        let negative: Vec<(f64, usize)> = (0..n)
            .filter(|&i| !free[i])
            .map(|i| (gradient[i] + pushback[i], i))
            .filter(|&(nu, _)| nu < -1e-9 * scale)
            .collect();
        let entering = negative
            .iter()
            .filter(|&&(_, i)| !refused[i])
            .min_by(|a, b| a.0.total_cmp(&b.0));
        match entering {
            Some(&(_, i)) => {
                free[i] = true;
                released = Some(i);
            }
            None if negative.is_empty() => return Ok(x),
            None => return Err(CutupError::NotConverged),
        }
    }
    Err(CutupError::NotConverged)
}

/// Moves variable `i` into the active set after a step of length `step`. If `i` was just
/// released and blocks without `x` moving, it is refused re-entry until `x` moves.
fn block(
    x: &mut DVector<f64>,
    free: &mut [bool],
    refused: &mut [bool],
    i: usize,
    step: f64,
    released: Option<usize>,
) {
    x[i] = 0.0;
    free[i] = false;
    if step > 0.0 {
        refused.fill(false);
    } else if released == Some(i) {
        refused[i] = true;
    }
}

/// The largest step `t ≤ ∞` keeping `x + t·d ≥ 0`, and the variable that reaches zero there,
/// or `None` if no variable decreases. Components of `d` within rounding of zero don't count
/// as decreasing.
fn ratio_test(x: &DVector<f64>, d: &DVector<f64>) -> Option<(f64, usize)> {
    let tolerance = 1e-12 * d.amax();
    (0..x.len())
        .filter(|&k| d[k] < -tolerance)
        .map(|k| (x[k] / -d[k], k))
        .min_by(|a, b| a.0.total_cmp(&b.0))
}

/// A direction `d` with `Qd = 0` and `Eᵀd = 0` along which `−cᵀx` decreases: the projection
/// of `c` onto that null space.
fn descent_ray(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
) -> Result<DVector<f64>, CutupError> {
    let (n, m) = (q.nrows(), e.ncols());
    let mut stacked = DMatrix::zeros(n + m, n);
    stacked.rows_mut(0, n).copy_from(&(q / k_scale(q)?));
    for j in 0..m {
        let column = e.column(j) / column_scale(e, j);
        stacked.row_mut(n + j).copy_from(&column.transpose());
    }
    let svd = stacked
        .try_svd(false, true, SVD_EPSILON, 10_000)
        .ok_or(CutupError::SingularCovariance)?;
    let v_t = svd.v_t.as_ref().ok_or(CutupError::SingularCovariance)?;
    let cutoff = svd.singular_values.max() * (n + m) as f64 * f64::EPSILON;
    let mut direction = DVector::zeros(n);
    for (k, &sigma) in svd.singular_values.iter().enumerate() {
        if sigma <= cutoff {
            let v = v_t.row(k).transpose();
            direction += &v * v.dot(c);
        }
    }
    // The KKT system was inconsistent, so such a direction exists up to rounding; without one
    // the system was too ill-conditioned to tell.
    if direction.norm() <= 1e-12 * c.norm() || direction.norm() == 0.0 {
        return Err(CutupError::SingularCovariance);
    }
    Ok(direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    fn budget(n: usize) -> (DMatrix<f64>, DVector<f64>) {
        (DMatrix::from_element(n, 1, 1.0), dvector![1.0])
    }

    fn assert_close(a: &DVector<f64>, b: &DVector<f64>, tol: f64) {
        assert!((a - b).amax() < tol, "{a:?} vs {b:?}");
    }

    #[test]
    fn interior_optimum_matches_kkt_solve() {
        let q = dmatrix![2.0, 0.5, 0.0; 0.5, 1.0, 0.2; 0.0, 0.2, 3.0];
        let c = dvector![0.1, 0.2, 0.0];
        let (e, b) = budget(3);
        let (expected, _) = solve_kkt(&q, &c, &e, &b).unwrap().unwrap();
        assert!(expected.iter().all(|&x| x > 0.0));
        let x = solve_nonnegative(&q, &c, &e, &b, dvector![1.0, 0.0, 0.0]).unwrap();
        assert_close(&x, &expected, 1e-14);
    }

    #[test]
    fn multipliers_satisfy_stationarity() {
        let k = dmatrix![2.0, 0.5; 0.5, 1.0];
        let c = dvector![0.3, -0.1];
        let e = dmatrix![1.0, 2.0; 1.0, -1.0];
        let b = dvector![1.0, 0.5];
        let (x, lambda) = solve_kkt(&k, &c, &e, &b).unwrap().unwrap();
        assert_close(&(&k * &x + &e * &lambda), &c, 1e-14);
        assert_close(&(e.transpose() * &x), &b, 1e-14);
    }

    #[test]
    fn negative_weight_is_clamped() {
        // Unconstrained, minimizing ½‖x‖² − 3x₀ over x₀ + x₁ = 1 gives (2, −1).
        let q = DMatrix::identity(2, 2);
        let c = dvector![3.0, 0.0];
        let (e, b) = budget(2);
        let x = solve_nonnegative(&q, &c, &e, &b, dvector![0.5, 0.5]).unwrap();
        assert_close(&x, &dvector![1.0, 0.0], 1e-15);
        assert_eq!(x[1], 0.0);
    }

    #[test]
    fn active_variables_are_released() {
        // From a vertex, both other variables must leave the active set to reach 1/3 each.
        let q = DMatrix::identity(3, 3);
        let (e, b) = budget(3);
        let x = solve_nonnegative(&q, &DVector::zeros(3), &e, &b, dvector![1.0, 0.0, 0.0]).unwrap();
        assert_close(&x, &DVector::from_element(3, 1.0 / 3.0), 1e-15);
    }

    #[test]
    fn linear_objective_follows_zero_curvature() {
        // With Q = 0 the free problem has no minimum, so the solver moves along rays until it
        // reaches the vertex with the largest cᵢ.
        let q = DMatrix::zeros(4, 4);
        let c = dvector![0.1, 0.4, -0.2, 0.3];
        let (e, b) = budget(4);
        let x = solve_nonnegative(&q, &c, &e, &b, DVector::from_element(4, 0.25)).unwrap();
        assert_close(&x, &dvector![0.0, 1.0, 0.0, 0.0], 1e-15);
    }

    #[test]
    fn ratio_test_ignores_rounding_noise() {
        let x = dvector![1.0, 1.0, 1.0];
        assert_eq!(ratio_test(&x, &dvector![1.0, -1e-20, 0.5]), None);
        assert_eq!(ratio_test(&x, &dvector![1.0, -1e-20, -0.5]), Some((2.0, 2)));
    }

    #[test]
    fn unbounded_objective_is_reported() {
        // x₀ = x₁ is the only constraint, so −(x₀ + x₁) falls without limit.
        let q = DMatrix::zeros(2, 2);
        let c = dvector![1.0, 1.0];
        let e = dmatrix![1.0; -1.0];
        let x = solve_nonnegative(&q, &c, &e, &dvector![0.0], dvector![1.0, 1.0]);
        assert_eq!(x, Err(CutupError::Unbounded));
    }

    #[test]
    fn singular_objective_with_no_descent_stays_feasible() {
        // Every portfolio has the same value under Q = 11ᵀ, so any feasible point is optimal.
        let q = DMatrix::from_element(3, 3, 1.0);
        let (e, b) = budget(3);
        let x = solve_nonnegative(&q, &DVector::zeros(3), &e, &b, dvector![0.2, 0.3, 0.5]).unwrap();
        assert!(x.iter().all(|&v| v >= 0.0), "{x:?}");
        assert!((x.sum() - 1.0).abs() < 1e-14);
    }
}
