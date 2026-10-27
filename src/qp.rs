//! Convex quadratic programs: equality-constrained ones solved by the null-space method, and
//! ones with nonnegative variables solved by a primal active-set method on top of it.

use crate::eigen::SymmetricEigen;
use crate::CutupError;
use nalgebra::{DMatrix, DVector};

/// A solution `x` and its constraint multipliers `λ`.
type Solution = (DVector<f64>, DVector<f64>);

/// Solves `minimize ½ xᵀKx − cᵀx subject to Eᵀx = b` for a positive semidefinite `K`,
/// returning `x` and the multipliers `λ` of the KKT system `Kx + Eλ = c`, `Eᵀx = b`.
///
/// This is the null-space method (Nocedal and Wright, 2006, §16.2). With `Y` an orthonormal
/// basis for the range of `E` and `Z` one for the null space of `Eᵀ`, the smallest feasible
/// point is `x₀ = Yu`, and every feasible point is `x₀ + Zv`. The objective is then a
/// quadratic in `v` with the positive semidefinite Hessian `ZᵀKZ`, solved with its
/// pseudo-inverse (eigenvalues below `size · ε` of the largest count as zero, matching
/// `numpy.linalg.pinv`). Since `Y` and `Z` are orthogonal, a singular `K` still gives an exact
/// optimum, the one with the smallest norm.
///
/// The solution is then checked against the KKT system. If it doesn't satisfy it, the problem
/// has no optimum, either because the objective is unbounded or because the constraints are
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
    if k.iter().chain(c.iter()).any(|x| !x.is_finite()) {
        return Err(CutupError::NonFiniteCovariance);
    }
    let problem = Scaled::new(k, e)?;
    let (k, e) = (&problem.k, &problem.e);
    let c = c / problem.k_scale;
    let b = DVector::from_fn(b.len(), |j, _| b[j] / problem.column_scales[j]);

    let constraints = Constraints::new(e);
    let x0 = constraints.smallest_feasible(&b);
    let z = &constraints.null;
    let gradient = z.transpose() * (&c - k * &x0);
    let (eigen, cutoff) = reduced_hessian(k, z)?;
    let mut v = DVector::zeros(z.ncols());
    for (i, &value) in eigen.values.iter().enumerate() {
        if value.abs() > cutoff {
            let w = eigen.vectors.column(i);
            v += w * (w.dot(&gradient) / value);
        }
    }
    let x = x0 + z * v;
    let multipliers = constraints.multipliers(&(&c - k * &x));

    let stationarity = (k * &x + e * &multipliers - &c).norm();
    let feasibility = (e.transpose() * &x - &b).norm();
    let size = k.norm() * x.norm()
        + e.norm() * (x.norm() + multipliers.norm())
        + c.norm()
        + b.norm()
        + noise / problem.k_scale;
    if stationarity + feasibility > 1e-8 * size {
        return Ok(None);
    }
    if x.iter().any(|w| !w.is_finite()) {
        return Err(CutupError::SingularCovariance);
    }
    let multipliers = DVector::from_fn(multipliers.len(), |j, _| {
        multipliers[j] * problem.k_scale / problem.column_scales[j]
    });
    Ok(Some((x, multipliers)))
}

/// `K` and `E` put on a comparable scale, so that the cutoffs treat them alike: `K` divided by
/// its average diagonal entry, and each column of `E` by its largest entry. Scaling `K` and `c`
/// together, or a column of `E` with its entry of `b`, leaves the solution unchanged and
/// scales the multipliers.
struct Scaled {
    k: DMatrix<f64>,
    e: DMatrix<f64>,
    k_scale: f64,
    column_scales: Vec<f64>,
}

impl Scaled {
    fn new(k: &DMatrix<f64>, e: &DMatrix<f64>) -> Result<Self, CutupError> {
        // A zero K, as the free block of a linear objective is, keeps scale 1.
        let average = k.trace() / k.nrows() as f64;
        let k_scale = if average == 0.0 {
            1.0
        } else if average.is_finite() && average > 0.0 {
            average
        } else {
            return Err(CutupError::SingularCovariance);
        };
        let column_scales: Vec<f64> = e
            .column_iter()
            .map(|column| match column.amax() {
                s if s > 0.0 => s,
                _ => 1.0,
            })
            .collect();
        let e = DMatrix::from_fn(e.nrows(), e.ncols(), |i, j| e[(i, j)] / column_scales[j]);
        Ok(Scaled {
            k: k / k_scale,
            e,
            k_scale,
            column_scales,
        })
    }
}

/// The eigendecomposition of the reduced Hessian `ZᵀKZ`, and the cutoff below which its
/// eigenvalues count as zero.
fn reduced_hessian(
    k: &DMatrix<f64>,
    z: &DMatrix<f64>,
) -> Result<(SymmetricEigen, f64), CutupError> {
    let eigen = SymmetricEigen::new(&(z.transpose() * k * z))?;
    // Relative to K as well, so that a reduced Hessian that is all rounding error is zero.
    let largest = eigen.values.amax().max(k.amax());
    let cutoff = largest * (k.nrows() + 1) as f64 * f64::EPSILON;
    Ok((eigen, cutoff))
}

/// The constraint matrix `E` as `E = YT`, with `Y` an orthonormal basis for its range, and
/// `Z` an orthonormal basis for the null space of `Eᵀ`, which completes `Y`.
///
/// `Y` comes from Gram–Schmidt with column pivoting and reorthogonalization: each step takes
/// the column of `E` with the most left over after removing the earlier basis vectors, until
/// what is left is rounding. A column that never gets taken depends on the others. `Z` comes
/// from a Householder QR of `Y`.
struct Constraints {
    range: DMatrix<f64>,
    null: DMatrix<f64>,
    /// `T`, with a row per basis vector and a column per column of `E`.
    coefficients: DMatrix<f64>,
    /// The column of `E` each basis vector was built from, in order.
    pivots: Vec<usize>,
}

impl Constraints {
    fn new(e: &DMatrix<f64>) -> Self {
        let (n, m) = e.shape();
        let tolerance =
            n.max(m) as f64 * f64::EPSILON * e.column_iter().map(|c| c.norm()).fold(0.0, f64::max);
        let mut left = e.clone();
        let mut coefficients = DMatrix::zeros(n.min(m), m);
        let mut basis: Vec<DVector<f64>> = Vec::new();
        let mut pivots = Vec::new();
        let mut remaining: Vec<usize> = (0..m).collect();

        while basis.len() < n {
            let Some((position, norm)) = remaining
                .iter()
                .map(|&j| left.column(j).norm())
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(&b.1))
            else {
                break;
            };
            if norm <= tolerance {
                break;
            }
            let j = remaining.remove(position);
            let row = basis.len();
            let y = left.column(j) / norm;
            coefficients[(row, j)] = norm;
            for &l in &remaining {
                // Twice, so that rounding in the first pass doesn't leave y in the column.
                for _ in 0..2 {
                    let t = y.dot(&left.column(l));
                    coefficients[(row, l)] += t;
                    let updated = left.column(l) - &y * t;
                    left.set_column(l, &updated);
                }
            }
            basis.push(y);
            pivots.push(j);
        }

        let rank = basis.len();
        let range = if rank == 0 {
            DMatrix::zeros(n, 0)
        } else {
            DMatrix::from_columns(&basis)
        };
        // The last n − rank columns of a full Q, with Q's first columns spanning range(Y).
        let null = if rank == 0 {
            DMatrix::identity(n, n)
        } else {
            let mut q_transpose = DMatrix::identity(n, n);
            range.clone().qr().q_tr_mul(&mut q_transpose);
            q_transpose.rows(rank, n - rank).transpose()
        };
        Constraints {
            range,
            null,
            coefficients: coefficients.rows(0, rank).into_owned(),
            pivots,
        }
    }

    /// The smallest `x` with `Eᵀx = b` on the pivot columns: `x = Yu` with `Tᵀu = b`, solved
    /// by forward substitution, since column `pᵢ` of `T` is zero below row `i`. A dependent
    /// column's equation is left to the caller to check.
    fn smallest_feasible(&self, b: &DVector<f64>) -> DVector<f64> {
        let mut u = DVector::zeros(self.pivots.len());
        for (i, &j) in self.pivots.iter().enumerate() {
            let known: f64 = (0..i).map(|k| self.coefficients[(k, j)] * u[k]).sum();
            u[i] = (b[j] - known) / self.coefficients[(i, j)];
        }
        &self.range * u
    }

    /// The `λ` that best solves `Eλ = g`: `Tλ = Yᵀg` on the pivot columns, solved by back
    /// substitution, with zero for dependent columns.
    fn multipliers(&self, g: &DVector<f64>) -> DVector<f64> {
        let h = self.range.transpose() * g;
        let mut lambda = DVector::zeros(self.coefficients.ncols());
        for (i, &j) in self.pivots.iter().enumerate().rev() {
            let known: f64 = self.pivots[i + 1..]
                .iter()
                .map(|&l| self.coefficients[(i, l)] * lambda[l])
                .sum();
            lambda[j] = (h[i] - known) / self.coefficients[(i, j)];
        }
        lambda
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
    let problem = Scaled::new(q, e)?;
    let z = Constraints::new(&problem.e).null;
    let (eigen, cutoff) = reduced_hessian(&problem.k, &z)?;
    let mut direction = DVector::zeros(q.nrows());
    for (i, &value) in eigen.values.iter().enumerate() {
        if value.abs() <= cutoff {
            let v = &z * eigen.vectors.column(i);
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
