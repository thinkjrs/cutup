//! Convex quadratic programs: equality-constrained ones solved by the null-space method, and
//! ones with bounded variables solved by a primal active-set method on top of it.

use crate::CutupError;
use crate::eigen::SymmetricEigen;
use nalgebra::{Cholesky, DMatrix, DVector, Dyn};

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
/// optimum, the one with the smallest norm in the rescaled variables below.
///
/// The variables are first rescaled to `u = Dx`, with `D = diag(sqrt(Kᵢᵢ))`, so that `K` has
/// a unit diagonal: a covariance becomes the correlation matrix. Then neither the solution
/// nor the cutoffs depend on each variable's units, and variables whose variances are many
/// orders of magnitude apart don't lose accuracy.
///
/// The solution is then checked against the KKT system. If it doesn't satisfy it, the problem
/// has no optimum, either because the objective is unbounded or because the constraints are
/// infeasible, and `Ok(None)` is returned. If it satisfies it there but not `Eᵀx = b` in the
/// original units, the problem is too ill-conditioned to solve, and
/// [`CutupError::SingularCovariance`] is returned.
pub(crate) fn solve_kkt(
    k: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<Option<Solution>, CutupError> {
    let units = Units::new(k, c, e, b);
    let Some((u, multipliers)) = solve_noisy_kkt(&units.k, &units.c, &units.e, &units.b, 0.0)?
    else {
        return Ok(None);
    };
    let x = checked(units.original(&u), e, b, &units.column_scales)?;
    Ok(Some((x, units.original_multipliers(&multipliers))))
}

/// `x`, if it satisfies `Eᵀx = b` in the original units: each constraint to within `1e-8` of
/// the size of its terms `|Eᵢⱼxᵢ|` and `|bⱼ|`, computed so that huge terms can't overflow.
///
/// [`solve_noisy_kkt`] checks its solution in [`Units`], where a variable with a near-zero variance
/// has a tiny entry in `E`. Converting back multiplies that variable's rounding by its inverse
/// scale, which can leave the constraint unmet by orders of magnitude more than the terms'
/// rounding. Such a problem is too ill-conditioned to solve, so it's
/// [`CutupError::SingularCovariance`]. `column_scales` are [`Units`]' largest entry of each
/// column of `E`.
fn checked(
    x: DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    column_scales: &[f64],
) -> Result<DVector<f64>, CutupError> {
    for ((column, &target), &scale) in e.column_iter().zip(b.iter()).zip(column_scales) {
        let terms = (column / scale).component_mul(&x);
        let target = target / scale;
        let largest = terms.amax().max(target.abs());
        if !largest.is_finite() {
            return Err(CutupError::SingularCovariance);
        }
        if largest == 0.0 {
            continue;
        }
        let residual = terms.iter().map(|t| t / largest).sum::<f64>() - target / largest;
        let size =
            terms.iter().map(|t| (t / largest).abs()).sum::<f64>() + (target / largest).abs();
        // A NaN entry, which `amax` skips over, leaves a NaN residual that no comparison fails.
        if residual.is_nan() || residual.abs() > 1e-8 * size {
            return Err(CutupError::SingularCovariance);
        }
    }
    Ok(x)
}

/// A problem in `x` restated in `u = Dx`, with `D = diag(sqrt(Kᵢᵢ))`: `K` becomes
/// `D⁻¹KD⁻¹`, with a unit diagonal, and `c` and the rows of `E` are divided by `D`. Bounds on
/// `x` scale by `D` like `x`, so `x ≥ 0` is unchanged. Each column of `E` and its entry of `b`
/// are also divided by the column's largest entry first, so that the division by `D` can't
/// overflow; that scales the multipliers, which are scaled back. A zero or non-finite `Kᵢᵢ` keeps its
/// variable's scale, so a linear objective's free block or a bad input passes through.
struct Units {
    k: DMatrix<f64>,
    c: DVector<f64>,
    e: DMatrix<f64>,
    b: DVector<f64>,
    scales: DVector<f64>,
    column_scales: Vec<f64>,
}

impl Units {
    fn new(k: &DMatrix<f64>, c: &DVector<f64>, e: &DMatrix<f64>, b: &DVector<f64>) -> Self {
        Units::scaled(k, c, e, b, diagonal_scales(k))
    }

    /// [`Units::new`] with the variables' scales given.
    fn scaled(
        k: &DMatrix<f64>,
        c: &DVector<f64>,
        e: &DMatrix<f64>,
        b: &DVector<f64>,
        scales: DVector<f64>,
    ) -> Self {
        let column_scales: Vec<f64> = e
            .column_iter()
            .map(|column| match column.amax() {
                s if s.is_finite() && s > 0.0 => s,
                _ => 1.0,
            })
            .collect();
        // Dividing by each scale in turn, so that tiny scales' product can't underflow.
        Units {
            k: DMatrix::from_fn(k.nrows(), k.ncols(), |i, j| {
                k[(i, j)] / scales[i] / scales[j]
            }),
            c: c.component_div(&scales),
            e: DMatrix::from_fn(e.nrows(), e.ncols(), |i, j| {
                e[(i, j)] / column_scales[j] / scales[i]
            }),
            b: DVector::from_fn(b.len(), |j, _| b[j] / column_scales[j]),
            scales,
            column_scales,
        }
    }

    /// `x = D⁻¹u`.
    fn original(&self, u: &DVector<f64>) -> DVector<f64> {
        u.component_div(&self.scales)
    }

    /// The multipliers of the original constraints.
    fn original_multipliers(&self, multipliers: &DVector<f64>) -> DVector<f64> {
        DVector::from_fn(multipliers.len(), |j, _| {
            multipliers[j] / self.column_scales[j]
        })
    }
}

/// `sqrt(Kᵢᵢ)` for each variable, or 1 where that's zero or not finite: [`Units`]' scales.
fn diagonal_scales(k: &DMatrix<f64>) -> DVector<f64> {
    k.diagonal().map(|v| match v.sqrt() {
        s if s.is_finite() && s > 0.0 => s,
        _ => 1.0,
    })
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
    let (range, _) = eigen.split(cutoff);
    let v = range
        .iter()
        .fold(DVector::zeros(z.ncols()), |v, (value, w)| {
            v + w * (w.dot(&gradient) / value)
        });
    let x = x0 + z * v;
    let multipliers = constraints.multipliers(&(&c - k * &x));

    let residual = kkt_residual((k, &c, e, &b), &x, &multipliers, noise / problem.k_scale);
    match within_rounding(residual) {
        None => return Err(CutupError::SingularCovariance),
        Some(false) => return Ok(None),
        Some(true) => {}
    }
    if x.iter().any(|w| !w.is_finite()) {
        return Err(CutupError::SingularCovariance);
    }
    let multipliers = DVector::from_fn(multipliers.len(), |j, _| {
        multipliers[j] * problem.k_scale / problem.column_scales[j]
    });
    Ok(Some((x, multipliers)))
}

/// How far `x` and `λ` are from solving `Kx + Eλ = c`, `Eᵀx = b`, and the size of the terms
/// that residual is a difference of: `‖K‖‖x‖ + ‖E‖(‖x‖ + ‖λ‖) + ‖c‖ + ‖b‖`, plus `noise` for a
/// `c` that may have cancelled.
fn kkt_residual(
    (k, c, e, b): (&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DVector<f64>),
    x: &DVector<f64>,
    multipliers: &DVector<f64>,
    noise: f64,
) -> (f64, f64) {
    let stationarity = norm(&(k * x + e * multipliers - c));
    let feasibility = norm(&(e.transpose() * x - b));
    let x_norm = norm(x);
    let size =
        norm(k) * x_norm + norm(e) * (x_norm + norm(multipliers)) + norm(c) + norm(b) + noise;
    (stationarity + feasibility, size)
}

/// Whether a [`kkt_residual`] is within `1e-8` of the size of its terms, or `None` if either
/// overflowed. A check that overflows can't tell a solution from an inconsistent system:
/// `∞ ≤ ∞` holds, so it would pass.
fn within_rounding((residual, size): (f64, f64)) -> Option<bool> {
    if !residual.is_finite() || !size.is_finite() {
        return None;
    }
    Some(residual <= 1e-8 * size)
}

/// The Euclidean norm of `m`'s entries, computed on `m / max|mᵢⱼ|` so that squaring entries
/// above about 1e154 doesn't overflow. Infinite or NaN if an entry is.
pub(crate) fn norm<R: nalgebra::Dim, C: nalgebra::Dim, S: nalgebra::RawStorage<f64, R, C>>(
    m: &nalgebra::Matrix<f64, R, C, S>,
) -> f64 {
    if m.iter().any(|v| v.is_nan()) {
        return f64::NAN;
    }
    let largest = m.amax();
    if largest == 0.0 || largest.is_infinite() {
        return largest;
    }
    largest * m.iter().map(|v| (v / largest).powi(2)).sum::<f64>().sqrt()
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
    let eigen =
        SymmetricEigen::new(&(z.transpose() * k * z)).ok_or(CutupError::SingularCovariance)?;
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
            let mut null = q_transpose.rows(rank, n - rank).transpose();
            // Householder leaves Z orthogonal to Y only to rounding of 1, so a tiny entry of
            // Z, where Y is near 1, can be wrong in all of its digits. That entry is a
            // near-zero-variance variable's, and converting back to x divides it by that
            // variable's tiny scale. Removing Y's part makes it accurate in itself: twice,
            // since the first pass can only cancel an error much larger than the entry.
            for _ in 0..2 {
                null -= &range * (range.transpose() * &null);
            }
            null
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

/// Lower and upper bounds `l ≤ x ≤ u` on each variable, and on linear combinations of them,
/// [`Rows`]. Any bound may be infinite.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) lower: DVector<f64>,
    pub(crate) upper: DVector<f64>,
    pub(crate) rows: Rows,
}

/// Two-sided limits `lⱼ ≤ aⱼᵀx ≤ uⱼ` on linear combinations of the variables, one for each
/// column `aⱼ` of `a`, such as a cap and a floor on a group of weights' sum.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Rows {
    pub(crate) a: DMatrix<f64>,
    pub(crate) lower: DVector<f64>,
    pub(crate) upper: DVector<f64>,
}

impl Bounds {
    /// `lower ≤ x ≤ upper`, with no [`Rows`].
    pub(crate) fn new(lower: DVector<f64>, upper: DVector<f64>) -> Self {
        let n = lower.len();
        Bounds {
            lower,
            upper,
            rows: Rows {
                a: DMatrix::zeros(n, 0),
                lower: DVector::zeros(0),
                upper: DVector::zeros(0),
            },
        }
    }

    /// `x ≥ 0`, with no upper bound.
    pub(crate) fn nonnegative(n: usize) -> Self {
        Bounds::new(DVector::zeros(n), DVector::from_element(n, f64::INFINITY))
    }
}

/// A cost `κᵢ|xᵢ − aᵢ|` on moving each variable away from `aᵢ`, added to the objective of
/// [`solve_piecewise`]: a kink in the objective at `a`. Zero slopes add nothing.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Kinks {
    pub(crate) at: DVector<f64>,
    pub(crate) slopes: DVector<f64>,
}

impl Kinks {
    /// No kinks, for `n` variables.
    pub(crate) fn none(n: usize) -> Self {
        Kinks {
            at: DVector::zeros(n),
            slopes: DVector::zeros(n),
        }
    }
}

/// One variable's range `[l, u]`, cut at its kink when that lies strictly inside, into
/// segments on which the objective is smooth. `slopes[j]` is the derivative of the kink's cost
/// on segment `j`: `−κ` below the kink and `+κ` above it.
#[derive(Debug, Clone)]
struct Segments {
    points: Vec<f64>,
    slopes: Vec<f64>,
}

impl Segments {
    fn new(lower: f64, upper: f64, at: f64, slope: f64) -> Self {
        if slope > 0.0 && lower < at && at < upper {
            Segments {
                points: vec![lower, at, upper],
                slopes: vec![-slope, slope],
            }
        } else {
            let side = if at <= lower { slope } else { -slope };
            Segments {
                points: vec![lower, upper],
                slopes: vec![if slope > 0.0 { side } else { 0.0 }],
            }
        }
    }

    fn last(&self) -> usize {
        self.points.len() - 1
    }
}

/// Where a variable is: held at one of its segments' end points, or free inside a segment.
#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    At(usize),
    Free(usize),
}

/// Which end of its segment a variable reaches.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Side {
    Lower,
    Upper,
}

/// Solves `minimize ½ xᵀQx − cᵀx subject to Eᵀx = b and l ≤ x ≤ u`, and the bounds' [`Rows`],
/// for a positive semidefinite `Q`, starting from `start`, which must be feasible. This is
/// [`solve_piecewise`] with no kinks.
pub(crate) fn solve_bounded(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &Bounds,
    start: DVector<f64>,
) -> Result<DVector<f64>, CutupError> {
    solve_piecewise(q, c, e, b, bounds, &Kinks::none(q.nrows()), start)
}

/// Solves `minimize ½ xᵀQx − cᵀx + Σᵢ κᵢ|xᵢ − aᵢ| subject to Eᵀx = b and l ≤ x ≤ u`, and the
/// bounds' [`Rows`], for a positive semidefinite `Q` and `κ ≥ 0`, starting from `start`, which
/// must be feasible. A bound may be infinite.
///
/// This is a primal active-set method (Nocedal and Wright, 2006, Algorithm 16.3). Each
/// variable's range is cut at its kink `aᵢ` into segments on which the objective is smooth,
/// with the kink's cost a linear term of `±κᵢ`. Variables in the active set are held at an end
/// of a segment, a bound or the kink, and each iteration solves the equality-constrained
/// problem in the others, each within its segment, with [`solve_kkt`]. If that solution stays
/// within the segments the method moves to it, and otherwise moves toward it until a variable
/// reaches an end, which joins the active set there. At an equality-constrained optimum, a
/// held variable leaves the active set if moving it into a segment next to it lowers the
/// objective: with `ν = Qx − c + Eλ` and that segment's slope `s`, if `ν + s < 0` going up or
/// `ν + s > 0` going down. At a lower bound that's `ν < 0`, at an upper bound `ν > 0`, and at
/// a kink `|ν| > κ`. When none does, `x` satisfies the KKT conditions and is optimal. When `Q`
/// is singular and the equality-constrained problem has no minimum, the method follows a
/// direction of zero curvature that lowers the objective instead.
///
/// A variable whose bounds are equal is fixed and never leaves the active set.
///
/// Each row `l ≤ aᵀx ≤ u` gets a slack variable `s = aᵀx` bounded by its limits, from
/// [`with_slacks`], so a row reaches a limit, is held there and is released as a variable is.
/// A slack that isn't held only measures its row, so each solve leaves it and its row out and
/// sets it from the step afterwards. That keeps the block of `Q` the solve factors free of the
/// slacks' zero curvature.
///
/// The returned `x` is clamped to the bounds, so rounding never leaves an entry outside them.
///
/// # Errors
///
/// * [`CutupError::Unbounded`] if the objective decreases without limit.
/// * [`CutupError::SingularCovariance`] if the linear algebra fails, or the solution misses
///   `Eᵀx = b` in the original units, as [`solve_kkt`]'s can.
/// * [`CutupError::NotConverged`] if the iteration cap is reached, or a variable that would
///   lower the objective can't enter without the solver cycling. Only rounding trouble on
///   extreme inputs should cause either.
pub(crate) fn solve_piecewise(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &Bounds,
    kinks: &Kinks,
    start: DVector<f64>,
) -> Result<DVector<f64>, CutupError> {
    let n = q.nrows();
    let rows = &bounds.rows;
    let Problem { q, c, e, b } = with_slacks((q, c, e, b), &rows.a);
    // In the units of [`solve_kkt`], so that the tolerances below compare like with like. The
    // kinks move with x, and their slopes scale like c.
    let units = Units::scaled(&q, &c, &e, &b, scales_with_slacks(&q, &rows.a));
    let segments: Vec<Segments> = (0..n)
        .map(|i| {
            let scale = units.scales[i];
            Segments::new(
                bounds.lower[i] * scale,
                bounds.upper[i] * scale,
                kinks.at[i] * scale,
                kinks.slopes[i] / scale,
            )
        })
        .chain((0..rows.a.ncols()).map(|j| {
            let scale = units.scales[n + j];
            Segments::new(rows.lower[j] * scale, rows.upper[j] * scale, 0.0, 0.0)
        }))
        .collect();
    let start = DVector::from_iterator(
        q.nrows(),
        start
            .iter()
            .copied()
            .chain(rows.a.tr_mul(&start).iter().copied()),
    )
    .component_mul(&units.scales);
    let u = solve_piecewise_in_units(
        (&units.k, &units.c, &units.e, &units.b),
        &segments,
        start,
        n,
    )?;
    // Against the problem's own constraints, not the slacks' rows, which only measure each
    // row's sum: the slacks don't come back. A row whose variables all end at 0 leaves its slack
    // at rounding of 0, which, against no other term, is all residual.
    let m = e.ncols() - rows.a.ncols();
    checked(
        units.original(&u).rows(0, n).into_owned(),
        &e.view((0, 0), (n, m)).into_owned(),
        &b.rows(0, m).into_owned(),
        &units.column_scales[..m],
    )
}

/// The problem `(Q, c, E, b)` with a slack variable `sⱼ = aⱼᵀx` for each column `aⱼ` of `a`,
/// after `x`: no curvature or linear term, and a constraint `aⱼᵀx − sⱼ = 0` each, after `E`'s.
/// Bounding `sⱼ` then bounds `aⱼᵀx`, and the active-set method holds a row at a limit as it
/// holds any variable at a bound.
fn with_slacks(
    (q, c, e, b): (&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DVector<f64>),
    a: &DMatrix<f64>,
) -> Problem {
    let (n, r) = a.shape();
    let m = e.ncols();
    let mut q_all = DMatrix::zeros(n + r, n + r);
    q_all.view_mut((0, 0), (n, n)).copy_from(q);
    let mut e_all = DMatrix::zeros(n + r, m + r);
    e_all.view_mut((0, 0), (n, m)).copy_from(e);
    e_all.view_mut((0, m), (n, r)).copy_from(a);
    e_all.view_mut((n, m), (r, r)).fill_diagonal(-1.0);
    let zeros = |v: &DVector<f64>| {
        DVector::from_iterator(
            v.len() + r,
            v.iter().copied().chain(std::iter::repeat_n(0.0, r)),
        )
    };
    Problem {
        q: q_all,
        c: zeros(c),
        e: e_all,
        b: zeros(b),
    }
}

/// A problem `(Q, c, E, b)` that [`with_slacks`] built. A struct rather than a tuple, so that
/// cargo-mutants doesn't try every combination of replacement values for its four parts.
struct Problem {
    q: DMatrix<f64>,
    c: DVector<f64>,
    e: DMatrix<f64>,
    b: DVector<f64>,
}

/// [`Units`]' scales for `q`, which [`with_slacks`] padded with a slack for each column of
/// `a`: [`diagonal_scales`] for the variables, and [`slack_scale`] for the slacks after them.
fn scales_with_slacks(q: &DMatrix<f64>, a: &DMatrix<f64>) -> DVector<f64> {
    let n = a.nrows();
    let mut scales = diagonal_scales(q);
    for (j, column) in a.column_iter().enumerate() {
        scales[n + j] = slack_scale(&scales.as_slice()[..n], column.iter().copied());
    }
    scales
}

/// The scale of the slack variable `s = aᵀx` in [`Units`], given the variables' `scales`:
/// `minᵢ scaleᵢ / |aᵢ|` over the variables in it, so that in units `s` is a combination
/// whose largest coefficient is 1, like a variable's own row. 1 if `a` is zero.
fn slack_scale(scales: &[f64], a: impl IntoIterator<Item = f64>) -> f64 {
    let scale = a
        .into_iter()
        .zip(scales)
        .filter(|(a, _)| *a != 0.0)
        .map(|(a, s)| s / a.abs())
        .fold(f64::INFINITY, f64::min);
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

/// The free `indices`, in order, split into the variables before `originals` and the slacks
/// from it on.
fn split_slacks(indices: &[usize], originals: usize) -> (&[usize], &[usize]) {
    indices.split_at(indices.partition_point(|&i| i < originals))
}

/// [`solve_piecewise`] for a problem already in [`Units`], whose variables after the first
/// `originals` are [`with_slacks`]' slacks, one for each of the last columns of `E`.
fn solve_piecewise_in_units(
    (q, c, e, b): (&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DVector<f64>),
    segments: &[Segments],
    start: DVector<f64>,
    originals: usize,
) -> Result<DVector<f64>, CutupError> {
    let n = q.nrows();
    let first_slack_column = e.ncols() - (n - originals);
    let fixed: Vec<bool> = segments
        .iter()
        .map(|s| s.points[0] >= s.points[s.last()])
        .collect();
    let mut x = DVector::from_fn(n, |i, _| {
        let s = &segments[i];
        if fixed[i] {
            s.points[0]
        } else {
            start[i].clamp(s.points[0], s.points[s.last()])
        }
    });
    let mut state: Vec<State> = (0..n)
        .map(|i| {
            let s = &segments[i];
            if fixed[i] || x[i] <= s.points[0] {
                return State::At(0);
            }
            if x[i] >= s.points[s.last()] {
                return State::At(s.last());
            }
            match s.points.iter().position(|&p| p == x[i]) {
                Some(p) => State::At(p),
                None => State::Free(s.points.iter().rposition(|&p| p < x[i]).unwrap_or(0)),
            }
        })
        .collect();
    let slope_amax = segments
        .iter()
        .flat_map(|s| s.slopes.iter())
        .fold(0.0f64, |m, s| m.max(s.abs()));
    // Variables that were released and then blocked at once without x moving. Releasing them
    // again would loop, so they stay in the active set until x moves.
    let mut refused = vec![false; n];
    let mut released = None;
    // Where every variable was last freed at once, so that returning there without x moving
    // stops instead of looping.
    let mut freed_at: Option<DVector<f64>> = None;
    let mut factoring = Factoring::new();

    for _ in 0..100 + 20 * n {
        let indices: Vec<usize> = (0..n)
            .filter(|&i| matches!(state[i], State::Free(_)))
            .collect();
        let (solved, slacks) = split_slacks(&indices, originals);
        // Every variable is held, as when starting from a vertex. Freeing all of them lets the
        // steps below block back the ones that belong where they are, one per step; at an
        // optimal vertex they block back onto it without x moving. Steps never block the last
        // free variable, so getting here again at the same x means the solver can't make
        // progress. Free slacks alone don't count: they only measure their rows.
        if solved.is_empty() {
            if fixed[..originals].iter().all(|&f| f) {
                return Ok(x);
            }
            if freed_at.as_ref() == Some(&x) {
                return Err(CutupError::NotConverged);
            }
            freed_at = Some(x.clone());
            for i in (0..n).filter(|&i| !fixed[i]) {
                if let State::At(p) = state[i] {
                    state[i] = State::Free(p.min(segments[i].last() - 1));
                }
            }
            continue;
        }
        // Each free variable's segment: its ends, and the slope of the kink's cost on it,
        // which joins the linear term.
        let free: Vec<(&Segments, usize)> = indices
            .iter()
            .map(|&i| match state[i] {
                State::Free(j) => (&segments[i], j),
                State::At(_) => unreachable!("only free variables are indexed"),
            })
            .collect();
        let lower: Vec<f64> = free.iter().map(|(s, j)| s.points[*j]).collect();
        let upper: Vec<f64> = free.iter().map(|(s, j)| s.points[j + 1]).collect();
        let slope = DVector::from_iterator(
            solved.len(),
            free.iter().map(|(s, j)| s.slopes[*j]).take(solved.len()),
        );
        let slack_columns: Vec<(usize, usize)> = slacks
            .iter()
            .map(|&i| (i, first_slack_column + i - originals))
            .collect();
        let kept = kept_columns(e, solved, &slack_columns, first_slack_column);
        let q_free = q.select_rows(solved).select_columns(solved);
        let c_free = c.select_rows(solved) - slope;
        let e_free = e.select_rows(solved).select_columns(&kept);
        let x_free = x.select_rows(&indices);
        let q_rows = q.select_rows(solved);
        // The variables without the slacks, whose sizes the tolerances below measure: a slack
        // sums up to all of them, and would loosen each tolerance up to that many times.
        let variables = x.rows(0, originals).into_owned();

        // Solve for the step from x rather than for the new point, so that when Q is singular
        // and the free problem has many minima, the target is the one closest to x. The
        // minimum-norm one could put a just-released variable outside its segment, blocking it
        // at once. The rows of Q over all of x include the pull of variables held at a
        // nonzero point.
        let gradient_free = &c_free - &q_rows * &x;
        let residuals = b - e.transpose() * &x;
        let residual = residuals.select_rows(&kept);
        let noise = step_noise(&q_rows, &variables, &c_free);
        // With Q positive definite on the free variables, from the factor kept across
        // iterations. Otherwise, or if that misses, from the eigendecomposition, which handles
        // a singular Q.
        let problem = (&q_free, &gradient_free, &e_free, &residual);
        let factored = factoring
            .update(q, solved)
            .and_then(|cholesky| solve_factored(cholesky, problem, noise));
        let solution = match factored {
            Some(solution) => Some(solution),
            None => {
                factoring.reset();
                solve_noisy_kkt(&q_free, &gradient_free, &e_free, &residual, noise)?
            }
        };
        let slack_steps = |step: DVector<f64>, residuals: Option<&DVector<f64>>| {
            with_slack_steps(step, (solved, &slack_columns), e, residuals)
        };
        let solution = solution.map(|(delta, multipliers)| {
            let mut all = DVector::zeros(e.ncols());
            for (k, &j) in kept.iter().enumerate() {
                all[j] = multipliers[k];
            }
            (slack_steps(delta, Some(&residuals)), all)
        });
        let Some((delta, multipliers)) = solution else {
            // Without a linear term on the free variables the free problem is bounded below:
            // the held variables only add Q_fh·x_h, which for a positive semidefinite Q lies in
            // the range of Q_ff. So it has a minimum, and the solve failed numerically.
            if c_free.iter().all(|&v| v == 0.0) {
                return Err(CutupError::SingularCovariance);
            }
            // No minimum on the free variables: move along a direction of zero curvature
            // until a variable reaches an end of its segment. The gradient includes the pull of
            // the held variables; its free variables' own part, Q_ff·x_f, is orthogonal to any
            // direction of zero curvature, so descent_ray leaves it out anyway.
            let direction = slack_steps(descent_ray(&q_free, &gradient_free, &e_free)?, None);
            let (step, blocking, side) =
                ratio_test(&x_free, &direction, &lower, &upper).ok_or(CutupError::Unbounded)?;
            for k in 0..indices.len() {
                x[indices[k]] = (x_free[k] + step * direction[k]).clamp(lower[k], upper[k]);
            }
            block(
                &mut x,
                &mut state,
                &mut refused,
                (indices[blocking], side),
                segments,
                step,
                released,
            );
            released = None;
            continue;
        };

        let target = &x_free + delta;

        // Step toward the target, stopping where the first variable reaches an end of its
        // segment. Entries within rounding of an end don't block, and are clamped to it
        // instead. Rounding is measured against all of x, since variables held at a nonzero
        // point feed the step: a lone free variable at its bound would otherwise block on a
        // step of 1e-17.
        let tolerance = 1e-12 * target.rows(0, solved.len()).amax().max(variables.amax());
        let blocking = (0..indices.len())
            .filter_map(|k| {
                if target[k] < lower[k] - tolerance {
                    Some((
                        (x_free[k] - lower[k]) / (x_free[k] - target[k]),
                        k,
                        Side::Lower,
                    ))
                } else if target[k] > upper[k] + tolerance {
                    Some((
                        (upper[k] - x_free[k]) / (target[k] - x_free[k]),
                        k,
                        Side::Upper,
                    ))
                } else {
                    None
                }
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((step, k, side)) = blocking {
            for j in 0..indices.len() {
                x[indices[j]] =
                    (x_free[j] + step * (target[j] - x_free[j])).clamp(lower[j], upper[j]);
            }
            block(
                &mut x,
                &mut state,
                &mut refused,
                (indices[k], side),
                segments,
                step,
                released,
            );
            released = None;
            continue;
        }
        let mut moved = false;
        for (k, &i) in indices.iter().enumerate() {
            let next = target[k].clamp(lower[k], upper[k]);
            moved |= (next - x[i]).abs() > tolerance;
            x[i] = next;
        }
        if moved {
            refused.fill(false);
        }

        // Release the held variable that lowers the objective fastest by moving into a
        // segment next to it, if any. A refused variable that would means x isn't optimal,
        // but the solver can't make progress on it.
        let gradient = q * &x - c;
        let pushback = e * &multipliers;
        // Like the step, the multipliers are differences of terms up to this size.
        let scale = (q.amax() * x.rows(0, originals).amax() + c.amax().max(slope_amax))
            .max(pushback.amax());
        let wrong: Vec<(f64, usize, usize)> = (0..n)
            .filter(|&i| !fixed[i])
            .filter_map(|i| {
                let State::At(p) = state[i] else {
                    return None;
                };
                let s = &segments[i];
                let nu = gradient[i] + pushback[i];
                // The objective's rate of decrease moving up into segment p, and down into
                // segment p − 1.
                let up = (p < s.last()).then(|| -(nu + s.slopes[p]));
                let down = (p > 0).then(|| nu + s.slopes[p - 1]);
                match (up, down) {
                    (Some(rate), _) if rate > 1e-9 * scale => Some((rate, i, p)),
                    (_, Some(rate)) if rate > 1e-9 * scale => Some((rate, i, p - 1)),
                    _ => None,
                }
            })
            .collect();
        let entering = wrong
            .iter()
            .filter(|&&(_, i, _)| !refused[i])
            .max_by(|a, b| a.0.total_cmp(&b.0));
        match entering {
            Some(&(_, i, segment)) => {
                state[i] = State::Free(segment);
                released = Some(i);
            }
            None if wrong.is_empty() => return Ok(x),
            None => return Err(CutupError::NotConverged),
        }
    }
    Err(CutupError::NotConverged)
}

/// The columns of `E` that a solve on the variables `solved` keeps: every one before
/// `first_slack`, `E`'s own, and each slack's row but those of the free `slacks`, given as
/// `(slack, column)`, and those with no entry on `solved`.
///
/// A free slack only measures its row, so it moves to keep its row once the step is known,
/// and the row's multiplier is 0. Without them, `Q`'s block on `solved` has no zero rows and
/// can be factored. A held slack's row on held variables alone has no entry on `solved`, so its
/// multiplier is 0 too, as for any dependent constraint, and keeping it would leave a zero
/// column in `E` that no factored solve takes.
fn kept_columns(
    e: &DMatrix<f64>,
    solved: &[usize],
    slacks: &[(usize, usize)],
    first_slack: usize,
) -> Vec<usize> {
    (0..e.ncols())
        .filter(|&j| {
            j < first_slack
                || (!slacks.iter().any(|&(_, column)| column == j)
                    && solved.iter().any(|&i| e[(i, j)] != 0.0))
        })
        .collect()
}

/// A step `δ` on the variables `solved`, extended to the free slacks: each `(i, j)` of
/// `slacks`, slack `i` with its constraint in column `j` of `E`, moves so that the constraint
/// holds after the step, `Eⱼᵀ(x + δ) = bⱼ`, given the `residuals` `b − Eᵀx`, or none for a
/// direction.
fn with_slack_steps(
    step: DVector<f64>,
    (solved, slacks): (&[usize], &[(usize, usize)]),
    e: &DMatrix<f64>,
    residuals: Option<&DVector<f64>>,
) -> DVector<f64> {
    let moves = slacks.iter().map(|&(i, j)| {
        let residual = residuals.map_or(0.0, |r| r[j]);
        let others: f64 = solved
            .iter()
            .zip(step.iter())
            .map(|(&k, d)| e[(k, j)] * d)
            .sum();
        (residual - others) / e[(i, j)]
    });
    let len = step.len() + slacks.len();
    DVector::from_iterator(len, step.iter().copied().chain(moves))
}

/// Solves [`solve_piecewise`]'s problem with the turnover `Σᵢ|xᵢ − aᵢ|` capped at `cap`, `a`
/// being the kinks' points. A multiplier `λ ≥ 0` on the cap adds `λ` to every slope, and the
/// turnover of the optimum with slopes `κ + λ` falls as `λ` grows, so [`search_multiplier`]
/// finds the `λ` that reaches the cap, each solve starting from the last solution within it.
///
/// The caller checks that some feasible `x` turns over at most `cap`. Then a large enough `λ`
/// reaches one: at the slopes `|Qa − c|` every variable that can stay at its kink does.
///
/// # Errors
///
/// Any error from [`solve_piecewise`] without the cap but [`CutupError::Unbounded`], which a
/// small `λ` only means is too small, or [`CutupError::NotConverged`] if no `λ` the solver can
/// handle, up to 2¹⁰⁰⁰ times that, reaches the cap.
pub(crate) fn solve_turnover_capped(
    q: &DMatrix<f64>,
    c: &DVector<f64>,
    e: &DMatrix<f64>,
    b: &DVector<f64>,
    bounds: &Bounds,
    (kinks, cap): (&Kinks, f64),
    start: DVector<f64>,
) -> Result<DVector<f64>, CutupError> {
    let turnover = |x: &DVector<f64>| (x - &kinks.at).abs().sum();
    // The optimum with every slope raised by `extra`, and its turnover. An objective with no
    // minimum runs the weights off without limit, past any cap, so it turns over infinitely.
    let solve = |extra: f64, start: DVector<f64>| {
        let raised = Kinks {
            at: kinks.at.clone(),
            slopes: kinks.slopes.add_scalar(extra),
        };
        match solve_piecewise(q, c, e, b, bounds, &raised, start) {
            Ok(x) => Ok((turnover(&x), Some(x))),
            Err(CutupError::Unbounded) => Ok((f64::INFINITY, None)),
            Err(error) => Err(error),
        }
    };
    search_multiplier(
        solve,
        // Summing the turnover rounds by a few ulps of the largest weight, so within that of
        // the cap counts as at it.
        (cap, |x: &DVector<f64>| {
            sum_rounding(x.len(), x.amax().max(1.0))
        }),
        kink_multiplier(q, c, kinks),
        start,
    )
}

/// The slope `max |Qa − c|` at which every variable that can stay at its kink `a` does, where
/// [`solve_turnover_capped`]'s search starts, or 1 if that's 0.
fn kink_multiplier(q: &DMatrix<f64>, c: &DVector<f64>, kinks: &Kinks) -> f64 {
    let pull = (q * &kinks.at - c).amax();
    if pull > 0.0 { pull } else { 1.0 }
}

/// How far a sum of `terms` terms, none larger than `size`, can round: a few ulps of `size`
/// for each.
pub(crate) fn sum_rounding(terms: usize, size: f64) -> f64 {
    4.0 * terms as f64 * f64::EPSILON * size
}

/// Finds the multiplier `λ ≥ 0` of a constraint `g(x) ≤ limit` on a convex problem whose
/// other constraints `solve(λ, start)` handles, returning the solution there. `solve` returns
/// the optimum of the objective plus `λg`, starting from `start`, with its `g`, or `(∞, None)`
/// if that objective has no minimum. `g` at the optimum falls as `λ` grows.
///
/// If `λ = 0` already meets the limit, within `slack` of the solution, that's the answer.
/// Otherwise `λ` is bracketed by doubling from `first`, and then found by alternating secant
/// and bisection steps, each solve starting from the last solution within the limit. When `g`
/// is piecewise linear in `λ`, as it is with the active set fixed, once both ends of the
/// bracket share an active set the secant step lands on the limit, and the search stops there.
///
/// # Errors
///
/// Any error from `solve` at `λ = 0`. At a larger `λ`, [`CutupError::Unbounded`] from `solve`,
/// and [`CutupError::NotConverged`] for any other error, or if no `λ` up to 2¹⁰⁰⁰ times `first`
/// meets the limit.
pub(crate) fn search_multiplier(
    mut solve: impl FnMut(f64, DVector<f64>) -> Result<(f64, Option<DVector<f64>>), CutupError>,
    (limit, slack): (f64, impl Fn(&DVector<f64>) -> f64),
    first: f64,
    start: DVector<f64>,
) -> Result<DVector<f64>, CutupError> {
    let (mut low_value, unconstrained) = solve(0.0, start.clone())?;
    if let Some(x) = unconstrained
        .as_ref()
        .filter(|x| low_value <= limit + slack(x))
    {
        return Ok(x.clone());
    }
    let mut low = 0.0;
    let mut warm = unconstrained.unwrap_or(start);
    let mut high = first;
    let (mut high_value, mut best) = loop {
        // The problem itself solved at λ = 0, so a failure only at a larger λ is its arithmetic
        // breaking down short of the limit: no λ the solver can handle reaches it. Unless the
        // objective has no minimum, which no larger λ changes.
        let solved = solve(high, warm.clone()).map_err(|error| match error {
            CutupError::Unbounded => error,
            _ => CutupError::NotConverged,
        })?;
        match solved {
            (value, Some(x)) if value <= limit + slack(&x) => break (value, x),
            (value, x) => {
                (low, low_value) = (high, value);
                warm = x.unwrap_or(warm);
            }
        }
        if high > f64::MAX / 2.0 {
            return Err(CutupError::NotConverged);
        }
        high *= 2.0;
    };

    for step in 0.. {
        if high_value >= limit - slack(&best) {
            return Ok(best);
        }
        let secant = low + (low_value - limit) / (low_value - high_value) * (high - low);
        let middle = if step % 2 == 0 && secant > low && secant < high {
            secant
        } else {
            low + (high - low) / 2.0
        };
        if middle <= low || middle >= high {
            return Ok(best);
        }
        match solve(middle, best.clone())? {
            (value, Some(x)) if value <= limit + slack(&x) => {
                (high, high_value, best) = (middle, value, x);
            }
            (value, _) => (low, low_value) = (middle, value),
        }
    }
    unreachable!("the loop returns once the bracket can't be split")
}

/// The size of the terms the free problem's linear term `c_f − Q_f·x` is a difference of,
/// `|Q_f|·|x| + |c_f|` in their largest entries, for [`solve_noisy_kkt`]'s `noise`.
fn step_noise(q_rows: &DMatrix<f64>, x: &DVector<f64>, c_free: &DVector<f64>) -> f64 {
    q_rows.amax() * x.amax() + c_free.amax()
}

/// The pivot `Lᵢᵢ²` of a Cholesky factor below which [`Factoring`] counts `Q`'s block on the
/// free variables as singular, leaving it to [`solve_noisy_kkt`]. In [`Units`], `Q` has a unit
/// diagonal, so this is relative. It's `2⁻²⁶`, about `1.5e-8`.
///
/// The smallest pivot only bounds the smallest eigenvalue from above, so a block can pass with
/// an eigenvalue below [`solve_noisy_kkt`]'s cutoff, whose direction that would treat as zero
/// curvature. The factored solve then takes the block as nonsingular, and [`within_rounding`]
/// checks its solution.
const PIVOT_FLOOR: f64 = 1.0 / 67_108_864.0;

/// Whether every pivot of `cholesky` is at least [`PIVOT_FLOOR`]. A NaN pivot isn't.
fn positive_definite(cholesky: &Cholesky<f64, Dyn>) -> bool {
    cholesky
        .l_dirty()
        .diagonal()
        .iter()
        .all(|&p| p * p >= PIVOT_FLOOR)
}

/// A Cholesky factor `LLᵀ` of `Q`'s block on [`solve_piecewise_in_units`]' free variables,
/// kept from one iteration to the next. The free set usually gains or loses one variable an
/// iteration, and updating `L` for that costs `O(k²)` for `k` free variables, where a fresh
/// factor, or [`solve_noisy_kkt`]'s eigendecomposition, costs `O(k³)`.
struct Factoring {
    /// The free variables in increasing order, and the factor of their block.
    factor: Option<(Vec<usize>, Cholesky<f64, Dyn>)>,
    /// The last free set whose block was singular, or nearly so. Every set containing it is
    /// too, since every principal block of a positive definite matrix is positive definite.
    singular: Option<Vec<usize>>,
    /// How many factors were computed afresh rather than updated.
    #[cfg(test)]
    factorizations: usize,
}

impl Factoring {
    fn new() -> Self {
        Factoring {
            factor: None,
            singular: None,
            #[cfg(test)]
            factorizations: 0,
        }
    }

    /// The factor of `q`'s block on `indices`, which are in increasing order, or `None` if
    /// that block is singular or nearly so. It's updated from the last one where [`updated`]
    /// can, and otherwise computed afresh unless the block contains the last singular one. An
    /// update that finds the block singular is retried afresh too, since over many updates the
    /// factor drifts by rounding.
    fn update(&mut self, q: &DMatrix<f64>, indices: &[usize]) -> Option<&Cholesky<f64, Dyn>> {
        let updated = self
            .factor
            .take()
            .and_then(|(held, cholesky)| updated(q, held, cholesky, indices));
        let factor = match updated {
            Some(cholesky) => cholesky,
            None => {
                let contains_singular = self
                    .singular
                    .as_ref()
                    .is_some_and(|s| s.iter().all(|i| indices.binary_search(i).is_ok()));
                if contains_singular {
                    return None;
                }
                #[cfg(test)]
                {
                    self.factorizations += 1;
                }
                let block = q.select_rows(indices).select_columns(indices);
                let Some(cholesky) = Cholesky::new(block).filter(positive_definite) else {
                    self.singular = Some(indices.to_vec());
                    return None;
                };
                cholesky
            }
        };
        Some(&self.factor.insert((indices.to_vec(), factor)).1)
    }

    /// Drops the factor, as when its solution missed [`within_rounding`], which a factor drifted
    /// over many updates can, so that the next one is computed afresh.
    fn reset(&mut self) {
        self.factor = None;
    }
}

/// `cholesky`, the factor of `q`'s block on `held`, updated to the block on `indices`, both in
/// increasing order: each variable that left is removed, then each that joined is inserted,
/// each in `O(k²)`. `None` if an insertion makes the block singular, or nearly so; stopping
/// there also keeps the next insertion from solving with a zero pivot, which panics.
///
/// Also `None`, for a fresh factor instead, if more than a sixth of `indices` join: `j`
/// insertions into a block of `k` cost about `2jk²`, and a fresh factor `k³/3`. That happens
/// when every variable is freed at once.
fn updated(
    q: &DMatrix<f64>,
    mut held: Vec<usize>,
    mut cholesky: Cholesky<f64, Dyn>,
    indices: &[usize],
) -> Option<Cholesky<f64, Dyn>> {
    let joining = indices
        .iter()
        .filter(|i| held.binary_search(i).is_err())
        .count();
    if 6 * joining > indices.len() {
        return None;
    }
    for position in (0..held.len()).rev() {
        if indices.binary_search(&held[position]).is_err() {
            cholesky = cholesky.remove_column(position);
            held.remove(position);
        }
    }
    // What is left is in `indices` in order, so each joining variable goes in at its position
    // there.
    for (position, &i) in indices.iter().enumerate() {
        if held.get(position) != Some(&i) {
            held.insert(position, i);
            let column = DVector::from_iterator(held.len(), held.iter().map(|&j| q[(j, i)]));
            cholesky = cholesky.insert_column(position, column);
            if !positive_definite(&cholesky) {
                return None;
            }
        }
    }
    Some(cholesky)
}

/// [`solve_noisy_kkt`]'s problem for a positive definite `K` with Cholesky factor `LLᵀ`, by the
/// range-space method (Nocedal and Wright, 2006, §16.2): with `W = L⁻¹E` and `h = L⁻¹c`, the
/// multipliers solve `WᵀWλ = Wᵀh − b`, and `x = L⁻ᵀ(h − Wλ)`. `None` if `E`'s columns are
/// dependent, or nearly so, since the multipliers aren't unique then, or if the solution fails
/// [`within_rounding`]: the caller solves the problem with [`solve_noisy_kkt`] instead.
fn solve_factored(
    cholesky: &Cholesky<f64, Dyn>,
    (k, c, e, b): (&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DVector<f64>),
    noise: f64,
) -> Option<Solution> {
    let l = cholesky.l_dirty();
    let mut w = l.solve_lower_triangular(e)?;
    // Each column of W at unit length, so that `WᵀW` has a unit diagonal and PIVOT_FLOOR is
    // relative to it, as it is for Q; the multipliers scale back by the lengths. A zero
    // column becomes NaN, which no factor accepts.
    let lengths = DVector::from_iterator(w.ncols(), w.column_iter().map(|column| column.norm()));
    for (mut column, length) in w.column_iter_mut().zip(lengths.iter()) {
        column /= *length;
    }
    let gram = Cholesky::new(w.transpose() * &w).filter(positive_definite)?;
    let solve = |c: &DVector<f64>, b: &DVector<f64>| -> Option<Solution> {
        let h = l.solve_lower_triangular(c)?;
        let scaled = gram.solve(&(w.transpose() * &h - b.component_div(&lengths)));
        let x = l.tr_solve_lower_triangular(&(h - &w * &scaled))?;
        Some((x, scaled.component_div(&lengths)))
    };
    let (x, multipliers) = solve(c, b)?;
    // `WᵀW` squares the conditioning of `W`, so where `E`'s columns are close to parallel, as
    // a budget and a target return on similar means are, `Eᵀx = b` holds only to that
    // conditioning. One step of iterative refinement, solving for the residuals, brings it to
    // rounding.
    let (dx, dm) = solve(&(c - k * &x - e * &multipliers), &(b - e.transpose() * &x))?;
    let (x, multipliers) = (x + dx, multipliers + dm);
    within_rounding(kkt_residual((k, c, e, b), &x, &multipliers, noise))?
        .then_some((x, multipliers))
}

/// Moves free variable `i` into the active set at the `side` end of its segment after a step
/// of length `step`. If `i` was just released and blocks without `x` moving, it is refused
/// re-entry until `x` moves.
fn block(
    x: &mut DVector<f64>,
    state: &mut [State],
    refused: &mut [bool],
    (i, side): (usize, Side),
    segments: &[Segments],
    step: f64,
    released: Option<usize>,
) {
    let State::Free(j) = state[i] else {
        unreachable!("only free variables block");
    };
    let point = match side {
        Side::Lower => j,
        Side::Upper => j + 1,
    };
    x[i] = segments[i].points[point];
    state[i] = State::At(point);
    if step > 0.0 {
        refused.fill(false);
    } else if released == Some(i) {
        refused[i] = true;
    }
}

/// The largest step `t ≤ ∞` keeping `x + t·d` within `[lower, upper]`, the entry that reaches
/// an end there, and which end, or `None` if no entry moves toward a finite end. Components of
/// `d` within rounding of zero don't count as moving.
fn ratio_test(
    x: &DVector<f64>,
    d: &DVector<f64>,
    lower: &[f64],
    upper: &[f64],
) -> Option<(f64, usize, Side)> {
    let tolerance = 1e-12 * d.amax();
    (0..x.len())
        .filter_map(|k| {
            if d[k] < -tolerance && lower[k].is_finite() {
                Some(((x[k] - lower[k]) / -d[k], k, Side::Lower))
            } else if d[k] > tolerance && upper[k].is_finite() {
                Some(((upper[k] - x[k]) / d[k], k, Side::Upper))
            } else {
                None
            }
        })
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
    let (_, null) = eigen.split(cutoff);
    let direction = null.iter().fold(DVector::zeros(q.nrows()), |direction, w| {
        let v = &z * w;
        let along = v.dot(c);
        direction + v * along
    });
    // The KKT system was inconsistent, so such a direction exists up to rounding; without one
    // the system was too ill-conditioned to tell.
    let length = norm(&direction);
    if length <= 1e-12 * norm(c) || length == 0.0 {
        return Err(CutupError::SingularCovariance);
    }
    Ok(direction)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    fn budget(n: usize) -> (DMatrix<f64>, DVector<f64>) {
        (DMatrix::from_element(n, 1, 1.0), dvector![1.0])
    }

    fn solve_nonnegative(
        q: &DMatrix<f64>,
        c: &DVector<f64>,
        e: &DMatrix<f64>,
        b: &DVector<f64>,
        start: DVector<f64>,
    ) -> Result<DVector<f64>, CutupError> {
        solve_bounded(q, c, e, b, &Bounds::nonnegative(q.nrows()), start)
    }

    fn capped(upper: DVector<f64>) -> Bounds {
        Bounds::new(DVector::zeros(upper.len()), upper)
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
    fn inconsistent_constraints_are_caught_at_any_scale() {
        // x = 1 and x = b₁ can't both hold. At b₁ ≈ 1e299 the residuals' squares overflow,
        // which made the check pass and returned x as if it were optimal.
        let k = dmatrix![1.0];
        let c = dvector![0.0];
        let e = dmatrix![1.0, 1.0];
        for b1 in [2.0, 1e20, 1e299] {
            assert_eq!(
                solve_kkt(&k, &c, &e, &dvector![1.0, b1]).unwrap(),
                None,
                "{b1}"
            );
        }
    }

    #[test]
    fn the_noise_scales_with_k() {
        // K = 4 is scaled to 1, so the noise of 2 counts as 0.5 against the terms' 5.24: x = 1
        // and x = 1 + 6.5e-8 then disagree by more than 1e-8 of them, but not of 7.24 or more.
        let (k, c, e) = (dmatrix![4.0], dvector![0.0], dmatrix![1.0, 1.0]);
        let b = dvector![1.0, 1.0 + 6.5e-8];
        assert_eq!(solve_noisy_kkt(&k, &c, &e, &b, 2.0), Ok(None));
        assert!(solve_noisy_kkt(&k, &c, &e, &b, 8.0).unwrap().is_some());
    }

    #[test]
    fn the_step_noise_is_the_size_of_its_terms() {
        // 3 · 2 + 4.
        let noise = step_noise(
            &dmatrix![1.0, -3.0],
            &dvector![2.0, -1.0],
            &dvector![0.5, -4.0],
        );
        assert_eq!(noise, 10.0);
    }

    #[test]
    fn null_space_is_accurate_in_its_tiny_entries() {
        // For E = (s, 1), Z ∝ (1, −s). Householder got −s to rounding of 1, so at s = 1e-12
        // it was 2e-5 off relative to s, which converting back to x multiplies by 1/s.
        for s in [1e-4, 1e-12, 1e-50, 1e-200] {
            let z = Constraints::new(&DMatrix::from_column_slice(2, 1, &[s, 1.0])).null;
            let ratio = -z[(1, 0)] / z[(0, 0)];
            assert!((ratio - s).abs() <= 1e-14 * s, "s = {s:e}: {ratio:e}");
        }
    }

    #[test]
    fn norm_does_not_overflow() {
        assert_eq!(norm(&dvector![3e300, 4e300]), 5e300);
        assert_eq!(norm(&dvector![3.0, 4.0]), 5.0);
        assert_eq!(norm(&DVector::<f64>::zeros(2)), 0.0);
        assert_eq!(norm(&dvector![f64::INFINITY, 1.0]), f64::INFINITY);
        assert!(norm(&dvector![f64::NAN, f64::INFINITY]).is_nan());
    }

    #[test]
    fn the_kkt_residual_is_measured_against_its_terms() {
        // Kx + Eλ − c = 8 + 5 − 1 and Eᵀx − b = 4 − 3; the terms are 2·4, 1·(4 + 5), 1, 3 and
        // the noise.
        let problem = (
            &dmatrix![2.0],
            &dvector![1.0],
            &dmatrix![1.0],
            &dvector![3.0],
        );
        let residual = kkt_residual(problem, &dvector![4.0], &dvector![5.0], 0.5);
        assert_eq!(residual, (13.0, 21.5));
    }

    #[test]
    fn a_residual_within_rounding_is_at_most_1e_8_of_its_terms() {
        let at = 1e-8 * 2.0;
        assert_eq!(within_rounding((at, 2.0)), Some(true));
        assert_eq!(
            within_rounding((f64::from_bits(at.to_bits() + 1), 2.0)),
            Some(false)
        );
        assert_eq!(within_rounding((f64::INFINITY, 2.0)), None);
        assert_eq!(within_rounding((f64::NAN, 2.0)), None);
        assert_eq!(within_rounding((0.0, f64::INFINITY)), None);
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
        let (lower, upper) = ([0.0; 3], [f64::INFINITY; 3]);
        assert_eq!(
            ratio_test(&x, &dvector![1.0, -1e-20, 0.5], &lower, &upper),
            None
        );
        assert_eq!(
            ratio_test(&x, &dvector![1.0, -1e-20, -0.5], &lower, &upper),
            Some((2.0, 2, Side::Lower))
        );
    }

    #[test]
    fn ratio_test_stops_at_upper_bounds() {
        let x = dvector![1.0, 1.0];
        assert_eq!(
            ratio_test(&x, &dvector![1.0, 1.0], &[0.0, 0.0], &[4.0, 1.5]),
            Some((0.5, 1, Side::Upper))
        );
    }

    #[test]
    fn ratio_test_ignores_infinite_floors() {
        let x = dvector![1.0, 1.0];
        let lower = [f64::NEG_INFINITY, 0.0];
        assert_eq!(
            ratio_test(&x, &dvector![-1.0, 0.0], &lower, &[2.0, 2.0]),
            None
        );
    }

    #[test]
    fn cap_binds_and_the_rest_moves_to_other_variables() {
        // Unconstrained in x ≥ 0 the optimum is (1, 0); capping x₀ at 0.8 leaves 0.2 for x₁.
        let q = DMatrix::identity(2, 2);
        let c = dvector![3.0, 0.0];
        let (e, b) = budget(2);
        let x = solve_bounded(
            &q,
            &c,
            &e,
            &b,
            &capped(dvector![0.8, 1.0]),
            dvector![0.5, 0.5],
        )
        .unwrap();
        assert_close(&x, &dvector![0.8, 0.2], 1e-15);
        assert_eq!(x[0], 0.8);
    }

    #[test]
    fn variables_are_released_from_upper_bounds() {
        // From the vertex at both caps, x₂ must leave its cap to reach equal weights.
        let q = DMatrix::identity(3, 3);
        let (e, b) = budget(3);
        let bounds = capped(dvector![0.5, 0.5, 0.5]);
        let start = dvector![0.0, 0.5, 0.5];
        let x = solve_bounded(&q, &DVector::zeros(3), &e, &b, &bounds, start).unwrap();
        assert_close(&x, &DVector::from_element(3, 1.0 / 3.0), 1e-15);
    }

    #[test]
    fn floors_and_fixed_variables_hold() {
        // Minimizing ½‖x‖² − 3x₀ wants x₀ large; x₁'s floor of 0.3 and x₂ fixed at 0.1 leave
        // x₀ at most 0.6.
        let q = DMatrix::identity(3, 3);
        let c = dvector![3.0, 0.0, 0.0];
        let (e, b) = budget(3);
        let bounds = Bounds::new(dvector![0.0, 0.3, 0.1], dvector![1.0, 1.0, 0.1]);
        let x = solve_bounded(&q, &c, &e, &b, &bounds, dvector![0.3, 0.6, 0.1]).unwrap();
        assert_close(&x, &dvector![0.6, 0.3, 0.1], 1e-15);
    }

    #[test]
    fn linear_objective_fills_caps_in_order() {
        // With Q = 0 the optimum puts as much as the caps allow on the largest cᵢ first.
        let q = DMatrix::zeros(4, 4);
        let c = dvector![0.1, 0.4, -0.2, 0.3];
        let (e, b) = budget(4);
        let bounds = capped(DVector::from_element(4, 0.4));
        let x = solve_bounded(&q, &c, &e, &b, &bounds, DVector::from_element(4, 0.25)).unwrap();
        assert_close(&x, &dvector![0.2, 0.4, 0.0, 0.4], 1e-15);
    }

    #[test]
    fn a_single_feasible_point_is_returned() {
        let q = DMatrix::identity(2, 2);
        let (e, b) = budget(2);
        let bounds = capped(dvector![0.25, 0.75]);
        let x = solve_bounded(
            &q,
            &DVector::zeros(2),
            &e,
            &b,
            &bounds,
            dvector![0.25, 0.75],
        )
        .unwrap();
        assert_eq!(x, dvector![0.25, 0.75]);
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

    /// Two variables summing to 1 with a diagonal `Q` and both kinks at `x₀ = a`: the
    /// objective in `t = x₀` is a parabola plus `(κ₀ + κ₁)|t − a|`, minimized in closed form.
    fn two_kinked(q: [f64; 2], c: [f64; 2], a: f64, kappa: [f64; 2]) -> f64 {
        let curvature = q[0] + q[1];
        let smooth = (c[0] - c[1] + q[1]) / curvature;
        let total = kappa[0] + kappa[1];
        if smooth > a + total / curvature {
            smooth - total / curvature
        } else if smooth < a - total / curvature {
            smooth + total / curvature
        } else {
            a
        }
    }

    #[test]
    fn kinks_match_the_closed_form() {
        let (e, b) = budget(2);
        let q = dmatrix![2.0, 0.0; 0.0, 1.0];
        for (c, a, kappa) in [
            ([0.9, 0.1], 0.3, [0.1, 0.05]),
            ([0.9, 0.1], 0.3, [0.5, 0.5]),
            ([-0.4, 0.3], 0.6, [0.02, 0.0]),
            ([0.2, 0.2], 0.5, [0.0, 0.0]),
        ] {
            let kinks = Kinks {
                at: dvector![a, 1.0 - a],
                slopes: DVector::from_row_slice(&kappa),
            };
            let expected = two_kinked([2.0, 1.0], c, a, kappa);
            for start in [dvector![0.5, 0.5], dvector![0.9, 0.1], dvector![a, 1.0 - a]] {
                let x = solve_piecewise(
                    &q,
                    &DVector::from_row_slice(&c),
                    &e,
                    &b,
                    &Bounds::nonnegative(2),
                    &kinks,
                    start,
                )
                .unwrap();
                assert_close(&x, &dvector![expected, 1.0 - expected], 1e-14);
            }
        }
    }

    #[test]
    fn zero_slopes_change_nothing() {
        let q = dmatrix![2.0, 0.5, 0.0; 0.5, 1.0, 0.2; 0.0, 0.2, 3.0];
        let c = dvector![0.4, 0.2, -0.5];
        let (e, b) = budget(3);
        let bounds = Bounds::nonnegative(3);
        let start = DVector::from_element(3, 1.0 / 3.0);
        let kinks = Kinks {
            at: dvector![0.2, 0.3, 0.5],
            slopes: DVector::zeros(3),
        };
        assert_eq!(
            solve_piecewise(&q, &c, &e, &b, &bounds, &kinks, start.clone()).unwrap(),
            solve_bounded(&q, &c, &e, &b, &bounds, start).unwrap()
        );
    }

    #[test]
    fn large_slopes_hold_every_variable_at_its_kink() {
        let q = dmatrix![2.0, 0.5, 0.0; 0.5, 1.0, 0.2; 0.0, 0.2, 3.0];
        let c = dvector![0.4, 0.2, -0.5];
        let (e, b) = budget(3);
        let at = dvector![0.2, 0.3, 0.5];
        let kinks = Kinks {
            at: at.clone(),
            slopes: DVector::from_element(3, 10.0),
        };
        for start in [DVector::from_element(3, 1.0 / 3.0), at.clone()] {
            let x =
                solve_piecewise(&q, &c, &e, &b, &Bounds::nonnegative(3), &kinks, start).unwrap();
            assert_close(&x, &at, 1e-15);
        }
    }

    #[test]
    fn kinks_work_without_bounds() {
        // Long-short: x₀ wants to go below 0.
        let (e, b) = budget(2);
        let q = dmatrix![1.0, 0.0; 0.0, 1.0];
        let c = dvector![-2.0, 1.0];
        let free = Bounds::new(
            DVector::from_element(2, f64::NEG_INFINITY),
            DVector::from_element(2, f64::INFINITY),
        );
        let kinks = Kinks {
            at: dvector![0.5, 0.5],
            slopes: dvector![0.25, 0.25],
        };
        let x = solve_piecewise(&q, &c, &e, &b, &free, &kinks, dvector![0.5, 0.5]).unwrap();
        let expected = two_kinked([1.0, 1.0], [-2.0, 1.0], 0.5, [0.25, 0.25]);
        assert!(expected < 0.0);
        assert_close(&x, &dvector![expected, 1.0 - expected], 1e-14);
    }

    #[test]
    fn a_kink_outside_the_bounds_is_one_sided() {
        // Holding 0.8 of x₀, capped at 0.5: x₀ can only fall, and selling it costs κ, so the
        // cost only pushes weight toward the cap.
        let (e, b) = budget(2);
        let q = dmatrix![1.0, 0.0; 0.0, 1.0];
        let bounds = capped(dvector![0.5, 1.0]);
        let kinks = Kinks {
            at: dvector![0.8, 0.2],
            slopes: dvector![0.1, 0.0],
        };
        let x = solve_piecewise(
            &q,
            &DVector::zeros(2),
            &e,
            &b,
            &bounds,
            &kinks,
            dvector![0.3, 0.7],
        )
        .unwrap();
        assert_close(&x, &dvector![0.5, 0.5], 1e-15);
    }

    #[test]
    fn a_binding_turnover_cap_is_met_exactly() {
        let q = dmatrix![2.0, 0.5, 0.0; 0.5, 1.0, 0.2; 0.0, 0.2, 3.0];
        let c = dvector![0.4, 0.2, -0.5];
        let (e, b) = budget(3);
        let bounds = Bounds::nonnegative(3);
        let at = dvector![0.1, 0.1, 0.8];
        let kinks = Kinks {
            at: at.clone(),
            slopes: DVector::zeros(3),
        };
        let start = DVector::from_element(3, 1.0 / 3.0);
        let free = solve_bounded(&q, &c, &e, &b, &bounds, start.clone()).unwrap();
        let unconstrained = (&free - &at).abs().sum();
        let cap = unconstrained / 2.0;
        let x =
            solve_turnover_capped(&q, &c, &e, &b, &bounds, (&kinks, cap), start.clone()).unwrap();
        assert!(((&x - &at).abs().sum() - cap).abs() < 1e-14);
        // A loose cap changes nothing.
        let loose = solve_turnover_capped(&q, &c, &e, &b, &bounds, (&kinks, 2.0), start).unwrap();
        assert_eq!(loose, free);
    }

    #[test]
    fn a_turnover_cap_is_a_cost_at_its_multiplier() {
        // On two variables the capped optimum moves t = x₀ from a by exactly cap / 2, and is
        // the closed form with the costs that stop it there.
        let (e, b) = budget(2);
        let q = dmatrix![2.0, 0.0; 0.0, 1.0];
        let c = dvector![0.9, 0.1];
        let kinks = Kinks {
            at: dvector![0.3, 0.7],
            slopes: dvector![0.0, 0.0],
        };
        let x = solve_turnover_capped(
            &q,
            &c,
            &e,
            &b,
            &Bounds::nonnegative(2),
            (&kinks, 0.2),
            dvector![0.5, 0.5],
        )
        .unwrap();
        assert!(two_kinked([2.0, 1.0], [0.9, 0.1], 0.3, [0.0, 0.0]) > 0.4);
        assert_close(&x, &dvector![0.4, 0.6], 1e-14);
    }

    #[test]
    fn the_multiplier_search_lands_on_a_linear_limit() {
        // g(λ) = 10 − λ, limit 3: doubling from 1 brackets λ = 7 between 4 and 8, where g is
        // 6 and 2, and the secant step lands on it.
        let mut tried = Vec::new();
        let solve = |lambda: f64, _: DVector<f64>| {
            tried.push(lambda);
            Ok((10.0 - lambda, Some(dvector![lambda])))
        };
        let x = search_multiplier(solve, (3.0, |_: &DVector<f64>| 0.0), 1.0, dvector![0.0]);
        assert_eq!(x, Ok(dvector![7.0]));
        assert_eq!(tried, [0.0, 1.0, 2.0, 4.0, 8.0, 7.0]);
    }

    #[test]
    fn the_multiplier_search_accepts_zero_within_its_slack() {
        // At λ = 0, g is past the limit only by less than the slack, so that's the answer.
        let solve = |lambda: f64, _: DVector<f64>| Ok((3.0005 - lambda, Some(dvector![lambda])));
        let x = search_multiplier(solve, (3.0, |_: &DVector<f64>| 1e-3), 1.0, dvector![0.0]);
        assert_eq!(x, Ok(dvector![0.0]));
    }

    #[test]
    fn an_objective_unbounded_past_zero_stays_unbounded() {
        // Any other failure past λ = 0 is the arithmetic breaking down, but an objective with
        // no minimum is reported as such.
        let solve = |lambda: f64, _: DVector<f64>| {
            if lambda == 0.0 {
                Ok((5.0, Some(dvector![0.0])))
            } else {
                Err(CutupError::Unbounded)
            }
        };
        let x = search_multiplier(solve, (3.0, |_: &DVector<f64>| 0.0), 1.0, dvector![0.0]);
        assert_eq!(x, Err(CutupError::Unbounded));
    }

    #[test]
    fn the_turnover_search_starts_where_every_variable_can_stay_at_its_kink() {
        // Qa − c = [2 − 1, 2 − 0], so the largest pull is 2.
        let kinks = Kinks {
            at: dvector![1.0, 2.0],
            slopes: DVector::zeros(2),
        };
        let q = dmatrix![2.0, 0.0; 0.0, 1.0];
        assert_eq!(kink_multiplier(&q, &dvector![1.0, 0.0], &kinks), 2.0);
        // With no pull at all, it starts at 1.
        assert_eq!(kink_multiplier(&q, &dvector![2.0, 2.0], &kinks), 1.0);
    }

    #[test]
    fn a_sum_rounds_by_a_few_ulps_of_its_terms() {
        assert_eq!(sum_rounding(3, 2.0), 24.0 * f64::EPSILON);
    }

    /// `AAᵀ + I` for a fixed `A` with entries from −2 to 2.
    fn positive_definite_matrix(n: usize) -> DMatrix<f64> {
        let a = DMatrix::from_fn(n, n, |i, j| ((7 * i + 3 * j) % 5) as f64 - 2.0);
        &a * a.transpose() + DMatrix::identity(n, n)
    }

    /// `n` variables, the first two the same, so a block with both is singular, with a second
    /// pivot of exactly 0.
    fn repeated_variable(n: usize) -> DMatrix<f64> {
        let mut q = DMatrix::identity(n, n);
        q[(0, 1)] = 1.0;
        q[(1, 0)] = 1.0;
        q
    }

    #[test]
    fn a_factor_updated_a_variable_at_a_time_matches_a_fresh_one() {
        let q = positive_definite_matrix(12);
        let all: Vec<usize> = (0..12).collect();
        let mut factoring = Factoring::new();
        // Variables leaving, joining, both at once, and two joining a block of 12: a sixth, as
        // many as an update takes.
        for indices in [
            all.clone(),
            vec![0, 2, 3, 5, 6, 7, 8, 9, 10, 11],
            vec![0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11],
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            all.clone(),
            (0..10).collect(),
            all.clone(),
        ] {
            let block = q.select_rows(&indices).select_columns(&indices);
            let l = factoring.update(&q, &indices).unwrap().l();
            assert!((&l * l.transpose() - block).amax() < 1e-12, "{indices:?}");
        }
        assert_eq!(factoring.factorizations, 1);
        // Eleven joining one is factored afresh.
        factoring.update(&q, &[0]).unwrap();
        factoring.update(&q, &all).unwrap();
        assert_eq!(factoring.factorizations, 2);
    }

    #[test]
    fn a_block_is_factored_down_to_the_pivot_floor() {
        // The floor is a power of 2, so its square root squares back to it exactly.
        for (pivot, factored) in [(PIVOT_FLOOR, true), (PIVOT_FLOOR / 2.0, false)] {
            let mut q = DMatrix::identity(7, 7);
            q[(6, 6)] = pivot;
            let all: Vec<usize> = (0..7).collect();
            assert_eq!(Factoring::new().update(&q, &all).is_some(), factored);
            // And inserted into the factor of the others.
            let mut factoring = Factoring::new();
            factoring.update(&q, &all[..6]).unwrap();
            assert_eq!(factoring.update(&q, &all).is_some(), factored);
        }
    }

    #[test]
    fn an_insertion_that_makes_the_block_singular_stops_there() {
        // Inserting the third variable after the second would solve with the second's zero
        // pivot, which panics. The update is retried afresh, which finds the block singular
        // too.
        let q = repeated_variable(13);
        let mut factoring = Factoring::new();
        let others: Vec<usize> = [0].into_iter().chain(3..13).collect();
        factoring.update(&q, &others).unwrap();
        let all: Vec<usize> = (0..13).collect();
        assert!(factoring.update(&q, &all).is_none());
        assert_eq!(factoring.factorizations, 2);
    }

    #[test]
    fn a_block_containing_a_singular_one_is_not_factored_again() {
        let q = repeated_variable(3);
        let mut factoring = Factoring::new();
        assert!(factoring.update(&q, &[0, 1, 2]).is_none());
        assert!(factoring.update(&q, &[0, 1, 2]).is_none());
        assert_eq!(factoring.factorizations, 1);
        // Without the second variable it's positive definite, and factored afresh.
        assert!(factoring.update(&q, &[0, 2]).is_some());
        assert_eq!(factoring.factorizations, 2);
        // With it again, too many join to update, and the block contains the singular one. A
        // block without the first variable doesn't.
        assert!(factoring.update(&q, &[0, 1, 2]).is_none());
        assert_eq!(factoring.factorizations, 2);
        assert!(factoring.update(&q, &[1, 2]).is_some());
        assert_eq!(factoring.factorizations, 3);
    }

    #[test]
    fn a_reset_factor_is_computed_afresh() {
        let q = positive_definite_matrix(3);
        let mut factoring = Factoring::new();
        factoring.update(&q, &[0, 1, 2]).unwrap();
        factoring.update(&q, &[0, 1, 2]).unwrap();
        assert_eq!(factoring.factorizations, 1);
        factoring.reset();
        factoring.update(&q, &[0, 1, 2]).unwrap();
        assert_eq!(factoring.factorizations, 2);
    }

    #[test]
    fn the_range_space_solve_matches_the_null_space_one() {
        // A budget and a target return on means 5e-7 apart: close to parallel, so WᵀW loses
        // digits that the refinement brings back.
        let q = positive_definite_matrix(3);
        let c = dvector![0.3, -0.2, 0.1];
        let mu = dvector![0.02, 0.02 - 5e-7, 0.019];
        let e = DMatrix::from_columns(&[DVector::from_element(3, 1.0), mu]);
        let b = dvector![1.0, 0.0199];
        let cholesky = Cholesky::new(q.clone()).unwrap();
        let (x, multipliers) = solve_factored(&cholesky, (&q, &c, &e, &b), 0.0).unwrap();
        let (expected, expected_multipliers) =
            solve_noisy_kkt(&q, &c, &e, &b, 0.0).unwrap().unwrap();
        assert_close(&x, &expected, 1e-12);
        let size = expected_multipliers.amax();
        assert_close(&multipliers, &expected_multipliers, 1e-12 * size);
        // A few ulps of the budget, not under one: below 1 the spacing is 1.1e-16, so a
        // bound of 1e-16 only passes on a sum that rounds to exactly 1.
        assert_close(&(e.transpose() * &x), &b, 4.0 * f64::EPSILON);
    }

    #[test]
    fn nearly_dependent_constraints_are_left_to_the_null_space_solve() {
        // At unit length the columns' WᵀW has a second pivot of about 2⁻³⁰: positive, so it
        // factors, but below the floor.
        let q = DMatrix::identity(2, 2);
        let e = DMatrix::from_columns(&[dvector![1.0, 1.0], dvector![1.0, 1.0 + 1.0 / 16384.0]]);
        let b = e.transpose() * dvector![1.0, 0.0];
        let cholesky = Cholesky::new(q.clone()).unwrap();
        assert_eq!(
            solve_factored(&cholesky, (&q, &dvector![0.0, 0.0], &e, &b), 0.0),
            None
        );
    }

    /// `bounds` with the rows `lower ≤ aᵀx ≤ upper`.
    fn with_rows(
        bounds: Bounds,
        a: DMatrix<f64>,
        lower: DVector<f64>,
        upper: DVector<f64>,
    ) -> Bounds {
        Bounds {
            rows: Rows { a, lower, upper },
            ..bounds
        }
    }

    #[test]
    fn a_row_caps_a_group_of_variables() {
        // Alone, ½‖x‖² − x₀ − x₁ with x ≥ 0 splits the budget between x₀ and x₁. Capping their
        // sum at 0.5 leaves the rest to x₂: x = c − ν1 − μ(1, 1, 0) with ν = −0.5 and μ = 1.25.
        let q = DMatrix::identity(3, 3);
        let c = dvector![1.0, 1.0, 0.0];
        let (e, b) = budget(3);
        let alone = solve_nonnegative(&q, &c, &e, &b, DVector::from_element(3, 1.0 / 3.0)).unwrap();
        assert_close(&alone, &dvector![0.5, 0.5, 0.0], 1e-15);
        let bounds = with_rows(
            Bounds::nonnegative(3),
            dmatrix![1.0; 1.0; 0.0],
            dvector![0.0],
            dvector![0.5],
        );
        let x = solve_bounded(&q, &c, &e, &b, &bounds, dvector![0.2, 0.2, 0.6]).unwrap();
        assert_close(&x, &dvector![0.25, 0.25, 0.5], 1e-15);
    }

    #[test]
    fn a_row_puts_a_floor_under_a_group_of_variables() {
        // Alone, ½‖x‖² − x₀ puts everything in x₀. A floor of 0.8 on x₁ + x₂ leaves it 0.2:
        // x = c − ν1 + μ(0, 1, 1) with ν = 0.8 and μ = 1.2.
        let q = DMatrix::identity(3, 3);
        let c = dvector![1.0, 0.0, 0.0];
        let (e, b) = budget(3);
        let bounds = with_rows(
            Bounds::nonnegative(3),
            dmatrix![0.0; 1.0; 1.0],
            dvector![0.8],
            dvector![f64::INFINITY],
        );
        let x = solve_bounded(&q, &c, &e, &b, &bounds, dvector![0.0, 0.5, 0.5]).unwrap();
        assert_close(&x, &dvector![0.2, 0.4, 0.4], 1e-15);
    }

    #[test]
    fn a_row_that_doesnt_bind_changes_nothing() {
        let q = positive_definite_matrix(5);
        let c = dvector![0.3, -0.2, 0.1, 0.4, 0.0];
        let (e, b) = budget(5);
        let start = DVector::from_element(5, 0.2);
        let alone = solve_nonnegative(&q, &c, &e, &b, start.clone()).unwrap();
        let bounds = with_rows(
            Bounds::nonnegative(5),
            dmatrix![1.0; 1.0; 0.0; 0.0; 1.0],
            dvector![0.0],
            dvector![1.0],
        );
        let x = solve_bounded(&q, &c, &e, &b, &bounds, start).unwrap();
        assert_close(&x, &alone, 1e-15);
    }

    #[test]
    fn a_row_with_equal_limits_holds_its_sum() {
        // x₀ + x₁ = 0.7 exactly, as a second budget would.
        let q = DMatrix::identity(3, 3);
        let c = dvector![0.0, 0.0, 1.0];
        let (e, b) = budget(3);
        let bounds = with_rows(
            Bounds::nonnegative(3),
            dmatrix![1.0; 1.0; 0.0],
            dvector![0.7],
            dvector![0.7],
        );
        let x = solve_bounded(&q, &c, &e, &b, &bounds, dvector![0.35, 0.35, 0.3]).unwrap();
        assert_close(&x, &dvector![0.35, 0.35, 0.3], 1e-15);
    }

    #[test]
    fn a_held_row_on_held_variables_changes_nothing() {
        // x₂ and x₃ start at 0, holding their row at its floor too. Alone on x₀ and x₁,
        // ½‖x‖² − x₀ − 0.5x₁ splits the budget as x = c − ν1 with ν = 0.25.
        let q = DMatrix::identity(4, 4);
        let c = dvector![1.0, 0.5, -1.0, -1.0];
        let (e, b) = budget(4);
        let bounds = with_rows(
            Bounds::nonnegative(4),
            dmatrix![0.0; 0.0; 1.0; 1.0],
            dvector![0.0],
            dvector![0.5],
        );
        let x = solve_bounded(&q, &c, &e, &b, &bounds, dvector![0.5, 0.5, 0.0, 0.0]).unwrap();
        assert_close(&x, &dvector![0.75, 0.25, 0.0, 0.0], 1e-15);
    }

    /// The problem with each row's sum as a variable of its own, built here rather than by
    /// [`with_slacks`], solved without rows: Q is singular there, so it goes through the
    /// eigendecomposition rather than the factor.
    fn solve_with_explicit_slacks(
        (q, c, e, b): (&DMatrix<f64>, &DVector<f64>, &DMatrix<f64>, &DVector<f64>),
        bounds: &Bounds,
        start: &DVector<f64>,
    ) -> Result<DVector<f64>, CutupError> {
        let (n, r) = bounds.rows.a.shape();
        let m = e.ncols();
        let q_all = DMatrix::from_fn(
            n + r,
            n + r,
            |i, j| if i < n && j < n { q[(i, j)] } else { 0.0 },
        );
        let c_all = DVector::from_fn(n + r, |i, _| if i < n { c[i] } else { 0.0 });
        let e_all = DMatrix::from_fn(n + r, m + r, |i, j| match (i < n, j < m) {
            (true, true) => e[(i, j)],
            (true, false) => bounds.rows.a[(i, j - m)],
            (false, true) => 0.0,
            (false, false) => {
                if i - n == j - m {
                    -1.0
                } else {
                    0.0
                }
            }
        });
        let b_all = DVector::from_fn(m + r, |j, _| if j < m { b[j] } else { 0.0 });
        let sums = bounds.rows.a.tr_mul(start);
        let all = |x: &DVector<f64>, rows: &DVector<f64>| {
            DVector::from_iterator(n + r, x.iter().chain(rows.iter()).copied())
        };
        let explicit = Bounds::new(
            all(&bounds.lower, &bounds.rows.lower),
            all(&bounds.upper, &bounds.rows.upper),
        );
        let x = solve_bounded(&q_all, &c_all, &e_all, &b_all, &explicit, all(start, &sums))?;
        Ok(x.rows(0, n).into_owned())
    }

    /// A point within `bounds` and their rows with `Eᵀx = b`, from the simplex method's first
    /// phase, or `None` if there's none.
    fn feasible_point(e: &DMatrix<f64>, b: &DVector<f64>, bounds: &Bounds) -> Option<DVector<f64>> {
        let (n, r) = bounds.rows.a.shape();
        let m = e.ncols();
        let a = DMatrix::from_fn(m + r, n + r, |i, j| match (i < m, j < n) {
            (true, true) => e[(j, i)],
            (true, false) => 0.0,
            (false, true) => bounds.rows.a[(j, i - m)],
            (false, false) => {
                if i - m == j - n {
                    -1.0
                } else {
                    0.0
                }
            }
        });
        let rhs = DVector::from_fn(m + r, |i, _| if i < m { b[i] } else { 0.0 });
        let all = |x: &DVector<f64>, rows: &DVector<f64>| {
            DVector::from_iterator(n + r, x.iter().chain(rows.iter()).copied())
        };
        let lower = all(&bounds.lower, &bounds.rows.lower);
        let upper = all(&bounds.upper, &bounds.rows.upper);
        match crate::lp::solve(&DVector::zeros(n + r), &a, &rhs, &lower, &upper).unwrap() {
            crate::lp::Outcome::Optimal(solution) => Some(solution.x.rows(0, n).into_owned()),
            _ => None,
        }
    }

    #[test]
    fn rows_match_their_sums_as_variables_on_random_problems() {
        // SplitMix64, so the problems are fixed.
        let mut state = 0x5eed_u64;
        let mut uniform = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let (n, r) = (6, 3);
        let (e, b) = budget(n);
        let mut solved = 0;
        for case in 0..200 {
            let factors = DMatrix::from_fn(n, n, |_, _| 2.0 * uniform() - 1.0);
            // Every tenth case singular, with rank 3.
            let rank = if case % 10 == 0 { 3 } else { n };
            let factors = factors.columns(0, rank).into_owned();
            let q = &factors * factors.transpose();
            let c = DVector::from_fn(n, |_, _| 2.0 * uniform() - 1.0);
            let a = DMatrix::from_fn(n, r, |_, _| if uniform() < 0.5 { 1.0 } else { 0.0 });
            let lower = DVector::from_fn(r, |_, _| 0.3 * uniform());
            let upper = DVector::from_fn(r, |j, _| lower[j] + 0.1 + 0.6 * uniform());
            let bounds = with_rows(
                Bounds::new(DVector::zeros(n), DVector::from_element(n, 0.6)),
                a,
                lower,
                upper,
            );
            let Some(start) = feasible_point(&e, &b, &bounds) else {
                continue;
            };
            let problem = (&q, &c, &e, &b);
            let x = solve_bounded(&q, &c, &e, &b, &bounds, start.clone());
            let expected = solve_with_explicit_slacks(problem, &bounds, &start);
            let (x, expected) = match (x, expected) {
                (Ok(x), Ok(expected)) => (x, expected),
                (x, expected) => panic!("case {case}: {x:?} vs {expected:?}"),
            };
            let objective = |x: &DVector<f64>| 0.5 * x.dot(&(&q * x)) - c.dot(x);
            // A singular Q can have many optima, so the objectives are compared.
            assert!(
                (objective(&x) - objective(&expected)).abs() < 1e-12,
                "case {case}: {x:?} vs {expected:?}"
            );
            if rank == n {
                assert_close(&x, &expected, 1e-9);
            }
            let sums = bounds.rows.a.tr_mul(&x);
            for j in 0..r {
                assert!(
                    sums[j] >= bounds.rows.lower[j] - 1e-12,
                    "case {case}: {sums:?}"
                );
                assert!(
                    sums[j] <= bounds.rows.upper[j] + 1e-12,
                    "case {case}: {sums:?}"
                );
            }
            assert!((x.sum() - 1.0).abs() < 1e-12);
            solved += 1;
        }
        assert!(solved > 100, "only {solved} feasible problems");
    }

    #[test]
    fn a_slack_is_scaled_by_its_smallest_member_per_unit() {
        let scales = [0.1, 0.2, 0.3];
        // min(0.1 / 1, 0.3 / 2).
        assert_eq!(slack_scale(&scales, [1.0, 0.0, 2.0]), 0.1);
        assert_eq!(slack_scale(&scales, [0.0, -4.0, 0.0]), 0.05);
        assert_eq!(slack_scale(&scales, [0.0, 0.0, 0.0]), 1.0);
        // A scale that underflows to 0 can't be divided by.
        assert_eq!(slack_scale(&[1e-300], [1e300]), 1.0);
    }

    #[test]
    fn the_free_slacks_come_after_the_free_variables() {
        let indices = [0, 2, 3, 5];
        assert_eq!(split_slacks(&indices, 3), (&[0, 2][..], &[3, 5][..]));
        assert_eq!(split_slacks(&indices, 6), (&indices[..], &[][..]));
    }

    #[test]
    fn each_slack_gets_its_own_rows_scale_after_the_variables() {
        // Variables scaled 2 and 3, and two rows: x₀ + x₁ and 6x₁.
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 1.0, 6.0]);
        let mut q = DMatrix::zeros(4, 4);
        q[(0, 0)] = 4.0;
        q[(1, 1)] = 9.0;
        assert_eq!(
            scales_with_slacks(&q, &a),
            DVector::from_vec(vec![2.0, 3.0, 2.0, 0.5])
        );
    }

    #[test]
    fn a_solve_keeps_held_rows_on_the_variables_it_solves_for() {
        // Columns 0 and 1 are E's own, 2 to 4 slacks' rows: 2's slack is free, and 4 has no
        // entry on variables 0 and 1. Column 0 has none either, but E's own are kept.
        let e = dmatrix![
            0.0, 1.0, 1.0, 1.0, 0.0;
            0.0, 1.0, 0.0, 1.0, 0.0;
            1.0, 1.0, 1.0, 0.0, 1.0
        ];
        assert_eq!(kept_columns(&e, &[0, 1], &[(5, 2)], 2), vec![0, 1, 3]);
        assert_eq!(kept_columns(&e, &[0, 1, 2], &[], 2), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn a_free_slack_moves_to_keep_its_row() {
        // Variables 0 and 1 are solved for, and slack 2 measures x₀ + 2x₁ in column 1.
        let e = dmatrix![1.0, 1.0; 1.0, 2.0; 0.0, -1.0];
        let step = dvector![0.1, -0.3];
        let slacks = [(2, 1)];
        // −Δs = 0.5 − (0.1 − 0.6).
        let residuals = dvector![0.0, 0.5];
        assert_eq!(
            with_slack_steps(step.clone(), (&[0, 1], &slacks), &e, Some(&residuals)),
            dvector![0.1, -0.3, -1.0]
        );
        // A direction: −Δs = 0 − (0.1 − 0.6).
        assert_eq!(
            with_slack_steps(step.clone(), (&[0, 1], &slacks), &e, None),
            dvector![0.1, -0.3, -0.5]
        );
        assert_eq!(
            with_slack_steps(step.clone(), (&[0, 1], &[]), &e, None),
            step
        );
    }
}
