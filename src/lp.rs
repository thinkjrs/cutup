//! Linear programs with bounded variables, solved by a two-phase revised simplex method.

use crate::CutupError;
use nalgebra::{DMatrix, DVector, Dyn, LU};

/// An optimal solution: `x`, the multipliers `y` of the equality constraints, and the
/// objective `cᵀx`. At the optimum `c − Aᵀy` is the reduced cost of each variable: nonnegative
/// at a lower bound, nonpositive at an upper bound, and zero strictly between.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Solution {
    pub(crate) x: DVector<f64>,
    pub(crate) y: DVector<f64>,
    pub(crate) objective: f64,
}

/// How a linear program ended.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Outcome {
    Optimal(Solution),
    /// No `x` satisfies the constraints and the bounds.
    Infeasible,
    /// The objective decreases without limit.
    Unbounded,
}

/// Where a variable is: in the basis at some row, or out of it at a bound, or, if it has
/// none, at zero.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Status {
    Basic,
    Lower,
    Upper,
    Free,
}

/// Solves `minimize cᵀx subject to Ax = b and l ≤ x ≤ u`, where a bound may be infinite.
///
/// This is the revised simplex method for bounded variables (Chvátal, 1983). Variables
/// out of the basis sit at one of their bounds, so bounds need no constraint rows: an `m × n`
/// problem keeps an `m × m` basis however many variables are bounded. Phase 1 starts from
/// artificial variables carrying the residual of `Ax = b`, one per row, and minimizes their
/// sum; if that can't reach zero, the problem is infeasible. Phase 2 then fixes the
/// artificial variables at zero and minimizes `cᵀx`.
///
/// The basis is factored once and then updated in product form: each pivot appends an eta
/// matrix, so a solve costs `O(m²)` plus `O(m)` per eta instead of a fresh `O(m³)`
/// factorization. The basis is factored afresh once [`REFACTOR`] eta matrices have built up,
/// or sooner after a pivot on an entry small enough to amplify rounding, and once more to
/// confirm an optimum or an unbounded direction, so either verdict comes from fresh factors.
/// Each iteration still recomputes the basic variables from the nonbasic ones, so rounding in
/// `x` doesn't accumulate.
///
/// The entering variable is the one whose reduced cost improves the objective fastest
/// (Dantzig's rule). On a degenerate vertex, where basic variables sit at their bounds, pivots
/// can leave `x` where it was, and that rule can cycle. So after a few such pivots in a row,
/// the bounds of the basic variables are widened, each by a tiny amount of its own, so they no
/// longer sit at them, and the solve carries on within the widened bounds. At the optimum
/// they're restored. That moves the basic variables, and any
/// left beyond a bound are brought back by dual simplex pivots, which keep the basis optimal.
/// If pivots still stall once every basic variable's bounds are widened, or after they're
/// restored, the lowest-indexed improving variable enters instead, with ties in the ratio test
/// going to the lowest index: Bland's rule, which can't cycle. It holds until `x` moves again.
///
/// Nonbasic variables start at their lower bound if it's finite, or else their upper bound,
/// or zero. [`solve_from`] can start some at their upper bound instead.
///
/// # Errors
///
/// [`CutupError::NotConverged`] if a basis can't be factored or the iteration cap is reached,
/// or if restoring the bounds leaves a basic variable beyond one that no pivot can bring back.
/// Only rounding trouble should cause either.
pub(crate) fn solve(
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
) -> Result<Outcome, CutupError> {
    solve_from(c, a, b, lower, upper, &[])
}

/// [`solve`], with the variables in `at_upper` starting at their upper bound where it's
/// finite.
///
/// The start matters on rows where `b` is zero. If it leaves such rows with no residual,
/// their artificial variables start in the basis at zero, a degenerate vertex. Widening the
/// bounds gets the solve off it, but a start that leaves every row a residual avoids it, and
/// can be faster.
///
/// # Errors
///
/// As for [`solve`].
pub(crate) fn solve_from(
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
    at_upper: &[usize],
) -> Result<Outcome, CutupError> {
    Ok(solve_perturbing(c, a, b, lower, upper, at_upper, PERTURBATION)?.0)
}

/// [`solve_from`], widening bounds on a degenerate vertex by `perturbation` rather than
/// [`PERTURBATION`], so that a test can widen them enough that restoring them breaks some.
fn solve_perturbing(
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
    at_upper: &[usize],
    perturbation: f64,
) -> Result<(Outcome, Option<Basis>), CutupError> {
    let (m, n) = a.shape();
    if crossed(lower, upper) {
        return Ok((Outcome::Infeasible, None));
    }
    if m == 0 {
        return Ok((unconstrained(c, lower, upper), None));
    }
    let (a, b, scales) = scaled(a, b);
    let (a, b) = (&a, &b);
    // Nonbasic variables start at a finite bound, the upper one if asked and it's finite, or at
    // zero if they have none. One rule gives both the value and the status, so they agree.
    let mut asked_upper = vec![false; n];
    for &j in at_upper {
        asked_upper[j] = true;
    }
    let (start, start_status): (Vec<f64>, Vec<Status>) = (0..n)
        .map(|j| match (lower[j].is_finite(), upper[j].is_finite()) {
            (_, true) if asked_upper[j] => (upper[j], Status::Upper),
            (true, _) => (lower[j], Status::Lower),
            (false, true) => (upper[j], Status::Upper),
            (false, false) => (0.0, Status::Free),
        })
        .unzip();
    let start = DVector::from_vec(start);
    let residual = b - a * &start;
    // Artificial variable i carries row i's residual, with the sign that makes it nonnegative.
    let mut columns = a.clone().resize_horizontally(n + m, 0.0);
    for i in 0..m {
        columns[(i, n + i)] = if residual[i] < 0.0 { -1.0 } else { 1.0 };
    }
    let mut problem = Problem {
        a_abs: columns.abs(),
        a: columns,
        b: b.clone(),
        lower: lower.clone().resize_vertically(n + m, 0.0),
        upper: upper.clone().resize_vertically(n + m, f64::INFINITY),
        x: start.resize_vertically(n + m, 0.0),
        basis: (n..n + m).collect(),
        status: start_status
            .into_iter()
            .chain(std::iter::repeat_n(Status::Basic, m))
            .collect(),
        perturbation,
        repairs: usize::MAX,
    };
    for i in 0..m {
        problem.x[n + i] = residual[i].abs();
    }

    let phase_one = DVector::from_fn(n + m, |j, _| if j >= n { 1.0 } else { 0.0 });
    // Phase 1 is bounded below by zero, and the ratio test never lets a variable past a
    // finite bound, so it always ends at an optimum: running off without limit would take
    // an artificial variable below zero.
    let Some(_) = problem.minimize(&phase_one, Perturbation::Off)? else {
        return Err(CutupError::NotConverged);
    };
    let infeasibility: f64 = problem.x.rows(n, m).sum();
    let scale = a.amax() * problem.x.rows(0, n).amax() + b.amax();
    if infeasibility > 1e-9 * scale.max(f64::MIN_POSITIVE) {
        return Ok((Outcome::Infeasible, None));
    }
    problem.drive_out_artificials(n)?;
    for j in n..n + m {
        problem.upper[j] = 0.0;
        if problem.status[j] != Status::Basic {
            problem.x[j] = 0.0;
        }
    }

    let cost = c.clone().resize_vertically(n + m, 0.0);
    problem.finish(c, &cost, Perturbation::Off, &scales)
}

/// [`solve`], from `warm`, an earlier optimum's [`Basis`] for this problem, if it can start
/// there, returning the basis it ends at, if it can be a later start.
///
/// The warm start fits a problem grown from an earlier one by rows, each with a variable of
/// its own that's basic in it, such as a slack for a new inequality: see [`Basis::add_row`].
/// The costs and the earlier rows haven't changed, so the earlier basis is still optimal for
/// them, if not feasible: the new rows' variables can start beyond their bounds. Dual simplex
/// pivots bring them back, keeping it optimal, as after restoring perturbed bounds, and then
/// the primal method carries on from there. Far fewer pivots than from scratch, which starts
/// from a basis of artificial variables.
///
/// Anything that goes wrong from the warm start, such as a basis that can't be factored, a
/// variable nothing can bring back, or a repair that takes more than [`repair_cap`] dual
/// pivots, falls back to solving from scratch. A warm optimum meets
/// the same tolerances as a cold one after its bounds are restored: a basic variable can end
/// up to `1e-9` of `max(1, |x|)` beyond one of its bounds, where a cold solve might land on
/// another vertex.
///
/// # Errors
///
/// As for [`solve`].
pub(crate) fn solve_warm(
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
    warm: Option<&Basis>,
) -> Result<(Outcome, Option<Basis>), CutupError> {
    // Nested rather than a let chain, which takes Rust 1.88.
    if let Some(warm) = warm {
        if let Some(solved) = from_basis(c, a, b, lower, upper, warm, repair_cap(a.nrows())) {
            return Ok(solved);
        }
    }
    solve_perturbing(c, a, b, lower, upper, &[], PERTURBATION)
}

/// [`solve_warm`] from `warm`, or `None` if it can't start or finish there, or if repairing the
/// basis takes more than `repairs` dual pivots.
fn from_basis(
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
    warm: &Basis,
    repairs: usize,
) -> Option<(Outcome, Option<Basis>)> {
    let (m, n) = a.shape();
    if m == 0 || warm.basis.len() != m || warm.status.len() != n || crossed(lower, upper) {
        return None;
    }
    // Each row's basic variable distinct, in range and marked basic, and no other marked so:
    // otherwise one marked basic outside the basis would be held at zero.
    let mut seen = vec![false; n];
    for &j in &warm.basis {
        if j >= n || seen[j] || warm.status[j] != Status::Basic {
            return None;
        }
        seen[j] = true;
    }
    if warm.status.iter().filter(|&&s| s == Status::Basic).count() != m {
        return None;
    }
    // Each nonbasic variable at the bound its status says, which must be finite.
    let mut x = DVector::zeros(n);
    for j in 0..n {
        x[j] = match warm.status[j] {
            Status::Lower if lower[j].is_finite() => lower[j],
            Status::Upper if upper[j].is_finite() => upper[j],
            Status::Free if !lower[j].is_finite() && !upper[j].is_finite() => 0.0,
            Status::Basic => 0.0,
            _ => return None,
        };
    }
    let (a, b, scales) = scaled(a, b);
    let mut problem = Problem {
        a_abs: a.abs(),
        a,
        b,
        lower: lower.clone(),
        upper: upper.clone(),
        x,
        basis: warm.basis.clone(),
        status: warm.status.clone(),
        perturbation: PERTURBATION,
        repairs,
    };
    problem.finish(c, c, Perturbation::Repairing, &scales).ok()
}

/// Each row of `a` and `b` divided by the row's largest entry in `a`, and those entries, so
/// that the tolerances compare like with like whatever A's scale: row i of Ax = b becomes
/// (aᵢ/sᵢ)x = bᵢ/sᵢ, and its multiplier is divided by sᵢ to undo it. An all-zero row keeps
/// scale 1.
fn scaled(a: &DMatrix<f64>, b: &DVector<f64>) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let (m, n) = a.shape();
    let scales = DVector::from_fn(m, |i, _| match a.row(i).amax() {
        s if s > 0.0 => s,
        _ => 1.0,
    });
    let a = DMatrix::from_fn(m, n, |i, j| a[(i, j)] / scales[i]);
    let b = b.component_div(&scales);
    (a, b, scales)
}

/// Where a solve ended: each row's basic variable and every variable's status, to start a later
/// solve from with [`solve_warm`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Basis {
    basis: Vec<usize>,
    status: Vec<Status>,
}

impl Basis {
    /// The basis for the problem with one more row, last, and one more variable, last, that's
    /// basic in it: such as a slack for a new inequality.
    pub(crate) fn add_row(&mut self) {
        self.basis.push(self.status.len());
        self.status.push(Status::Basic);
    }
}

/// Whether some variable's lower bound is above its upper bound, or either is NaN, so that no
/// `x` is within them. The simplex method would otherwise hold such a variable at one bound,
/// past the other, and with no rows `f64::clamp` would panic on it.
fn crossed(lower: &DVector<f64>, upper: &DVector<f64>) -> bool {
    lower
        .iter()
        .zip(upper.iter())
        .any(|(l, u)| l.is_nan() || u.is_nan() || l > u)
}

/// The optimum with no constraints but the bounds: each variable at the bound its cost
/// favors, or as close to zero as its bounds allow at zero cost. Unbounded if a favored bound
/// is infinite.
fn unconstrained(c: &DVector<f64>, lower: &DVector<f64>, upper: &DVector<f64>) -> Outcome {
    let x = DVector::from_fn(c.len(), |j, _| match c[j] {
        cost if cost > 0.0 => lower[j],
        cost if cost < 0.0 => upper[j],
        _ => 0.0f64.clamp(lower[j], upper[j]),
    });
    if x.iter().any(|v| !v.is_finite()) {
        return Outcome::Unbounded;
    }
    Outcome::Optimal(Solution {
        objective: c.dot(&x),
        x,
        y: DVector::zeros(0),
    })
}

/// How many pivots in a row may leave `x` where it was before the bounds are perturbed, or,
/// once they can't be, Bland's rule takes over.
const STALL: usize = 10;

/// How far [`Problem::perturb`] widens a bound `l`, relative to `max(1, |l|)`: between this and
/// twice it, different for each bound. Far above rounding, so that the steps it allows count
/// as moving `x`, and small enough that restoring the bounds seldom leaves a basic variable
/// beyond one.
const PERTURBATION: f64 = 1e-7;

/// Where [`Problem::minimize`] is with perturbing the bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Perturbation {
    /// The bounds are the problem's own.
    Off,
    /// Some bounds are widened.
    On,
    /// The bounds were widened and are restored, and dual pivots repair any basic variable
    /// left beyond its own.
    Repairing,
    /// The bounds are restored, every basic variable is within them or within rounding of
    /// them, and they stay as they are.
    Done,
}

/// How many pivots [`Problem::minimize`] takes at most, for `m` rows and `total` variables.
fn iteration_cap(m: usize, total: usize) -> usize {
    1000 + 100 * (m + total)
}

/// How many dual pivots a warm start takes repairing its basis, for `m` rows, before it gives
/// up for a solve from scratch. The dual pivots have no Bland's rule, as the primal ones do, so
/// a dual-degenerate basis could cycle until the iteration cap. Measured on 0.3.12, no repair
/// took more than 7 pivots in 32,498, all `MinCdar`'s, warm and cold, in its unit and property
/// tests. Past this, the warm start has cost this many pivots rather than the iteration cap's.
/// A solve from scratch has no fallback, and no cap but the iteration cap, as before.
fn repair_cap(m: usize) -> usize {
    100 + 10 * m
}

/// How many eta matrices the basis's factors may carry before it's factored afresh.
const REFACTOR: usize = 32;

/// The basis `B` as the LU factors of an earlier basis `B₀` times eta matrices `E₁ ⋯ Eₖ`, so
/// `B = B₀E₁⋯Eₖ`. Each `Eᵢ` is the identity with column `r` replaced by `d`, the entering
/// column in the basis before it, `B⁻¹aⱼ`, with `r` the row it entered at.
struct Factors {
    lu: LU<f64, Dyn, Dyn>,
    /// `lu`'s `L` and `U`, unpacked once rather than copied out on every transposed solve.
    l: DMatrix<f64>,
    u: DMatrix<f64>,
    etas: Vec<(usize, DVector<f64>)>,
}

impl Factors {
    /// Factors the columns of `a` in `basis`.
    fn new(a: &DMatrix<f64>, basis: &[usize]) -> Self {
        let lu = a.select_columns(basis).lu();
        Factors {
            l: lu.l(),
            u: lu.u(),
            lu,
            etas: Vec::new(),
        }
    }

    /// The factors of `basis`, whose column at `row` was just replaced by one with `d = B⁻¹aⱼ`
    /// in the basis before: one more eta matrix, or a fresh factorization once [`REFACTOR`]
    /// have built up. An eta matrix divides by its pivot `dᵣ`, so one below `1e-6` of the
    /// column's largest entry would amplify rounding by over a million in every later solve,
    /// and the new basis is factored afresh then too.
    fn update(&mut self, a: &DMatrix<f64>, basis: &[usize], row: usize, d: DVector<f64>) {
        if self.etas.len() >= REFACTOR || d[row].abs() <= 1e-6 * d.amax() {
            *self = Factors::new(a, basis);
        } else {
            self.etas.push((row, d));
        }
    }

    /// `B⁻¹b`: `B₀⁻¹b`, then each `Eᵢ⁻¹` in turn.
    fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, CutupError> {
        let mut x = self.lu.solve(b).ok_or(CutupError::NotConverged)?;
        for (r, d) in &self.etas {
            let pivot = x[*r] / d[*r];
            x.axpy(-pivot, d, 1.0);
            x[*r] = pivot;
        }
        Ok(x)
    }

    /// `B⁻ᵀc`: each `Eᵢ⁻ᵀ` from the last, then `B₀⁻ᵀ`. `Eᵢᵀ` is the identity with row `r`
    /// replaced by `dᵀ`, so only entry `r` changes, to `(cᵣ − Σ_{i≠r} dᵢcᵢ) / dᵣ`.
    fn solve_transpose(&self, c: DVector<f64>) -> Result<DVector<f64>, CutupError> {
        let mut y = c;
        for (r, d) in self.etas.iter().rev() {
            let others = d.dot(&y) - d[*r] * y[*r];
            y[*r] = (y[*r] - others) / d[*r];
        }
        // PB₀ = LU, so B₀ᵀ = UᵀLᵀP, solved by two triangular solves and the inverse
        // permutation.
        if !(self.u.tr_solve_upper_triangular_mut(&mut y)
            && self.l.tr_solve_lower_triangular_mut(&mut y))
        {
            return Err(CutupError::NotConverged);
        }
        self.lu.p().inv_permute_rows(&mut y);
        Ok(y)
    }
}

/// A step at which a variable reaches a bound: its length, the variable, and, for a basic
/// variable, its row and the bound it reaches.
type Block = (f64, usize, Option<(usize, Status)>);

/// A linear program in the state the simplex method moves through.
struct Problem {
    a: DMatrix<f64>,
    /// `|A|`, entrywise, for the reduced costs' tolerances.
    a_abs: DMatrix<f64>,
    b: DVector<f64>,
    lower: DVector<f64>,
    upper: DVector<f64>,
    x: DVector<f64>,
    /// The basic variable of each row.
    basis: Vec<usize>,
    status: Vec<Status>,
    /// How far [`Problem::perturb`] widens a bound, as [`PERTURBATION`] describes.
    perturbation: f64,
    /// How many dual pivots [`Problem::minimize`] takes repairing the basis before it gives up
    /// with [`CutupError::NotConverged`]: [`repair_cap`] for a warm start, which then falls
    /// back to a solve from scratch, and no limit for that solve.
    repairs: usize,
}

impl Problem {
    /// The solution in the problem's first `n` variables, with the multipliers `y` of the rows
    /// as [`scaled`] scaled them, unscaled.
    fn solution(&self, c: &DVector<f64>, y: &DVector<f64>, scales: &DVector<f64>) -> Solution {
        let x = self.x.rows(0, c.len()).into_owned();
        Solution {
            objective: c.dot(&x),
            x,
            y: y.component_div(scales),
        }
    }

    /// Minimizes `cost` from `start`, as [`Problem::minimize`], and gives the outcome in the
    /// problem's own variables, `c`'s, with the basis it ends at if a later solve can start
    /// there.
    fn finish(
        &mut self,
        c: &DVector<f64>,
        cost: &DVector<f64>,
        start: Perturbation,
        scales: &DVector<f64>,
    ) -> Result<(Outcome, Option<Basis>), CutupError> {
        Ok(match self.minimize(cost, start)? {
            None => (Outcome::Unbounded, None),
            Some(y) => (
                Outcome::Optimal(self.solution(c, &y, scales)),
                self.basis_of(c.len()),
            ),
        })
    }

    /// The basis as a start for a later solve, if every basic variable is one of the problem's
    /// first `n`, rather than an artificial one, which a later solve wouldn't have.
    fn basis_of(&self, n: usize) -> Option<Basis> {
        self.basis.iter().all(|&j| j < n).then(|| Basis {
            basis: self.basis.clone(),
            status: self.status[..n].to_vec(),
        })
    }

    /// Pivots each artificial variable still basic after phase 1, the first `n` variables being
    /// the problem's own, out of the basis for one of those with an entry in its row of `B⁻¹A`.
    /// Phase 1 ended feasible, so the artificial is at zero to within its tolerance, and it
    /// leaves at zero: the pivot is made only where that moves the entering variable and every
    /// basic one by no more than rounding, so that it moves nothing. Left in, it would be basic and fixed at zero in phase 2, where a
    /// change in its row that's only rounding can be the one thing to block a step, and
    /// pivoting on it leaves the basis singular. Nor could a later solve start from a basis
    /// with it, as [`Problem::basis_of`] says. The entry largest in size enters, for the
    /// steadiest pivot. A row with no entry beyond rounding is a combination of the others, and
    /// keeps its artificial, which nothing then moves.
    fn drive_out_artificials(&mut self, n: usize) -> Result<(), CutupError> {
        // Most solves end phase 1 with none basic, and need no factors for it.
        if self.basis.iter().all(|&j| j < n) {
            return Ok(());
        }
        let mut factors = Factors::new(&self.a, &self.basis);
        for row in 0..self.basis.len() {
            if self.basis[row] < n {
                continue;
            }
            // The artificial's own entry is 1.
            let alpha = self.tableau_row(row, &factors)?;
            let pivot = 1e-9 * alpha.amax();
            let Some(entering) = (0..n)
                .filter(|&j| self.status[j] != Status::Basic && alpha[j].abs() > pivot)
                .max_by(|&p, &q| alpha[p].abs().total_cmp(&alpha[q].abs()).then(q.cmp(&p)))
            else {
                continue;
            };
            // Taking the artificial the rest of the way to zero moves the entering variable by
            // δ = x_k / d_r, and each basic variable by d_i·δ.
            let d = factors.solve(&self.a.column(entering).into_owned())?;
            let delta = self.x[self.basis[row]] / d[row];
            let rounding = |change: f64, x: f64| change.abs() <= 1e-12 * x.abs().max(1.0);
            let moves_nothing = rounding(delta, self.x[entering])
                && (0..d.len()).all(|i| rounding(d[i] * delta, self.x[self.basis[i]]));
            if moves_nothing {
                self.exchange(row, entering, (Status::Lower, 0.0), d, &mut factors);
            }
        }
        Ok(())
    }

    /// Row `row` of `B⁻¹A`, from `B⁻ᵀeᵣ`.
    fn tableau_row(&self, row: usize, factors: &Factors) -> Result<DVector<f64>, CutupError> {
        let unit = DVector::from_fn(self.basis.len(), |i, _| f64::from(i == row));
        Ok(self.a.tr_mul(&factors.solve_transpose(unit)?))
    }

    /// Brings `entering`, with `d = B⁻¹aⱼ`, into the basis at `row`, the variable it replaces
    /// leaving with `leaving`'s status and value.
    fn exchange(
        &mut self,
        row: usize,
        entering: usize,
        leaving: (Status, f64),
        d: DVector<f64>,
        factors: &mut Factors,
    ) {
        let k = self.basis[row];
        (self.status[k], self.x[k]) = leaving;
        self.status[entering] = Status::Basic;
        self.basis[row] = entering;
        factors.update(&self.a, &self.basis, row, d);
    }

    /// Minimizes `costᵀx` from the current basis, returning the multipliers `y` at the
    /// optimum, or `None` if the objective decreases without limit.
    ///
    /// From [`Perturbation::Repairing`], it first brings each basic variable beyond its bounds
    /// back with dual simplex pivots, as after restoring perturbed bounds: so a basis that's
    /// optimal for the costs but not feasible, such as an earlier optimum with a new row, is a
    /// start. Then, with no bounds widened yet, it may widen them as from [`Perturbation::Off`].
    fn minimize(
        &mut self,
        cost: &DVector<f64>,
        start: Perturbation,
    ) -> Result<Option<DVector<f64>>, CutupError> {
        let (m, total) = self.a.shape();
        // Pivots in a row that didn't move x. Past a few, the bounds are widened, or if they
        // can't be, Bland's rule takes over until one does: only a run of such pivots can
        // cycle, and Bland's can't.
        let mut stalled = 0;
        let mut repairs = 0;
        let mut factors = Factors::new(&self.a, &self.basis);
        // The problem's own bounds, to restore after perturbing them, and which variables'
        // bounds are widened.
        let (lower, upper) = (self.lower.clone(), self.upper.clone());
        let mut perturbed = vec![false; total];
        let mut perturbation = start;
        for _ in 0..iteration_cap(m, total) {
            // The basic variables from the nonbasic ones, which sit exactly at their bounds.
            let nonbasic = DVector::from_fn(total, |j, _| {
                if self.status[j] == Status::Basic {
                    0.0
                } else {
                    self.x[j]
                }
            });
            let x_basis = factors.solve(&(&self.b - &self.a * nonbasic))?;
            for (i, &j) in self.basis.iter().enumerate() {
                self.x[j] = x_basis[i];
            }
            // Bᵀy = c_B from the same factors.
            let cost_basis = DVector::from_fn(m, |i, _| cost[self.basis[i]]);
            let y = factors.solve_transpose(cost_basis)?;
            let reduced = cost - self.a.tr_mul(&y);

            // Restoring the bounds moves the basic variables, and can leave some beyond theirs.
            // The basis is still optimal for the costs, which haven't changed, so dual simplex
            // pivots repair it, each keeping it optimal. One beyond its bound only by rounding
            // may have no variable to enter, and is left there.
            if perturbation == Perturbation::Repairing {
                if self.repair(&reduced, &mut factors)? {
                    repairs += 1;
                    if repairs > self.repairs {
                        return Err(CutupError::NotConverged);
                    }
                    stalled = 0;
                    continue;
                }
                // A warm start never widened any bounds, so it may still widen them on a
                // degenerate vertex rather than leave each stall to Bland's rule.
                perturbation = if perturbed.contains(&true) {
                    Perturbation::Done
                } else {
                    Perturbation::Off
                };
            }
            // A run of pivots that don't move x is on a degenerate vertex, where basic variables
            // sit at their bounds. Widening those bounds, each by a different amount, moves the
            // vertex apart, so later steps move x. Only when the run reaches STALL: past it,
            // Bland's rule has taken over, and widening a variable it brought into the basis
            // would hand the run back to Dantzig's, which can cycle.
            if stalled == STALL
                && matches!(perturbation, Perturbation::Off | Perturbation::On)
                && self.perturb(&mut perturbed)
            {
                perturbation = Perturbation::On;
                stalled = 0;
            }

            // The variable that improves the objective fastest, and the direction it moves in
            // (Dantzig's rule), or while stalled the lowest-indexed one (Bland's). Each reduced
            // cost cⱼ − aⱼᵀy is a difference of terms up to |cⱼ| + |aⱼ|ᵀ|y|, so it only counts
            // beyond rounding of that: a single tolerance would stop early on variables whose
            // terms are all small, as for an asset whose returns are orders of magnitude below
            // the others'. But never below rounding of the problem as a whole, which such a
            // variable can't tell apart from a true reduced cost: entering on that noise moves
            // nothing, and pivots on rounding.
            let magnitude = cost.abs() + self.a_abs.tr_mul(&y.abs());
            let floor = 1e-14 * magnitude.amax();
            let improving = (0..total).filter_map(|j| {
                let tolerance = (1e-9 * magnitude[j]).max(floor);
                let movable = self.upper[j] > self.lower[j];
                match self.status[j] {
                    Status::Lower if movable && reduced[j] < -tolerance => Some((j, 1.0)),
                    Status::Upper if movable && reduced[j] > tolerance => Some((j, -1.0)),
                    Status::Free if reduced[j].abs() > tolerance => Some((j, -reduced[j].signum())),
                    _ => None,
                }
            });
            let entering = if stalled < STALL {
                improving.max_by(|p, q| reduced[p.0].abs().total_cmp(&reduced[q.0].abs()))
            } else {
                improving.min_by_key(|p| p.0)
            };
            // The optimum is confirmed, and `x` and `y` returned, only from fresh factors, so
            // the eta matrices' rounding never reaches the solution.
            let Some((j, sign)) = entering else {
                if perturbation == Perturbation::On {
                    self.restore(&lower, &upper);
                    perturbation = Perturbation::Repairing;
                    continue;
                }
                if factors.etas.is_empty() {
                    return Ok(Some(y));
                }
                factors = Factors::new(&self.a, &self.basis);
                continue;
            };

            // Moving x_j by sign·t moves the basic variables by t·change, with d = B⁻¹aⱼ.
            let d = factors.solve(&self.a.column(j).into_owned())?;
            let change = d.scale(-sign);
            let pivot = 1e-9 * change.amax();
            // The step at which each variable reaches a bound, the entering one included, by
            // variable index: the smallest step wins, and among equal steps the lowest index.
            // Changes within rounding of zero don't block, so that a pivot is never on noise.
            let own = (self.upper[j] - self.lower[j])
                .is_finite()
                .then(|| (self.upper[j] - self.lower[j], j, None));
            let ratio = |i: usize, counts: &dyn Fn(f64) -> bool| {
                let k = self.basis[i];
                let d = change[i];
                if d < 0.0 && counts(d) && self.lower[k].is_finite() {
                    Some((
                        ((self.x[k] - self.lower[k]) / -d).max(0.0),
                        k,
                        Some((i, Status::Lower)),
                    ))
                } else if d > 0.0 && counts(d) && self.upper[k].is_finite() {
                    Some((
                        ((self.upper[k] - self.x[k]) / d).max(0.0),
                        k,
                        Some((i, Status::Upper)),
                    ))
                } else {
                    None
                }
            };
            let smallest = |candidates: Vec<Block>| {
                candidates
                    .into_iter()
                    .chain(own)
                    .min_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)))
            };
            let mut blocking = smallest(
                (0..m)
                    .filter_map(|i| ratio(i, &|d| d.abs() > pivot))
                    .collect(),
            );
            // But a change that small still moves its variable, and a long enough step can
            // carry it past a bound. If one would end up beyond rounding of its bound, it
            // blocks too.
            let step = blocking.map_or(f64::INFINITY, |(step, _, _)| step);
            let overshoots = |i: usize| {
                let k = self.basis[i];
                let next = self.x[k] + step * change[i];
                let slack = 1e-12 * self.x[k].abs().max(1.0);
                change[i] != 0.0 && (next < self.lower[k] - slack || next > self.upper[k] + slack)
            };
            if (0..m).any(|i| change[i].abs() <= pivot && overshoots(i)) {
                // With nothing else to block the step, only a change that small stops it
                // running off without limit, and the eta matrices' rounding can make one where
                // the change is nothing: then pivoting on it leaves the basis singular. So such
                // a block, like unboundedness, counts only from fresh factors.
                if blocking.is_none() && !factors.etas.is_empty() {
                    factors = Factors::new(&self.a, &self.basis);
                    continue;
                }
                blocking = smallest(
                    (0..m)
                        .filter_map(|i| ratio(i, &|d| d.abs() > pivot || overshoots(i)))
                        .collect(),
                );
            }
            // Unboundedness, like the optimum, is declared only from fresh factors: the eta
            // matrices' rounding could hide a change that blocks.
            let Some((step, _, leaving)) = blocking else {
                if factors.etas.is_empty() {
                    return Ok(None);
                }
                factors = Factors::new(&self.a, &self.basis);
                continue;
            };

            // A step stalls when the leaving variable moves within rounding of nothing. The
            // basic variables are recomputed each pivot, so even a degenerate step comes out
            // as noise rather than exactly zero.
            let stalls = leaving.is_some_and(|(row, _)| {
                let k = self.basis[row];
                step * change[row].abs() <= 1e-12 * self.x[k].abs().max(1.0)
            });
            stalled = if stalls { stalled + 1 } else { 0 };
            // The entering variable is set here only if it reaches its other bound: if it
            // enters the basis, the next pivot solves for it with the other basic variables.
            match leaving {
                // The entering variable reaches its other bound before anything blocks it.
                None => {
                    self.status[j] = if sign > 0.0 {
                        Status::Upper
                    } else {
                        Status::Lower
                    };
                    self.x[j] = if sign > 0.0 {
                        self.upper[j]
                    } else {
                        self.lower[j]
                    };
                }
                Some((row, side)) => {
                    let k = self.basis[row];
                    self.status[k] = side;
                    self.x[k] = match side {
                        Status::Upper => self.upper[k],
                        _ => self.lower[k],
                    };
                    self.status[j] = Status::Basic;
                    self.basis[row] = j;
                    factors.update(&self.a, &self.basis, row, d);
                }
            }
        }
        Err(CutupError::NotConverged)
    }

    /// Widens the finite bounds of each basic variable not yet perturbed, returning whether
    /// any had a finite bound to widen. Each bound `l` moves out by `self.perturbation` times
    /// `max(1, |l|)` times a factor from 1 to 2 that differs from bound to bound, so that no two
    /// variables reach their bounds on the same step. Widening keeps `x` within them.
    fn perturb(&mut self, perturbed: &mut [bool]) -> bool {
        // The fractional parts of the multiples of an irrational number spread evenly over
        // [0, 1).
        let spread = |k: usize| 1.0 + (k as f64 * std::f64::consts::SQRT_2).fract();
        let mut any = false;
        for &j in &self.basis {
            if perturbed[j] {
                continue;
            }
            perturbed[j] = true;
            if self.lower[j].is_finite() {
                self.lower[j] -= self.perturbation * spread(2 * j) * self.lower[j].abs().max(1.0);
                any = true;
            }
            if self.upper[j].is_finite() {
                self.upper[j] +=
                    self.perturbation * spread(2 * j + 1) * self.upper[j].abs().max(1.0);
                any = true;
            }
        }
        any
    }

    /// Restores the problem's own bounds, moving each nonbasic variable back onto its bound.
    fn restore(&mut self, lower: &DVector<f64>, upper: &DVector<f64>) {
        self.lower.copy_from(lower);
        self.upper.copy_from(upper);
        for j in 0..self.x.len() {
            match self.status[j] {
                Status::Lower => self.x[j] = lower[j],
                Status::Upper => self.x[j] = upper[j],
                Status::Basic | Status::Free => {}
            }
        }
    }

    /// The row of the basic variable furthest beyond one of its bounds, relative to
    /// `max(1, |x|)`, and how far, if any is beyond it by more than `1e-12` of that.
    fn most_infeasible(&self) -> Option<(usize, f64)> {
        let beyond = |i: usize| {
            let k = self.basis[i];
            (self.lower[k] - self.x[k]).max(self.x[k] - self.upper[k]) / self.x[k].abs().max(1.0)
        };
        (0..self.basis.len())
            .map(|i| (i, beyond(i)))
            .filter(|&(_, excess)| excess > 1e-12)
            .max_by(|p, q| p.1.total_cmp(&q.1))
    }

    /// After the bounds are restored, a dual pivot on the basic variable furthest beyond its
    /// bounds, if any is beyond rounding of them, returning whether it took one. If none can
    /// enter, one within `1e-9` of `max(1, |x|)` is left there.
    ///
    /// # Errors
    ///
    /// [`CutupError::NotConverged`] if a basis can't be factored, or if nothing can bring back
    /// a basic variable further beyond its bound than that.
    fn repair(
        &mut self,
        reduced: &DVector<f64>,
        factors: &mut Factors,
    ) -> Result<bool, CutupError> {
        let Some((row, excess)) = self.most_infeasible() else {
            return Ok(false);
        };
        if self.dual_pivot(row, reduced, factors)? {
            return Ok(true);
        }
        if excess > 1e-9 {
            return Err(CutupError::NotConverged);
        }
        Ok(false)
    }

    /// One pivot of the dual simplex method: the basic variable at `row`, beyond a bound, leaves
    /// the basis at that bound, and the nonbasic variable that enters is the one that keeps
    /// every reduced cost's sign, so the basis stays optimal for the costs.
    ///
    /// Row `row` of `B⁻¹A` is `α = Aᵀρ`, with `ρ = B⁻ᵀeᵣ`. Moving a nonbasic `xⱼ` by `t` moves
    /// the leaving variable by `−αⱼt`, so `xⱼ` can bring it back only if it's free to move the
    /// way that takes. The reduced costs then change by `−(dⱼ/αⱼ)α`, so the one with the
    /// smallest `|dⱼ/αⱼ|` enters: any other would push it past zero.
    ///
    /// Returns whether it pivoted. No variable can bring the leaving one back only if the
    /// problem is infeasible, which after phase 1 only rounding can make it.
    ///
    /// # Errors
    ///
    /// [`CutupError::NotConverged`] if a basis can't be factored.
    fn dual_pivot(
        &mut self,
        row: usize,
        reduced: &DVector<f64>,
        factors: &mut Factors,
    ) -> Result<bool, CutupError> {
        let k = self.basis[row];
        // The bound the leaving variable is brought back to, and whether that's up.
        let bound = self.x[k].max(self.lower[k]).min(self.upper[k]);
        let rises = (bound - self.x[k]).is_sign_positive();
        let alpha = self.tableau_row(row, factors)?;
        let pivot = 1e-9 * alpha.amax();
        let Some(entering) = (0..self.x.len())
            .filter(|&j| alpha[j].abs() > pivot)
            .filter(|&j| {
                // The direction xⱼ must move in: against αⱼ to raise the leaving variable.
                let up = alpha[j].is_sign_negative() == rises;
                let movable = self.upper[j] > self.lower[j];
                match self.status[j] {
                    Status::Lower => movable && up,
                    Status::Upper => movable && !up,
                    Status::Free => true,
                    Status::Basic => false,
                }
            })
            .min_by(|&p, &q| {
                (reduced[p] / alpha[p])
                    .abs()
                    .total_cmp(&(reduced[q] / alpha[q]).abs())
                    .then(alpha[q].abs().total_cmp(&alpha[p].abs()))
                    .then(p.cmp(&q))
            })
        else {
            return Ok(false);
        };

        let d = factors.solve(&self.a.column(entering).into_owned())?;
        let side = if rises { Status::Lower } else { Status::Upper };
        self.exchange(row, entering, (side, bound), d, factors);
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};
    #[cfg(target_arch = "wasm32")]
    use wasm_bindgen_test::wasm_bindgen_test as test;

    const INF: f64 = f64::INFINITY;

    fn optimal(outcome: Outcome) -> Solution {
        match outcome {
            Outcome::Optimal(solution) => solution,
            other => panic!("expected an optimum, got {other:?}"),
        }
    }

    #[test]
    fn updated_factors_solve_like_fresh_ones() {
        // A well-conditioned 5 × 9 matrix, and a run of column swaps into its basis.
        let a = DMatrix::from_fn(5, 9, |i, j| {
            if i == j {
                4.0
            } else {
                ((i * 7 + j * 3) % 5) as f64 / 5.0 - 0.4
            }
        });
        let mut basis = vec![0, 1, 2, 3, 4];
        let mut factors = Factors::new(&a, &basis);
        let b = DVector::from_fn(5, |i, _| i as f64 - 1.5);
        for (row, j) in [(1, 6), (3, 8), (0, 5), (1, 7), (4, 1)] {
            let d = factors.solve(&a.column(j).into_owned()).unwrap();
            basis[row] = j;
            factors.update(&a, &basis, row, d);
            let fresh = Factors::new(&a, &basis);
            let (x, y) = (
                factors.solve(&b).unwrap(),
                factors.solve_transpose(b.clone()).unwrap(),
            );
            assert!((&x - fresh.solve(&b).unwrap()).amax() < 1e-13, "{x}");
            assert!(
                (&y - fresh.solve_transpose(b.clone()).unwrap()).amax() < 1e-13,
                "{y}"
            );
        }
        assert_eq!(factors.etas.len(), 5);
    }

    #[test]
    fn factors_are_refreshed_every_so_often_and_after_a_tiny_pivot() {
        let a = DMatrix::<f64>::identity(3, 3);
        let basis = [0, 1, 2];
        let mut factors = Factors::new(&a, &basis);
        let d = dvector![0.5, 1.0, 0.25];
        for count in 1..=REFACTOR {
            factors.update(&a, &basis, 0, d.clone());
            assert_eq!(factors.etas.len(), count);
        }
        // One more is factored afresh.
        factors.update(&a, &basis, 0, d.clone());
        assert!(factors.etas.is_empty());
        // So is a pivot at 1e-6 of the column's largest entry, but not one just above it.
        factors.update(&a, &basis, 0, dvector![2e-6, 2.0, 0.0]);
        assert!(factors.etas.is_empty());
        factors.update(&a, &basis, 0, dvector![2.2e-6, 2.0, 0.0]);
        assert_eq!(factors.etas.len(), 1);
    }

    #[test]
    fn the_iteration_cap_grows_with_the_problem() {
        // 1000 pivots, and 100 more per row and per variable: a guard against cycling that
        // rounding gets past Bland's rule, far above what any test problem needs.
        assert_eq!(iteration_cap(2, 3), 1500);
        assert_eq!(iteration_cap(0, 0), 1000);
    }

    #[test]
    fn the_repair_cap_grows_with_the_rows() {
        // 100 dual pivots, and 10 more per row: over ten times what any repair has taken.
        assert_eq!(repair_cap(0), 100);
        assert_eq!(repair_cap(13), 230);
    }

    #[test]
    fn a_small_lp_reaches_its_vertex() {
        // maximize 3x + 2y subject to x + y ≤ 4, x + 3y ≤ 6, x ≤ 3, as equalities with slacks.
        let a = dmatrix![1.0, 1.0, 1.0, 0.0; 1.0, 3.0, 0.0, 1.0];
        let solution = optimal(
            solve(
                &dvector![-3.0, -2.0, 0.0, 0.0],
                &a,
                &dvector![4.0, 6.0],
                &dvector![0.0, 0.0, 0.0, 0.0],
                &dvector![3.0, INF, INF, INF],
            )
            .unwrap(),
        );
        assert!((solution.x[0] - 3.0).abs() < 1e-14);
        assert!((solution.x[1] - 1.0).abs() < 1e-14);
        assert!((solution.objective + 11.0).abs() < 1e-14);
    }

    #[test]
    fn infeasibility_is_measured_against_the_problems_scale() {
        // x₀ − x₁ = 0 with x₀ ≥ l and x₁ ≤ u, and x₀ = b where given: infeasible by l − u.
        let solve_gap = |l: f64, u: f64, b: Option<f64>| {
            let (a, rhs) = match b {
                Some(b) => (dmatrix![1.0, -1.0; 1.0, 0.0], dvector![0.0, b]),
                None => (dmatrix![1.0, -1.0], dvector![0.0]),
            };
            solve(
                &dvector![0.0, 0.0],
                &a,
                &rhs,
                &dvector![l, 0.0],
                &dvector![INF, u],
            )
            .unwrap()
        };
        // Within 1e-9 of weights of 1e8, or of a right-hand side of 1e8, is rounding.
        optimal(solve_gap(1e8, 1e8 - 1e-2, None));
        optimal(solve_gap(1e8, 1e8 - 1e-2, Some(1e8)));
        // At weights of 1e-8, a gap of 1e-12 is not.
        assert!(matches!(
            solve_gap(1e-8, 1e-8 - 1e-12, None),
            Outcome::Infeasible
        ));
    }

    #[test]
    fn infeasibility_at_the_tolerance_itself_is_rounding() {
        // 0 = r alongside x = 0.5, a scale of 1: nothing can take the first row's artificial
        // variable below r. At exactly 1e-9 of the scale that's still rounding. Beyond it, the
        // problem is infeasible.
        let off_by = |r: f64| {
            solve(
                &dvector![1.0],
                &dmatrix![0.0; 1.0],
                &dvector![r, 0.5],
                &dvector![0.0],
                &dvector![1.0],
            )
            .unwrap()
        };
        assert_eq!(optimal(off_by(1e-9)).x[0], 0.5);
        assert_eq!(off_by(1.5e-9), Outcome::Infeasible);
    }

    #[test]
    fn without_constraints_each_variable_takes_the_bound_its_cost_favors() {
        // The last has a cost and a floor below zero: it takes the floor, not the zero that a
        // variable without a cost would.
        let bounds = (
            dvector![0.5, 0.0, -2.0, 1.0, -1.5],
            dvector![3.0, 2.0, 5.0, 4.0, 2.5],
        );
        let solution = optimal(
            solve(
                &dvector![1.0, -1.0, 0.0, 0.0, 1.0],
                &DMatrix::zeros(0, 5),
                &DVector::zeros(0),
                &bounds.0,
                &bounds.1,
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[0.5, 2.0, 0.0, 1.0, -1.5]);
        assert!(matches!(
            solve(
                &dvector![-1.0],
                &DMatrix::zeros(0, 1),
                &DVector::zeros(0),
                &dvector![0.0],
                &dvector![INF],
            )
            .unwrap(),
            Outcome::Unbounded
        ));
    }

    #[test]
    fn a_small_lp_reaches_its_vertex_at_any_cost_scale() {
        // a_small_lp_reaches_its_vertex with its costs scaled: the reduced costs' tolerance
        // scales with them.
        let a = dmatrix![1.0, 1.0, 1.0, 0.0; 1.0, 3.0, 0.0, 1.0];
        for scale in [1e-8, 1e8] {
            let solution = optimal(
                solve(
                    &(dvector![-3.0, -2.0, 0.0, 0.0] * scale),
                    &a,
                    &dvector![4.0, 6.0],
                    &dvector![0.0, 0.0, 0.0, 0.0],
                    &dvector![3.0, INF, INF, INF],
                )
                .unwrap(),
            );
            assert!((solution.x[0] - 3.0).abs() < 1e-14, "{scale}");
            assert!((solution.x[1] - 1.0).abs() < 1e-14, "{scale}");
        }
    }

    #[test]
    fn an_entering_variable_can_stop_at_its_other_bound() {
        // maximize x subject to x + s = 4, 1 ≤ x ≤ 3: x reaches 3 before s reaches 0.
        let solution = optimal(
            solve(
                &dvector![-1.0, 0.0],
                &dmatrix![1.0, 1.0],
                &dvector![4.0],
                &dvector![1.0, 0.0],
                &dvector![3.0, INF],
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[3.0, 1.0]);
    }

    #[test]
    fn an_entering_variables_range_is_the_difference_of_its_bounds() {
        // x₀, in no row, runs from 1e308 to 1.2e308. The bounds' sum overflows, but its range,
        // their difference, doesn't, and it stops at its cap rather than running off.
        let solution = optimal(
            solve(
                &dvector![-1.0, 0.0],
                &dmatrix![0.0, 1.0],
                &dvector![1.0],
                &dvector![1e308, 0.0],
                &dvector![1.2e308, INF],
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[1.2e308, 1.0]);
    }

    #[test]
    fn the_steepest_reduced_cost_enters_first() {
        // 0.5x₀ + x₁ = 1 with no costs, so the solve ends where phase 1 does. Either variable
        // lowers the row's residual, x₁ twice as fast, so it enters (Dantzig's rule) and meets
        // the row alone. The lowest-indexed, x₀, would meet it at (2, 0).
        let solution = optimal(
            solve(
                &dvector![0.0, 0.0],
                &dmatrix![0.5, 1.0],
                &dvector![1.0],
                &dvector![0.0, 0.0],
                &dvector![INF, INF],
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[0.0, 1.0]);
    }

    #[test]
    fn only_the_variables_asked_start_at_their_upper_bound() {
        // x₀ + 2x₁ = 2 within [0, 1] and [0, 5], with no costs, so the solve ends where phase 1
        // does. From the lower bounds, x₁ rises to 1 and meets the row alone. With x₀ started
        // at its cap, the row is half met already, and x₁ rises to 0.5.
        let (c, a, b) = (dvector![0.0, 0.0], dmatrix![1.0, 2.0], dvector![2.0]);
        let (lower, upper) = (dvector![0.0, 0.0], dvector![1.0, 5.0]);
        let from_lower = optimal(solve(&c, &a, &b, &lower, &upper).unwrap());
        assert_eq!(from_lower.x.as_slice(), &[0.0, 1.0]);
        let from_upper = optimal(solve_from(&c, &a, &b, &lower, &upper, &[0]).unwrap());
        assert_eq!(from_upper.x.as_slice(), &[1.0, 0.5]);
    }

    #[test]
    fn a_reduced_cost_within_rounding_of_its_terms_doesnt_enter() {
        // 0.5x₀ + x₁ = 1: phase 1 meets the row with x₁, at a cost of 2. x₀ would meet it for
        // 2e-10 less, but its reduced cost, c₀ − 0.5y = −1e-10, is a difference of terms of 1:
        // within 1e-9 of them it's rounding, and x₀ stays out.
        let solution = optimal(
            solve(
                &dvector![1.0 - 1e-10, 2.0],
                &dmatrix![0.5, 1.0],
                &dvector![1.0],
                &dvector![0.0, 0.0],
                &dvector![INF, INF],
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[0.0, 1.0]);
    }

    #[test]
    fn a_free_variable_with_no_reduced_cost_stays_out() {
        // x₀ + 0.5x₁ = 1 with no costs and x₁ free: phase 1 meets the row with x₀, and x₁
        // stays at zero. With no costs every tolerance in phase 2 is zero, and so is x₁'s
        // reduced cost, which is no reason to move it: it would take x₀ off without limit.
        let solution = optimal(
            solve(
                &dvector![0.0, 0.0],
                &dmatrix![1.0, 0.5],
                &dvector![1.0],
                &dvector![0.0, -INF],
                &dvector![INF, INF],
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[1.0, 0.0]);
    }

    #[test]
    fn a_tiny_change_still_blocks_a_long_step() {
        // maximize x subject to x + s₁ = 1e10 and 1e-10·x + s₂ = 0.5. s₂ changes by only
        // 1e-10 per unit of x, within rounding of the pivot, but over the step of 1e10 that s₁
        // allows it would reach −0.5, so it blocks first, at x = 5e9.
        let solution = optimal(
            solve(
                &dvector![-1.0, 0.0, 0.0],
                &dmatrix![1.0, 1.0, 0.0; 1e-10, 0.0, 1.0],
                &dvector![1e10, 0.5],
                &dvector![0.0, 0.0, 0.0],
                &dvector![INF, INF, INF],
            )
            .unwrap(),
        );
        assert!((solution.x[0] - 5e9).abs() <= 1e-6, "{}", solution.x);
        assert!(solution.x[2].abs() <= 1e-12, "{}", solution.x);
    }

    #[test]
    fn an_all_zero_row_is_kept() {
        // 0 = 0 alongside x₀ + x₁ = 1: minimize x₀.
        let solution = optimal(
            solve(
                &dvector![1.0, 0.0],
                &dmatrix![0.0, 0.0; 1.0, 1.0],
                &dvector![0.0, 1.0],
                &dvector![0.0, 0.0],
                &dvector![INF, INF],
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[0.0, 1.0]);
    }

    #[test]
    fn beales_cycling_example_terminates() {
        // Beale (1955): the textbook rule cycles on this degenerate vertex. Widening the bounds
        // gets the solve off it.
        let a = dmatrix![
            0.25, -60.0, -0.04, 9.0, 1.0, 0.0, 0.0;
            0.5, -90.0, -0.02, 3.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        ];
        let (c, b) = (
            dvector![-0.75, 150.0, -0.02, 6.0, 0.0, 0.0, 0.0],
            dvector![0.0, 0.0, 1.0],
        );
        let (lower, upper) = (DVector::zeros(7), DVector::from_element(7, INF));
        let solution = optimal(solve(&c, &a, &b, &lower, &upper).unwrap());
        assert!((solution.objective + 0.05).abs() < 1e-14);
        // So does Bland's rule, which takes over when widening the bounds by nothing doesn't.
        let solution = optimal(
            solve_perturbing(&c, &a, &b, &lower, &upper, &[], 0.0)
                .unwrap()
                .0,
        );
        assert!((solution.objective + 0.05).abs() < 1e-14);
    }

    /// A problem with one row per basic variable, at `x` with the given bounds and statuses,
    /// for testing the steps of [`Problem::minimize`] on their own.
    fn problem(
        (lower, upper): (DVector<f64>, DVector<f64>),
        x: DVector<f64>,
        basis: Vec<usize>,
        status: Vec<Status>,
        perturbation: f64,
    ) -> Problem {
        let a = DMatrix::zeros(basis.len(), x.len());
        Problem {
            a_abs: a.clone(),
            a,
            b: DVector::zeros(basis.len()),
            lower,
            upper,
            x,
            basis,
            status,
            perturbation,
            repairs: usize::MAX,
        }
    }

    #[test]
    fn perturbing_widens_each_basic_variables_bounds_once() {
        use Status::{Basic, Lower, Upper};
        let mut problem = problem(
            (dvector![0.0, -3.0, 0.0, -INF], dvector![2.0, INF, 1.0, INF]),
            dvector![0.5, 2.0, 1.0, 0.0],
            vec![0, 1],
            vec![Basic, Basic, Upper, Lower],
            0.1,
        );
        let mut perturbed = vec![false; 4];
        assert!(problem.perturb(&mut perturbed));
        // Each bound moves out by 0.1 times max(1, |bound|) times 1 plus the fractional part of
        // a multiple of √2: the 0th for x₀'s floor, the 1st for its cap, the 2nd for x₁'s
        // floor. Infinite bounds stay infinite, and nonbasic variables keep theirs.
        let spread = |k: f64| 1.0 + (k * std::f64::consts::SQRT_2).fract();
        assert_eq!(problem.lower[0], -0.1);
        assert!((problem.upper[0] - (2.0 + 0.1 * spread(1.0) * 2.0)).abs() < 1e-15);
        assert!((problem.lower[1] - (-3.0 - 0.1 * spread(2.0) * 3.0)).abs() < 1e-15);
        assert_eq!(problem.upper[1], INF);
        assert_eq!(problem.lower.rows(2, 2), dvector![0.0, -INF]);
        assert_eq!(problem.upper.rows(2, 2), dvector![1.0, INF]);
        assert_eq!(perturbed, [true, true, false, false]);
        // x stays where it was, now strictly within the bounds.
        assert_eq!(problem.x, dvector![0.5, 2.0, 1.0, 0.0]);

        // Variables already perturbed aren't again, so with nothing new in the basis there's
        // nothing to do.
        let (lower, upper) = (problem.lower.clone(), problem.upper.clone());
        assert!(!problem.perturb(&mut perturbed));
        assert_eq!((&problem.lower, &problem.upper), (&lower, &upper));
        // A variable that has entered the basis since is.
        problem.basis[1] = 2;
        problem.status[2] = Basic;
        assert!(problem.perturb(&mut perturbed));
        assert!(problem.lower[2] < 0.0 && problem.upper[2] > 1.0);
        assert_eq!(problem.lower[1], lower[1]);
        // A free variable has no bound to widen, so one entering leaves nothing to do, and
        // Bland's rule takes over.
        problem.basis[0] = 3;
        problem.status[3] = Basic;
        assert!(!problem.perturb(&mut perturbed));
        assert!(perturbed[3]);
    }

    #[test]
    fn restoring_puts_nonbasic_variables_back_on_their_bounds() {
        use Status::{Basic, Free, Lower, Upper};
        let mut problem = problem(
            (
                dvector![-0.1, -0.2, -0.3, -INF],
                dvector![1.1, 1.2, 1.3, INF],
            ),
            dvector![0.5, -0.2, 1.3, 7.0],
            vec![0],
            vec![Basic, Lower, Upper, Free],
            0.1,
        );
        let (lower, upper) = (DVector::zeros(4), dvector![1.0, 1.0, 1.0, INF]);
        problem.restore(&lower, &upper);
        assert_eq!((&problem.lower, &problem.upper), (&lower, &upper));
        // The basic variable is left for the next pivot to recompute, and the free one where
        // it was.
        assert_eq!(problem.x, dvector![0.5, 0.0, 1.0, 7.0]);
    }

    #[test]
    fn the_most_infeasible_basic_variable_is_found_beyond_rounding() {
        use Status::Basic;
        let mut problem = problem(
            (DVector::zeros(3), DVector::from_element(3, 1.0)),
            dvector![-2e-12, 1.0 + 5e-12, 0.5],
            vec![0, 1, 2],
            vec![Basic; 3],
            0.1,
        );
        let (row, excess) = problem.most_infeasible().unwrap();
        assert_eq!(row, 1);
        assert!((excess - 5e-12).abs() < 1e-16, "{excess}");
        // Relative to max(1, |x|): 5e-10 beyond a cap of 1000 is within rounding of it, and so
        // is 1e-12 beyond a floor of 0.
        problem.upper[1] = 1000.0;
        problem.x[1] = 1000.0 + 5e-10;
        problem.x[0] = -1e-12;
        assert_eq!(problem.most_infeasible(), None);
    }

    /// One row, `x₀ + Σαⱼxⱼ = 0`, with `x₀` basic and beyond its bounds of `[0, 1]` at `x0`,
    /// so `B = [1]` and row 0 of `B⁻¹A` is the row itself. Its reduced costs are optimal for
    /// each variable's status.
    fn one_row(x0: f64) -> (Problem, DVector<f64>) {
        use Status::{Basic, Free, Lower, Upper};
        // x₄ is fixed, and α₇ is exactly 1e-9 of the largest |α|, at the pivot tolerance.
        let alpha = [1.0, -0.5, 1.0, 1.0, -4.0, -0.2, -1.0, -1e-9 * 4.0];
        let mut problem = problem(
            (
                dvector![0.0, 0.0, 0.0, 0.0, 0.5, -INF, 0.0, 0.0],
                dvector![1.0, 1.0, 1.0, 1.0, 0.5, INF, 1.0, 1.0],
            ),
            dvector![x0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0],
            vec![0],
            vec![Basic, Lower, Upper, Lower, Lower, Free, Upper, Lower],
            0.1,
        );
        problem.a = DMatrix::from_row_slice(1, 8, &alpha);
        let reduced = dvector![0.0, 1.0, -1.0, 0.1, 0.2, 0.3, -0.05, 1e-12];
        (problem, reduced)
    }

    #[test]
    fn a_dual_pivot_keeps_every_reduced_cost_on_its_side() {
        // x₀ = −1 must rise. x₀ moves by −αⱼ per unit of xⱼ, so x₁ (α = −0.5) can raise it from
        // its floor and x₂ (α = 1) from its cap, at |dⱼ/αⱼ| of 2 and 1, and the free x₅ at
        // 1.5. x₃ and x₆ would move x₀ the wrong way, x₄ is fixed, and α₇ is within rounding,
        // so none of them can enter however small their ratio. x₂ enters, and x₀ leaves at
        // its floor.
        let (mut problem, reduced) = one_row(-1.0);
        let mut factors = Factors::new(&problem.a, &problem.basis);
        assert!(problem.dual_pivot(0, &reduced, &mut factors).unwrap());
        assert_eq!(problem.basis, [2]);
        assert_eq!((problem.status[0], problem.x[0]), (Status::Lower, 0.0));
        assert_eq!(problem.status[2], Status::Basic);
        assert_eq!(factors.etas.len(), 1);

        // x₀ = 2 must fall: now x₃ from its floor and x₆ from its cap can lower it, at 0.1 and
        // 0.05, and x₆ enters.
        let (mut problem, reduced) = one_row(2.0);
        let mut factors = Factors::new(&problem.a, &problem.basis);
        assert!(problem.dual_pivot(0, &reduced, &mut factors).unwrap());
        assert_eq!(problem.basis, [6]);
        assert_eq!((problem.status[0], problem.x[0]), (Status::Upper, 1.0));
    }

    #[test]
    fn a_basic_variable_nothing_can_bring_back_is_left_only_within_rounding() {
        use Status::{Basic, Lower};
        // x₀ + x₁ = 0 with x₁ at its floor: raising x₁ only lowers x₀, so nothing can bring
        // x₀ back up to 0.
        let at = |x0: f64| {
            let mut problem = problem(
                (DVector::zeros(2), DVector::from_element(2, 1.0)),
                dvector![x0, 0.0],
                vec![0],
                vec![Basic, Lower],
                0.1,
            );
            problem.a = dmatrix![1.0, 1.0];
            let mut factors = Factors::new(&problem.a, &problem.basis);
            problem.repair(&dvector![0.0, 1.0], &mut factors)
        };
        // 1e-9 below is within rounding, so the repair is done; just beyond that, it fails. At
        // 0 there's nothing to repair.
        assert_eq!(at(-1e-9), Ok(false));
        assert_eq!(at(-1.1e-9), Err(CutupError::NotConverged));
        assert_eq!(at(0.0), Ok(false));
    }

    #[test]
    fn restoring_wide_bounds_is_repaired_by_dual_pivots() {
        // A degenerate problem, b = Ax₀ with x₀ mostly at its bounds. Widened by 1e-2, the
        // solve leaves a basic variable beyond its restored bound, which a dual pivot brings
        // back; widened by the usual 1e-7, or not at all, it doesn't. Each reaches the same
        // optimum.
        let a = dmatrix![
            -0.5, 1.0, 1.0, 1.0, -1.5, -0.5, 1.0, -0.5, 1.5, 0.5, 0.5;
            1.0, 0.5, 0.0, 1.5, 0.0, -0.5, -1.5, -1.0, 0.5, -1.5, 1.5;
            1.5, -2.0, 1.0, 1.5, 0.5, 0.5, 1.0, -1.0, 0.5, 1.5, 0.0;
            0.5, -1.0, -2.0, -0.5, 0.0, 1.0, -0.5, 1.5, -1.5, -1.0, 0.5;
            0.5, 1.5, 0.0, 0.0, 0.0, -1.5, -2.0, 1.5, -1.0, -2.0, -1.5;
            -2.0, -1.0, 1.5, 1.5, -1.0, 0.5, -2.0, -0.5, -1.5, -0.5, -2.0;
            1.5, -2.0, -1.0, 0.5, 0.0, 0.5, -2.0, -2.0, 0.0, -1.0, -1.5;
            0.5, -1.5, -0.5, -2.0, -2.0, 1.5, 1.0, -1.0, -1.0, -1.0, -1.5
        ];
        let x0 = dvector![
            0.0, 0.25, 0.0, 0.0, 0.75, 0.75, -0.5, -0.5, -0.25, 0.25, 0.5
        ];
        let b = &a * &x0;
        let c = dvector![0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.5, 1.0, 0.5, -1.0];
        let lower = dvector![
            0.0, 0.25, -1.0, 0.0, 0.75, 0.75, -0.5, -0.5, -0.25, 0.25, -0.5
        ];
        let upper = dvector![2.0, INF, 0.0, INF, 2.75, INF, INF, -0.5, INF, INF, 0.5];
        let plain = optimal(
            solve_perturbing(&c, &a, &b, &lower, &upper, &[], 0.0)
                .unwrap()
                .0,
        );
        assert_optimal(&plain, (&c, &a, &b), (&lower, &upper));
        for perturbation in [PERTURBATION, 1e-2] {
            let solution = optimal(
                solve_perturbing(&c, &a, &b, &lower, &upper, &[], perturbation)
                    .unwrap()
                    .0,
            );
            assert_optimal(&solution, (&c, &a, &b), (&lower, &upper));
            assert!((solution.objective - plain.objective).abs() < 1e-12);
        }
    }

    #[test]
    fn a_degenerate_start_reaches_the_same_optimum_as_a_better_one() {
        // Min CVaR's dual (see `MinCvar`), long-only, for 30 assets over 60 scenarios: a row
        // −Rᵀz − ν − λ⁺ᵢ + λ⁻ᵢ = 0 per asset, and Σz = 1, maximizing ν − Σλ⁻. With every zₜ at
        // 0, the asset rows start without a residual, a degenerate vertex. Starting the worst
        // three scenarios at their cap of 1/3, as `MinCvar` does, avoids it.
        let (n, t) = (30, 60);
        let mut state = 7u64;
        let mut draw = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let returns = DMatrix::from_fn(t, n, |_, _| 0.04 * draw());
        let columns = t + 1 + 2 * n;
        let mut a = DMatrix::zeros(n + 1, columns);
        for i in 0..n {
            for s in 0..t {
                a[(i, s)] = -returns[(s, i)];
            }
            a[(i, t)] = -1.0;
            a[(i, t + 1 + i)] = -1.0;
            a[(i, t + 1 + n + i)] = 1.0;
        }
        a.row_mut(n).columns_mut(0, t).fill(1.0);
        let mut b = DVector::zeros(n + 1);
        b[n] = 1.0;
        let c = DVector::from_fn(columns, |j, _| match j {
            _ if j == t => -1.0,
            _ if j > t + n => 1.0,
            _ => 0.0,
        });
        let lower = DVector::from_fn(columns, |j, _| if j == t { -INF } else { 0.0 });
        let upper = DVector::from_fn(columns, |j, _| if j < t { 1.0 / 3.0 } else { INF });
        let mut losses: Vec<usize> = (0..t).collect();
        losses.sort_by(|&p, &q| returns.row(p).sum().total_cmp(&returns.row(q).sum()));

        let plain = optimal(solve(&c, &a, &b, &lower, &upper).unwrap());
        let tail = optimal(solve_from(&c, &a, &b, &lower, &upper, &losses[..3]).unwrap());
        assert_optimal(&plain, (&c, &a, &b), (&lower, &upper));
        assert_optimal(&tail, (&c, &a, &b), (&lower, &upper));
        assert!(
            (plain.objective - tail.objective).abs() < 1e-14,
            "{} vs {}",
            plain.objective,
            tail.objective
        );
    }

    #[test]
    fn beales_example_shifted_off_zero_terminates() {
        // The same problem in x − l: nonzero bounds make degenerate steps come out as rounding
        // noise rather than exactly zero, which mustn't count as progress.
        let a = dmatrix![
            0.25, -60.0, -0.04, 9.0, 1.0, 0.0, 0.0;
            0.5, -90.0, -0.02, 3.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        ];
        let c = dvector![-0.75, 150.0, -0.02, 6.0, 0.0, 0.0, 0.0];
        let l = dvector![
            -4.105497651910723,
            -2.782490163151331,
            0.4245830870739675,
            1.5584902031795,
            0.13554998649659034,
            3.5429669233119574,
            -2.4326451470286417
        ];
        let solution = optimal(
            solve(
                &c,
                &a,
                &(&a * &l + dvector![0.0, 0.0, 1.0]),
                &l,
                &DVector::from_element(7, INF),
            )
            .unwrap(),
        );
        assert!((solution.objective - (c.dot(&l) - 0.05)).abs() < 1e-12);
    }

    #[test]
    fn infeasible_and_unbounded_problems_are_reported() {
        let a = dmatrix![1.0, 1.0];
        let bounds = (dvector![0.0, 0.0], dvector![1.0, 1.0]);
        assert_eq!(
            solve(
                &dvector![1.0, 1.0],
                &a,
                &dvector![3.0],
                &bounds.0,
                &bounds.1
            )
            .unwrap(),
            Outcome::Infeasible
        );
        assert_eq!(
            solve(
                &dvector![-1.0, 0.0],
                &dmatrix![1.0, -1.0],
                &dvector![0.0],
                &dvector![0.0, 0.0],
                &dvector![INF, INF],
            )
            .unwrap(),
            Outcome::Unbounded
        );
    }

    #[test]
    fn free_variables_and_upper_bounds() {
        // minimize x − y subject to x + y = 1, with x free and y ≤ 5: y = 5, x = −4.
        let solution = optimal(
            solve(
                &dvector![1.0, -1.0],
                &dmatrix![1.0, 1.0],
                &dvector![1.0],
                &dvector![-INF, -INF],
                &dvector![INF, 5.0],
            )
            .unwrap(),
        );
        assert!((solution.x[0] + 4.0).abs() < 1e-14);
        assert!((solution.x[1] - 5.0).abs() < 1e-14);
        // The multiplier prices the constraint: c₀ − y = 0 for the basic free variable.
        assert!((solution.y[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn a_fixed_variable_stays_fixed() {
        let solution = optimal(
            solve(
                &dvector![-1.0, -1.0],
                &dmatrix![1.0, 1.0],
                &dvector![1.0],
                &dvector![0.25, 0.0],
                &dvector![0.25, 1.0],
            )
            .unwrap(),
        );
        assert_eq!(solution.x[0], 0.25);
        assert!((solution.x[1] - 0.75).abs() < 1e-15);
    }

    #[test]
    fn redundant_rows_are_fine() {
        // The second row is twice the first.
        let solution = optimal(
            solve(
                &dvector![1.0, 2.0],
                &dmatrix![1.0, 1.0; 2.0, 2.0],
                &dvector![1.0, 2.0],
                &dvector![0.0, 0.0],
                &dvector![INF, INF],
            )
            .unwrap(),
        );
        assert!((solution.x[0] - 1.0).abs() < 1e-15);
        assert!((solution.objective - 1.0).abs() < 1e-15);
    }

    #[test]
    fn a_redundant_row_keeps_its_artificial_variable() {
        // The second row is twice the first, so nothing can take its artificial variable's
        // place, and the basis isn't a start for a later solve.
        let (outcome, basis) = solve_warm(
            &dvector![1.0, 2.0],
            &dmatrix![1.0, 1.0; 2.0, 2.0],
            &dvector![1.0, 2.0],
            &dvector![0.0, 0.0],
            &dvector![INF, INF],
            None,
        )
        .unwrap();
        assert!((optimal(outcome).objective - 1.0).abs() < 1e-15);
        assert!(basis.is_none());
    }

    #[test]
    fn rows_met_at_bounds_leave_no_artificial_variable_basic() {
        // MinCdar's program at a target only one portfolio reaches, from the CI case: the
        // budget and the target pin the weights at (1, 0), each at a bound, so phase 1 ends
        // with both rows' artificial variables basic at zero. In phase 2, a change of 1e-14 in
        // the target's row, only rounding, was all that blocked a step, and pivoting on it left
        // the basis singular.
        let mu = [0.010448579610984887, -0.015468362554214288];
        let a = dmatrix![
            1.0, 1.0, 0.0, 0.0, 0.0;
            mu[0], mu[1], 0.0, 0.0, 0.0;
            0.36147408335656184, 0.6526561616879752, -1.0, 1.0, 0.0;
            0.374356670354395, 0.5057168653383007, -1.0, 0.0, 1.0
        ];
        let (outcome, basis) = solve_warm(
            &dvector![0.0, 0.0, 1.0, 0.0, 0.0],
            &a,
            &dvector![1.0, mu[0], 0.0, 0.0],
            &dvector![0.0, 0.0, -INF, 0.0, 0.0],
            &dvector![1.0, 1.0, INF, INF, INF],
            None,
        )
        .unwrap();
        let solution = optimal(outcome);
        // A weight that ends basic is solved for, and can be a unit in the last place off.
        assert!(
            (solution.x[0] - 1.0).abs() <= 2.0 * f64::EPSILON,
            "{}",
            solution.x
        );
        assert!(solution.x[1].abs() <= 2.0 * f64::EPSILON, "{}", solution.x);
        assert!((solution.objective - 0.374356670354395).abs() <= 1e-15);
        // Every basic variable is the problem's own, so a later round can start here.
        assert!(basis.unwrap().basis.iter().all(|&j| j < 5));
    }

    #[test]
    fn the_largest_entry_takes_an_artificial_variables_place() {
        // x₀ + 4x₁ = 5 within [0, 1]: phase 1 raises x₁ to its cap and then x₀ to its own,
        // where the row is met, and leaves the row's artificial variable basic at zero, with
        // nothing that could lower it further. Either could take its place, and x₁, with the
        // larger entry, does. With no costs, nothing moves after. The row is scaled to
        // x₀/4 + x₁ = 5/4, so each step is exact.
        let (outcome, basis) = solve_warm(
            &dvector![0.0, 0.0],
            &dmatrix![1.0, 4.0],
            &dvector![5.0],
            &dvector![0.0, 0.0],
            &dvector![1.0, 1.0],
            None,
        )
        .unwrap();
        assert_eq!(optimal(outcome).x.as_slice(), [1.0, 1.0]);
        assert_eq!(basis.unwrap().basis, [1]);
    }

    #[test]
    fn an_artificial_variable_leaves_only_if_that_moves_nothing() {
        // c₀x₀ + c₁x₁ + a = b with a basic, x₀ at its floor and x₁ at 0: the one with the larger
        // entry would take a's place, and move by a over its entry. Phase 1 can end with a
        // anywhere within its tolerance of zero.
        let problem = |a: f64, c: [f64; 2], floor: f64| Problem {
            a: dmatrix![c[0], c[1], 1.0],
            a_abs: dmatrix![c[0].abs(), c[1].abs(), 1.0],
            b: dvector![a],
            lower: dvector![floor, 0.0, 0.0],
            upper: dvector![INF, INF, INF],
            x: dvector![floor, 0.0, a],
            basis: vec![2],
            status: vec![Status::Lower, Status::Lower, Status::Basic],
            perturbation: PERTURBATION,
            repairs: usize::MAX,
        };
        let basis_after = |a: f64, c: [f64; 2], floor: f64| {
            let mut problem = problem(a, c, floor);
            problem.drive_out_artificials(2).unwrap();
            if problem.basis != [2] {
                assert_eq!(problem.status[2], Status::Lower);
                assert_eq!(problem.x[2], 0.0);
            }
            problem.basis
        };
        // At rounding of zero, it leaves, and x₁ moves by 5e-18.
        assert_eq!(basis_after(1e-17, [1.0, 2.0], 0.0), [1]);
        // At 1e-10, within phase 1's tolerance but not rounding, x₁ would move by 5e-11, and it
        // stays.
        assert_eq!(basis_after(1e-10, [1.0, 2.0], 0.0), [2]);
        // With an entry of 2000, x₁ would move by only 5e-14, but a itself would still have
        // 1e-10 to go to reach zero, and it stays too.
        assert_eq!(basis_after(1e-10, [1.0, 2000.0], 0.0), [2]);
        // At 1e-13, a is rounding, but over an entry of 0.01, x₀ would move by 1e-11, which
        // isn't, and it stays.
        assert_eq!(basis_after(1e-13, [0.01, 0.001], 0.0), [2]);
        // Rounding is relative to the variable: x₀ at a floor of 1000 moves by 1e-10, within
        // rounding of it, and it leaves.
        assert_eq!(basis_after(1e-13, [0.001, 0.0], 1000.0), [0]);
        // At exactly zero, nothing moves, but entries within rounding of zero, below 1e-9 of
        // the artificial's own, don't pivot.
        assert_eq!(basis_after(0.0, [1e-12, 1e-13], 0.0), [2]);
        // Nor does one at exactly 1e-9 of it. Just above, it does.
        assert_eq!(basis_after(0.0, [0.0, 1e-9], 0.0), [2]);
        assert_eq!(basis_after(0.0, [0.0, 2e-9], 0.0), [1]);
    }

    #[test]
    fn an_entry_within_rounding_of_its_rows_largest_takes_no_artificial_variables_place() {
        // x₀ is basic in the second row, on an entry of 1e-3, and the first row keeps its
        // artificial variable, at zero. That row of B⁻¹A has −1000 under the second row's
        // artificial variable and 1e-7 under x₁, which would count against the artificial's
        // own entry of 1. Against the row's largest it's within 1e-9, and doesn't pivot.
        let a = dmatrix![1.0, 1e-7, 1.0, 0.0; 1e-3, 0.0, 0.0, 1.0];
        let mut problem = Problem {
            a_abs: a.abs(),
            a,
            b: DVector::zeros(2),
            lower: DVector::zeros(4),
            upper: DVector::from_element(4, INF),
            x: DVector::zeros(4),
            basis: vec![2, 0],
            status: vec![Status::Basic, Status::Lower, Status::Basic, Status::Lower],
            perturbation: PERTURBATION,
            repairs: usize::MAX,
        };
        problem.drive_out_artificials(2).unwrap();
        assert_eq!(problem.basis, [2, 0]);
    }

    /// Checks that `solution` is optimal for the problem: feasible, with reduced costs of the
    /// right sign at each bound and zero between, so that the dual objective equals the primal.
    fn assert_optimal(
        solution: &Solution,
        (c, a, b): (&DVector<f64>, &DMatrix<f64>, &DVector<f64>),
        (lower, upper): (&DVector<f64>, &DVector<f64>),
    ) {
        let x = &solution.x;
        let scale = a.amax() * x.amax().max(1.0) + b.amax().max(1.0);
        assert!((a * x - b).amax() <= 1e-9 * scale, "infeasible: {x:?}");
        let reduced = c - a.transpose() * &solution.y;
        let tolerance = 1e-9 * (c.amax() + a.amax() * solution.y.amax()).max(1.0);
        let mut dual = b.dot(&solution.y);
        for j in 0..x.len() {
            assert!(
                x[j] >= lower[j] - 1e-12 && x[j] <= upper[j] + 1e-12,
                "x{j} out of bounds"
            );
            let at_lower = (x[j] - lower[j]).abs() <= 1e-9 * scale;
            let at_upper = (x[j] - upper[j]).abs() <= 1e-9 * scale;
            match (at_lower, at_upper) {
                (true, true) => {}
                (true, false) => assert!(reduced[j] >= -tolerance, "x{j}: {}", reduced[j]),
                (false, true) => assert!(reduced[j] <= tolerance, "x{j}: {}", reduced[j]),
                (false, false) => assert!(reduced[j].abs() <= tolerance, "x{j}: {}", reduced[j]),
            }
            dual += reduced[j] * x[j];
        }
        assert!((dual - solution.objective).abs() <= 1e-9 * scale * c.amax().max(1.0));
    }

    /// A program: its costs, rows, right-hand sides, and lower and upper bounds.
    type Program = (
        DVector<f64>,
        DMatrix<f64>,
        DVector<f64>,
        DVector<f64>,
        DVector<f64>,
    );

    /// The small program above, then the same with a row more, `y ≤ 0.5` with its slack: its
    /// optimum moves from (3, 1) to (3, 0.5).
    fn grown() -> [Program; 2] {
        [
            (
                dvector![-3.0, -2.0, 0.0, 0.0],
                dmatrix![1.0, 1.0, 1.0, 0.0; 1.0, 3.0, 0.0, 1.0],
                dvector![4.0, 6.0],
                dvector![0.0, 0.0, 0.0, 0.0],
                dvector![3.0, INF, INF, INF],
            ),
            (
                dvector![-3.0, -2.0, 0.0, 0.0, 0.0],
                dmatrix![
                    1.0, 1.0, 1.0, 0.0, 0.0;
                    1.0, 3.0, 0.0, 1.0, 0.0;
                    0.0, 1.0, 0.0, 0.0, 1.0
                ],
                dvector![4.0, 6.0, 0.5],
                dvector![0.0, 0.0, 0.0, 0.0, 0.0],
                dvector![3.0, INF, INF, INF, INF],
            ),
        ]
    }

    #[test]
    fn a_warm_start_from_an_earlier_optimum_with_a_new_row_reaches_the_new_optimum() {
        let [(c, a, b, lower, upper), (c2, a2, b2, lower2, upper2)] = grown();
        let (first, basis) = solve_warm(&c, &a, &b, &lower, &upper, None).unwrap();
        assert!((optimal(first).objective + 11.0).abs() < 1e-14);
        let mut basis = basis.expect("the optimum's basis has no artificial variable");
        basis.add_row();
        // The earlier optimum violates the new row, y = 1 > 0.5, so its slack starts at −0.5,
        // and the warm start itself, not the fallback, repairs it.
        let (warm, next) = from_basis(&c2, &a2, &b2, &lower2, &upper2, &basis, usize::MAX).unwrap();
        let warm = optimal(warm);
        assert!((warm.x[0] - 3.0).abs() < 1e-14 && (warm.x[1] - 0.5).abs() < 1e-14);
        assert!((warm.objective + 10.0).abs() < 1e-14);
        assert_optimal(&warm, (&c2, &a2, &b2), (&lower2, &upper2));
        assert!(next.is_some());
        let cold = optimal(solve(&c2, &a2, &b2, &lower2, &upper2).unwrap());
        assert!((warm.objective - cold.objective).abs() < 1e-14);
    }

    #[test]
    fn a_warm_start_gives_up_past_its_repair_cap() {
        let [(c, a, b, lower, upper), (c2, a2, b2, lower2, upper2)] = grown();
        let (_, basis) = solve_warm(&c, &a, &b, &lower, &upper, None).unwrap();
        let mut basis = basis.unwrap();
        basis.add_row();
        // Bringing the new row in takes one dual pivot. Allowed none, the warm start gives up,
        // for `solve_warm` to solve from scratch; allowed one, it finishes.
        let warm = |repairs: usize| from_basis(&c2, &a2, &b2, &lower2, &upper2, &basis, repairs);
        assert!(warm(0).is_none());
        assert!(warm(1).is_some());
        // And a solve from scratch that repairs its restored bounds has no cap: see
        // `restoring_wide_bounds_is_repaired_by_dual_pivots`.
    }

    #[test]
    fn a_basis_that_cant_start_falls_back_to_solving_from_scratch() {
        let [(c, a, b, lower, upper), (c2, a2, b2, lower2, upper2)] = grown();
        let (_, basis) = solve_warm(&c, &a, &b, &lower, &upper, None).unwrap();
        let basis = basis.unwrap();
        let cold = optimal(solve(&c2, &a2, &b2, &lower2, &upper2).unwrap());
        // Without the new row's slack, the basis has a row too few for the grown program.
        assert!(from_basis(&c2, &a2, &b2, &lower2, &upper2, &basis, usize::MAX).is_none());
        let (outcome, _) = solve_warm(&c2, &a2, &b2, &lower2, &upper2, Some(&basis)).unwrap();
        assert_eq!(optimal(outcome).objective, cold.objective);
        // A variable at a bound it doesn't have: y, s₀ and s₁ have no upper bound, and at the
        // optimum (3, 1) one of them is nonbasic.
        let mut unbounded = basis.clone();
        unbounded.add_row();
        let j = (1..4)
            .find(|&j| unbounded.status[j] != Status::Basic)
            .expect("a nonbasic variable with no upper bound");
        unbounded.status[j] = Status::Upper;
        assert!(from_basis(&c2, &a2, &b2, &lower2, &upper2, &unbounded, usize::MAX).is_none());
        // A basic variable marked at a bound, with a nonbasic one marked basic in its place: the
        // basis itself is fine and the count of basic variables right, but the two disagree.
        let mut mislabeled = basis.clone();
        mislabeled.add_row();
        let k = (0..5)
            .find(|&k| mislabeled.status[k] != Status::Basic)
            .expect("a nonbasic variable");
        mislabeled.status[k] = Status::Basic;
        mislabeled.status[mislabeled.basis[0]] = Status::Lower;
        assert!(from_basis(&c2, &a2, &b2, &lower2, &upper2, &mislabeled, usize::MAX).is_none());
        // A basis naming a column the program doesn't have.
        let mut beyond = basis.clone();
        beyond.add_row();
        beyond.basis[0] = 5;
        assert!(from_basis(&c2, &a2, &b2, &lower2, &upper2, &beyond, usize::MAX).is_none());
        // A basis with one column twice, and so a variable marked basic outside it.
        let mut repeated = basis;
        repeated.add_row();
        repeated.basis[1] = repeated.basis[0];
        assert!(from_basis(&c2, &a2, &b2, &lower2, &upper2, &repeated, usize::MAX).is_none());
        let (outcome, _) = solve_warm(&c2, &a2, &b2, &lower2, &upper2, Some(&repeated)).unwrap();
        assert_eq!(optimal(outcome).objective, cold.objective);
        // A basis that's consistent but singular: x's column is s₀'s plus s₁'s, so the
        // factorization can't solve with it.
        let singular = Basis {
            basis: vec![0, 2, 3],
            status: vec![
                Status::Basic,
                Status::Lower,
                Status::Basic,
                Status::Basic,
                Status::Lower,
            ],
        };
        assert!(from_basis(&c2, &a2, &b2, &lower2, &upper2, &singular, usize::MAX).is_none());
        let (outcome, _) = solve_warm(&c2, &a2, &b2, &lower2, &upper2, Some(&singular)).unwrap();
        assert_eq!(optimal(outcome).objective, cold.objective);
    }

    #[test]
    fn a_warm_start_holds_each_nonbasic_variable_at_a_bound_it_has() {
        // minimize x subject to x + z = 2: x ≤ 5, y free and in no row, z ≥ 0 basic, w ≥ 0 with
        // no upper bound and in no row, and v ≤ 0 with no lower bound and in no row.
        let c = dvector![1.0, 0.0, 0.0, 0.0, 0.0];
        let a = dmatrix![1.0, 0.0, 1.0, 0.0, 0.0];
        let b = dvector![2.0];
        let lower = dvector![0.0, -INF, 0.0, 0.0, -INF];
        let upper = dvector![5.0, INF, INF, INF, 0.0];
        let start = |status: [Status; 5]| Basis {
            basis: vec![2],
            status: status.to_vec(),
        };
        use Status::{Basic, Free, Lower, Upper};
        let cold = optimal(solve(&c, &a, &b, &lower, &upper).unwrap());
        // Each at a bound it has, and y free: the warm start itself solves it.
        let valid = start([Lower, Free, Basic, Lower, Upper]);
        let (warm, _) = from_basis(&c, &a, &b, &lower, &upper, &valid, usize::MAX).unwrap();
        assert_eq!(optimal(warm).objective, cold.objective);
        // Free, though it has a bound: x has both, w one. At a lower bound v doesn't have.
        for status in [
            [Free, Free, Basic, Lower, Upper],
            [Lower, Free, Basic, Free, Upper],
            [Lower, Free, Basic, Lower, Lower],
        ] {
            let bad = start(status);
            assert!(
                from_basis(&c, &a, &b, &lower, &upper, &bad, usize::MAX).is_none(),
                "{status:?}"
            );
            let (outcome, _) = solve_warm(&c, &a, &b, &lower, &upper, Some(&bad)).unwrap();
            assert_eq!(optimal(outcome).objective, cold.objective, "{status:?}");
        }
        // A status for each of too few variables, with the right number of rows.
        let short = Basis {
            basis: vec![2],
            status: vec![Lower, Free, Basic],
        };
        assert!(from_basis(&c, &a, &b, &lower, &upper, &short, usize::MAX).is_none());
        // No rows at all.
        let none = Basis {
            basis: vec![],
            status: vec![Lower, Free, Lower, Lower, Upper],
        };
        let rowless = DMatrix::zeros(0, 5);
        assert!(from_basis(&c, &rowless, &dvector![], &lower, &upper, &none, usize::MAX).is_none());
    }

    #[test]
    fn a_basis_with_an_artificial_variable_is_no_start() {
        // The problem's own variables are 0 and 1; 2 is the first artificial one.
        let problem = |basis: Vec<usize>| Problem {
            a: DMatrix::zeros(1, 3),
            a_abs: DMatrix::zeros(1, 3),
            b: dvector![0.0],
            lower: dvector![0.0, 0.0, 0.0],
            upper: dvector![INF, INF, INF],
            x: dvector![0.0, 0.0, 0.0],
            basis,
            status: vec![Status::Basic, Status::Basic, Status::Basic],
            perturbation: PERTURBATION,
            repairs: usize::MAX,
        };
        assert!(problem(vec![2]).basis_of(2).is_none());
        let own = problem(vec![1]).basis_of(2).unwrap();
        assert_eq!(own.basis, [1]);
        assert_eq!(own.status.len(), 2);
    }

    #[test]
    fn matches_linprog() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/lp.json")).unwrap();
        let floats = |v: &serde_json::Value| -> Vec<f64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        };
        let bounds = |v: &serde_json::Value, missing: f64| -> DVector<f64> {
            DVector::from_iterator(
                v.as_array().unwrap().len(),
                v.as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap_or(missing)),
            )
        };
        let cases = fixture["cases"].as_array().unwrap();
        assert!(cases.len() >= 70);
        for (index, case) in cases.iter().enumerate() {
            let rows: Vec<Vec<f64>> = case["a"].as_array().unwrap().iter().map(floats).collect();
            let a = DMatrix::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j]);
            let c = DVector::from_vec(floats(&case["c"]));
            let b = DVector::from_vec(floats(&case["b"]));
            let lower = bounds(&case["lower"], -INF);
            let upper = bounds(&case["upper"], INF);
            let outcome = solve(&c, &a, &b, &lower, &upper).unwrap();
            match (case["outcome"].as_str().unwrap(), outcome) {
                ("optimal", Outcome::Optimal(solution)) => {
                    let expected = case["objective"].as_f64().unwrap();
                    assert!(
                        (solution.objective - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                        "case {index}: {} vs {expected}",
                        solution.objective
                    );
                    assert_optimal(&solution, (&c, &a, &b), (&lower, &upper));
                }
                ("infeasible", Outcome::Infeasible) | ("unbounded", Outcome::Unbounded) => {}
                (expected, got) => panic!("case {index}: expected {expected}, got {got:?}"),
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    proptest::proptest! {
        #[test]
        fn optima_satisfy_the_optimality_conditions(
            m in 1usize..5,
            extra in 1usize..6,
            seed in proptest::collection::vec(-1.0f64..1.0, 200),
        ) {
            // A feasible, bounded problem: b = Ax₀ for x₀ within the bounds, and every
            // variable boxed below, with nonnegative costs on those unbounded above.
            let n = m + extra;
            let mut values = seed.into_iter().cycle();
            let mut next = || values.next().unwrap();
            let a = DMatrix::from_fn(m, n, |_, _| (next() * 4.0).round() / 2.0);
            let lower = DVector::from_fn(n, |_, _| next());
            let upper = DVector::from_fn(n, |j, _| {
                let width = next();
                if width < -0.3 { INF } else { lower[j] + 1.0 + width }
            });
            let x0 = DVector::from_fn(n, |j, _| lower[j] + 0.5);
            let b = &a * x0;
            let c = DVector::from_fn(n, |j, _| {
                let cost = next();
                if upper[j].is_infinite() { cost.abs() } else { cost }
            });
            let solution = optimal(solve(&c, &a, &b, &lower, &upper).unwrap());
            assert_optimal(&solution, (&c, &a, &b), (&lower, &upper));
        }
    }

    #[test]
    fn tiny_changes_still_block_at_their_bounds() {
        // x₀ = 1e-12·x₁ is a change far below the pivot tolerance, but a step of 1e8 in x₁
        // carries x₀ to 1e-4, past its cap of 1e-6. It must block there instead.
        let solution = optimal(
            solve(
                &dvector![0.0, -1.0, 0.0],
                &dmatrix![1.0, -1e-12, 0.0; 0.0, -1e3, 1.0],
                &dvector![0.0, 0.0],
                &dvector![0.0, 0.0, -INF],
                &dvector![1e-6, 1e8, INF],
            )
            .unwrap(),
        );
        assert!(solution.x[0] <= 1e-6 * (1.0 + 1e-12), "{:?}", solution.x);
        assert!((solution.x[1] - 1e6).abs() <= 1e-6, "{:?}", solution.x);
    }

    /// Where `x` ends when `x₀` is raised from zero as far as it goes: maximizes `x₀ ≤ cap`
    /// subject to `column·x₀ + s = b`, from the basis of the slacks `s`, which have the bounds
    /// given. So the first pivot's ratio test has `column` to weigh, with no phase 1 before it.
    fn raised<const M: usize>(
        column: [f64; M],
        b: [f64; M],
        cap: f64,
        (lower, upper): ([f64; M], [f64; M]),
    ) -> DVector<f64> {
        let a = DMatrix::from_fn(M, M + 1, |i, j| match j {
            0 => column[i],
            _ => f64::from(j == i + 1),
        });
        let with_slacks = |own: f64, slacks: [f64; M]| {
            DVector::from_fn(M + 1, |j, _| if j == 0 { own } else { slacks[j - 1] })
        };
        let mut problem = Problem {
            a_abs: a.abs(),
            a,
            b: DVector::from_column_slice(&b),
            lower: with_slacks(0.0, lower),
            upper: with_slacks(cap, upper),
            x: DVector::zeros(M + 1),
            basis: (1..=M).collect(),
            status: std::iter::once(Status::Lower)
                .chain(std::iter::repeat_n(Status::Basic, M))
                .collect(),
            perturbation: PERTURBATION,
            repairs: usize::MAX,
        };
        let cost = with_slacks(-1.0, [0.0; M]);
        let bounded = problem.minimize(&cost, Perturbation::Off).unwrap();
        assert!(bounded.is_some());
        problem.x
    }

    #[test]
    fn a_change_counts_against_the_largest_in_its_column() {
        // x₀ rises toward a cap of 1e-4 while the second slack, at its floor of 0, falls by
        // 1e-9 per unit of it. Against a largest change of 1 that's exactly at the pivot
        // tolerance, so it's rounding, and over so short a step the slack would end only 1e-13
        // below its floor: x₀ reaches its cap. Against a largest change of 0.01 it's a change
        // like any other, and blocks x₀ where it is.
        let x0 =
            |largest: f64| raised([largest, 1e-9], [1.0, 0.0], 1e-4, ([0.0, 0.0], [INF, INF]))[0];
        assert_eq!(x0(1.0), 1e-4);
        assert_eq!(x0(0.01), 0.0);
    }

    #[test]
    fn a_tiny_change_blocks_only_beyond_rounding_of_its_bound() {
        let floors = ([0.0, 0.0], [INF, INF]);
        // The second slack starts at its floor of 0 and falls by 1e-10 per unit of x₀, within
        // rounding of the largest change, 1. Over a step of 1e-4 it ends 1e-14 below its
        // floor, which is rounding too, and x₀ reaches its cap.
        assert_eq!(raised([1.0, 1e-10], [1.0, 0.0], 1e-4, floors)[0], 1e-4);
        // Rounding is up to 1e-12 of max(1, |x|). A step of 1 at 1e-12 per unit ends exactly
        // there, and isn't blocked. A step of 1.5 ends beyond it, and is, where it starts.
        assert_eq!(raised([1.0, 1e-12], [2.0, 0.0], 1.0, floors)[0], 1.0);
        assert_eq!(raised([1.0, 1e-12], [2.0, 0.0], 1.5, floors)[0], 0.0);
        // The same for a slack that rises from its cap of 0.
        let caps = ([0.0, -INF], [INF, 0.0]);
        assert_eq!(raised([1.0, -1e-12], [2.0, 0.0], 1.0, caps)[0], 1.0);
        assert_eq!(raised([1.0, -1e-12], [2.0, 0.0], 1.5, caps)[0], 0.0);
        // And relative to the variable: 1e-11 below a floor of 1000 is within rounding of it.
        let high = ([0.0, 1000.0], [INF, INF]);
        assert_eq!(raised([1.0, 1e-10], [1.0, 1000.0], 0.1, high)[0], 0.1);
    }

    #[test]
    fn a_tiny_change_within_rounding_of_its_bound_doesnt_block_when_another_does() {
        // x₀ can rise by 10 before the first slack reaches zero. The third falls by 1e-10 per
        // unit from 9.7e-10, and would end well below its floor of 0, so it blocks, at 9.7.
        // The second falls from just above its floor of 1000, by a change at the pivot
        // tolerance or below it, and would reach the floor at 9.5. But at 9.7, as at 10, it's
        // within rounding of 1000, so it doesn't block, though the third's blocking has the
        // ratio test taken again.
        for (change, above) in [(1e-9, 9.5e-9), (5e-10, 4.75e-9)] {
            let x = raised(
                [1.0, change, 1e-10],
                [10.0, 1000.0 + above, 9.7e-10],
                INF,
                ([0.0, 1000.0, 0.0], [INF; 3]),
            );
            assert!((x[0] - 9.7).abs() < 1e-9, "{change}: {x}");
        }
    }

    /// The problem [`Problem::minimize`] leaves after [`STALL`] pivots and then running off
    /// without limit. Each `xᵢ`, at zero, enters row `i` of `entry·xᵢ + sᵢ = b`, where its
    /// slack `sᵢ ≥ floor` leaves; they cost the most, so they go first. Then `w` and `u`, in
    /// no row, are left: `w` improves faster but stops at a cap of 1, and `u`, the
    /// lower-indexed, has nothing to stop it. An unbounded solve returns with the bounds as
    /// they are, so they show whether the run of pivots was counted as stalled, and `w`
    /// which rule chose after it.
    fn ray_after_a_run(entry: f64, b: f64, floor: f64, free: bool) -> Problem {
        // The columns: the xᵢ, their slacks, then u and w.
        let (m, u, total) = (STALL, 2 * STALL, 2 * STALL + 2);
        let mut a = DMatrix::zeros(m, total);
        for i in 0..m {
            a[(i, i)] = entry;
            a[(i, m + i)] = 1.0;
        }
        let per_kind = |x: f64, slack: f64, u_value: f64, w_value: f64| {
            DVector::from_fn(total, |j, _| match j {
                _ if j < m => x,
                _ if j < u => slack,
                _ if j == u => u_value,
                _ => w_value,
            })
        };
        let mut problem = Problem {
            a_abs: a.abs(),
            a,
            b: DVector::from_element(m, b),
            lower: per_kind(if free { -INF } else { 0.0 }, floor, 0.0, 0.0),
            upper: per_kind(INF, INF, INF, 1.0),
            x: DVector::zeros(total),
            basis: (m..u).collect(),
            status: (0..total)
                .map(|j| match j {
                    _ if j < m && free => Status::Free,
                    _ if (m..u).contains(&j) => Status::Basic,
                    _ => Status::Lower,
                })
                .collect(),
            perturbation: PERTURBATION,
            repairs: usize::MAX,
        };
        let cost = per_kind(-3.0, 0.0, -1.0, -2.0);
        assert_eq!(problem.minimize(&cost, Perturbation::Off), Ok(None));
        problem
    }

    #[test]
    fn a_step_stalls_when_the_leaving_variable_moves_within_rounding_of_nothing() {
        // After STALL stalled pivots the bounds of the basic variables, the xᵢ, are widened,
        // so x₀'s floor ends below zero only if its run was counted as stalled.
        let floor_of_x0 =
            |entry: f64, b: f64, floor: f64| ray_after_a_run(entry, b, floor, false).lower[0];
        // Each slack leaves from 0.5 above its floor: x moves, and no bound is widened.
        assert_eq!(floor_of_x0(1.0, 0.5, 0.0), 0.0);
        // From 1e-13 above, it moves by rounding only, though with an entry of 0.01 the step
        // that takes it there is 1e-11.
        assert!(floor_of_x0(0.01, 1e-13, 0.0) < 0.0);
        // Rounding is relative to the variable: from 1e-10 above a floor of 1000 it is too.
        assert!(floor_of_x0(1.0, 1000.0 + 1e-10, 1000.0) < 0.0);
    }

    #[test]
    fn blands_rule_takes_over_once_stall_pivots_have_stalled_with_no_bounds_to_widen() {
        let w = 2 * STALL + 1;
        // Free variables enter on STALL degenerate pivots, and have no bounds to widen. So
        // the lowest-indexed improving variable enters next: u, which runs off at once, and w
        // is left where it was.
        assert_eq!(ray_after_a_run(1.0, 0.0, 0.0, true).x[w], 0.0);
        // After pivots that moved x, the fastest-improving one does, and w reaches its cap
        // before u runs off.
        assert_eq!(ray_after_a_run(1.0, 0.5, 0.0, false).x[w], 1.0);
    }

    #[test]
    fn tiny_coefficients_are_feasible() {
        let solution = optimal(
            solve(
                &dvector![1.0, 2.0],
                &dmatrix![1e-10, 1e-10],
                &dvector![2e-20],
                &dvector![0.0, 0.0],
                &dvector![INF, INF],
            )
            .unwrap(),
        );
        assert!((solution.x[0] - 2e-10).abs() < 1e-24);
        // The multiplier is in the original units: c₀ = 1e-10·y.
        assert!((solution.y[0] - 1e10).abs() < 1e-3);
    }

    #[test]
    fn starting_at_upper_bounds_reaches_the_same_optimum() {
        // Pick at most two of four items, at most one of the last two, for the most value.
        let c = dvector![-3.0, -1.0, -2.0, -4.0, 0.0, 0.0];
        let a = dmatrix![
            1.0, 1.0, 1.0, 1.0, 1.0, 0.0;
            0.0, 0.0, 1.0, 1.0, 0.0, 1.0
        ];
        let b = dvector![2.0, 1.0];
        let lower = DVector::zeros(6);
        let upper = dvector![1.0, 1.0, 1.0, 1.0, INF, INF];
        let from_lower = optimal(solve(&c, &a, &b, &lower, &upper).unwrap());
        assert_optimal(&from_lower, (&c, &a, &b), (&lower, &upper));
        assert_eq!(from_lower.objective, -7.0);
        // Starting at the upper bound of variables 1 and 2, the wrong ones, and of variable 4,
        // which has none and so starts at its lower bound.
        let from_upper = optimal(solve_from(&c, &a, &b, &lower, &upper, &[1, 2, 4]).unwrap());
        assert_optimal(&from_upper, (&c, &a, &b), (&lower, &upper));
        assert_eq!(from_upper.x, from_lower.x);
        assert_eq!(from_upper.objective, -7.0);
    }

    #[test]
    fn no_rows_leaves_only_the_bounds() {
        let none = DMatrix::zeros(0, 3);
        let solution = optimal(
            solve(
                &dvector![1.0, -1.0, 0.0],
                &none,
                &DVector::zeros(0),
                &dvector![0.5, 0.0, -2.0],
                &dvector![1.0, 2.0, -1.0],
            )
            .unwrap(),
        );
        assert_eq!(solution.x, dvector![0.5, 2.0, -1.0]);
        assert_eq!(solution.objective, -1.5);
        assert_eq!(
            solve(
                &dvector![-1.0],
                &DMatrix::zeros(0, 1),
                &DVector::zeros(0),
                &dvector![0.0],
                &dvector![INF],
            )
            .unwrap(),
            Outcome::Unbounded
        );
    }

    #[test]
    fn bounds_that_cross_are_infeasible() {
        // x₁'s floor of 2 is above its cap of 1, so no x is within the bounds: with no rows,
        // where clamping 0 into them would panic, and with a row x₀ + x₁ = 3 that x₀ = 1 and
        // x₁ at its floor would otherwise meet.
        let (lower, upper) = (dvector![0.0, 2.0], dvector![INF, 1.0]);
        let c = dvector![1.0, 0.0];
        let none = DMatrix::zeros(0, 2);
        let infeasible = |a: &DMatrix<f64>, b: &DVector<f64>| {
            solve(&c, a, b, &lower, &upper).unwrap() == Outcome::Infeasible
        };
        assert!(infeasible(&none, &DVector::zeros(0)));
        assert!(infeasible(&dmatrix![1.0, 1.0], &dvector![3.0]));
        // So is a NaN bound, which no x is within either.
        assert_eq!(
            solve(
                &c,
                &none,
                &DVector::zeros(0),
                &dvector![0.0, f64::NAN],
                &upper
            )
            .unwrap(),
            Outcome::Infeasible
        );
    }
}
