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
    solve_perturbing(c, a, b, lower, upper, at_upper, PERTURBATION)
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
) -> Result<Outcome, CutupError> {
    let (m, n) = a.shape();
    if m == 0 {
        return Ok(unconstrained(c, lower, upper));
    }
    // Each row divided by its largest entry, so that the tolerances below compare like with
    // like whatever A's scale: row i of Ax = b becomes (aᵢ/sᵢ)x = bᵢ/sᵢ, and its multiplier is
    // divided by sᵢ to undo it. An all-zero row keeps scale 1.
    let scales = DVector::from_fn(m, |i, _| match a.row(i).amax() {
        s if s > 0.0 => s,
        _ => 1.0,
    });
    let a = &DMatrix::from_fn(m, n, |i, j| a[(i, j)] / scales[i]);
    let b = &b.component_div(&scales);
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
    };
    for i in 0..m {
        problem.x[n + i] = residual[i].abs();
    }

    let phase_one = DVector::from_fn(n + m, |j, _| if j >= n { 1.0 } else { 0.0 });
    // Phase 1 is bounded below by zero, and the ratio test never lets a variable past a
    // finite bound, so it always ends at an optimum: running off without limit would take
    // an artificial variable below zero.
    let Some(_) = problem.minimize(&phase_one)? else {
        return Err(CutupError::NotConverged);
    };
    let infeasibility: f64 = problem.x.rows(n, m).sum();
    let scale = a.amax() * problem.x.rows(0, n).amax() + b.amax();
    if infeasibility > 1e-9 * scale.max(f64::MIN_POSITIVE) {
        return Ok(Outcome::Infeasible);
    }
    for j in n..n + m {
        problem.upper[j] = 0.0;
        if problem.status[j] != Status::Basic {
            problem.x[j] = 0.0;
        }
    }

    let cost = c.clone().resize_vertically(n + m, 0.0);
    match problem.minimize(&cost)? {
        None => Ok(Outcome::Unbounded),
        Some(y) => {
            let x = problem.x.rows(0, n).into_owned();
            Ok(Outcome::Optimal(Solution {
                objective: c.dot(&x),
                x,
                y: y.component_div(&scales),
            }))
        }
    }
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
}

impl Problem {
    /// Minimizes `costᵀx` from the current basis, returning the multipliers `y` at the
    /// optimum, or `None` if the objective decreases without limit.
    fn minimize(&mut self, cost: &DVector<f64>) -> Result<Option<DVector<f64>>, CutupError> {
        let (m, total) = self.a.shape();
        // Pivots in a row that didn't move x. Past a few, the bounds are widened, or if they
        // can't be, Bland's rule takes over until one does: only a run of such pivots can
        // cycle, and Bland's can't.
        let mut stalled = 0;
        let mut factors = Factors::new(&self.a, &self.basis);
        // The problem's own bounds, to restore after perturbing them, and which variables'
        // bounds are widened.
        let (lower, upper) = (self.lower.clone(), self.upper.clone());
        let mut perturbed = vec![false; total];
        let mut perturbation = Perturbation::Off;
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
                    stalled = 0;
                    continue;
                }
                perturbation = Perturbation::Done;
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
        let alpha = self.a.tr_mul(&factors.solve_transpose(DVector::from_fn(
            self.basis.len(),
            |i, _| {
                if i == row { 1.0 } else { 0.0 }
            },
        ))?);
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
        self.status[k] = if rises { Status::Lower } else { Status::Upper };
        self.x[k] = bound;
        self.status[entering] = Status::Basic;
        self.basis[row] = entering;
        factors.update(&self.a, &self.basis, row, d);
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
    fn without_constraints_each_variable_takes_the_bound_its_cost_favors() {
        let bounds = (dvector![0.5, 0.0, -2.0, 1.0], dvector![3.0, 2.0, 5.0, 4.0]);
        let solution = optimal(
            solve(
                &dvector![1.0, -1.0, 0.0, 0.0],
                &DMatrix::zeros(0, 4),
                &DVector::zeros(0),
                &bounds.0,
                &bounds.1,
            )
            .unwrap(),
        );
        assert_eq!(solution.x.as_slice(), &[0.5, 2.0, 0.0, 1.0]);
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
        let solution = optimal(solve_perturbing(&c, &a, &b, &lower, &upper, &[], 0.0).unwrap());
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
        let plain = optimal(solve_perturbing(&c, &a, &b, &lower, &upper, &[], 0.0).unwrap());
        assert_optimal(&plain, (&c, &a, &b), (&lower, &upper));
        for perturbation in [PERTURBATION, 1e-2] {
            let solution =
                optimal(solve_perturbing(&c, &a, &b, &lower, &upper, &[], perturbation).unwrap());
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
}
