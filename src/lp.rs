//! Linear programs with bounded variables, solved by a two-phase revised simplex method.

use crate::CutupError;
use nalgebra::{DMatrix, DVector};

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
/// artificial variables at zero and minimizes `cᵀx`. Each iteration factors the basis afresh
/// and recomputes the basic variables from the nonbasic ones, so rounding doesn't accumulate.
///
/// The entering variable is the lowest-indexed one whose reduced cost improves the objective,
/// and ties in the ratio test go to the lowest index: Bland's rule, which can't cycle on a
/// degenerate vertex.
///
/// # Errors
///
/// [`CutupError::NotConverged`] if a basis can't be factored or the iteration cap is reached,
/// which only rounding trouble should cause.
pub(crate) fn solve(
    c: &DVector<f64>,
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lower: &DVector<f64>,
    upper: &DVector<f64>,
) -> Result<Outcome, CutupError> {
    let (m, n) = a.shape();
    // Nonbasic variables start at a finite bound, or at zero if they have none.
    let start = DVector::from_fn(n, |j, _| {
        if lower[j].is_finite() {
            lower[j]
        } else if upper[j].is_finite() {
            upper[j]
        } else {
            0.0
        }
    });
    let residual = b - a * &start;
    // Artificial variable i carries row i's residual, with the sign that makes it nonnegative.
    let mut columns = a.clone().resize_horizontally(n + m, 0.0);
    for i in 0..m {
        columns[(i, n + i)] = if residual[i] < 0.0 { -1.0 } else { 1.0 };
    }
    let mut problem = Problem {
        a: columns,
        b: b.clone(),
        lower: lower.clone().resize_vertically(n + m, 0.0),
        upper: upper.clone().resize_vertically(n + m, f64::INFINITY),
        x: start.resize_vertically(n + m, 0.0),
        basis: (n..n + m).collect(),
        status: (0..n + m)
            .map(|j| match j {
                j if j >= n => Status::Basic,
                j if lower[j].is_finite() => Status::Lower,
                j if upper[j].is_finite() => Status::Upper,
                _ => Status::Free,
            })
            .collect(),
    };
    for i in 0..m {
        problem.x[n + i] = residual[i].abs();
    }

    let phase_one = DVector::from_fn(n + m, |j, _| if j >= n { 1.0 } else { 0.0 });
    // Phase 1 is bounded below by zero, so it always ends at an optimum.
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
                y,
            }))
        }
    }
}

/// A linear program in the state the simplex method moves through.
struct Problem {
    a: DMatrix<f64>,
    b: DVector<f64>,
    lower: DVector<f64>,
    upper: DVector<f64>,
    x: DVector<f64>,
    /// The basic variable of each row.
    basis: Vec<usize>,
    status: Vec<Status>,
}

impl Problem {
    /// Minimizes `costᵀx` from the current basis, returning the multipliers `y` at the
    /// optimum, or `None` if the objective decreases without limit.
    fn minimize(&mut self, cost: &DVector<f64>) -> Result<Option<DVector<f64>>, CutupError> {
        let (m, total) = self.a.shape();
        for _ in 0..1000 + 100 * (m + total) {
            let basis = self.a.select_columns(&self.basis);
            let lu = basis.clone().lu();
            // The basic variables from the nonbasic ones, which sit exactly at their bounds.
            let nonbasic = DVector::from_fn(total, |j, _| {
                if self.status[j] == Status::Basic {
                    0.0
                } else {
                    self.x[j]
                }
            });
            let x_basis = lu
                .solve(&(&self.b - &self.a * nonbasic))
                .ok_or(CutupError::NotConverged)?;
            for (i, &j) in self.basis.iter().enumerate() {
                self.x[j] = x_basis[i];
            }
            let cost_basis = DVector::from_fn(m, |i, _| cost[self.basis[i]]);
            let y = basis
                .transpose()
                .lu()
                .solve(&cost_basis)
                .ok_or(CutupError::NotConverged)?;
            let reduced = cost - self.a.transpose() * &y;

            // Bland's rule: the lowest-indexed variable that improves the objective, and the
            // direction it moves in.
            let tolerance = 1e-9 * cost.amax().max(f64::MIN_POSITIVE);
            let entering = (0..total).find_map(|j| {
                let movable = self.upper[j] > self.lower[j];
                match self.status[j] {
                    Status::Lower if movable && reduced[j] < -tolerance => Some((j, 1.0)),
                    Status::Upper if movable && reduced[j] > tolerance => Some((j, -1.0)),
                    Status::Free if reduced[j].abs() > tolerance => Some((j, -reduced[j].signum())),
                    _ => None,
                }
            });
            let Some((j, sign)) = entering else {
                return Ok(Some(y));
            };

            // Moving x_j by sign·t moves the basic variables by t·change.
            let change = -lu
                .solve(&self.a.column(j).into_owned())
                .ok_or(CutupError::NotConverged)?
                * sign;
            let pivot = 1e-9 * change.amax();
            // The step at which each variable reaches a bound, the entering one included, by
            // variable index: the smallest step wins, and among equal steps the lowest index.
            let own = (self.upper[j] - self.lower[j])
                .is_finite()
                .then(|| (self.upper[j] - self.lower[j], j, None));
            let blocking = (0..m)
                .filter_map(|i| {
                    let k = self.basis[i];
                    let d = change[i];
                    if d < -pivot && self.lower[k].is_finite() {
                        Some((
                            ((self.x[k] - self.lower[k]) / -d).max(0.0),
                            k,
                            Some((i, Status::Lower)),
                        ))
                    } else if d > pivot && self.upper[k].is_finite() {
                        Some((
                            ((self.upper[k] - self.x[k]) / d).max(0.0),
                            k,
                            Some((i, Status::Upper)),
                        ))
                    } else {
                        None
                    }
                })
                .chain(own)
                .min_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
            let Some((step, _, leaving)) = blocking else {
                return Ok(None);
            };

            self.x[j] += sign * step;
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
                }
            }
        }
        Err(CutupError::NotConverged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dmatrix, dvector};

    const INF: f64 = f64::INFINITY;

    fn optimal(outcome: Outcome) -> Solution {
        match outcome {
            Outcome::Optimal(solution) => solution,
            other => panic!("expected an optimum, got {other:?}"),
        }
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
    fn beales_cycling_example_terminates() {
        // Beale (1955): the textbook rule cycles on this degenerate vertex; Bland's can't.
        let a = dmatrix![
            0.25, -60.0, -0.04, 9.0, 1.0, 0.0, 0.0;
            0.5, -90.0, -0.02, 3.0, 0.0, 1.0, 0.0;
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0
        ];
        let solution = optimal(
            solve(
                &dvector![-0.75, 150.0, -0.02, 6.0, 0.0, 0.0, 0.0],
                &a,
                &dvector![0.0, 0.0, 1.0],
                &DVector::zeros(7),
                &DVector::from_element(7, INF),
            )
            .unwrap(),
        );
        assert!((solution.objective + 0.05).abs() < 1e-14);
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
}
