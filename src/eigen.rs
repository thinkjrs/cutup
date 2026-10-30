//! Symmetric eigendecomposition by the cyclic Jacobi method.
//!
//! nalgebra's SVD and symmetric eigendecomposition can both return factors that reconstruct
//! the matrix only to 1e-3 or worse, on ordinary covariance and KKT matrices. Jacobi rotations
//! are backward stable: the factors reconstruct the matrix to within rounding of its norm, so
//! cutup uses them wherever it needs a pseudo-inverse or a null space. They start from
//! nalgebra's eigenvectors, which are close enough that a sweep or two usually finishes the
//! job.

use nalgebra::{DMatrix, DVector};

/// Sweeps allowed before giving up. Convergence is quadratic, so well under 20 are needed in
/// practice.
const MAX_SWEEPS: usize = 100;

/// Eigenvectors spanning a range, each with its eigenvalue, and ones spanning a null space.
type Split = (Vec<(f64, DVector<f64>)>, Vec<DVector<f64>>);

/// QR iterations per row allowed for nalgebra's eigendecomposition, which only supplies the
/// starting point, as in LAPACK. Past this, Jacobi starts from the identity instead.
const START_ITERATIONS_PER_ROW: usize = 30;

/// `A = V diag(λ) Vᵀ` for a symmetric `A`: eigenvalues `λ`, unordered, and orthonormal
/// eigenvectors in the matching columns of `V`.
pub(crate) struct SymmetricEigen {
    pub(crate) values: DVector<f64>,
    pub(crate) vectors: DMatrix<f64>,
}

impl SymmetricEigen {
    /// Decomposes the symmetric matrix `a` by cyclic Jacobi rotations (Golub and Van Loan,
    /// 2013, Algorithm 8.5.3). Each rotation zeroes one off-diagonal pair, and sweeps repeat
    /// until the off-diagonal part is below rounding of the whole.
    ///
    /// The rotations apply to `VᵀAV`, where `V` is nalgebra's eigenvector matrix made exactly
    /// orthonormal by a QR decomposition, or the identity if nalgebra doesn't converge. Any
    /// orthonormal `V` gives the same eigenvalues; a good one leaves little to rotate. `A` is
    /// first divided by its largest entry, so the stopping test can't overflow or underflow.
    ///
    /// Returns `None` if `a` has NaN or infinite entries, or the sweeps run out.
    pub(crate) fn new(a: &DMatrix<f64>) -> Option<Self> {
        let n = a.nrows();
        if a.iter().any(|x| !x.is_finite()) {
            return None;
        }
        let scale = a.amax();
        if scale == 0.0 {
            return Some(SymmetricEigen {
                values: DVector::zeros(n),
                vectors: DMatrix::identity(n, n),
            });
        }
        let a = a / scale;
        let start = a
            .clone()
            .try_symmetric_eigen(5.0 * f64::EPSILON, START_ITERATIONS_PER_ROW * n);
        let mut v = match start {
            Some(start) => start.eigenvectors.qr().q(),
            None => DMatrix::identity(n, n),
        };
        let rotated = v.transpose() * a * &v;
        let mut a = (&rotated + rotated.transpose()) * 0.5;
        let tolerance = f64::EPSILON * a.norm();

        for _ in 0..MAX_SWEEPS {
            let off_diagonal = (0..n)
                .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
                .map(|(p, q)| a[(p, q)] * a[(p, q)])
                .sum::<f64>();
            if off_diagonal.is_nan() {
                break;
            }
            if (2.0 * off_diagonal).sqrt() <= tolerance {
                return Some(SymmetricEigen {
                    values: a.diagonal() * scale,
                    vectors: v,
                });
            }
            for p in 0..n {
                for q in p + 1..n {
                    rotate(&mut a, &mut v, p, q);
                }
            }
        }
        None
    }

    /// For a positive semidefinite matrix, splits the eigenvectors into those of eigenvalues
    /// above `cutoff`, paired with their eigenvalue, which span the range, and the rest, which
    /// span the null space. Eigenvalues at or below `cutoff`, negative ones included, are
    /// rounding error.
    pub(crate) fn split(&self, cutoff: f64) -> Split {
        let (mut range, mut null) = (Vec::new(), Vec::new());
        for (k, &value) in self.values.iter().enumerate() {
            let vector = self.vectors.column(k).into_owned();
            if value > cutoff {
                range.push((value, vector));
            } else {
                null.push(vector);
            }
        }
        (range, null)
    }
}

/// Applies the Jacobi rotation that zeroes `a[(p, q)]`: `A ← JᵀAJ` and `V ← VJ`.
fn rotate(a: &mut DMatrix<f64>, v: &mut DMatrix<f64>, p: usize, q: usize) {
    let apq = a[(p, q)];
    if apq == 0.0 {
        return;
    }
    // t = tan θ is the smaller root of t² + 2τt − 1 = 0, which keeps the rotation below 45°.
    // When τ is so large that τ² overflows, t ≈ 1 / 2τ.
    let tau = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
    let t = if tau.abs() > 1e150 {
        0.5 / tau
    } else {
        tau.signum() / (tau.abs() + (tau * tau + 1.0).sqrt())
    };
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;

    let n = a.nrows();
    for k in 0..n {
        let (akp, akq) = (a[(k, p)], a[(k, q)]);
        a[(k, p)] = c * akp - s * akq;
        a[(k, q)] = s * akp + c * akq;
    }
    for k in 0..n {
        let (apk, aqk) = (a[(p, k)], a[(q, k)]);
        a[(p, k)] = c * apk - s * aqk;
        a[(q, k)] = s * apk + c * aqk;
    }
    a[(p, q)] = 0.0;
    a[(q, p)] = 0.0;
    for k in 0..n {
        let (vkp, vkq) = (v[(k, p)], v[(k, q)]);
        v[(k, p)] = c * vkp - s * vkq;
        v[(k, q)] = s * vkp + c * vkq;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dmatrix;
    use proptest::prelude::*;

    /// Checks that the decomposition reconstructs `a` and that `V` is orthonormal, both to
    /// within rounding.
    fn assert_decomposes(a: &DMatrix<f64>) {
        let n = a.nrows();
        let eigen = SymmetricEigen::new(a).unwrap();
        let v = &eigen.vectors;
        let rebuilt = v * DMatrix::from_diagonal(&eigen.values) * v.transpose();
        let scale = a.norm().max(f64::MIN_POSITIVE);
        assert!(
            (&rebuilt - a).norm() <= 1e-14 * scale * n as f64,
            "{a} rebuilt as {rebuilt}"
        );
        let identity = DMatrix::<f64>::identity(n, n);
        assert!((v.transpose() * v - identity).amax() <= 1e-14 * n as f64);
    }

    #[test]
    fn known_spectrum() {
        let eigen = SymmetricEigen::new(&dmatrix![2.0, 1.0; 1.0, 2.0]).unwrap();
        let mut values: Vec<f64> = eigen.values.iter().copied().collect();
        values.sort_by(f64::total_cmp);
        assert!((values[0] - 1.0).abs() < 1e-14 && (values[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn diagonal_and_empty_matrices() {
        let eigen = SymmetricEigen::new(&DMatrix::from_diagonal_element(3, 3, 4.0)).unwrap();
        assert!(
            eigen
                .values
                .iter()
                .all(|&value| (value - 4.0).abs() < 1e-14)
        );
        assert_decomposes(&DMatrix::from_diagonal(&DVector::from_column_slice(&[
            3.0, -1.0, 0.0, 2.0,
        ])));
        assert_eq!(
            SymmetricEigen::new(&DMatrix::zeros(0, 0))
                .unwrap()
                .values
                .len(),
            0
        );
        assert_decomposes(&DMatrix::zeros(4, 4));
    }

    #[test]
    fn rejects_non_finite_matrices() {
        let a = dmatrix![1.0, f64::NAN; f64::NAN, 1.0];
        assert!(SymmetricEigen::new(&a).is_none());
    }

    #[test]
    fn extreme_scales() {
        // At 1e300 the squared entries overflow, and at 1e-300 they underflow, unless the
        // matrix is scaled first.
        let a = dmatrix![2.0, 1.0; 1.0, 2.0];
        for scale in [1e300, 1e-300] {
            let eigen = SymmetricEigen::new(&(&a * scale)).unwrap();
            let mut values: Vec<f64> = eigen.values.iter().map(|v| v / scale).collect();
            values.sort_by(f64::total_cmp);
            assert!((values[0] - 1.0).abs() < 1e-14 && (values[1] - 3.0).abs() < 1e-14);
        }
    }

    #[test]
    fn split_puts_negative_rounding_in_the_null_space() {
        let eigen = SymmetricEigen {
            values: DVector::from_column_slice(&[2.0, -1e-3, 1e-16]),
            vectors: DMatrix::identity(3, 3),
        };
        let (range, null) = eigen.split(1e-15);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].0, 2.0);
        assert_eq!(null.len(), 2);
    }

    #[test]
    fn singular_and_indefinite_matrices() {
        // Rank one: one eigenvalue of 14, the rest zero.
        let x = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let rank_one = &x * x.transpose();
        assert_decomposes(&rank_one);
        let eigen = SymmetricEigen::new(&rank_one).unwrap();
        assert!((eigen.values.amax() - 14.0).abs() < 1e-13);
        assert!((eigen.values.sum() - 14.0).abs() < 1e-13);

        // A KKT matrix, which has negative eigenvalues.
        assert_decomposes(&dmatrix![
            1.0, 0.5, 1.0;
            0.5, 2.0, 1.0;
            1.0, 1.0, 0.0
        ]);
    }

    proptest! {
        #[test]
        fn decomposes_random_symmetric_matrices(
            n in 1usize..12,
            values in prop::collection::vec(-1.0f64..1.0, 144),
            rank in 0usize..12,
            exponent in -8i32..8,
        ) {
            // Gram matrices of `rank` random vectors, so many are singular, plus an
            // indefinite symmetric part, at varied scales.
            let x = DMatrix::from_fn(n, rank.min(n), |i, j| values[i * 12 + j]);
            let mut a = &x * x.transpose();
            if rank % 2 == 1 {
                a += DMatrix::from_fn(n, n, |i, j| values[(i + j) % 144] * 0.1);
            }
            a *= 10f64.powi(exponent);
            assert_decomposes(&a);
        }
    }
}
