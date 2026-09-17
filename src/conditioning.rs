//! Stable Gaussian conditioning through Cholesky factorization.

use nalgebra::{DMatrix, DVector, Dyn, linalg::Cholesky};

use crate::ConditioningError;

/// Reusable factorization of a symmetric positive-definite linear system.
///
/// The constructor accepts a dense matrix in row-major order, optionally adds a
/// fixed non-negative jitter value to its diagonal, and computes a Cholesky
/// factorization. Solves reuse that factorization and never form an explicit
/// matrix inverse.
#[derive(Debug, Clone)]
pub struct GaussianConditioner {
    dimension: usize,
    jitter: f64,
    cholesky: Cholesky<f64, Dyn>,
}

impl GaussianConditioner {
    /// Construct a Gaussian conditioning system from a flattened square matrix.
    ///
    /// `matrix` is interpreted in row-major order. The supplied `jitter` is
    /// added exactly once to each diagonal entry before factorization.
    ///
    /// # Errors
    ///
    /// Returns [`ConditioningError`] when dimensions are inconsistent, values
    /// are non-finite, jitter is negative, or Cholesky factorization fails.
    pub fn new(matrix: &[f64], dimension: usize, jitter: f64) -> Result<Self, ConditioningError> {
        if dimension == 0 {
            return Err(ConditioningError::ZeroDimension);
        }

        let expected_len = dimension
            .checked_mul(dimension)
            .ok_or(ConditioningError::MatrixDimensionMismatch)?;
        if matrix.len() != expected_len {
            return Err(ConditioningError::MatrixDimensionMismatch);
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(ConditioningError::NonFiniteMatrixEntry);
        }
        if !jitter.is_finite() {
            return Err(ConditioningError::NonFiniteJitter);
        }
        if jitter < 0.0 {
            return Err(ConditioningError::NegativeJitter);
        }

        let mut regularized = DMatrix::from_row_slice(dimension, dimension, matrix);
        if jitter > 0.0 {
            for index in 0..dimension {
                regularized[(index, index)] += jitter;
            }
        }

        let cholesky = Cholesky::new(regularized).ok_or(ConditioningError::NotPositiveDefinite)?;

        Ok(Self {
            dimension,
            jitter,
            cholesky,
        })
    }

    /// Return the system dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the fixed diagonal jitter used during factorization.
    #[must_use]
    pub const fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Solve `K x = rhs` using the stored Cholesky factorization.
    ///
    /// # Errors
    ///
    /// Returns [`ConditioningError`] when the right-hand side has the wrong
    /// dimension or contains non-finite entries.
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, ConditioningError> {
        if rhs.len() != self.dimension {
            return Err(ConditioningError::RightHandSideDimensionMismatch);
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(ConditioningError::NonFiniteRightHandSideEntry);
        }

        let right_hand_side = DVector::from_column_slice(rhs);
        let solution = self.cholesky.solve(&right_hand_side);

        Ok(solution.iter().copied().collect())
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact round-trips of constructor inputs are intended
mod tests {
    use super::GaussianConditioner;
    use crate::ConditioningError;

    const TOLERANCE: f64 = 1.0e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOLERANCE * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    #[test]
    fn solves_known_spd_system_without_inverse() {
        let conditioner =
            GaussianConditioner::new(&[4.0, 1.0, 1.0, 3.0], 2, 0.0).expect("matrix is SPD");
        let solution = conditioner.solve(&[1.0, 2.0]).expect("rhs is valid");

        assert_close(solution[0], 1.0 / 11.0);
        assert_close(solution[1], 7.0 / 11.0);
    }

    #[test]
    fn factorization_is_reused_for_multiple_right_hand_sides() {
        let conditioner =
            GaussianConditioner::new(&[2.0, 0.5, 0.5, 1.5], 2, 0.0).expect("matrix is SPD");

        let first = conditioner.solve(&[1.0, 0.0]).expect("rhs is valid");
        let second = conditioner.solve(&[0.0, 1.0]).expect("rhs is valid");

        assert_close(2.0 * first[0] + 0.5 * first[1], 1.0);
        assert_close(0.5 * first[0] + 1.5 * first[1], 0.0);
        assert_close(2.0 * second[0] + 0.5 * second[1], 0.0);
        assert_close(0.5 * second[0] + 1.5 * second[1], 1.0);
    }

    #[test]
    fn rejects_invalid_matrix_contracts() {
        assert!(matches!(
            GaussianConditioner::new(&[], 0, 0.0),
            Err(ConditioningError::ZeroDimension)
        ));
        assert!(matches!(
            GaussianConditioner::new(&[1.0, 0.0, 0.0], 2, 0.0),
            Err(ConditioningError::MatrixDimensionMismatch)
        ));
        assert!(matches!(
            GaussianConditioner::new(&[1.0, f64::NAN, 0.0, 1.0], 2, 0.0),
            Err(ConditioningError::NonFiniteMatrixEntry)
        ));
    }

    #[test]
    fn rejects_invalid_jitter() {
        assert!(matches!(
            GaussianConditioner::new(&[1.0], 1, f64::NAN),
            Err(ConditioningError::NonFiniteJitter)
        ));
        assert!(matches!(
            GaussianConditioner::new(&[1.0], 1, -1.0e-6),
            Err(ConditioningError::NegativeJitter)
        ));
    }

    #[test]
    fn singular_matrix_fails_without_jitter() {
        assert!(matches!(
            GaussianConditioner::new(&[1.0, 1.0, 1.0, 1.0], 2, 0.0),
            Err(ConditioningError::NotPositiveDefinite)
        ));
    }

    #[test]
    fn explicit_jitter_can_regularize_singular_matrix() {
        let conditioner = GaussianConditioner::new(&[1.0, 1.0, 1.0, 1.0], 2, 1.0e-6)
            .expect("positive jitter makes matrix positive definite");
        let solution = conditioner.solve(&[2.0, 2.0]).expect("rhs is valid");

        assert_eq!(conditioner.jitter(), 1.0e-6);
        assert_close((1.0 + 1.0e-6) * solution[0] + solution[1], 2.0);
        assert_close(solution[0] + (1.0 + 1.0e-6) * solution[1], 2.0);
    }

    #[test]
    fn rejects_invalid_right_hand_side() {
        let conditioner =
            GaussianConditioner::new(&[2.0, 0.0, 0.0, 3.0], 2, 0.0).expect("matrix is SPD");

        assert_eq!(
            conditioner.solve(&[1.0]),
            Err(ConditioningError::RightHandSideDimensionMismatch)
        );
        assert_eq!(
            conditioner.solve(&[1.0, f64::INFINITY]),
            Err(ConditioningError::NonFiniteRightHandSideEntry)
        );
    }
}
