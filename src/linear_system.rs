use nalgebra::DMatrix;

use crate::LinearSolverError;

const SYMMETRY_TOLERANCE: f64 = 64.0 * f64::EPSILON;

/// Dense symmetric positive-definite linear system `A x = b`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpdLinearSystem {
    dimension: usize,
    matrix: Vec<f64>,
    rhs: Vec<f64>,
}

impl SpdLinearSystem {
    /// Construct and validate an SPD linear system from a row-major matrix.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] for invalid dimensions, non-finite values,
    /// non-symmetry, or failure of Cholesky positive-definiteness validation.
    pub fn new(matrix: &[f64], rhs: &[f64], dimension: usize) -> Result<Self, LinearSolverError> {
        if dimension == 0 {
            return Err(LinearSolverError::ZeroDimension);
        }
        let expected_len = dimension
            .checked_mul(dimension)
            .ok_or(LinearSolverError::MatrixDimensionMismatch)?;
        if matrix.len() != expected_len {
            return Err(LinearSolverError::MatrixDimensionMismatch);
        }
        if rhs.len() != dimension {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }
        if matrix.iter().any(|value| !value.is_finite()) {
            return Err(LinearSolverError::NonFiniteMatrixEntry);
        }
        if rhs.iter().any(|value| !value.is_finite()) {
            return Err(LinearSolverError::NonFiniteVectorEntry);
        }

        let matrix_view = DMatrix::from_row_slice(dimension, dimension, matrix);
        let scale = matrix
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tolerance = SYMMETRY_TOLERANCE * scale;
        for row in 0..dimension {
            for column in (row + 1)..dimension {
                if (matrix_view[(row, column)] - matrix_view[(column, row)]).abs() > tolerance {
                    return Err(LinearSolverError::NonSymmetricSystemMatrix);
                }
            }
        }
        if matrix_view.cholesky().is_none() {
            return Err(LinearSolverError::SystemMatrixNotPositiveDefinite);
        }

        Ok(Self {
            dimension,
            matrix: matrix.to_vec(),
            rhs: rhs.to_vec(),
        })
    }

    /// Return the system dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the row-major system matrix.
    #[must_use]
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Return the right-hand side.
    #[must_use]
    pub fn rhs(&self) -> &[f64] {
        &self.rhs
    }
}

#[cfg(test)]
mod tests {
    use super::SpdLinearSystem;
    use crate::LinearSolverError;

    #[test]
    fn accepts_spd_system() {
        let system =
            SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2).expect("matrix is SPD");
        assert_eq!(system.dimension(), 2);
        assert_eq!(system.rhs(), &[1.0, 2.0]);
    }

    #[test]
    fn rejects_non_symmetric_matrix() {
        assert_eq!(
            SpdLinearSystem::new(&[2.0, 1.0, 0.0, 2.0], &[1.0, 1.0], 2),
            Err(LinearSolverError::NonSymmetricSystemMatrix)
        );
    }

    #[test]
    fn rejects_non_positive_definite_matrix() {
        assert_eq!(
            SpdLinearSystem::new(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0], 2),
            Err(LinearSolverError::SystemMatrixNotPositiveDefinite)
        );
    }
}
