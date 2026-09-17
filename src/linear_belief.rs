use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::{LinearSolverError, SpdLinearSystem};

const COVARIANCE_TOLERANCE: f64 = 128.0 * f64::EPSILON;

/// Gaussian belief over the unknown solution vector of a linear system.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianLinearBelief {
    mean: Vec<f64>,
    covariance: Vec<f64>,
    dimension: usize,
}

impl GaussianLinearBelief {
    /// Construct a Gaussian solution belief from a mean and row-major covariance matrix.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] for invalid dimensions, non-finite values,
    /// non-symmetry, or a covariance that is not positive semidefinite within
    /// numerical tolerance.
    pub fn new(
        mean: &[f64],
        covariance: &[f64],
        dimension: usize,
    ) -> Result<Self, LinearSolverError> {
        if dimension == 0 {
            return Err(LinearSolverError::ZeroDimension);
        }
        if mean.len() != dimension {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }
        let expected_len = dimension
            .checked_mul(dimension)
            .ok_or(LinearSolverError::MatrixDimensionMismatch)?;
        if covariance.len() != expected_len {
            return Err(LinearSolverError::MatrixDimensionMismatch);
        }
        if mean.iter().any(|value| !value.is_finite()) {
            return Err(LinearSolverError::NonFiniteVectorEntry);
        }
        if covariance.iter().any(|value| !value.is_finite()) {
            return Err(LinearSolverError::NonFiniteMatrixEntry);
        }

        let covariance_matrix = DMatrix::from_row_slice(dimension, dimension, covariance);
        let scale = covariance
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tolerance = COVARIANCE_TOLERANCE * scale;
        for row in 0..dimension {
            for column in (row + 1)..dimension {
                if (covariance_matrix[(row, column)] - covariance_matrix[(column, row)]).abs()
                    > tolerance
                {
                    return Err(LinearSolverError::NonSymmetricCovariance);
                }
            }
        }

        let eigenvalues = SymmetricEigen::new(covariance_matrix).eigenvalues;
        if eigenvalues.iter().any(|value| *value < -tolerance) {
            return Err(LinearSolverError::CovarianceNotPositiveSemidefinite);
        }

        Ok(Self {
            mean: mean.to_vec(),
            covariance: covariance.to_vec(),
            dimension,
        })
    }

    /// Return the belief dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Return the current solution mean.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Return the row-major covariance matrix.
    #[must_use]
    pub fn covariance(&self) -> &[f64] {
        &self.covariance
    }

    /// Condition the solution belief on one exact projection `s^T A x = s^T b`.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] when dimensions are inconsistent, the search
    /// direction is non-finite, or the requested projection has no remaining
    /// uncertainty under the current covariance.
    pub fn condition_on_projection(
        &self,
        system: &SpdLinearSystem,
        search_direction: &[f64],
    ) -> Result<Self, LinearSolverError> {
        if system.dimension() != self.dimension || search_direction.len() != self.dimension {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }
        if search_direction.iter().any(|value| !value.is_finite()) {
            return Err(LinearSolverError::NonFiniteVectorEntry);
        }

        let dimension = self.dimension;
        let matrix = DMatrix::from_row_slice(dimension, dimension, system.matrix());
        let rhs = DVector::from_column_slice(system.rhs());
        let search = DVector::from_column_slice(search_direction);
        let mean = DVector::from_column_slice(&self.mean);
        let covariance = DMatrix::from_row_slice(dimension, dimension, &self.covariance);

        let observation_vector = matrix.transpose() * &search;
        let covariance_observation = &covariance * &observation_vector;
        let observation_variance = observation_vector.dot(&covariance_observation);
        let scale = covariance
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tolerance = COVARIANCE_TOLERANCE * scale;
        if observation_variance <= tolerance {
            return Err(LinearSolverError::DegenerateObservation);
        }

        let observed_value = search.dot(&rhs);
        let predicted_value = observation_vector.dot(&mean);
        let innovation = observed_value - predicted_value;
        let gain = &covariance_observation / observation_variance;
        let updated_mean = mean + &gain * innovation;
        let updated_covariance = covariance
            - (&covariance_observation * covariance_observation.transpose())
                / observation_variance;

        let mut covariance_flat = Vec::with_capacity(dimension * dimension);
        for row in 0..dimension {
            for column in 0..dimension {
                let symmetric_value =
                    0.5 * (updated_covariance[(row, column)] + updated_covariance[(column, row)]);
                covariance_flat.push(symmetric_value);
            }
        }

        Self::new(updated_mean.as_slice(), &covariance_flat, dimension)
    }
}

#[cfg(test)]
mod tests {
    use super::GaussianLinearBelief;
    use crate::{LinearSolverError, SpdLinearSystem};

    const TOLERANCE: f64 = 1.0e-11;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= TOLERANCE * expected.abs().max(1.0),
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    #[test]
    fn projection_conditioning_makes_observation_exact() {
        let system = SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2)
            .expect("system is valid");
        let belief = GaussianLinearBelief::new(&[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 2)
            .expect("belief is valid");
        let updated = belief
            .condition_on_projection(&system, &[1.0, 0.0])
            .expect("projection is informative");

        let observed = 4.0 * updated.mean()[0] + updated.mean()[1];
        assert_close(observed, 1.0);
    }

    #[test]
    fn projection_conditioning_reduces_total_variance() {
        let system = SpdLinearSystem::new(&[2.0, 0.5, 0.5, 1.5], &[1.0, -1.0], 2)
            .expect("system is valid");
        let belief = GaussianLinearBelief::new(&[0.0, 0.0], &[2.0, 0.2, 0.2, 1.0], 2)
            .expect("belief is valid");
        let updated = belief
            .condition_on_projection(&system, &[1.0, 0.0])
            .expect("projection is informative");

        let prior_trace = belief.covariance()[0] + belief.covariance()[3];
        let posterior_trace = updated.covariance()[0] + updated.covariance()[3];
        assert!(posterior_trace < prior_trace);
    }

    #[test]
    fn orthogonal_uncertainty_can_remain_after_one_projection() {
        let system = SpdLinearSystem::new(&[1.0, 0.0, 0.0, 1.0], &[2.0, -3.0], 2)
            .expect("system is valid");
        let belief = GaussianLinearBelief::new(&[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 2)
            .expect("belief is valid");
        let updated = belief
            .condition_on_projection(&system, &[1.0, 0.0])
            .expect("projection is informative");

        assert_close(updated.mean()[0], 2.0);
        assert_close(updated.covariance()[0], 0.0);
        assert_close(updated.covariance()[3], 1.0);
    }

    #[test]
    fn repeated_exact_projection_is_degenerate() {
        let system = SpdLinearSystem::new(&[1.0, 0.0, 0.0, 1.0], &[2.0, -3.0], 2)
            .expect("system is valid");
        let belief = GaussianLinearBelief::new(&[0.0, 0.0], &[1.0, 0.0, 0.0, 1.0], 2)
            .expect("belief is valid");
        let updated = belief
            .condition_on_projection(&system, &[1.0, 0.0])
            .expect("projection is informative");

        assert_eq!(
            updated.condition_on_projection(&system, &[1.0, 0.0]),
            Err(LinearSolverError::DegenerateObservation)
        );
    }
}
