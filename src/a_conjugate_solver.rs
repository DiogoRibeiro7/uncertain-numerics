use nalgebra::{DMatrix, DVector};

use crate::{GaussianLinearBelief, LinearSolveTermination, LinearSolverError, SpdLinearSystem};

/// One A-conjugate projection-conditioning step.
#[derive(Debug, Clone, PartialEq)]
pub struct AConjugateLinearSolveStep {
    residual_norm_before: f64,
    residual_norm_after: f64,
    covariance_trace_after: f64,
    search_direction: Vec<f64>,
}

impl AConjugateLinearSolveStep {
    #[must_use]
    pub const fn residual_norm_before(&self) -> f64 {
        self.residual_norm_before
    }

    #[must_use]
    pub const fn residual_norm_after(&self) -> f64 {
        self.residual_norm_after
    }

    #[must_use]
    pub const fn covariance_trace_after(&self) -> f64 {
        self.covariance_trace_after
    }

    #[must_use]
    pub fn search_direction(&self) -> &[f64] {
        &self.search_direction
    }
}

/// Result of an A-conjugate probabilistic linear solve.
#[derive(Debug, Clone, PartialEq)]
pub struct AConjugateLinearSolveResult {
    belief: GaussianLinearBelief,
    steps: Vec<AConjugateLinearSolveStep>,
    termination: LinearSolveTermination,
}

impl AConjugateLinearSolveResult {
    #[must_use]
    pub const fn belief(&self) -> &GaussianLinearBelief {
        &self.belief
    }

    #[must_use]
    pub fn steps(&self) -> &[AConjugateLinearSolveStep] {
        &self.steps
    }

    #[must_use]
    pub const fn termination(&self) -> LinearSolveTermination {
        self.termination
    }
}

/// Probabilistic linear solver using residual directions orthogonalized in the `A` inner product.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AConjugateProjectionSolver {
    residual_tolerance: f64,
    covariance_trace_tolerance: f64,
    max_iterations: usize,
}

impl AConjugateProjectionSolver {
    /// Construct a solver with explicit numerical-accuracy and uncertainty tolerances.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] when either tolerance is non-finite or negative.
    pub fn new(
        residual_tolerance: f64,
        covariance_trace_tolerance: f64,
        max_iterations: usize,
    ) -> Result<Self, LinearSolverError> {
        if !residual_tolerance.is_finite() || !covariance_trace_tolerance.is_finite() {
            return Err(LinearSolverError::NonFiniteTolerance);
        }
        if residual_tolerance < 0.0 || covariance_trace_tolerance < 0.0 {
            return Err(LinearSolverError::NegativeTolerance);
        }
        Ok(Self {
            residual_tolerance,
            covariance_trace_tolerance,
            max_iterations,
        })
    }

    /// Solve using residual directions made pairwise `A`-conjugate by explicit Gram-Schmidt.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] for incompatible belief dimensions or failed
    /// conditioning updates.
    pub fn solve(
        &self,
        system: &SpdLinearSystem,
        initial_belief: &GaussianLinearBelief,
    ) -> Result<AConjugateLinearSolveResult, LinearSolverError> {
        if system.dimension() != initial_belief.dimension() {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }

        let dimension = system.dimension();
        let matrix = DMatrix::from_row_slice(dimension, dimension, system.matrix());
        let rhs = DVector::from_column_slice(system.rhs());
        let mut belief = initial_belief.clone();
        let mut steps = Vec::new();
        let mut directions: Vec<DVector<f64>> = Vec::new();

        let mut residual = residual(&matrix, &rhs, belief.mean());
        let mut residual_norm = residual.norm();
        if residual_norm <= self.residual_tolerance {
            return Ok(AConjugateLinearSolveResult {
                belief,
                steps,
                termination: LinearSolveTermination::ResidualToleranceReached,
            });
        }
        if covariance_trace(&belief) <= self.covariance_trace_tolerance {
            return Ok(AConjugateLinearSolveResult {
                belief,
                steps,
                termination: LinearSolveTermination::CovarianceTraceToleranceReached,
            });
        }

        for _ in 0..self.max_iterations {
            let mut search = residual.clone();
            for previous in &directions {
                let denominator = a_inner(previous, previous, &matrix);
                let tolerance = 128.0 * f64::EPSILON * denominator.abs().max(1.0);
                if denominator.abs() <= tolerance {
                    continue;
                }
                let coefficient = a_inner(&search, previous, &matrix) / denominator;
                search -= previous * coefficient;
            }

            let search_norm = search.norm();
            let tolerance = 128.0 * f64::EPSILON * residual_norm.max(1.0);
            if search_norm <= tolerance {
                return Ok(AConjugateLinearSolveResult {
                    belief,
                    steps,
                    termination: LinearSolveTermination::NoInformativeDirection,
                });
            }
            search /= search_norm;

            let updated = match belief.condition_on_projection(system, search.as_slice()) {
                Ok(updated) => updated,
                Err(LinearSolverError::DegenerateObservation) => {
                    return Ok(AConjugateLinearSolveResult {
                        belief,
                        steps,
                        termination: LinearSolveTermination::NoInformativeDirection,
                    });
                }
                Err(error) => return Err(error),
            };

            let next_residual = residual(&matrix, &rhs, updated.mean());
            let next_residual_norm = next_residual.norm();
            let trace = covariance_trace(&updated);
            steps.push(AConjugateLinearSolveStep {
                residual_norm_before: residual_norm,
                residual_norm_after: next_residual_norm,
                covariance_trace_after: trace,
                search_direction: search.as_slice().to_vec(),
            });
            directions.push(search);
            belief = updated;
            residual = next_residual;
            residual_norm = next_residual_norm;

            if residual_norm <= self.residual_tolerance {
                return Ok(AConjugateLinearSolveResult {
                    belief,
                    steps,
                    termination: LinearSolveTermination::ResidualToleranceReached,
                });
            }
            if trace <= self.covariance_trace_tolerance {
                return Ok(AConjugateLinearSolveResult {
                    belief,
                    steps,
                    termination: LinearSolveTermination::CovarianceTraceToleranceReached,
                });
            }
        }

        Ok(AConjugateLinearSolveResult {
            belief,
            steps,
            termination: LinearSolveTermination::IterationBudgetReached,
        })
    }
}

fn residual(matrix: &DMatrix<f64>, rhs: &DVector<f64>, mean: &[f64]) -> DVector<f64> {
    rhs - matrix * DVector::from_column_slice(mean)
}

fn covariance_trace(belief: &GaussianLinearBelief) -> f64 {
    let dimension = belief.dimension();
    (0..dimension)
        .map(|index| belief.covariance()[index * dimension + index])
        .sum()
}

fn a_inner(left: &DVector<f64>, right: &DVector<f64>, matrix: &DMatrix<f64>) -> f64 {
    left.dot(&(matrix * right))
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use super::AConjugateProjectionSolver;
    use crate::{GaussianLinearBelief, LinearSolveTermination, SpdLinearSystem};

    fn identity_belief(dimension: usize) -> GaussianLinearBelief {
        let mut covariance = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            covariance[index * dimension + index] = 1.0;
        }
        GaussianLinearBelief::new(&vec![0.0; dimension], &covariance, dimension)
            .expect("identity belief is valid")
    }

    #[test]
    fn search_directions_are_pairwise_a_conjugate() {
        let system = SpdLinearSystem::new(
            &[4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0],
            &[1.0, 2.0, -1.0],
            3,
        )
        .expect("system is valid");
        let solver = AConjugateProjectionSolver::new(0.0, 0.0, 3).expect("solver is valid");
        let result = solver
            .solve(&system, &identity_belief(3))
            .expect("solve should succeed");
        let matrix = DMatrix::from_row_slice(3, 3, system.matrix());

        for left_index in 0..result.steps().len() {
            for right_index in 0..left_index {
                let left = DVector::from_column_slice(result.steps()[left_index].search_direction());
                let right = DVector::from_column_slice(result.steps()[right_index].search_direction());
                let inner = left.dot(&(&matrix * right));
                assert!(inner.abs() <= 1.0e-10, "A-inner product was {inner:.16e}");
            }
        }
    }

    #[test]
    fn solves_small_spd_system_within_dimension_steps() {
        let system = SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2)
            .expect("system is valid");
        let solver = AConjugateProjectionSolver::new(1.0e-12, 0.0, 2).expect("solver is valid");
        let result = solver
            .solve(&system, &identity_belief(2))
            .expect("solve should succeed");

        assert_eq!(result.termination(), LinearSolveTermination::ResidualToleranceReached);
        assert!(result.steps().len() <= 2);
        assert!((result.belief().mean()[0] - 1.0 / 11.0).abs() < 1.0e-10);
        assert!((result.belief().mean()[1] - 7.0 / 11.0).abs() < 1.0e-10);
    }

    #[test]
    fn covariance_trace_remains_non_increasing() {
        let system = SpdLinearSystem::new(
            &[3.0, 0.5, 0.0, 0.5, 2.0, 0.25, 0.0, 0.25, 1.5],
            &[1.0, -2.0, 0.5],
            3,
        )
        .expect("system is valid");
        let solver = AConjugateProjectionSolver::new(0.0, 0.0, 3).expect("solver is valid");
        let result = solver
            .solve(&system, &identity_belief(3))
            .expect("solve should succeed");

        let mut previous = 3.0;
        for step in result.steps() {
            assert!(step.covariance_trace_after() <= previous + 1.0e-12);
            previous = step.covariance_trace_after();
        }
    }
}
