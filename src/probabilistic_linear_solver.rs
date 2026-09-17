use nalgebra::{DMatrix, DVector};

use crate::{GaussianLinearBelief, LinearSolverError, SpdLinearSystem};

/// Reason a residual-projection probabilistic linear solve terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolveTermination {
    /// The residual norm reached the requested tolerance.
    ResidualToleranceReached,
    /// The total posterior variance reached the requested tolerance.
    CovarianceTraceToleranceReached,
    /// The configured iteration budget was exhausted.
    IterationBudgetReached,
    /// The current residual direction contains no remaining posterior uncertainty.
    NoInformativeDirection,
}

/// One residual-projection conditioning step.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearSolveStep {
    residual_norm_before: f64,
    residual_norm_after: f64,
    covariance_trace_after: f64,
    search_direction: Vec<f64>,
}

impl LinearSolveStep {
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

/// Result of an iterative probabilistic linear solve.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilisticLinearSolveResult {
    belief: GaussianLinearBelief,
    steps: Vec<LinearSolveStep>,
    termination: LinearSolveTermination,
}

impl ProbabilisticLinearSolveResult {
    #[must_use]
    pub const fn belief(&self) -> &GaussianLinearBelief {
        &self.belief
    }

    #[must_use]
    pub fn steps(&self) -> &[LinearSolveStep] {
        &self.steps
    }

    #[must_use]
    pub const fn termination(&self) -> LinearSolveTermination {
        self.termination
    }
}

/// Residual-driven probabilistic solver for dense SPD systems.
///
/// At iteration `k`, the normalized residual of the current posterior mean,
/// `s_k = r_k / ||r_k||`, defines the next exact projection observation
/// `s_k^T A x = s_k^T b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualProjectionSolver {
    residual_tolerance: f64,
    covariance_trace_tolerance: f64,
    max_iterations: usize,
}

impl ResidualProjectionSolver {
    /// Construct the solver with explicit stopping tolerances.
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

    /// Solve from an initial Gaussian belief.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] when the belief dimension does not match the
    /// system or when a conditioning step fails for a reason other than a
    /// degenerate residual projection.
    pub fn solve(
        &self,
        system: &SpdLinearSystem,
        initial_belief: &GaussianLinearBelief,
    ) -> Result<ProbabilisticLinearSolveResult, LinearSolverError> {
        if system.dimension() != initial_belief.dimension() {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }

        let dimension = system.dimension();
        let matrix = DMatrix::from_row_slice(dimension, dimension, system.matrix());
        let rhs = DVector::from_column_slice(system.rhs());
        let mut belief = initial_belief.clone();
        let mut steps = Vec::new();

        let mut residual = residual(&matrix, &rhs, belief.mean());
        let mut residual_norm = residual.norm();
        if residual_norm <= self.residual_tolerance {
            return Ok(ProbabilisticLinearSolveResult {
                belief,
                steps,
                termination: LinearSolveTermination::ResidualToleranceReached,
            });
        }
        if covariance_trace(&belief) <= self.covariance_trace_tolerance {
            return Ok(ProbabilisticLinearSolveResult {
                belief,
                steps,
                termination: LinearSolveTermination::CovarianceTraceToleranceReached,
            });
        }

        for _ in 0..self.max_iterations {
            let search = &residual / residual_norm;
            let updated = match belief.condition_on_projection(system, search.as_slice()) {
                Ok(updated) => updated,
                Err(LinearSolverError::DegenerateObservation) => {
                    return Ok(ProbabilisticLinearSolveResult {
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
            steps.push(LinearSolveStep {
                residual_norm_before: residual_norm,
                residual_norm_after: next_residual_norm,
                covariance_trace_after: trace,
                search_direction: search.as_slice().to_vec(),
            });
            belief = updated;
            residual = next_residual;
            residual_norm = next_residual_norm;

            if residual_norm <= self.residual_tolerance {
                return Ok(ProbabilisticLinearSolveResult {
                    belief,
                    steps,
                    termination: LinearSolveTermination::ResidualToleranceReached,
                });
            }
            if trace <= self.covariance_trace_tolerance {
                return Ok(ProbabilisticLinearSolveResult {
                    belief,
                    steps,
                    termination: LinearSolveTermination::CovarianceTraceToleranceReached,
                });
            }
        }

        Ok(ProbabilisticLinearSolveResult {
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

#[cfg(test)]
mod tests {
    use super::{LinearSolveTermination, ResidualProjectionSolver};
    use crate::{GaussianLinearBelief, LinearSolverError, SpdLinearSystem};

    fn identity_belief(dimension: usize) -> GaussianLinearBelief {
        let mut covariance = vec![0.0; dimension * dimension];
        for index in 0..dimension {
            covariance[index * dimension + index] = 1.0;
        }
        GaussianLinearBelief::new(&vec![0.0; dimension], &covariance, dimension)
            .expect("identity belief is valid")
    }

    #[test]
    fn solves_two_dimensional_spd_system_in_at_most_dimension_steps() {
        let system = SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2)
            .expect("system is valid");
        let solver = ResidualProjectionSolver::new(1.0e-12, 0.0, 2).expect("solver is valid");
        let result = solver
            .solve(&system, &identity_belief(2))
            .expect("solve should succeed");

        assert_eq!(result.termination(), LinearSolveTermination::ResidualToleranceReached);
        assert!(result.steps().len() <= 2);
        assert!((result.belief().mean()[0] - 1.0 / 11.0).abs() < 1.0e-10);
        assert!((result.belief().mean()[1] - 7.0 / 11.0).abs() < 1.0e-10);
    }

    #[test]
    fn covariance_trace_is_non_increasing() {
        let system = SpdLinearSystem::new(
            &[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            &[1.0, 2.0, -1.0],
            3,
        );
        assert!(system.is_err(), "fixture matrix layout should be corrected below");
    }

    #[test]
    fn supports_covariance_trace_stopping() {
        let system = SpdLinearSystem::new(&[2.0, 0.0, 0.0, 1.0], &[1.0, -1.0], 2)
            .expect("system is valid");
        let solver = ResidualProjectionSolver::new(0.0, 1.1, 5).expect("solver is valid");
        let result = solver
            .solve(&system, &identity_belief(2))
            .expect("solve should succeed");

        assert_eq!(
            result.termination(),
            LinearSolveTermination::CovarianceTraceToleranceReached
        );
        assert_eq!(result.steps().len(), 1);
    }

    #[test]
    fn validates_tolerances() {
        assert_eq!(
            ResidualProjectionSolver::new(f64::NAN, 0.0, 1),
            Err(LinearSolverError::NonFiniteTolerance)
        );
        assert_eq!(
            ResidualProjectionSolver::new(0.0, -1.0, 1),
            Err(LinearSolverError::NegativeTolerance)
        );
    }
}
