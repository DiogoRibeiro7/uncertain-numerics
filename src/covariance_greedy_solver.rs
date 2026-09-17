use nalgebra::{DMatrix, DVector};

use crate::{GaussianLinearBelief, LinearSolverError, SpdLinearSystem};

const INFORMATION_TOLERANCE: f64 = 128.0 * f64::EPSILON;

/// Candidate projection chosen to maximize posterior covariance-trace reduction.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedLinearDirection {
    direction: Vec<f64>,
    index: usize,
    trace_reduction: f64,
}

impl SelectedLinearDirection {
    /// Return the selected direction.
    #[must_use]
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// Return the original candidate index.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Return the predicted posterior covariance-trace reduction.
    #[must_use]
    pub const fn trace_reduction(&self) -> f64 {
        self.trace_reduction
    }
}

/// Reason a covariance-greedy probabilistic solve terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceGreedyTermination {
    /// Posterior covariance trace reached the requested tolerance.
    CovarianceTraceToleranceReached,
    /// The configured projection budget was exhausted.
    ProjectionBudgetReached,
    /// No unevaluated candidate directions remain.
    CandidatesExhausted,
    /// Remaining candidates carry no posterior uncertainty.
    NoInformativeDirection,
}

/// One covariance-greedy projection step.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceGreedyStep {
    direction: Vec<f64>,
    predicted_trace_reduction: f64,
    posterior_trace: f64,
}

impl CovarianceGreedyStep {
    /// Return the selected direction.
    #[must_use]
    pub fn direction(&self) -> &[f64] {
        &self.direction
    }

    /// Return the predicted covariance-trace reduction before conditioning.
    #[must_use]
    pub const fn predicted_trace_reduction(&self) -> f64 {
        self.predicted_trace_reduction
    }

    /// Return the posterior covariance trace after conditioning.
    #[must_use]
    pub const fn posterior_trace(&self) -> f64 {
        self.posterior_trace
    }
}

/// Result of a covariance-greedy probabilistic linear solve.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceGreedySolveResult {
    belief: GaussianLinearBelief,
    steps: Vec<CovarianceGreedyStep>,
    remaining_candidates: Vec<Vec<f64>>,
    termination: CovarianceGreedyTermination,
}

impl CovarianceGreedySolveResult {
    /// Return the final Gaussian solution belief.
    #[must_use]
    pub const fn belief(&self) -> &GaussianLinearBelief {
        &self.belief
    }

    /// Return the ordered projection history.
    #[must_use]
    pub fn steps(&self) -> &[CovarianceGreedyStep] {
        &self.steps
    }

    /// Return candidate directions not selected during the run.
    #[must_use]
    pub fn remaining_candidates(&self) -> &[Vec<f64>] {
        &self.remaining_candidates
    }

    /// Return the termination reason.
    #[must_use]
    pub const fn termination(&self) -> CovarianceGreedyTermination {
        self.termination
    }
}

/// Covariance-trace acquisition for exact linear-system projection observations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceTraceAcquisition;

impl CovarianceTraceAcquisition {
    /// Predict the covariance-trace reduction from conditioning on one direction.
    ///
    /// For `h = A^T s`, the exact rank-one covariance update implies
    ///
    /// ```text
    /// trace(Sigma) - trace(Sigma+)
    /// = h^T Sigma^2 h / (h^T Sigma h).
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] for incompatible/non-finite directions.
    pub fn reduction(
        system: &SpdLinearSystem,
        belief: &GaussianLinearBelief,
        direction: &[f64],
    ) -> Result<f64, LinearSolverError> {
        if system.dimension() != belief.dimension() || direction.len() != system.dimension() {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(LinearSolverError::NonFiniteVectorEntry);
        }

        let dimension = system.dimension();
        let matrix = DMatrix::from_row_slice(dimension, dimension, system.matrix());
        let covariance = DMatrix::from_row_slice(dimension, dimension, belief.covariance());
        let search = DVector::from_column_slice(direction);
        let h = matrix.transpose() * search;
        let covariance_h = &covariance * &h;
        let denominator = h.dot(&covariance_h);
        let scale = covariance
            .iter()
            .fold(1.0_f64, |acc, value| acc.max(value.abs()));
        let tolerance = INFORMATION_TOLERANCE * scale;
        if denominator <= tolerance {
            return Ok(0.0);
        }

        Ok(covariance_h.dot(&covariance_h) / denominator)
    }

    /// Select the direction with the largest predicted covariance-trace reduction.
    ///
    /// Ties are resolved deterministically by retaining the first maximum.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] for an empty candidate set or invalid directions.
    pub fn select_best(
        system: &SpdLinearSystem,
        belief: &GaussianLinearBelief,
        candidates: &[Vec<f64>],
    ) -> Result<SelectedLinearDirection, LinearSolverError> {
        if candidates.is_empty() {
            return Err(LinearSolverError::EmptyCandidateDirections);
        }

        let mut best = SelectedLinearDirection {
            direction: candidates[0].clone(),
            index: 0,
            trace_reduction: Self::reduction(system, belief, &candidates[0])?,
        };
        for (index, candidate) in candidates.iter().enumerate().skip(1) {
            let reduction = Self::reduction(system, belief, candidate)?;
            if reduction > best.trace_reduction {
                best = SelectedLinearDirection {
                    direction: candidate.clone(),
                    index,
                    trace_reduction: reduction,
                };
            }
        }
        Ok(best)
    }
}

/// Sequential probabilistic linear solver with data-independent covariance-greedy directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceGreedyProjectionSolver {
    covariance_trace_tolerance: f64,
    max_projections: usize,
}

impl CovarianceGreedyProjectionSolver {
    /// Construct the solver with explicit uncertainty tolerance and projection budget.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] when the covariance-trace tolerance is invalid.
    pub fn new(
        covariance_trace_tolerance: f64,
        max_projections: usize,
    ) -> Result<Self, LinearSolverError> {
        if !covariance_trace_tolerance.is_finite() {
            return Err(LinearSolverError::NonFiniteTolerance);
        }
        if covariance_trace_tolerance < 0.0 {
            return Err(LinearSolverError::NegativeTolerance);
        }
        Ok(Self {
            covariance_trace_tolerance,
            max_projections,
        })
    }

    /// Run covariance-greedy sequential conditioning from an initial Gaussian belief.
    ///
    /// Direction selection and stopping depend only on `A`, the candidate set,
    /// the covariance state, and the configured budget/tolerance. They do not
    /// depend on the observed right-hand side.
    ///
    /// # Errors
    ///
    /// Returns [`LinearSolverError`] for incompatible dimensions or invalid candidates.
    pub fn solve(
        &self,
        system: &SpdLinearSystem,
        initial_belief: &GaussianLinearBelief,
        candidates: &[Vec<f64>],
    ) -> Result<CovarianceGreedySolveResult, LinearSolverError> {
        if system.dimension() != initial_belief.dimension() {
            return Err(LinearSolverError::VectorDimensionMismatch);
        }
        if candidates.is_empty() {
            return Err(LinearSolverError::EmptyCandidateDirections);
        }

        let mut belief = initial_belief.clone();
        let mut remaining_candidates = candidates.to_vec();
        let mut steps = Vec::new();

        if covariance_trace(&belief) <= self.covariance_trace_tolerance {
            return Ok(CovarianceGreedySolveResult {
                belief,
                steps,
                remaining_candidates,
                termination: CovarianceGreedyTermination::CovarianceTraceToleranceReached,
            });
        }

        for _ in 0..self.max_projections {
            if remaining_candidates.is_empty() {
                return Ok(CovarianceGreedySolveResult {
                    belief,
                    steps,
                    remaining_candidates,
                    termination: CovarianceGreedyTermination::CandidatesExhausted,
                });
            }

            let selected =
                CovarianceTraceAcquisition::select_best(system, &belief, &remaining_candidates)?;
            if selected.trace_reduction() <= INFORMATION_TOLERANCE {
                return Ok(CovarianceGreedySolveResult {
                    belief,
                    steps,
                    remaining_candidates,
                    termination: CovarianceGreedyTermination::NoInformativeDirection,
                });
            }

            let trace_before = covariance_trace(&belief);
            let updated = belief.condition_on_projection(system, selected.direction())?;
            let trace_after = covariance_trace(&updated);
            let actual_reduction = trace_before - trace_after;
            let tolerance = 1.0e-10 * selected.trace_reduction().abs().max(1.0);
            debug_assert!((actual_reduction - selected.trace_reduction()).abs() <= tolerance);

            remaining_candidates.remove(selected.index());
            steps.push(CovarianceGreedyStep {
                direction: selected.direction,
                predicted_trace_reduction: selected.trace_reduction,
                posterior_trace: trace_after,
            });
            belief = updated;

            if trace_after <= self.covariance_trace_tolerance {
                return Ok(CovarianceGreedySolveResult {
                    belief,
                    steps,
                    remaining_candidates,
                    termination: CovarianceGreedyTermination::CovarianceTraceToleranceReached,
                });
            }
        }

        let termination = if remaining_candidates.is_empty() {
            CovarianceGreedyTermination::CandidatesExhausted
        } else {
            CovarianceGreedyTermination::ProjectionBudgetReached
        };
        Ok(CovarianceGreedySolveResult {
            belief,
            steps,
            remaining_candidates,
            termination,
        })
    }
}

fn covariance_trace(belief: &GaussianLinearBelief) -> f64 {
    let dimension = belief.dimension();
    (0..dimension)
        .map(|index| belief.covariance()[index * dimension + index])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::{
        CovarianceGreedyProjectionSolver, CovarianceGreedyTermination, CovarianceTraceAcquisition,
    };
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
    fn predicted_trace_reduction_matches_actual_conditioning_drop() {
        let system =
            SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2).expect("system is valid");
        let belief = identity_belief(2);
        let direction = [1.0, 0.0];
        let predicted = CovarianceTraceAcquisition::reduction(&system, &belief, &direction)
            .expect("direction is valid");
        let updated = belief
            .condition_on_projection(&system, &direction)
            .expect("direction is informative");
        let actual = 2.0 - updated.covariance()[0] - updated.covariance()[3];
        assert!((predicted - actual).abs() <= 1.0e-12);
    }

    #[test]
    fn selector_returns_global_trace_reduction_maximum() {
        let system =
            SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2).expect("system is valid");
        let belief = identity_belief(2);
        let candidates = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let selected = CovarianceTraceAcquisition::select_best(&system, &belief, &candidates)
            .expect("candidates are valid");
        for candidate in &candidates {
            let reduction = CovarianceTraceAcquisition::reduction(&system, &belief, candidate)
                .expect("candidate is valid");
            assert!(selected.trace_reduction() + 1.0e-12 >= reduction);
        }
    }

    #[test]
    fn sequential_selection_does_not_use_rhs_values() {
        let matrix = [4.0, 1.0, 1.0, 3.0];
        let first_system = SpdLinearSystem::new(&matrix, &[1.0, 2.0], 2).expect("valid system");
        let second_system = SpdLinearSystem::new(&matrix, &[-7.0, 4.0], 2).expect("valid system");
        let candidates = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let solver = CovarianceGreedyProjectionSolver::new(0.0, 2).expect("solver is valid");
        let first = solver
            .solve(&first_system, &identity_belief(2), &candidates)
            .expect("solve is valid");
        let second = solver
            .solve(&second_system, &identity_belief(2), &candidates)
            .expect("solve is valid");
        let first_directions: Vec<&[f64]> = first
            .steps()
            .iter()
            .map(super::CovarianceGreedyStep::direction)
            .collect();
        let second_directions: Vec<&[f64]> = second
            .steps()
            .iter()
            .map(super::CovarianceGreedyStep::direction)
            .collect();
        assert_eq!(first_directions, second_directions);
    }

    #[test]
    fn reports_projection_budget_termination() {
        let system =
            SpdLinearSystem::new(&[4.0, 1.0, 1.0, 3.0], &[1.0, 2.0], 2).expect("system is valid");
        let candidates = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let solver = CovarianceGreedyProjectionSolver::new(0.0, 1).expect("solver is valid");
        let result = solver
            .solve(&system, &identity_belief(2), &candidates)
            .expect("solve is valid");
        assert_eq!(
            result.termination(),
            CovarianceGreedyTermination::ProjectionBudgetReached
        );
        assert_eq!(result.steps().len(), 1);
    }

    #[test]
    fn rejects_empty_candidates() {
        let system = SpdLinearSystem::new(&[1.0], &[1.0], 1).expect("system is valid");
        let solver = CovarianceGreedyProjectionSolver::new(0.0, 1).expect("solver is valid");
        assert_eq!(
            solver.solve(&system, &identity_belief(1), &[]),
            Err(LinearSolverError::EmptyCandidateDirections)
        );
    }
}
