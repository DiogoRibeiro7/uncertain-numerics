use crate::{
    ActiveDesignError, BayesianQuadrature, ScalarNormalPosterior, VarianceReductionAcquisition,
};

/// Reason a sequential active Bayesian-quadrature run terminated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveTermination {
    /// The posterior integral variance reached the requested tolerance.
    VarianceToleranceReached,
    /// The configured budget of new function evaluations was exhausted.
    EvaluationBudgetReached,
    /// No unevaluated candidates remain.
    CandidatesExhausted,
}

/// One function evaluation selected by the active design.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveDesignStep {
    point: f64,
    value: f64,
    predicted_variance_reduction: f64,
    posterior_variance: f64,
}

impl ActiveDesignStep {
    /// Selected evaluation point.
    #[must_use]
    pub const fn point(self) -> f64 {
        self.point
    }

    /// Observed function value at the selected point.
    #[must_use]
    pub const fn value(self) -> f64 {
        self.value
    }

    /// Predicted variance reduction before the point was evaluated.
    #[must_use]
    pub const fn predicted_variance_reduction(self) -> f64 {
        self.predicted_variance_reduction
    }

    /// Posterior integral variance after adding this observation.
    #[must_use]
    pub const fn posterior_variance(self) -> f64 {
        self.posterior_variance
    }
}

/// Result of a sequential active Bayesian-quadrature run.
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveDesignResult {
    nodes: Vec<f64>,
    values: Vec<f64>,
    remaining_candidates: Vec<f64>,
    steps: Vec<ActiveDesignStep>,
    posterior: ScalarNormalPosterior,
    termination: ActiveTermination,
}

impl ActiveDesignResult {
    /// Final observation nodes, including actively selected points.
    #[must_use]
    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    /// Final observed function values.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Candidates not selected during this run.
    #[must_use]
    pub fn remaining_candidates(&self) -> &[f64] {
        &self.remaining_candidates
    }

    /// Ordered active-evaluation history.
    #[must_use]
    pub fn steps(&self) -> &[ActiveDesignStep] {
        &self.steps
    }

    /// Final integral posterior.
    #[must_use]
    pub const fn posterior(&self) -> ScalarNormalPosterior {
        self.posterior
    }

    /// Reason the active design stopped.
    #[must_use]
    pub const fn termination(&self) -> ActiveTermination {
        self.termination
    }
}

/// Sequential active Bayesian quadrature over a finite candidate set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveBayesianQuadrature {
    quadrature: BayesianQuadrature,
    acquisition: VarianceReductionAcquisition,
}

impl ActiveBayesianQuadrature {
    /// Construct active Bayesian quadrature from one consistent kernel/measure setup.
    #[must_use]
    pub const fn new(quadrature: BayesianQuadrature) -> Self {
        let acquisition = VarianceReductionAcquisition::new(
            quadrature.kernel(),
            quadrature.measure(),
            quadrature.jitter(),
        );
        Self {
            quadrature,
            acquisition,
        }
    }

    /// Return the underlying Bayesian-quadrature configuration.
    #[must_use]
    pub const fn quadrature(&self) -> BayesianQuadrature {
        self.quadrature
    }

    /// Run sequential active selection and function evaluation.
    ///
    /// The loop stops when the posterior integral variance is no greater than
    /// `variance_tolerance`, when `max_new_evaluations` points have been added,
    /// or when the finite candidate set is exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`ActiveDesignError`] for invalid stopping tolerance, non-finite
    /// candidates or function evaluations, candidate-selection failures, or
    /// posterior-construction failures.
    pub fn run<F>(
        &self,
        initial_nodes: &[f64],
        initial_values: &[f64],
        candidates: &[f64],
        max_new_evaluations: usize,
        variance_tolerance: f64,
        mut function: F,
    ) -> Result<ActiveDesignResult, ActiveDesignError>
    where
        F: FnMut(f64) -> f64,
    {
        if !variance_tolerance.is_finite() {
            return Err(ActiveDesignError::NonFiniteVarianceTolerance);
        }
        if variance_tolerance < 0.0 {
            return Err(ActiveDesignError::NegativeVarianceTolerance);
        }
        if candidates.iter().any(|candidate| !candidate.is_finite()) {
            return Err(ActiveDesignError::NonFiniteCandidate);
        }

        let mut nodes = initial_nodes.to_vec();
        let mut values = initial_values.to_vec();
        let mut remaining_candidates = candidates.to_vec();
        let mut steps = Vec::new();
        let mut posterior = self.quadrature.posterior(&nodes, &values)?;

        if posterior.variance() <= variance_tolerance {
            return Ok(ActiveDesignResult {
                nodes,
                values,
                remaining_candidates,
                steps,
                posterior,
                termination: ActiveTermination::VarianceToleranceReached,
            });
        }

        for _ in 0..max_new_evaluations {
            if remaining_candidates.is_empty() {
                return Ok(ActiveDesignResult {
                    nodes,
                    values,
                    remaining_candidates,
                    steps,
                    posterior,
                    termination: ActiveTermination::CandidatesExhausted,
                });
            }

            let selected = self
                .acquisition
                .select_best(&nodes, &remaining_candidates)?;
            let point = selected.point();
            let value = function(point);
            if !value.is_finite() {
                return Err(ActiveDesignError::NonFiniteFunctionValue);
            }

            remaining_candidates.remove(selected.index());
            nodes.push(point);
            values.push(value);
            posterior = self.quadrature.posterior(&nodes, &values)?;
            steps.push(ActiveDesignStep {
                point,
                value,
                predicted_variance_reduction: selected.variance_reduction(),
                posterior_variance: posterior.variance(),
            });

            if posterior.variance() <= variance_tolerance {
                return Ok(ActiveDesignResult {
                    nodes,
                    values,
                    remaining_candidates,
                    steps,
                    posterior,
                    termination: ActiveTermination::VarianceToleranceReached,
                });
            }
        }

        let termination = if remaining_candidates.is_empty() {
            ActiveTermination::CandidatesExhausted
        } else {
            ActiveTermination::EvaluationBudgetReached
        };

        Ok(ActiveDesignResult {
            nodes,
            values,
            remaining_candidates,
            steps,
            posterior,
            termination,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ActiveBayesianQuadrature, ActiveTermination};
    use crate::{ActiveDesignError, BayesianQuadrature, GaussianMeasure, RbfKernel};

    fn fixture() -> ActiveBayesianQuadrature {
        let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
        ActiveBayesianQuadrature::new(BayesianQuadrature::new(kernel, measure, 1.0e-12))
    }

    #[test]
    fn sequential_run_adds_selected_candidates_and_removes_them() {
        let active = fixture();
        let result = active
            .run(
                &[-1.0, 1.0],
                &[1.0, 1.0],
                &[-0.5, 0.0, 0.5, 2.0],
                2,
                0.0,
                |x| x * x,
            )
            .expect("active design should be valid");

        assert_eq!(result.nodes().len(), 4);
        assert_eq!(result.values().len(), 4);
        assert_eq!(result.steps().len(), 2);
        assert_eq!(result.remaining_candidates().len(), 2);
        for step in result.steps() {
            assert!(result.nodes().contains(&step.point()));
            assert!(!result.remaining_candidates().contains(&step.point()));
            assert!(step.predicted_variance_reduction() >= 0.0);
        }
    }

    #[test]
    fn posterior_variance_is_non_increasing_across_active_steps() {
        let active = fixture();
        let initial = active
            .quadrature()
            .posterior(&[-1.0, 1.0], &[1.0, 1.0])
            .expect("initial posterior is valid");
        let result = active
            .run(
                &[-1.0, 1.0],
                &[1.0, 1.0],
                &[-2.0, -0.5, 0.0, 0.5, 2.0],
                3,
                0.0,
                f64::cos,
            )
            .expect("active design should be valid");

        let mut previous = initial.variance();
        for step in result.steps() {
            assert!(step.posterior_variance() <= previous + 1.0e-12);
            previous = step.posterior_variance();
        }
    }

    #[test]
    fn stops_immediately_when_variance_tolerance_is_already_met() {
        let active = fixture();
        let initial = active
            .quadrature()
            .posterior(&[-1.0, 0.0, 1.0], &[1.0, 0.0, 1.0])
            .expect("initial posterior is valid");
        let result = active
            .run(
                &[-1.0, 0.0, 1.0],
                &[1.0, 0.0, 1.0],
                &[-0.5, 0.5],
                5,
                initial.variance(),
                |x| x * x,
            )
            .expect("active design should be valid");

        assert_eq!(
            result.termination(),
            ActiveTermination::VarianceToleranceReached
        );
        assert!(result.steps().is_empty());
    }

    #[test]
    fn reports_evaluation_budget_termination() {
        let active = fixture();
        let result = active
            .run(&[-1.0, 1.0], &[1.0, 1.0], &[-0.5, 0.0, 0.5], 1, 0.0, |x| {
                x * x
            })
            .expect("active design should be valid");

        assert_eq!(
            result.termination(),
            ActiveTermination::EvaluationBudgetReached
        );
        assert_eq!(result.steps().len(), 1);
    }

    #[test]
    fn reports_candidate_exhaustion() {
        let active = fixture();
        let result = active
            .run(&[-1.0, 1.0], &[1.0, 1.0], &[0.0], 3, 0.0, |x| x * x)
            .expect("active design should be valid");

        assert_eq!(result.termination(), ActiveTermination::CandidatesExhausted);
        assert_eq!(result.steps().len(), 1);
        assert!(result.remaining_candidates().is_empty());
    }

    #[test]
    fn rejects_invalid_tolerance_and_function_values() {
        let active = fixture();
        assert_eq!(
            active.run(&[-1.0, 1.0], &[1.0, 1.0], &[0.0], 1, -1.0, |x| x),
            Err(ActiveDesignError::NegativeVarianceTolerance)
        );
        assert_eq!(
            active.run(&[-1.0, 1.0], &[1.0, 1.0], &[0.0], 1, 0.0, |_| f64::NAN),
            Err(ActiveDesignError::NonFiniteFunctionValue)
        );
    }
}
