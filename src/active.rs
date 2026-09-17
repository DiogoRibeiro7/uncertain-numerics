//! Active Bayesian quadrature acquisition based on posterior integral-variance reduction.

use crate::{
    ActiveSelectionError, BayesianQuadratureError, GaussianConditioner, GaussianMeasure,
    KernelMean, RbfKernel, ScalarKernel,
};

/// Candidate selected by posterior integral-variance reduction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectedCandidate {
    point: f64,
    variance_reduction: f64,
    index: usize,
}

impl SelectedCandidate {
    /// Return the selected candidate location.
    #[must_use]
    pub const fn point(self) -> f64 {
        self.point
    }

    /// Return the predicted posterior integral-variance reduction.
    #[must_use]
    pub const fn variance_reduction(self) -> f64 {
        self.variance_reduction
    }

    /// Return the selected candidate index in the supplied candidate slice.
    #[must_use]
    pub const fn index(self) -> usize {
        self.index
    }
}

/// Expected reduction in posterior integral variance from evaluating one candidate node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceReductionAcquisition {
    kernel: RbfKernel,
    measure: GaussianMeasure,
    jitter: f64,
}

impl VarianceReductionAcquisition {
    /// Construct a variance-reduction acquisition function.
    #[must_use]
    pub const fn new(kernel: RbfKernel, measure: GaussianMeasure, jitter: f64) -> Self {
        Self {
            kernel,
            measure,
            jitter,
        }
    }

    /// Return the configured kernel.
    #[must_use]
    pub const fn kernel(&self) -> RbfKernel {
        self.kernel
    }

    /// Return the configured integration measure.
    #[must_use]
    pub const fn measure(&self) -> GaussianMeasure {
        self.measure
    }

    /// Return the explicit diagonal jitter.
    #[must_use]
    pub const fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Evaluate the posterior integral-variance reduction at `candidate`.
    pub fn reduction(
        &self,
        nodes: &[f64],
        candidate: f64,
    ) -> Result<f64, BayesianQuadratureError> {
        validate_nodes(nodes)?;
        if !candidate.is_finite() {
            return Err(BayesianQuadratureError::NonFiniteObservationNode);
        }

        let (conditioner, solved_mean) = self.prepare(nodes)?;
        self.reduction_with_prepared(nodes, candidate, &conditioner, &solved_mean)
    }

    /// Select the candidate with the largest predicted variance reduction.
    ///
    /// Ties are resolved deterministically by keeping the first maximum in the
    /// supplied candidate slice.
    ///
    /// # Errors
    ///
    /// Returns [`ActiveSelectionError`] for empty/non-finite candidate sets or
    /// when acquisition evaluation fails for the current design.
    pub fn select_best(
        &self,
        nodes: &[f64],
        candidates: &[f64],
    ) -> Result<SelectedCandidate, ActiveSelectionError> {
        if candidates.is_empty() {
            return Err(ActiveSelectionError::EmptyCandidates);
        }
        if candidates.iter().any(|candidate| !candidate.is_finite()) {
            return Err(ActiveSelectionError::NonFiniteCandidate);
        }

        validate_nodes(nodes)?;
        let (conditioner, solved_mean) = self.prepare(nodes)?;

        let mut best = SelectedCandidate {
            point: candidates[0],
            variance_reduction: self.reduction_with_prepared(
                nodes,
                candidates[0],
                &conditioner,
                &solved_mean,
            )?,
            index: 0,
        };

        for (index, &candidate) in candidates.iter().enumerate().skip(1) {
            let reduction = self.reduction_with_prepared(
                nodes,
                candidate,
                &conditioner,
                &solved_mean,
            )?;
            if reduction > best.variance_reduction {
                best = SelectedCandidate {
                    point: candidate,
                    variance_reduction: reduction,
                    index,
                };
            }
        }

        Ok(best)
    }

    fn prepare(
        &self,
        nodes: &[f64],
    ) -> Result<(GaussianConditioner, Vec<f64>), BayesianQuadratureError> {
        let dimension = nodes.len();
        let mut gram = Vec::with_capacity(dimension * dimension);
        for &left in nodes {
            for &right in nodes {
                gram.push(self.kernel.covariance(left, right));
            }
        }

        let kernel_mean: Vec<f64> = nodes
            .iter()
            .map(|&node| self.kernel.kernel_mean(&self.measure, node))
            .collect();
        let conditioner = GaussianConditioner::new(&gram, dimension, self.jitter)?;
        let solved_mean = conditioner.solve(&kernel_mean)?;
        Ok((conditioner, solved_mean))
    }

    fn reduction_with_prepared(
        &self,
        nodes: &[f64],
        candidate: f64,
        conditioner: &GaussianConditioner,
        solved_mean: &[f64],
    ) -> Result<f64, BayesianQuadratureError> {
        let candidate_covariance: Vec<f64> = nodes
            .iter()
            .map(|&node| self.kernel.covariance(node, candidate))
            .collect();
        let solved_candidate = conditioner.solve(&candidate_covariance)?;

        let posterior_integral_covariance = self.kernel.kernel_mean(&self.measure, candidate)
            - dot(&candidate_covariance, solved_mean);
        let predictive_variance = self.kernel.covariance(candidate, candidate) + self.jitter
            - dot(&candidate_covariance, &solved_candidate);

        let tolerance = 64.0 * f64::EPSILON * self.kernel.signal_variance().max(1.0);
        if predictive_variance <= tolerance {
            return Ok(0.0);
        }

        Ok(
            posterior_integral_covariance * posterior_integral_covariance
                / predictive_variance,
        )
    }
}

fn validate_nodes(nodes: &[f64]) -> Result<(), BayesianQuadratureError> {
    if nodes.is_empty() {
        return Err(BayesianQuadratureError::EmptyObservations);
    }
    if nodes.iter().any(|value| !value.is_finite()) {
        return Err(BayesianQuadratureError::NonFiniteObservationNode);
    }
    Ok(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::VarianceReductionAcquisition;
    use crate::{ActiveSelectionError, BayesianQuadrature, GaussianMeasure, RbfKernel};

    const TOLERANCE: f64 = 1.0e-11;

    fn fixture() -> VarianceReductionAcquisition {
        let kernel = RbfKernel::new(1.0, 1.1).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(0.2, 1.0).expect("measure parameters are valid");
        VarianceReductionAcquisition::new(kernel, measure, 1.0e-12)
    }

    #[test]
    fn reduction_is_non_negative() {
        let acquisition = fixture();
        let nodes = [-1.0, 0.0, 1.0];
        for candidate in [-2.0, -0.4, 0.5, 1.7] {
            assert!(acquisition.reduction(&nodes, candidate).expect("valid") >= 0.0);
        }
    }

    #[test]
    fn existing_node_has_negligible_reduction() {
        let acquisition = fixture();
        let reduction = acquisition.reduction(&[-1.0, 0.0, 1.0], 0.0).expect("valid");
        assert!(reduction <= 1.0e-10);
    }

    #[test]
    fn criterion_matches_actual_posterior_variance_drop() {
        let acquisition = fixture();
        let nodes = [-1.25, -0.1, 1.1];
        let values = [0.4, -0.3, 0.8];
        let candidate = 0.55;
        let quadrature = BayesianQuadrature::new(
            acquisition.kernel(),
            acquisition.measure(),
            acquisition.jitter(),
        );
        let before = quadrature.posterior(&nodes, &values).expect("valid");
        let mut augmented_nodes = nodes.to_vec();
        augmented_nodes.push(candidate);
        let mut augmented_values = values.to_vec();
        augmented_values.push(-0.2);
        let after = quadrature
            .posterior(&augmented_nodes, &augmented_values)
            .expect("valid");
        let predicted_drop = acquisition.reduction(&nodes, candidate).expect("valid");
        assert!((predicted_drop - (before.variance() - after.variance())).abs() <= TOLERANCE);
    }

    #[test]
    fn selector_returns_global_candidate_maximum() {
        let acquisition = fixture();
        let nodes = [-1.0, 0.0, 1.0];
        let candidates = [-2.0, -0.6, 0.4, 1.8];
        let selected = acquisition.select_best(&nodes, &candidates).expect("valid");
        for &candidate in &candidates {
            let reduction = acquisition.reduction(&nodes, candidate).expect("valid");
            assert!(selected.variance_reduction() + TOLERANCE >= reduction);
        }
        assert_eq!(selected.point(), candidates[selected.index()]);
    }

    #[test]
    fn selector_uses_first_maximum_for_ties() {
        let acquisition = fixture();
        let nodes = [-1.0, 0.0, 1.0];
        let candidates = [0.0, 0.0, 0.0];
        let selected = acquisition.select_best(&nodes, &candidates).expect("valid");
        assert_eq!(selected.index(), 0);
    }

    #[test]
    fn selector_rejects_invalid_candidate_sets() {
        let acquisition = fixture();
        assert_eq!(
            acquisition.select_best(&[-1.0, 0.0, 1.0], &[]),
            Err(ActiveSelectionError::EmptyCandidates)
        );
        assert_eq!(
            acquisition.select_best(&[-1.0, 0.0, 1.0], &[0.5, f64::NAN]),
            Err(ActiveSelectionError::NonFiniteCandidate)
        );
    }
}
