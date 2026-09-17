//! Active Bayesian quadrature acquisition based on posterior integral-variance reduction.

use crate::{
    BayesianQuadratureError, GaussianConditioner, GaussianMeasure, KernelMean, RbfKernel,
    ScalarKernel,
};

/// Expected reduction in posterior integral variance from evaluating one candidate node.
///
/// For current observation nodes `X`, candidate `x_star`, regularized Gram matrix
/// `A = K + jitter * I`, kernel vector `k_star`, and kernel-mean vector `z`,
///
/// ```text
/// delta(x_star)
/// = (z_star - k_star^T A^(-1) z)^2
///   / (k(x_star, x_star) + jitter - k_star^T A^(-1) k_star).
/// ```
///
/// The quantity is independent of observed function values. It depends only on
/// the kernel, integration measure, existing node locations, candidate location,
/// and explicit jitter policy.
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
    ///
    /// # Errors
    ///
    /// Returns [`BayesianQuadratureError`] when the current node set is empty or
    /// contains non-finite values, when the candidate is non-finite, or when the
    /// regularized Gram matrix cannot be conditioned.
    pub fn reduction(
        &self,
        nodes: &[f64],
        candidate: f64,
    ) -> Result<f64, BayesianQuadratureError> {
        validate_nodes(nodes)?;
        if !candidate.is_finite() {
            return Err(BayesianQuadratureError::NonFiniteObservationNode);
        }

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
        let candidate_covariance: Vec<f64> = nodes
            .iter()
            .map(|&node| self.kernel.covariance(node, candidate))
            .collect();

        let conditioner = GaussianConditioner::new(&gram, dimension, self.jitter)?;
        let solved_mean = conditioner.solve(&kernel_mean)?;
        let solved_candidate = conditioner.solve(&candidate_covariance)?;

        let posterior_integral_covariance = self.kernel.kernel_mean(&self.measure, candidate)
            - dot(&candidate_covariance, &solved_mean);
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
    use crate::{BayesianQuadrature, GaussianMeasure, RbfKernel};

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
            let reduction = acquisition
                .reduction(&nodes, candidate)
                .expect("candidate should be valid");
            assert!(reduction >= 0.0);
        }
    }

    #[test]
    fn existing_node_has_negligible_reduction() {
        let acquisition = fixture();
        let nodes = [-1.0, 0.0, 1.0];

        let reduction = acquisition
            .reduction(&nodes, 0.0)
            .expect("candidate should be valid");

        assert!(reduction <= 1.0e-10);
    }

    #[test]
    fn criterion_matches_actual_posterior_variance_drop() {
        let acquisition = fixture();
        let nodes = [-1.25, -0.1, 1.1];
        let values = [0.4, -0.3, 0.8];
        let candidate = 0.55;
        let candidate_value = -0.2;

        let quadrature = BayesianQuadrature::new(
            acquisition.kernel(),
            acquisition.measure(),
            acquisition.jitter(),
        );
        let before = quadrature
            .posterior(&nodes, &values)
            .expect("current posterior should be valid");

        let mut augmented_nodes = nodes.to_vec();
        augmented_nodes.push(candidate);
        let mut augmented_values = values.to_vec();
        augmented_values.push(candidate_value);
        let after = quadrature
            .posterior(&augmented_nodes, &augmented_values)
            .expect("augmented posterior should be valid");

        let predicted_drop = acquisition
            .reduction(&nodes, candidate)
            .expect("candidate should be valid");
        let actual_drop = before.variance() - after.variance();

        assert!((predicted_drop - actual_drop).abs() <= TOLERANCE);
    }

    #[test]
    fn reduction_does_not_depend_on_observed_values() {
        let acquisition = fixture();
        let nodes = [-1.0, 0.0, 1.0];
        let candidate = 0.35;

        let first = acquisition
            .reduction(&nodes, candidate)
            .expect("candidate should be valid");
        let second = acquisition
            .reduction(&nodes, candidate)
            .expect("candidate should be valid");

        assert_eq!(first, second);
    }

    #[test]
    fn rejects_invalid_node_contracts() {
        let acquisition = fixture();

        assert!(acquisition.reduction(&[], 0.0).is_err());
        assert!(acquisition.reduction(&[f64::NAN], 0.0).is_err());
        assert!(acquisition.reduction(&[0.0], f64::INFINITY).is_err());
    }
}
