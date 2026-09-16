//! End-to-end one-dimensional Bayesian quadrature for the first supported pair.

use crate::{
    BayesianQuadratureError, GaussianConditioner, GaussianMeasure, KernelIntegral, KernelMean,
    RbfKernel, ScalarKernel, ScalarNormalPosterior,
};

/// Bayesian quadrature with an RBF covariance kernel and Gaussian integration measure.
///
/// The current implementation assumes a zero Gaussian-process prior mean. This is
/// intentionally explicit rather than hidden behind a generic prior-mean abstraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BayesianQuadrature {
    kernel: RbfKernel,
    measure: GaussianMeasure,
    jitter: f64,
}

impl BayesianQuadrature {
    /// Construct the first supported Bayesian quadrature configuration.
    #[must_use]
    pub const fn new(kernel: RbfKernel, measure: GaussianMeasure, jitter: f64) -> Self {
        Self {
            kernel,
            measure,
            jitter,
        }
    }

    /// Return the RBF kernel.
    #[must_use]
    pub const fn kernel(&self) -> RbfKernel {
        self.kernel
    }

    /// Return the Gaussian integration measure.
    #[must_use]
    pub const fn measure(&self) -> GaussianMeasure {
        self.measure
    }

    /// Return the fixed diagonal jitter used by Gaussian conditioning.
    #[must_use]
    pub const fn jitter(&self) -> f64 {
        self.jitter
    }

    /// Compute the posterior distribution of the integral from observed function values.
    ///
    /// For observations `y = f(X)` and zero prior mean,
    ///
    /// ```text
    /// posterior_mean = z^T (K + jitter I)^(-1) y
    /// posterior_var  = kappa - z^T (K + jitter I)^(-1) z
    /// ```
    ///
    /// The inverse is never formed explicitly; both systems are solved from one
    /// reusable Cholesky factorization.
    ///
    /// # Errors
    ///
    /// Returns [`BayesianQuadratureError`] for invalid observations, conditioning
    /// failures, or an invalid posterior variance.
    pub fn posterior(
        &self,
        nodes: &[f64],
        values: &[f64],
    ) -> Result<ScalarNormalPosterior, BayesianQuadratureError> {
        validate_observations(nodes, values)?;

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
        let alpha = conditioner.solve(values)?;
        let v = conditioner.solve(&kernel_mean)?;

        let posterior_mean = dot(&kernel_mean, &alpha);
        let raw_variance = self.kernel.kernel_integral(&self.measure) - dot(&kernel_mean, &v);
        let posterior_variance = non_negative_roundoff_variance(
            raw_variance,
            self.kernel.kernel_integral(&self.measure),
        )?;

        Ok(ScalarNormalPosterior::new(
            posterior_mean,
            posterior_variance,
        )?)
    }
}

fn validate_observations(
    nodes: &[f64],
    values: &[f64],
) -> Result<(), BayesianQuadratureError> {
    if nodes.is_empty() {
        return Err(BayesianQuadratureError::EmptyObservations);
    }
    if nodes.len() != values.len() {
        return Err(BayesianQuadratureError::ObservationLengthMismatch);
    }
    if nodes.iter().any(|value| !value.is_finite()) {
        return Err(BayesianQuadratureError::NonFiniteObservationNode);
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(BayesianQuadratureError::NonFiniteObservationValue);
    }

    Ok(())
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right).map(|(x, y)| x * y).sum()
}

fn non_negative_roundoff_variance(
    value: f64,
    scale: f64,
) -> Result<f64, BayesianQuadratureError> {
    if value >= 0.0 {
        return Ok(value);
    }

    let tolerance = 64.0 * f64::EPSILON * scale.abs().max(1.0);
    if value >= -tolerance {
        Ok(0.0)
    } else {
        Err(BayesianQuadratureError::MateriallyNegativePosteriorVariance {
            value,
            tolerance,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{BayesianQuadrature, non_negative_roundoff_variance};
    use crate::{BayesianQuadratureError, ConditioningError, GaussianMeasure, RbfKernel};

    const TOLERANCE: f64 = 1.0e-11;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOLERANCE * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    fn fixture() -> BayesianQuadrature {
        let kernel = RbfKernel::new(1.0, 1.2).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
        BayesianQuadrature::new(kernel, measure, 1.0e-12)
    }

    #[test]
    fn one_observation_matches_closed_form_conditioning() {
        let quadrature = fixture();
        let posterior = quadrature
            .posterior(&[0.0], &[2.0])
            .expect("one-node posterior is valid");

        let z = quadrature.kernel().kernel_mean(&quadrature.measure(), 0.0);
        let kappa = quadrature.kernel().kernel_integral(&quadrature.measure());
        let denominator = quadrature.kernel().signal_variance() + quadrature.jitter();

        assert_close(posterior.mean(), z * 2.0 / denominator);
        assert_close(posterior.variance(), kappa - z * z / denominator);
    }

    #[test]
    fn zero_observations_produce_zero_posterior_mean() {
        let quadrature = fixture();
        let posterior = quadrature
            .posterior(&[-1.0, 0.0, 1.0], &[0.0, 0.0, 0.0])
            .expect("posterior is valid");

        assert_close(posterior.mean(), 0.0);
        assert!(posterior.variance() >= 0.0);
    }

    #[test]
    fn posterior_variance_does_not_depend_on_observed_values() {
        let quadrature = fixture();
        let first = quadrature
            .posterior(&[-0.75, 0.25, 1.5], &[1.0, 2.0, -1.0])
            .expect("posterior is valid");
        let second = quadrature
            .posterior(&[-0.75, 0.25, 1.5], &[10.0, -5.0, 3.0])
            .expect("posterior is valid");

        assert_close(first.variance(), second.variance());
    }

    #[test]
    fn posterior_variance_is_no_greater_than_prior_integral_variance() {
        let quadrature = fixture();
        let posterior = quadrature
            .posterior(&[-2.0, -0.5, 0.5, 2.0], &[1.0, 0.5, -0.5, -1.0])
            .expect("posterior is valid");
        let prior_variance = quadrature.kernel().kernel_integral(&quadrature.measure());

        assert!(posterior.variance() >= 0.0);
        assert!(posterior.variance() <= prior_variance + TOLERANCE);
    }

    #[test]
    fn rejects_invalid_observations() {
        let quadrature = fixture();

        assert_eq!(
            quadrature.posterior(&[], &[]),
            Err(BayesianQuadratureError::EmptyObservations)
        );
        assert_eq!(
            quadrature.posterior(&[0.0], &[1.0, 2.0]),
            Err(BayesianQuadratureError::ObservationLengthMismatch)
        );
        assert_eq!(
            quadrature.posterior(&[f64::NAN], &[1.0]),
            Err(BayesianQuadratureError::NonFiniteObservationNode)
        );
        assert_eq!(
            quadrature.posterior(&[0.0], &[f64::INFINITY]),
            Err(BayesianQuadratureError::NonFiniteObservationValue)
        );
    }

    #[test]
    fn duplicate_nodes_fail_without_jitter() {
        let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
        let quadrature = BayesianQuadrature::new(kernel, measure, 0.0);

        let result = quadrature.posterior(&[0.0, 0.0], &[1.0, 1.0]);

        assert!(matches!(
            result,
            Err(BayesianQuadratureError::Conditioning(
                ConditioningError::NotPositiveDefinite
            ))
        ));
    }

    #[test]
    fn duplicate_nodes_can_be_regularized_with_explicit_jitter() {
        let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
        let quadrature = BayesianQuadrature::new(kernel, measure, 1.0e-8);

        let posterior = quadrature
            .posterior(&[0.0, 0.0], &[1.0, 1.0])
            .expect("jitter regularizes duplicate nodes");

        assert!(posterior.mean().is_finite());
        assert!(posterior.variance() >= 0.0);
    }

    #[test]
    fn tiny_negative_variance_is_clamped_to_zero() {
        let scale = 2.0;
        let tiny_negative = -32.0 * f64::EPSILON * scale;

        assert_eq!(
            non_negative_roundoff_variance(tiny_negative, scale),
            Ok(0.0)
        );
    }

    #[test]
    fn material_negative_variance_is_rejected() {
        let result = non_negative_roundoff_variance(-1.0e-6, 1.0);

        assert!(matches!(
            result,
            Err(BayesianQuadratureError::MateriallyNegativePosteriorVariance { .. })
        ));
    }
}
