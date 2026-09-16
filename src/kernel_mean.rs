//! Analytic kernel means for supported kernel and probability-measure pairs.

use crate::{GaussianMeasure, RbfKernel};

/// Analytic mean embedding of a scalar kernel under a probability measure.
///
/// For a kernel `k` and probability measure `p`, the kernel mean at `x` is
///
/// ```text
/// z(x) = integral k(x, u) p(u) du.
/// ```
///
/// The trait is implemented only for kernel/measure pairs with an explicit
/// analytic contract. Unsupported pairs therefore fail at compile time rather
/// than silently falling back to an unrelated numerical approximation.
pub trait KernelMean<M> {
    /// Evaluate the analytic kernel mean at `x`.
    #[must_use]
    fn kernel_mean(&self, measure: &M, x: f64) -> f64;
}

impl KernelMean<GaussianMeasure> for RbfKernel {
    fn kernel_mean(&self, measure: &GaussianMeasure, x: f64) -> f64 {
        let length_scale_squared = self.length_scale() * self.length_scale();
        let combined_variance = length_scale_squared + measure.variance();
        let centered = x - measure.mean();
        let scale = self.signal_variance() * (length_scale_squared / combined_variance).sqrt();
        let exponent = -(centered * centered) / (2.0 * combined_variance);

        scale * exponent.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::KernelMean;
    use crate::{ContinuousProbabilityMeasure, GaussianMeasure, RbfKernel, ScalarKernel};

    const TOLERANCE: f64 = 1.0e-12;
    const QUADRATURE_TOLERANCE: f64 = 1.0e-9;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    fn simpson_integral<F>(function: F, lower: f64, upper: f64, intervals: u32) -> f64
    where
        F: Fn(f64) -> f64,
    {
        assert!(intervals > 0);
        assert_eq!(intervals % 2, 0);

        let step = (upper - lower) / f64::from(intervals);
        let mut weighted_sum = function(lower) + function(upper);

        for index in 1..intervals {
            let x = lower + f64::from(index) * step;
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            weighted_sum += weight * function(x);
        }

        weighted_sum * step / 3.0
    }

    #[test]
    fn kernel_mean_at_measure_mean_has_closed_form_scale() {
        let kernel = RbfKernel::new(2.5, 0.75).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(-1.25, 1.6).expect("measure parameters are valid");
        let length_scale_squared = kernel.length_scale() * kernel.length_scale();
        let expected = kernel.signal_variance()
            * (length_scale_squared / (length_scale_squared + measure.variance())).sqrt();

        assert_close(
            kernel.kernel_mean(&measure, measure.mean()),
            expected,
            TOLERANCE,
        );
    }

    #[test]
    fn kernel_mean_is_symmetric_around_measure_mean() {
        let kernel = RbfKernel::new(1.7, 0.9).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(2.0, 1.3).expect("measure parameters are valid");

        assert_close(
            kernel.kernel_mean(&measure, 0.75),
            kernel.kernel_mean(&measure, 3.25),
            TOLERANCE,
        );
    }

    #[test]
    fn kernel_mean_is_translation_invariant() {
        let kernel = RbfKernel::new(1.2, 1.8).expect("kernel parameters are valid");
        let original = GaussianMeasure::new(-0.5, 0.7).expect("measure parameters are valid");
        let shifted = GaussianMeasure::new(9.5, 0.7).expect("measure parameters are valid");

        assert_close(
            kernel.kernel_mean(&original, 1.25),
            kernel.kernel_mean(&shifted, 11.25),
            TOLERANCE,
        );
    }

    #[test]
    fn kernel_mean_is_positive_and_bounded_by_signal_variance() {
        let kernel = RbfKernel::new(3.0, 0.6).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(0.0, 2.0).expect("measure parameters are valid");

        for x in [-8.0, -2.0, 0.0, 1.5, 10.0] {
            let mean = kernel.kernel_mean(&measure, x);
            assert!(mean > 0.0);
            assert!(mean <= kernel.signal_variance());
        }
    }

    #[test]
    fn analytic_kernel_mean_matches_independent_numerical_quadrature() {
        let cases = [
            (1.0, 1.0, 0.0, 1.0, 0.0),
            (2.5, 0.4, -1.0, 2.0, 0.75),
            (0.7, 3.0, 4.0, 0.25, 5.2),
            (1.8, 0.8, 1.5, 3.5, -2.0),
        ];

        for (signal_variance, length_scale, measure_mean, measure_variance, x) in cases {
            let kernel = RbfKernel::new(signal_variance, length_scale)
                .expect("kernel parameters are valid");
            let measure = GaussianMeasure::new(measure_mean, measure_variance)
                .expect("measure parameters are valid");
            let standard_deviation = measure.standard_deviation();
            let lower = measure.mean() - 10.0 * standard_deviation;
            let upper = measure.mean() + 10.0 * standard_deviation;

            let numerical = simpson_integral(
                |integration_point| {
                    kernel.covariance(x, integration_point) * measure.density(integration_point)
                },
                lower,
                upper,
                20_000,
            );
            let analytic = kernel.kernel_mean(&measure, x);

            assert_close(analytic, numerical, QUADRATURE_TOLERANCE);
        }
    }
}
