//! Analytic double kernel integrals for supported kernel and probability-measure pairs.

use crate::{GaussianMeasure, RbfKernel};

/// Analytic double integral of a scalar kernel under a probability measure.
///
/// For a kernel `k` and probability measure `p`, this quantity is
///
/// ```text
/// kappa = double integral k(x, x') p(x) p(x') dx dx'.
/// ```
///
/// The trait is implemented only for kernel/measure pairs with an explicit
/// analytic contract.
pub trait KernelIntegral<M> {
    /// Evaluate the analytic double kernel integral.
    #[must_use]
    fn kernel_integral(&self, measure: &M) -> f64;
}

impl KernelIntegral<GaussianMeasure> for RbfKernel {
    fn kernel_integral(&self, measure: &GaussianMeasure) -> f64 {
        let length_scale_squared = self.length_scale() * self.length_scale();
        let denominator = length_scale_squared + 2.0 * measure.variance();

        self.signal_variance() * (length_scale_squared / denominator).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::KernelIntegral;
    use crate::{ContinuousProbabilityMeasure, GaussianMeasure, RbfKernel, ScalarKernel};

    const TOLERANCE: f64 = 1.0e-12;
    const QUADRATURE_TOLERANCE: f64 = 1.0e-8;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    fn simpson_weights(intervals: u32) -> Vec<f64> {
        assert!(intervals > 0);
        assert_eq!(intervals % 2, 0);

        let capacity = usize::try_from(intervals + 1).expect("interval count fits in usize");
        let mut weights = Vec::with_capacity(capacity);
        for index in 0..=intervals {
            let weight = if index == 0 || index == intervals {
                1.0
            } else if index % 2 == 0 {
                2.0
            } else {
                4.0
            };
            weights.push(weight);
        }
        weights
    }

    fn simpson_double_integral<F>(
        function: F,
        lower: f64,
        upper: f64,
        intervals: u32,
    ) -> f64
    where
        F: Fn(f64, f64) -> f64,
    {
        let step = (upper - lower) / f64::from(intervals);
        let weights = simpson_weights(intervals);
        let mut weighted_sum = 0.0;

        for (i, &weight_x) in weights.iter().enumerate() {
            let i_u32 = u32::try_from(i).expect("index fits in u32");
            let x = lower + f64::from(i_u32) * step;

            for (j, &weight_y) in weights.iter().enumerate() {
                let j_u32 = u32::try_from(j).expect("index fits in u32");
                let y = lower + f64::from(j_u32) * step;
                weighted_sum += weight_x * weight_y * function(x, y);
            }
        }

        weighted_sum * step * step / 9.0
    }

    #[test]
    fn kernel_integral_matches_closed_form() {
        let kernel = RbfKernel::new(2.5, 0.75).expect("kernel parameters are valid");
        let measure = GaussianMeasure::new(1.25, 1.6).expect("measure parameters are valid");
        let length_scale_squared = kernel.length_scale() * kernel.length_scale();
        let expected = kernel.signal_variance()
            * (length_scale_squared / (length_scale_squared + 2.0 * measure.variance())).sqrt();

        assert_close(kernel.kernel_integral(&measure), expected, TOLERANCE);
    }

    #[test]
    fn kernel_integral_is_invariant_to_measure_mean() {
        let kernel = RbfKernel::new(1.8, 0.9).expect("kernel parameters are valid");
        let left = GaussianMeasure::new(-10.0, 2.0).expect("measure parameters are valid");
        let right = GaussianMeasure::new(25.0, 2.0).expect("measure parameters are valid");

        assert_close(
            kernel.kernel_integral(&left),
            kernel.kernel_integral(&right),
            TOLERANCE,
        );
    }

    #[test]
    fn kernel_integral_is_positive_and_bounded_by_signal_variance() {
        let kernel = RbfKernel::new(3.0, 0.6).expect("kernel parameters are valid");

        for variance in [0.01, 0.25, 1.0, 10.0, 100.0] {
            let measure = GaussianMeasure::new(0.0, variance).expect("measure parameters are valid");
            let integral = kernel.kernel_integral(&measure);

            assert!(integral > 0.0);
            assert!(integral <= kernel.signal_variance());
        }
    }

    #[test]
    fn kernel_integral_decreases_as_measure_variance_increases() {
        let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
        let narrow = GaussianMeasure::new(0.0, 0.1).expect("measure parameters are valid");
        let medium = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
        let wide = GaussianMeasure::new(0.0, 10.0).expect("measure parameters are valid");

        assert!(kernel.kernel_integral(&narrow) > kernel.kernel_integral(&medium));
        assert!(kernel.kernel_integral(&medium) > kernel.kernel_integral(&wide));
    }

    #[test]
    fn analytic_kernel_integral_matches_independent_numerical_quadrature() {
        let cases = [
            (1.0, 1.0, 0.0, 1.0),
            (2.5, 0.4, -1.0, 2.0),
            (0.7, 3.0, 4.0, 0.25),
        ];

        for (signal_variance, length_scale, measure_mean, measure_variance) in cases {
            let kernel = RbfKernel::new(signal_variance, length_scale)
                .expect("kernel parameters are valid");
            let measure = GaussianMeasure::new(measure_mean, measure_variance)
                .expect("measure parameters are valid");
            let standard_deviation = measure.standard_deviation();
            let lower = measure.mean() - 8.0 * standard_deviation;
            let upper = measure.mean() + 8.0 * standard_deviation;

            let numerical = simpson_double_integral(
                |x, y| {
                    kernel.covariance(x, y) * measure.density(x) * measure.density(y)
                },
                lower,
                upper,
                400,
            );
            let analytic = kernel.kernel_integral(&measure);

            assert_close(analytic, numerical, QUADRATURE_TOLERANCE);
        }
    }
}
