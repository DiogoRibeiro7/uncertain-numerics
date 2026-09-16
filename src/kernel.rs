//! Covariance kernels for one-dimensional probabilistic numerical methods.

use crate::KernelError;

/// Contract for a scalar covariance kernel.
///
/// Implementations are expected to represent symmetric positive-semidefinite
/// kernels on finite scalar inputs. The trait deliberately exposes only pointwise
/// covariance evaluation; matrix construction belongs to a later numerical layer.
pub trait ScalarKernel {
    /// Evaluate the covariance `k(x, y)`.
    ///
    /// Callers should provide finite coordinates. Implementations may propagate
    /// IEEE-754 non-finite values when supplied non-finite inputs.
    #[must_use]
    fn covariance(&self, x: f64, y: f64) -> f64;
}

/// Squared-exponential (radial-basis-function) covariance kernel.
///
/// The parameterization is
///
/// ```text
/// k(x, y) = signal_variance * exp(-0.5 * ((x - y) / length_scale)^2),
/// ```
///
/// with strictly positive finite signal variance and length scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RbfKernel {
    signal_variance: f64,
    length_scale: f64,
}

impl RbfKernel {
    /// Construct an RBF kernel with validated hyperparameters.
    ///
    /// # Errors
    ///
    /// Returns [`KernelError`] when either parameter is non-finite or not
    /// strictly positive.
    pub fn new(signal_variance: f64, length_scale: f64) -> Result<Self, KernelError> {
        if !signal_variance.is_finite() {
            return Err(KernelError::NonFiniteSignalVariance);
        }
        if signal_variance <= 0.0 {
            return Err(KernelError::NonPositiveSignalVariance);
        }
        if !length_scale.is_finite() {
            return Err(KernelError::NonFiniteLengthScale);
        }
        if length_scale <= 0.0 {
            return Err(KernelError::NonPositiveLengthScale);
        }

        Ok(Self {
            signal_variance,
            length_scale,
        })
    }

    /// Return the signal variance \(\sigma^2\).
    #[must_use]
    pub const fn signal_variance(&self) -> f64 {
        self.signal_variance
    }

    /// Return the length scale \(\ell\).
    #[must_use]
    pub const fn length_scale(&self) -> f64 {
        self.length_scale
    }
}

impl ScalarKernel for RbfKernel {
    fn covariance(&self, x: f64, y: f64) -> f64 {
        let scaled_distance = (x - y) / self.length_scale;
        self.signal_variance * (-0.5 * scaled_distance * scaled_distance).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::{RbfKernel, ScalarKernel};
    use crate::KernelError;

    const TOLERANCE: f64 = 1.0e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOLERANCE * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    #[test]
    fn rejects_invalid_signal_variance() {
        assert_eq!(
            RbfKernel::new(f64::NAN, 1.0),
            Err(KernelError::NonFiniteSignalVariance)
        );
        assert_eq!(
            RbfKernel::new(f64::INFINITY, 1.0),
            Err(KernelError::NonFiniteSignalVariance)
        );
        assert_eq!(
            RbfKernel::new(0.0, 1.0),
            Err(KernelError::NonPositiveSignalVariance)
        );
        assert_eq!(
            RbfKernel::new(-1.0, 1.0),
            Err(KernelError::NonPositiveSignalVariance)
        );
    }

    #[test]
    fn rejects_invalid_length_scale() {
        assert_eq!(
            RbfKernel::new(1.0, f64::NAN),
            Err(KernelError::NonFiniteLengthScale)
        );
        assert_eq!(
            RbfKernel::new(1.0, f64::INFINITY),
            Err(KernelError::NonFiniteLengthScale)
        );
        assert_eq!(
            RbfKernel::new(1.0, 0.0),
            Err(KernelError::NonPositiveLengthScale)
        );
        assert_eq!(
            RbfKernel::new(1.0, -1.0),
            Err(KernelError::NonPositiveLengthScale)
        );
    }

    #[test]
    fn exposes_validated_parameters() {
        let kernel = RbfKernel::new(2.5, 0.75).expect("parameters are valid");

        assert_close(kernel.signal_variance(), 2.5);
        assert_close(kernel.length_scale(), 0.75);
    }

    #[test]
    fn diagonal_equals_signal_variance() {
        let kernel = RbfKernel::new(3.0, 1.4).expect("parameters are valid");

        for x in [-10.0, -1.0, 0.0, 0.5, 12.0] {
            assert_close(kernel.covariance(x, x), 3.0);
        }
    }

    #[test]
    fn covariance_is_symmetric() {
        let kernel = RbfKernel::new(1.7, 0.9).expect("parameters are valid");

        for (x, y) in [(-3.0, 0.5), (-0.2, 4.1), (1.0, 1.0), (8.0, -2.0)] {
            assert_close(kernel.covariance(x, y), kernel.covariance(y, x));
        }
    }

    #[test]
    fn covariance_is_stationary() {
        let kernel = RbfKernel::new(1.2, 2.3).expect("parameters are valid");
        let shift = 17.0;

        assert_close(
            kernel.covariance(-1.5, 3.25),
            kernel.covariance(-1.5 + shift, 3.25 + shift),
        );
    }

    #[test]
    fn covariance_decays_with_distance() {
        let kernel = RbfKernel::new(2.0, 1.0).expect("parameters are valid");
        let at_zero = kernel.covariance(0.0, 0.0);
        let at_one = kernel.covariance(0.0, 1.0);
        let at_two = kernel.covariance(0.0, 2.0);

        assert!(at_zero > at_one);
        assert!(at_one > at_two);
        assert!(at_two > 0.0);
    }

    #[test]
    fn representative_gram_matrix_is_positive_semidefinite() {
        let kernel = RbfKernel::new(1.3, 0.8).expect("parameters are valid");
        let points = [-1.0, -0.25, 0.5, 2.0];
        let coefficient_sets = [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5, 0.25],
            [-2.0, 0.75, 1.25, -0.5],
            [1.0, 1.0, 1.0, 1.0],
        ];

        for coefficients in coefficient_sets {
            let quadratic_form = coefficients
                .iter()
                .enumerate()
                .map(|(i, &left)| {
                    coefficients
                        .iter()
                        .enumerate()
                        .map(|(j, &right)| {
                            left * kernel.covariance(points[i], points[j]) * right
                        })
                        .sum::<f64>()
                })
                .sum::<f64>();

            assert!(
                quadratic_form >= -TOLERANCE,
                "quadratic form should be non-negative, got {quadratic_form:.16e}"
            );
        }
    }
}
