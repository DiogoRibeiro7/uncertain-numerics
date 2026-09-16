//! Normalized one-dimensional probability measures.

use core::f64::consts::TAU;

use crate::MeasureError;

/// Contract for a normalized continuous probability measure on the real line.
///
/// Implementations expose pointwise density evaluation. The normalization
/// requirement is semantic: implementors represent probability measures whose
/// total mass is one.
pub trait ContinuousProbabilityMeasure {
    /// Evaluate the probability density at `x`.
    #[must_use]
    fn density(&self, x: f64) -> f64;
}

/// Gaussian probability measure \(N(\mu, \sigma^2)\).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianMeasure {
    mean: f64,
    variance: f64,
}

impl GaussianMeasure {
    /// Construct a Gaussian probability measure.
    ///
    /// # Errors
    ///
    /// Returns [`MeasureError`] when the mean is non-finite or the variance is
    /// non-finite or not strictly positive.
    pub fn new(mean: f64, variance: f64) -> Result<Self, MeasureError> {
        if !mean.is_finite() {
            return Err(MeasureError::NonFiniteMean);
        }
        if !variance.is_finite() {
            return Err(MeasureError::NonFiniteVariance);
        }
        if variance <= 0.0 {
            return Err(MeasureError::NonPositiveVariance);
        }

        Ok(Self { mean, variance })
    }

    /// Return the Gaussian mean \(\mu\).
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Return the Gaussian variance \(\sigma^2\).
    #[must_use]
    pub const fn variance(&self) -> f64 {
        self.variance
    }

    /// Return the Gaussian standard deviation \(\sigma\).
    #[must_use]
    pub fn standard_deviation(&self) -> f64 {
        self.variance.sqrt()
    }
}

impl ContinuousProbabilityMeasure for GaussianMeasure {
    fn density(&self, x: f64) -> f64 {
        let centered = x - self.mean;
        let exponent = -(centered * centered) / (2.0 * self.variance);
        exponent.exp() / (TAU * self.variance).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::{ContinuousProbabilityMeasure, GaussianMeasure};
    use crate::MeasureError;

    const TOLERANCE: f64 = 1.0e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= TOLERANCE * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    #[test]
    fn rejects_non_finite_mean() {
        assert_eq!(
            GaussianMeasure::new(f64::NAN, 1.0),
            Err(MeasureError::NonFiniteMean)
        );
        assert_eq!(
            GaussianMeasure::new(f64::INFINITY, 1.0),
            Err(MeasureError::NonFiniteMean)
        );
    }

    #[test]
    fn rejects_invalid_variance() {
        assert_eq!(
            GaussianMeasure::new(0.0, f64::NAN),
            Err(MeasureError::NonFiniteVariance)
        );
        assert_eq!(
            GaussianMeasure::new(0.0, f64::INFINITY),
            Err(MeasureError::NonFiniteVariance)
        );
        assert_eq!(
            GaussianMeasure::new(0.0, 0.0),
            Err(MeasureError::NonPositiveVariance)
        );
        assert_eq!(
            GaussianMeasure::new(0.0, -1.0),
            Err(MeasureError::NonPositiveVariance)
        );
    }

    #[test]
    fn exposes_parameters() {
        let measure = GaussianMeasure::new(1.5, 4.0).expect("parameters are valid");

        assert_close(measure.mean(), 1.5);
        assert_close(measure.variance(), 4.0);
        assert_close(measure.standard_deviation(), 2.0);
    }

    #[test]
    fn density_matches_standard_normal_at_mean() {
        let measure = GaussianMeasure::new(0.0, 1.0).expect("parameters are valid");
        let expected = 1.0 / (2.0 * core::f64::consts::PI).sqrt();

        assert_close(measure.density(0.0), expected);
    }

    #[test]
    fn density_is_symmetric_around_mean() {
        let measure = GaussianMeasure::new(2.0, 1.5).expect("parameters are valid");

        assert_close(measure.density(1.25), measure.density(2.75));
    }

    #[test]
    fn density_decreases_away_from_mean() {
        let measure = GaussianMeasure::new(-0.5, 2.0).expect("parameters are valid");
        let at_mean = measure.density(-0.5);
        let one_away = measure.density(0.5);
        let two_away = measure.density(1.5);

        assert!(at_mean > one_away);
        assert!(one_away > two_away);
        assert!(two_away > 0.0);
    }

    #[test]
    fn density_is_finite_for_finite_inputs() {
        let measure = GaussianMeasure::new(0.0, 0.25).expect("parameters are valid");

        for x in [-100.0, -2.0, 0.0, 3.0, 100.0] {
            assert!(measure.density(x).is_finite());
            assert!(measure.density(x) >= 0.0);
        }
    }
}
