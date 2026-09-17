use crate::PosteriorError;

/// Gaussian posterior for a scalar computational quantity.
///
/// This type is intentionally generic: Bayesian quadrature can use it for an
/// integral, while later probabilistic numerical methods can use the same
/// representation for other scalar quantities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarNormalPosterior {
    mean: f64,
    variance: f64,
}

impl ScalarNormalPosterior {
    /// Construct a validated scalar Gaussian posterior.
    ///
    /// # Errors
    ///
    /// Returns [`PosteriorError`] when the mean or variance is non-finite, or
    /// when the variance is negative.
    pub fn new(mean: f64, variance: f64) -> Result<Self, PosteriorError> {
        if !mean.is_finite() {
            return Err(PosteriorError::NonFiniteMean);
        }
        if !variance.is_finite() {
            return Err(PosteriorError::NonFiniteVariance);
        }
        if variance < 0.0 {
            return Err(PosteriorError::NegativeVariance);
        }

        Ok(Self { mean, variance })
    }

    /// Posterior mean of the computational quantity.
    #[must_use]
    pub const fn mean(self) -> f64 {
        self.mean
    }

    /// Posterior variance of the computational quantity.
    #[must_use]
    pub const fn variance(self) -> f64 {
        self.variance
    }

    /// Posterior standard deviation of the computational quantity.
    #[must_use]
    pub fn standard_deviation(self) -> f64 {
        self.variance.sqrt()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // exact round-trips of constructor inputs are intended
mod tests {
    use super::*;

    #[test]
    fn constructs_valid_posterior() {
        let posterior = ScalarNormalPosterior::new(1.25, 0.09).expect("valid posterior");

        assert_eq!(posterior.mean(), 1.25);
        assert_eq!(posterior.variance(), 0.09);
        assert!((posterior.standard_deviation() - 0.3).abs() < 1.0e-12);
    }

    #[test]
    fn accepts_zero_variance() {
        let posterior = ScalarNormalPosterior::new(2.0, 0.0).expect("zero variance is valid");

        assert_eq!(posterior.standard_deviation(), 0.0);
    }

    #[test]
    fn rejects_negative_variance() {
        assert_eq!(
            ScalarNormalPosterior::new(0.0, -1.0),
            Err(PosteriorError::NegativeVariance)
        );
    }

    #[test]
    fn rejects_non_finite_parameters() {
        assert_eq!(
            ScalarNormalPosterior::new(f64::NAN, 1.0),
            Err(PosteriorError::NonFiniteMean)
        );
        assert_eq!(
            ScalarNormalPosterior::new(0.0, f64::INFINITY),
            Err(PosteriorError::NonFiniteVariance)
        );
    }
}
