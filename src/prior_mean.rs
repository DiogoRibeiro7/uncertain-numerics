//! Prior mean functions for Gaussian-process integrand models.

use crate::{ContinuousProbabilityMeasure, GaussianMeasure, PriorMeanError};

/// Prior mean function of a Gaussian-process integrand model, together with
/// its analytic integral against a probability measure.
///
/// For a mean function `m` and probability measure `p`, Bayesian quadrature
/// needs the pointwise values `m(x_i)` at the observation nodes and the prior
/// integral
///
/// ```text
/// I_m = integral m(x) p(x) dx.
/// ```
///
/// Like [`KernelMean`](crate::KernelMean), the trait is implemented only for
/// mean/measure pairs with an explicit analytic contract, so unsupported pairs
/// fail at compile time rather than falling back to numerical integration.
pub trait PriorMean<M> {
    /// Evaluate the prior mean at `x`.
    #[must_use]
    fn value(&self, x: f64) -> f64;

    /// Evaluate the analytic integral of the prior mean under `measure`.
    #[must_use]
    fn integral(&self, measure: &M) -> f64;
}

/// The zero prior mean, used by [`BayesianQuadrature::new`](crate::BayesianQuadrature::new).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ZeroMean;

impl<M: ContinuousProbabilityMeasure> PriorMean<M> for ZeroMean {
    fn value(&self, _x: f64) -> f64 {
        0.0
    }

    fn integral(&self, _measure: &M) -> f64 {
        0.0
    }
}

/// A constant prior mean `m(x) = c`.
///
/// Because every [`ContinuousProbabilityMeasure`] is normalized, the integral
/// of a constant mean is the constant itself under any supported measure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConstantMean {
    constant: f64,
}

impl ConstantMean {
    /// Construct a constant prior mean.
    ///
    /// # Errors
    ///
    /// Returns [`PriorMeanError`] when the constant is not finite.
    pub fn new(constant: f64) -> Result<Self, PriorMeanError> {
        if !constant.is_finite() {
            return Err(PriorMeanError::NonFiniteConstant);
        }
        Ok(Self { constant })
    }

    /// Return the constant `c`.
    #[must_use]
    pub const fn constant(&self) -> f64 {
        self.constant
    }
}

impl<M: ContinuousProbabilityMeasure> PriorMean<M> for ConstantMean {
    fn value(&self, _x: f64) -> f64 {
        self.constant
    }

    fn integral(&self, _measure: &M) -> f64 {
        self.constant
    }
}

/// An affine prior mean `m(x) = intercept + slope * x`.
///
/// Under a Gaussian measure `N(mu, sigma^2)` its integral is
/// `intercept + slope * mu`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMean {
    intercept: f64,
    slope: f64,
}

impl AffineMean {
    /// Construct an affine prior mean.
    ///
    /// # Errors
    ///
    /// Returns [`PriorMeanError`] when either coefficient is not finite.
    pub fn new(intercept: f64, slope: f64) -> Result<Self, PriorMeanError> {
        if !intercept.is_finite() {
            return Err(PriorMeanError::NonFiniteIntercept);
        }
        if !slope.is_finite() {
            return Err(PriorMeanError::NonFiniteSlope);
        }
        Ok(Self { intercept, slope })
    }

    /// Return the intercept.
    #[must_use]
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Return the slope.
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }
}

impl PriorMean<GaussianMeasure> for AffineMean {
    fn value(&self, x: f64) -> f64 {
        self.intercept + self.slope * x
    }

    fn integral(&self, measure: &GaussianMeasure) -> f64 {
        self.intercept + self.slope * measure.mean()
    }
}

#[cfg(test)]
mod tests {
    use super::{AffineMean, ConstantMean, PriorMean, ZeroMean};
    use crate::{ContinuousProbabilityMeasure, GaussianMeasure, PriorMeanError};

    const TOLERANCE: f64 = 1.0e-12;
    const QUADRATURE_TOLERANCE: f64 = 1.0e-9;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "expected {expected:.16e}, got {actual:.16e}"
        );
    }

    /// Composite Simpson rule for `integral m(x) p(x) dx` over `[lower, upper]`.
    fn simpson<F: Fn(f64) -> f64>(
        function: F,
        measure: &GaussianMeasure,
        lower: f64,
        upper: f64,
        intervals: u32,
    ) -> f64 {
        assert_eq!(intervals % 2, 0);
        let step = (upper - lower) / f64::from(intervals);
        let mut sum =
            function(lower) * measure.density(lower) + function(upper) * measure.density(upper);
        for index in 1..intervals {
            let x = lower + f64::from(index) * step;
            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
            sum += weight * function(x) * measure.density(x);
        }
        sum * step / 3.0
    }

    #[test]
    fn zero_mean_is_zero_everywhere() {
        let measure = GaussianMeasure::new(0.3, 2.0).expect("measure is valid");
        for x in [-5.0, 0.0, 2.5] {
            assert_close(
                PriorMean::<GaussianMeasure>::value(&ZeroMean, x),
                0.0,
                TOLERANCE,
            );
        }
        assert_close(ZeroMean.integral(&measure), 0.0, TOLERANCE);
    }

    #[test]
    fn constant_mean_integrates_to_itself() {
        let mean = ConstantMean::new(-1.75).expect("constant is valid");
        let measure = GaussianMeasure::new(4.0, 0.5).expect("measure is valid");

        assert_close(mean.constant(), -1.75, TOLERANCE);
        assert_close(
            PriorMean::<GaussianMeasure>::value(&mean, 10.0),
            -1.75,
            TOLERANCE,
        );
        assert_close(mean.integral(&measure), -1.75, TOLERANCE);
    }

    #[test]
    fn affine_mean_matches_deterministic_quadrature() {
        let mean = AffineMean::new(0.8, -2.3).expect("coefficients are valid");
        let measure = GaussianMeasure::new(1.2, 0.7).expect("measure is valid");
        let width = 12.0 * measure.standard_deviation();
        let numerical = simpson(
            |x| mean.value(x),
            &measure,
            measure.mean() - width,
            measure.mean() + width,
            40_000,
        );

        assert_close(mean.intercept(), 0.8, TOLERANCE);
        assert_close(mean.slope(), -2.3, TOLERANCE);
        assert_close(mean.value(2.0), 0.8 - 4.6, TOLERANCE);
        assert_close(mean.integral(&measure), 0.8 - 2.3 * 1.2, TOLERANCE);
        assert_close(mean.integral(&measure), numerical, QUADRATURE_TOLERANCE);
    }

    #[test]
    fn rejects_non_finite_parameters() {
        assert_eq!(
            ConstantMean::new(f64::NAN),
            Err(PriorMeanError::NonFiniteConstant)
        );
        assert_eq!(
            AffineMean::new(f64::INFINITY, 1.0),
            Err(PriorMeanError::NonFiniteIntercept)
        );
        assert_eq!(
            AffineMean::new(1.0, f64::NEG_INFINITY),
            Err(PriorMeanError::NonFiniteSlope)
        );
    }
}
