use core::fmt;

use crate::{ConditioningError, PosteriorError};

/// Errors raised while constructing a Bayesian quadrature posterior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BayesianQuadratureError {
    /// No function observations were supplied.
    EmptyObservations,
    /// Observation nodes and function values have different lengths.
    ObservationLengthMismatch,
    /// At least one observation node is not finite.
    NonFiniteObservationNode,
    /// At least one observed function value is not finite.
    NonFiniteObservationValue,
    /// Gaussian conditioning failed.
    Conditioning(ConditioningError),
    /// The computed posterior could not be represented as a valid scalar Gaussian posterior.
    Posterior(PosteriorError),
    /// The posterior variance is negative by more than the documented roundoff tolerance.
    MateriallyNegativePosteriorVariance {
        /// Raw variance before roundoff handling.
        value: f64,
        /// Maximum negative magnitude treated as floating-point roundoff.
        tolerance: f64,
    },
}

impl fmt::Display for BayesianQuadratureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservations => write!(f, "at least one observation is required"),
            Self::ObservationLengthMismatch => {
                write!(f, "observation nodes and values must have equal lengths")
            }
            Self::NonFiniteObservationNode => {
                write!(f, "observation nodes must be finite")
            }
            Self::NonFiniteObservationValue => {
                write!(f, "observed function values must be finite")
            }
            Self::Conditioning(error) => write!(f, "Gaussian conditioning failed: {error}"),
            Self::Posterior(error) => write!(f, "posterior construction failed: {error}"),
            Self::MateriallyNegativePosteriorVariance { value, tolerance } => write!(
                f,
                "posterior variance {value:.16e} is negative beyond roundoff tolerance {tolerance:.16e}"
            ),
        }
    }
}

impl std::error::Error for BayesianQuadratureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Conditioning(error) => Some(error),
            Self::Posterior(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ConditioningError> for BayesianQuadratureError {
    fn from(error: ConditioningError) -> Self {
        Self::Conditioning(error)
    }
}

impl From<PosteriorError> for BayesianQuadratureError {
    fn from(error: PosteriorError) -> Self {
        Self::Posterior(error)
    }
}
