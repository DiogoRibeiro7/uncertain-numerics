use core::fmt;

use crate::{ActiveSelectionError, BayesianQuadratureError};

/// Errors raised while running sequential active Bayesian quadrature.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActiveDesignError {
    /// The posterior-variance stopping tolerance is not finite.
    NonFiniteVarianceTolerance,
    /// The posterior-variance stopping tolerance is negative.
    NegativeVarianceTolerance,
    /// At least one candidate point is not finite.
    NonFiniteCandidate,
    /// A user-supplied function evaluation returned a non-finite value.
    NonFiniteFunctionValue,
    /// Candidate selection failed.
    Selection(ActiveSelectionError),
    /// Bayesian-quadrature posterior construction failed.
    Posterior(BayesianQuadratureError),
}

impl fmt::Display for ActiveDesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteVarianceTolerance => {
                write!(f, "posterior-variance stopping tolerance must be finite")
            }
            Self::NegativeVarianceTolerance => {
                write!(f, "posterior-variance stopping tolerance must be non-negative")
            }
            Self::NonFiniteCandidate => write!(f, "candidate points must be finite"),
            Self::NonFiniteFunctionValue => {
                write!(f, "active function evaluations must be finite")
            }
            Self::Selection(error) => write!(f, "active candidate selection failed: {error}"),
            Self::Posterior(error) => write!(f, "Bayesian quadrature failed: {error}"),
        }
    }
}

impl std::error::Error for ActiveDesignError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Selection(error) => Some(error),
            Self::Posterior(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ActiveSelectionError> for ActiveDesignError {
    fn from(error: ActiveSelectionError) -> Self {
        Self::Selection(error)
    }
}

impl From<BayesianQuadratureError> for ActiveDesignError {
    fn from(error: BayesianQuadratureError) -> Self {
        Self::Posterior(error)
    }
}
