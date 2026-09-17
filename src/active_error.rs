use core::fmt;

use crate::BayesianQuadratureError;

/// Errors raised while selecting active Bayesian-quadrature candidates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActiveSelectionError {
    /// No candidate points were supplied.
    EmptyCandidates,
    /// At least one candidate is not finite.
    NonFiniteCandidate,
    /// Acquisition evaluation failed for the current design.
    Acquisition(BayesianQuadratureError),
}

impl fmt::Display for ActiveSelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidates => write!(f, "at least one candidate is required"),
            Self::NonFiniteCandidate => write!(f, "candidate points must be finite"),
            Self::Acquisition(error) => write!(f, "acquisition evaluation failed: {error}"),
        }
    }
}

impl std::error::Error for ActiveSelectionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Acquisition(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BayesianQuadratureError> for ActiveSelectionError {
    fn from(error: BayesianQuadratureError) -> Self {
        Self::Acquisition(error)
    }
}
