use core::fmt;

/// Errors raised while constructing probability measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasureError {
    /// The measure mean is not finite.
    NonFiniteMean,
    /// The measure variance is not finite.
    NonFiniteVariance,
    /// The measure variance is zero or negative.
    NonPositiveVariance,
}

impl fmt::Display for MeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteMean => write!(f, "measure mean must be finite"),
            Self::NonFiniteVariance => write!(f, "measure variance must be finite"),
            Self::NonPositiveVariance => write!(f, "measure variance must be strictly positive"),
        }
    }
}

impl std::error::Error for MeasureError {}
