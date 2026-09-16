use core::fmt;

/// Errors raised while constructing probabilistic numerical results.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PosteriorError {
    /// The posterior mean is not finite.
    NonFiniteMean,
    /// The posterior variance is not finite.
    NonFiniteVariance,
    /// The posterior variance is negative.
    NegativeVariance,
}

impl fmt::Display for PosteriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteMean => write!(f, "posterior mean must be finite"),
            Self::NonFiniteVariance => write!(f, "posterior variance must be finite"),
            Self::NegativeVariance => write!(f, "posterior variance must be non-negative"),
        }
    }
}

impl std::error::Error for PosteriorError {}
