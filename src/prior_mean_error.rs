use core::fmt;

/// Errors raised while constructing prior mean functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PriorMeanError {
    /// The constant prior mean is not finite.
    NonFiniteConstant,
    /// The affine prior mean intercept is not finite.
    NonFiniteIntercept,
    /// The affine prior mean slope is not finite.
    NonFiniteSlope,
}

impl fmt::Display for PriorMeanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteConstant => write!(f, "constant prior mean must be finite"),
            Self::NonFiniteIntercept => write!(f, "affine prior mean intercept must be finite"),
            Self::NonFiniteSlope => write!(f, "affine prior mean slope must be finite"),
        }
    }
}

impl std::error::Error for PriorMeanError {}
