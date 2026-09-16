use core::fmt;

/// Errors raised while constructing covariance kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError {
    /// The RBF signal variance is not finite.
    NonFiniteSignalVariance,
    /// The RBF signal variance is zero or negative.
    NonPositiveSignalVariance,
    /// The RBF length scale is not finite.
    NonFiniteLengthScale,
    /// The RBF length scale is zero or negative.
    NonPositiveLengthScale,
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteSignalVariance => write!(f, "signal variance must be finite"),
            Self::NonPositiveSignalVariance => {
                write!(f, "signal variance must be strictly positive")
            }
            Self::NonFiniteLengthScale => write!(f, "length scale must be finite"),
            Self::NonPositiveLengthScale => write!(f, "length scale must be strictly positive"),
        }
    }
}

impl std::error::Error for KernelError {}
