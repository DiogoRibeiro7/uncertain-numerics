use core::fmt;

/// Errors raised while constructing or using a Gaussian conditioning system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditioningError {
    /// The requested matrix dimension is zero.
    ZeroDimension,
    /// The flattened matrix length does not equal `dimension * dimension`.
    MatrixDimensionMismatch,
    /// At least one matrix entry is not finite.
    NonFiniteMatrixEntry,
    /// The jitter value is not finite.
    NonFiniteJitter,
    /// The jitter value is negative.
    NegativeJitter,
    /// Cholesky factorization failed because the regularized matrix is not positive definite.
    NotPositiveDefinite,
    /// The right-hand-side length does not equal the matrix dimension.
    RightHandSideDimensionMismatch,
    /// At least one right-hand-side entry is not finite.
    NonFiniteRightHandSideEntry,
}

impl fmt::Display for ConditioningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "conditioning matrix dimension must be positive"),
            Self::MatrixDimensionMismatch => write!(
                f,
                "flattened conditioning matrix length must equal dimension squared"
            ),
            Self::NonFiniteMatrixEntry => {
                write!(f, "conditioning matrix entries must be finite")
            }
            Self::NonFiniteJitter => write!(f, "conditioning jitter must be finite"),
            Self::NegativeJitter => {
                write!(f, "conditioning jitter must be non-negative")
            }
            Self::NotPositiveDefinite => write!(
                f,
                "conditioning matrix is not positive definite after applying jitter"
            ),
            Self::RightHandSideDimensionMismatch => write!(
                f,
                "right-hand-side length must equal conditioning matrix dimension"
            ),
            Self::NonFiniteRightHandSideEntry => {
                write!(f, "right-hand-side entries must be finite")
            }
        }
    }
}

impl std::error::Error for ConditioningError {}
