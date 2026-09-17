use core::fmt;

/// Errors raised by probabilistic linear-system primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearSolverError {
    /// The system dimension is zero.
    ZeroDimension,
    /// A flattened square matrix has the wrong length.
    MatrixDimensionMismatch,
    /// A vector has the wrong length.
    VectorDimensionMismatch,
    /// At least one matrix entry is not finite.
    NonFiniteMatrixEntry,
    /// At least one vector entry is not finite.
    NonFiniteVectorEntry,
    /// A solver tolerance is not finite.
    NonFiniteTolerance,
    /// A solver tolerance is negative.
    NegativeTolerance,
    /// No candidate projection directions were supplied.
    EmptyCandidateDirections,
    /// The system matrix is not symmetric within numerical tolerance.
    NonSymmetricSystemMatrix,
    /// The system matrix is not positive definite.
    SystemMatrixNotPositiveDefinite,
    /// The covariance matrix is not symmetric within numerical tolerance.
    NonSymmetricCovariance,
    /// The covariance matrix is not positive semidefinite within numerical tolerance.
    CovarianceNotPositiveSemidefinite,
    /// The observation direction contains no remaining prior uncertainty.
    DegenerateObservation,
}

impl fmt::Display for LinearSolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "linear-system dimension must be positive"),
            Self::MatrixDimensionMismatch => write!(f, "flattened matrix length must equal dimension squared"),
            Self::VectorDimensionMismatch => write!(f, "vector length must equal the system dimension"),
            Self::NonFiniteMatrixEntry => write!(f, "matrix entries must be finite"),
            Self::NonFiniteVectorEntry => write!(f, "vector entries must be finite"),
            Self::NonFiniteTolerance => write!(f, "solver tolerances must be finite"),
            Self::NegativeTolerance => write!(f, "solver tolerances must be non-negative"),
            Self::EmptyCandidateDirections => write!(f, "at least one candidate projection direction is required"),
            Self::NonSymmetricSystemMatrix => write!(f, "system matrix must be symmetric"),
            Self::SystemMatrixNotPositiveDefinite => write!(f, "system matrix must be positive definite"),
            Self::NonSymmetricCovariance => write!(f, "covariance matrix must be symmetric"),
            Self::CovarianceNotPositiveSemidefinite => write!(f, "covariance matrix must be positive semidefinite"),
            Self::DegenerateObservation => write!(f, "observation direction has no remaining prior uncertainty"),
        }
    }
}

impl std::error::Error for LinearSolverError {}
