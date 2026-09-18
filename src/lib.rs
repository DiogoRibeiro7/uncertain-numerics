//! Probabilistic numerical methods with explicit uncertainty over computational
//! quantities.
//!
//! `uncertain-numerics` treats numerical computation as an inference problem.
//! Instead of returning only a point estimate of an integral or of the solution
//! of a linear system, each method returns a validated Gaussian posterior: its
//! mean is the estimate and its variance states how much the computation still
//! does not know. Every statistical and numerical assumption behind that
//! posterior is explicit, documented, and tested.
//!
//! # What is implemented
//!
//! | Area | Entry points |
//! | --- | --- |
//! | One-dimensional Bayesian quadrature | [`BayesianQuadrature`], [`RbfKernel`], [`GaussianMeasure`], [`PriorMean`], [`ScalarNormalPosterior`] |
//! | Active Bayesian quadrature | [`ActiveBayesianQuadrature`], [`VarianceReductionAcquisition`] |
//! | Probabilistic linear solvers | [`SpdLinearSystem`], [`GaussianLinearBelief`], [`ResidualProjectionSolver`], [`AConjugateProjectionSolver`], [`CovarianceGreedyProjectionSolver`] |
//! | Building blocks | [`GaussianConditioner`], [`KernelMean`], [`KernelIntegral`], [`ScalarKernel`], [`ContinuousProbabilityMeasure`] |
//!
//! # Example
//!
//! Infer the integral of `cos(x)` against a standard Gaussian measure from seven
//! function evaluations, together with its posterior uncertainty:
//!
//! ```
//! use uncertain_numerics::{BayesianQuadrature, GaussianMeasure, RbfKernel};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let kernel = RbfKernel::new(1.0, 1.0)?; // signal variance, length scale
//! let measure = GaussianMeasure::new(0.0, 1.0)?; // p(x) = N(0, 1)
//! let quadrature = BayesianQuadrature::new(kernel, measure, 1.0e-10);
//!
//! let nodes = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
//! let values: Vec<f64> = nodes.iter().copied().map(f64::cos).collect();
//! let posterior = quadrature.posterior(&nodes, &values)?;
//!
//! // Exactly, E[cos X] = exp(-1/2) for X ~ N(0, 1).
//! let exact = (-0.5_f64).exp();
//! println!("{} ± {}", posterior.mean(), posterior.standard_deviation());
//! assert!((posterior.mean() - exact).abs() < 3.0 * posterior.standard_deviation());
//! # Ok(())
//! # }
//! ```
//!
//! # Numerical policy
//!
//! - Inputs are validated at the API boundary. Invalid values produce typed
//!   errors that implement [`std::error::Error`]; they never propagate as `NaN`.
//! - Linear systems are solved from a reusable Cholesky factorization. Explicit
//!   inverses are never formed.
//! - Jitter is a caller-supplied constant added to the diagonal. Because it
//!   changes the posterior, it is never increased automatically to make a
//!   factorization succeed.
//! - Posterior variances that are negative by more than machine roundoff are
//!   reported as errors rather than clamped.
//!
//! # Interpreting the uncertainty
//!
//! Posterior uncertainty is conditional on the numerical model. Under the
//! assumed prior the reported intervals are empirically calibrated, which the
//! integration tests verify. Under misspecification, for example an RBF length
//! scale that is far too smooth for the integrand, the posterior can be
//! confidently wrong. The `docs/` directory of the repository records the known
//! failure modes and the calibration status of each linear-solver policy.
//!
//! # Minimum supported Rust version
//!
//! Rust 1.89. Raising the MSRV is treated as at least a minor version bump.

// Applied here rather than in `[lints.rust]` so that examples and integration
// tests, which have no crate-level docs, are not affected.
#![warn(missing_docs)]

mod a_conjugate_solver;
mod active;
mod active_design;
mod active_design_error;
mod active_error;
mod bayesian_quadrature;
mod bayesian_quadrature_error;
mod conditioning;
mod conditioning_error;
mod covariance_greedy_solver;
mod error;
mod kernel;
mod kernel_error;
mod kernel_integral;
mod kernel_mean;
mod linear_belief;
mod linear_solver_error;
mod linear_system;
mod measure;
mod measure_error;
mod posterior;
mod prior_mean;
mod prior_mean_error;
mod probabilistic_linear_solver;

pub use a_conjugate_solver::{
    AConjugateLinearSolveResult, AConjugateLinearSolveStep, AConjugateProjectionSolver,
};
pub use active::{SelectedCandidate, VarianceReductionAcquisition};
pub use active_design::{
    ActiveBayesianQuadrature, ActiveDesignResult, ActiveDesignStep, ActiveTermination,
};
pub use active_design_error::ActiveDesignError;
pub use active_error::ActiveSelectionError;
pub use bayesian_quadrature::BayesianQuadrature;
pub use bayesian_quadrature_error::BayesianQuadratureError;
pub use conditioning::GaussianConditioner;
pub use conditioning_error::ConditioningError;
pub use covariance_greedy_solver::{
    CovarianceGreedyProjectionSolver, CovarianceGreedySolveResult, CovarianceGreedyStep,
    CovarianceGreedyTermination, CovarianceTraceAcquisition, SelectedLinearDirection,
};
pub use error::PosteriorError;
pub use kernel::{RbfKernel, ScalarKernel};
pub use kernel_error::KernelError;
pub use kernel_integral::KernelIntegral;
pub use kernel_mean::KernelMean;
pub use linear_belief::GaussianLinearBelief;
pub use linear_solver_error::LinearSolverError;
pub use linear_system::SpdLinearSystem;
pub use measure::{ContinuousProbabilityMeasure, GaussianMeasure};
pub use measure_error::MeasureError;
pub use posterior::ScalarNormalPosterior;
pub use prior_mean::{AffineMean, ConstantMean, PriorMean, ZeroMean};
pub use prior_mean_error::PriorMeanError;
pub use probabilistic_linear_solver::{
    LinearSolveStep, LinearSolveTermination, ProbabilisticLinearSolveResult,
    ResidualProjectionSolver,
};

/// Compiles and runs every Rust code block in the README as a doctest, so the
/// examples shown on GitHub and crates.io can never drift from the real API.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
mod readme_doctests {}
