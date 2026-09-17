#![doc = include_str!("../README.md")]

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
pub use probabilistic_linear_solver::{
    LinearSolveStep, LinearSolveTermination, ProbabilisticLinearSolveResult,
    ResidualProjectionSolver,
};
