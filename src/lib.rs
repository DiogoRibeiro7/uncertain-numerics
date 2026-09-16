#![doc = include_str!("../README.md")]

mod conditioning;
mod conditioning_error;
mod error;
mod kernel;
mod kernel_error;
mod kernel_integral;
mod kernel_mean;
mod measure;
mod measure_error;
mod posterior;

pub use conditioning::GaussianConditioner;
pub use conditioning_error::ConditioningError;
pub use error::PosteriorError;
pub use kernel::{RbfKernel, ScalarKernel};
pub use kernel_error::KernelError;
pub use kernel_integral::KernelIntegral;
pub use kernel_mean::KernelMean;
pub use measure::{ContinuousProbabilityMeasure, GaussianMeasure};
pub use measure_error::MeasureError;
pub use posterior::ScalarNormalPosterior;
