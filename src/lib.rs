#![doc = include_str!("../README.md")]

mod error;
mod kernel;
mod kernel_error;
mod measure;
mod measure_error;
mod posterior;

pub use error::PosteriorError;
pub use kernel::{RbfKernel, ScalarKernel};
pub use kernel_error::KernelError;
pub use measure::{ContinuousProbabilityMeasure, GaussianMeasure};
pub use measure_error::MeasureError;
pub use posterior::ScalarNormalPosterior;
