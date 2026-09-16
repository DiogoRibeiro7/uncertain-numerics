#![doc = include_str!("../README.md")]

mod error;
mod posterior;

pub use error::PosteriorError;
pub use posterior::ScalarNormalPosterior;
