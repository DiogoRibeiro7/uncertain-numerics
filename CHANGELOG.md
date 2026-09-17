# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ScalarNormalPosterior`: validated Gaussian posterior for a scalar computational quantity.
- `RbfKernel` and `GaussianMeasure` with analytic kernel mean and integrated kernel variance for the RBF/Gaussian pair.
- `GaussianConditioner`: Cholesky-based conditioning with explicit fixed jitter and typed failure modes.
- `BayesianQuadrature`: one-dimensional Bayesian quadrature returning posterior mean and variance.
- `VarianceReductionAcquisition` and `ActiveBayesianQuadrature`: posterior-variance-reduction acquisition, deterministic candidate selection, sequential evaluation loop, and explicit stopping rules.
- `SpdLinearSystem` and `GaussianLinearBelief`: validated SPD systems and Gaussian solution beliefs with exact projection conditioning.
- `ResidualProjectionSolver`, `AConjugateProjectionSolver`, and `CovarianceGreedyProjectionSolver`: probabilistic linear solvers with residual, A-conjugate, and data-independent covariance-greedy search policies.
- Scientific validation suites: analytic fixtures, independent deterministic quadrature, posterior calibration, misspecification sensitivity, numerical stability, active-versus-fixed benchmarks, linear-solver calibration, and policy equivalence.
- Runnable examples for Bayesian quadrature, active design, and probabilistic linear solvers.
- Project infrastructure: dual MIT/Apache-2.0 licensing, cross-platform CI with MSRV, documentation, and `cargo-deny` checks, tag-driven release workflow, Dependabot, issue and pull request templates, contributing guide, security policy, and code of conduct.

### Fixed

- The library failed to compile because the residual and A-conjugate solvers shadowed their `residual` helper with a local binding before calling it.
- Integration tests failed to compile because of ambiguous floating-point literals.
- Source formatting and `clippy::pedantic` findings that broke the CI pipeline.

[Unreleased]: https://github.com/DiogoRibeiro7/uncertain-numerics/commits/main
