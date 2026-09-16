# Roadmap

`uncertain-numerics` is a probabilistic-numerics library for Rust. The project begins with Bayesian quadrature and expands only after the statistical and numerical contracts of each layer are validated.

## M0 — Crate foundation

- [x] establish crate identity and scope;
- [x] add a validated scalar Gaussian posterior type;
- [ ] add CI for formatting, linting, tests, and documentation;
- [ ] choose and document the project license before publication;
- [ ] verify crates.io package-name availability immediately before release.

Exit criterion: a dependency-light crate with explicit invariants and a clean public API baseline.

## M1 — One-dimensional Bayesian quadrature

### M1.1 Kernels

- [x] define a scalar kernel trait;
- [x] implement squared-exponential/RBF kernel;
- [x] validate symmetry, positivity assumptions, and parameter constraints;
- [ ] add Matérn kernels only when their integration contracts are clear.

### M1.2 Probability measures

- [x] define a normalized one-dimensional continuous probability-measure abstraction;
- [x] implement Gaussian measure;
- [ ] implement finite uniform measure;
- [ ] separate normalized probability measures from generic weighted integration domains.

### M1.3 Analytic kernel integrals

For a kernel `k` and integration measure `p`, expose

\[
z_i = \int k(x_i, x)p(x)\,dx
\]

and

\[
\kappa = \iint k(x,x')p(x)p(x')\,dx\,dx'.
\]

- [x] derive and implement the analytic RBF/Gaussian kernel mean `z(x)`;
- [x] test the RBF/Gaussian kernel mean against independent deterministic quadrature;
- [x] make unsupported analytic kernel/measure pairs absent from the type-level contract;
- [ ] derive and implement the RBF/Gaussian initial integral variance `kappa`;
- [ ] test `kappa` against independent deterministic quadrature.

### M1.4 Gaussian conditioning

- [ ] introduce a minimal linear-algebra dependency only after the required operations are fixed;
- [ ] use factorization/solves rather than explicit matrix inversion;
- [ ] define jitter/regularization semantics explicitly;
- [ ] surface numerical failures rather than silently repairing them.

### M1.5 Bayesian quadrature posterior

For observations `y = f(X)`, compute

\[
\mathbb E[I\mid y]
= I_m + z^\top K^{-1}(y-m)
\]

and

\[
\operatorname{Var}(I\mid y)
= \kappa-z^\top K^{-1}z.
\]

- [ ] expose a stable `BayesianQuadrature` API;
- [ ] distinguish prior mean assumptions from kernel assumptions;
- [ ] return posterior uncertainty as a first-class result;
- [ ] prevent small negative variances caused by numerical roundoff from becoming hidden semantic errors.

Exit criterion: end-to-end Bayesian quadrature for at least one analytically integrated kernel/measure pair.

## M2 — Scientific validation

- [ ] exact polynomial and Gaussian-integral fixtures;
- [ ] deterministic quadrature comparisons;
- [ ] simulation study of posterior interval coverage;
- [ ] sensitivity to node placement and kernel hyperparameters;
- [ ] numerical-stability stress tests;
- [ ] document failure modes where posterior uncertainty is miscalibrated.

Exit criterion: the library demonstrates not only numerical accuracy but also whether its uncertainty statements are empirically calibrated.

## M3 — Active Bayesian quadrature

- [ ] posterior-variance reduction criterion;
- [ ] sequential node selection;
- [ ] stopping rules based on computational uncertainty;
- [ ] deterministic-vs-active benchmark suite.

## M4 — Broader probabilistic numerics

Only after Bayesian quadrature is mature:

- [ ] probabilistic linear solvers;
- [ ] probabilistic ODE solvers;
- [ ] uncertainty propagation between numerical subproblems;
- [ ] PDE methods where the mathematical contract is sufficiently clear.

## Non-goals for the early project

The initial crate will not attempt to become a generic machine-learning framework, dataframe library, Gaussian-process package, or replacement for classical numerical-integration crates. Those components may be dependencies or comparison targets, but the focus remains uncertainty over numerical computation itself.
