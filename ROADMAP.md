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
- [x] derive and implement the RBF/Gaussian initial integral variance `kappa`;
- [x] test `kappa` against independent deterministic quadrature.

### M1.4 Gaussian conditioning

- [x] introduce a minimal linear-algebra dependency while preserving the Rust 1.85 MSRV;
- [x] use Cholesky factorization/solves rather than explicit matrix inversion;
- [x] define jitter as an explicit fixed diagonal addition with no automatic escalation;
- [x] surface dimension, finiteness, jitter, and positive-definiteness failures explicitly.

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

- [x] expose the first `BayesianQuadrature` API for the RBF/Gaussian pair;
- [x] make the zero prior-mean assumption explicit rather than implicit;
- [x] return posterior uncertainty as a first-class `ScalarNormalPosterior`;
- [x] clamp only machine-scale negative posterior variance to zero and reject materially negative values;
- [ ] generalize prior-mean support beyond zero mean;
- [ ] generalize orchestration across additional analytically integrated kernel/measure pairs.

Exit criterion: end-to-end Bayesian quadrature for at least one analytically integrated kernel/measure pair.

## M2 — Scientific validation

- [x] exact constant, affine, quadratic, and off-centre Gaussian fixtures under a non-standard Gaussian measure;
- [x] independent deterministic Simpson quadrature comparisons for the initial fixture set;
- [x] deterministic simulation study of 50%, 80%, and 95% posterior interval coverage under the assumed GP prior;
- [x] verify the standardized integral error has approximately unit second moment under the assumed GP prior;
- [x] study deterministic misspecification with oscillatory, polynomial, and narrow-local-feature integrands;
- [x] demonstrate sensitivity to RBF length scale and node placement;
- [x] stress-test nearly duplicate nodes, extreme smoothness, and explicit jitter regularization;
- [x] document concrete failure modes where posterior uncertainty becomes overconfident under misspecification.

Numerical-stability policy: jitter is an explicit regularization parameter. It may improve matrix conditioning, but because it changes the posterior it must remain visible to the caller rather than being silently increased until factorization succeeds.

Exit criterion: the library demonstrates not only numerical accuracy but also whether its uncertainty statements are empirically calibrated, and exposes rather than hides important numerical-stability limits.

## M3 — Active Bayesian quadrature

For current nodes `X`, candidate `x_*`, regularized Gram matrix `A = K + jitter I`, candidate covariance vector `k_*`, and kernel-mean vector `z`, use

\[
\Delta(x_*) =
\frac{\left(z_* - k_*^\top A^{-1} z\right)^2}
{k(x_*,x_*) + \mathrm{jitter} - k_*^\top A^{-1} k_*}
\]

as the expected reduction in posterior integral variance.

- [x] implement the posterior-variance reduction criterion;
- [x] verify the criterion matches the actual posterior-variance drop after augmenting the node set;
- [x] make the criterion independent of observed function values;
- [ ] sequential node selection over a candidate set;
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
