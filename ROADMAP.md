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
- [x] select the maximum-reduction node from a deterministic candidate set;
- [x] reuse one Cholesky factorization across all candidate evaluations;
- [x] define deterministic first-maximum tie breaking and explicit candidate validation;
- [x] sequential active-design loop with function evaluation callback;
- [x] stopping rules based on posterior integral variance, evaluation budget, and candidate exhaustion;
- [x] deterministic active-vs-fixed benchmark at equal evaluation budget.

The sequential design records every selected point, function value, predicted variance reduction, and resulting posterior variance. This keeps stopping decisions and benchmark comparisons auditable.

The equal-budget benchmark deliberately evaluates both aggregate and per-function behavior. Active placement is required to reduce posterior variance and aggregate mean squared integration error for the benchmark suite, but it is not claimed to dominate fixed placement on every individual integrand.

Exit criterion: active Bayesian quadrature has a validated acquisition rule, deterministic finite-candidate selection, explicit stopping semantics, and an equal-budget benchmark demonstrating aggregate benefit without claiming universal function-wise dominance.

## M4 — Broader probabilistic numerics

### M4.1 Probabilistic linear solvers

For an SPD system `A x = b`, represent uncertainty over the exact solution as

\[
x \sim \mathcal N(m, \Sigma).
\]

Given a noiseless linear observation along search direction `s`,

\[
s^\top A x = s^\top b,
\]

condition the Gaussian belief with

\[
h=A^\top s,
\qquad
m^+=m+\frac{\Sigma h}{h^\top\Sigma h}\left(s^\top b-h^\top m\right),
\]

\[
\Sigma^+
=
\Sigma-\frac{(\Sigma h)(\Sigma h)^\top}{h^\top\Sigma h}.
\]

The first iterative policy uses the current normalized residual as the next search direction,

\[
s_k = \frac{b-Am_k}{\|b-Am_k\|_2},
\]

and treats posterior covariance trace as an uncertainty diagnostic distinct from residual norm.

The second policy starts from the current residual and explicitly orthogonalizes it against all previous search directions in the SPD matrix inner product

\[
\langle u,v\rangle_A = u^\top A v,
\]

so the retained search directions satisfy

\[
s_i^\top A s_j \approx 0,
\qquad i\ne j.
\]

- [x] define a validated dense SPD linear-system contract;
- [x] define a Gaussian solution belief with PSD covariance validation;
- [x] implement exact noiseless projection conditioning;
- [x] verify the conditioned projection is satisfied exactly within floating-point tolerance;
- [x] verify uncertainty remains in uninformed directions after one projection;
- [x] implement a residual-driven iterative projection policy;
- [x] support stopping by residual tolerance, covariance-trace tolerance, iteration budget, or lack of an informative direction;
- [x] verify covariance trace is non-increasing while residual norm need not be;
- [x] implement an explicitly A-conjugate residual-orthogonalization policy;
- [x] verify pairwise A-conjugacy numerically and retain uncertainty-based stopping semantics;
- [ ] compare the A-conjugate and residual-driven probabilistic policies at equal projection budgets;
- [ ] validate uncertainty calibration on synthetic SPD systems;
- [ ] benchmark against classical conjugate gradients at equal matrix-vector budgets.

### M4.2 Later methods

- [ ] probabilistic ODE solvers;
- [ ] uncertainty propagation between numerical subproblems;
- [ ] PDE methods where the mathematical contract is sufficiently clear.

## Non-goals for the early project

The initial crate will not attempt to become a generic machine-learning framework, dataframe library, Gaussian-process package, or replacement for classical numerical-integration crates. Those components may be dependencies or comparison targets, but the focus remains uncertainty over numerical computation itself.
