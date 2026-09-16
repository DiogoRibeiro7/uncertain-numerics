# uncertain-numerics

Probabilistic numerical methods in Rust, with explicit uncertainty over computational quantities.

## Scope

`uncertain-numerics` treats numerical computation as an inference problem. The first milestone is Bayesian quadrature: infer an integral together with posterior uncertainty rather than returning only a point approximation.

For

\[
I = \int f(x)\,p(x)\,dx,
\]

we place a probabilistic model on the integrand and infer a posterior distribution over `I` from function evaluations.

The initial implementation will focus on a mathematically auditable scalar setting before adding broader numerical methods.

## Design principles

- expose uncertainty over the computational quantity, not only a numerical estimate;
- keep statistical and numerical assumptions explicit;
- validate algorithms against analytic reference problems where possible;
- keep the core independent of dataframe ecosystems;
- avoid unnecessary dependencies until the numerical abstractions are stable;
- treat calibration and numerical stability as part of correctness.

## Initial milestone

Version `0.1.x` is scoped to one-dimensional Bayesian quadrature with a Gaussian-process prior. The planned core is:

1. scalar Gaussian posterior representation;
2. kernels and probability measures;
3. kernel means and integrated kernel variance;
4. stable Gaussian conditioning;
5. Bayesian-quadrature posterior mean and variance;
6. analytic regression tests and empirical calibration checks.

Later milestones may extend the same probabilistic-numerics principles to active quadrature, linear solvers, ODEs, and PDEs.

See [`ROADMAP.md`](ROADMAP.md) for the project sequence.

## Status

The crate is at the bootstrap stage. The public API is intentionally small while the mathematical contracts are established.
