# uncertain-numerics

[![CI](https://github.com/DiogoRibeiro7/uncertain-numerics/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/uncertain-numerics/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/uncertain-numerics.svg)](https://crates.io/crates/uncertain-numerics)
[![docs.rs](https://img.shields.io/docsrs/uncertain-numerics)](https://docs.rs/uncertain-numerics)
[![MSRV](https://img.shields.io/crates/msrv/uncertain-numerics)](#minimum-supported-rust-version)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

Probabilistic numerical methods in Rust, with explicit uncertainty over computational quantities.

`uncertain-numerics` treats numerical computation as an inference problem. Instead of returning only a point estimate of an integral or of the solution of a linear system, every method returns a validated Gaussian posterior: its mean is the estimate and its variance states how much the computation still does not know. Each statistical and numerical assumption behind that posterior is explicit, tested against analytic references, and checked for calibration.

## Highlights

- **Bayesian quadrature** in one dimension with a Gaussian-process prior (RBF kernel, Gaussian integration measure) and closed-form kernel integrals.
- **Active Bayesian quadrature**: a posterior-variance-reduction acquisition rule, deterministic candidate selection, a sequential evaluation loop, and explicit stopping rules.
- **Probabilistic linear solvers** for dense symmetric positive-definite systems: exact projection conditioning of a Gaussian solution belief with residual-driven, A-conjugate, and covariance-greedy search policies.
- **Auditable numerics**: Cholesky solves instead of explicit inverses, jitter as a visible parameter that is never escalated silently, and typed errors instead of `NaN` for every invalid input.
- **Scientific validation shipped as tests**: analytic fixtures, independent deterministic quadrature, coverage and calibration studies, misspecification and numerical-stability regressions.
- **Small footprint**: one dependency (`nalgebra`), `unsafe` forbidden, MSRV 1.89.

## Installation

```sh
cargo add uncertain-numerics
```

or, in `Cargo.toml`:

```toml
[dependencies]
uncertain-numerics = "0.1"
```

## Quick start

All three snippets below are compiled and run as doctests, and each has a runnable counterpart in [`examples/`](https://github.com/DiogoRibeiro7/uncertain-numerics/tree/main/examples).

### Bayesian quadrature

Infer $I = \int f(x)\,p(x)\,dx$ from a handful of function evaluations and get the posterior uncertainty alongside the estimate.

```rust
use uncertain_numerics::{BayesianQuadrature, GaussianMeasure, RbfKernel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Prior over the integrand: zero-mean Gaussian process with an RBF kernel.
    let kernel = RbfKernel::new(1.0, 1.0)?; // signal variance, length scale
    // Integration measure p(x) = N(0, 1).
    let measure = GaussianMeasure::new(0.0, 1.0)?;
    // Jitter is an explicit, fixed diagonal regularizer. It is never escalated silently.
    let quadrature = BayesianQuadrature::new(kernel, measure, 1.0e-10);

    // Observe f(x) = cos(x) at seven nodes. Exactly, E[cos X] = exp(-1/2) for X ~ N(0, 1).
    let nodes = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let values: Vec<f64> = nodes.iter().copied().map(f64::cos).collect();
    let exact = (-0.5_f64).exp();

    let posterior = quadrature.posterior(&nodes, &values)?;
    println!("E[I | y]  = {:.6}   (exact {exact:.6})", posterior.mean());
    println!("sd[I | y] = {:.3e}", posterior.standard_deviation());

    // The posterior is honest about its own error here: the exact value lies well
    // inside the reported uncertainty.
    assert!((posterior.mean() - exact).abs() < 3.0 * posterior.standard_deviation());
    Ok(())
}
```

### Active Bayesian quadrature

Let the acquisition rule choose where to evaluate next. Every step records the selected point, the predicted variance reduction, and the resulting posterior variance, so stopping decisions are auditable.

```rust
use uncertain_numerics::{
    ActiveBayesianQuadrature, BayesianQuadrature, GaussianMeasure, RbfKernel,
};

// A Gaussian bump centred at 0.5. Against p(x) = N(0, 1) its integral has a closed form.
fn integrand(x: f64) -> f64 {
    let centered = x - 0.5;
    (-centered * centered).exp()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kernel = RbfKernel::new(1.0, 1.0)?;
    let measure = GaussianMeasure::new(0.0, 1.0)?;
    let quadrature = BayesianQuadrature::new(kernel, measure, 1.0e-10);
    let active = ActiveBayesianQuadrature::new(quadrature);

    // Start from three evaluations; allow up to six more from a fixed candidate grid,
    // stopping early once the posterior variance of the integral drops below 1e-6.
    let initial_nodes = [-1.0, 0.0, 1.0];
    let initial_values: Vec<f64> = initial_nodes.iter().copied().map(integrand).collect();
    let candidates: Vec<f64> = (0..=23).map(|i| -2.875 + 0.25 * f64::from(i)).collect();

    let result = active.run(
        &initial_nodes,
        &initial_values,
        &candidates,
        6,
        1.0e-6,
        integrand,
    )?;

    for step in result.steps() {
        println!(
            "x = {:+.3}  predicted reduction = {:.3e}  posterior variance = {:.3e}",
            step.point(),
            step.predicted_variance_reduction(),
            step.posterior_variance(),
        );
    }

    let exact = (1.0_f64 / 3.0).sqrt() * (-0.25_f64 / 3.0).exp();
    let posterior = result.posterior();
    println!("stopped because: {:?}", result.termination());
    println!(
        "E[I | y] = {:.6} ± {:.3e}   (exact {exact:.6})",
        posterior.mean(),
        posterior.standard_deviation(),
    );
    Ok(())
}
```

### Probabilistic linear solvers

Represent the unknown solution of $A x = b$ as a Gaussian belief and condition it on exact projections $s^\top A x = s^\top b$. The posterior covariance trace is a diagnostic of the uncertainty that remains, distinct from the residual norm.

```rust
use uncertain_numerics::{
    CovarianceGreedyProjectionSolver, GaussianLinearBelief, ResidualProjectionSolver,
    SpdLinearSystem,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A x = b with A symmetric positive definite, stored row-major.
    let system = SpdLinearSystem::new(
        &[4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
        &[1.0, 2.0, 3.0],
        3,
    )?;
    // Prior belief x ~ N(0, I).
    let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let prior = GaussianLinearBelief::new(&[0.0; 3], &identity, 3)?;

    // Residual-driven policy: each step observes the projection along the normalized residual.
    let solver = ResidualProjectionSolver::new(1.0e-10, 0.0, 3)?;
    let result = solver.solve(&system, &prior)?;
    println!("residual policy: mean = {:?}", result.belief().mean());
    println!("stopped because: {:?}", result.termination());
    for (index, step) in result.steps().iter().enumerate() {
        println!(
            "  step {index}: residual {:.3e} -> {:.3e}, covariance trace {:.3e}",
            step.residual_norm_before(),
            step.residual_norm_after(),
            step.covariance_trace_after(),
        );
    }

    // Covariance-greedy policy: directions are chosen without looking at b, so the
    // posterior covariance keeps its calibrated interpretation under the assumed prior.
    let candidates: Vec<Vec<f64>> = (0..3)
        .map(|axis| {
            let mut direction = vec![0.0; 3];
            direction[axis] = 1.0;
            direction
        })
        .collect();
    let greedy = CovarianceGreedyProjectionSolver::new(0.0, 2)?;
    let result = greedy.solve(&system, &prior, &candidates)?;
    println!("covariance-greedy: mean = {:?}", result.belief().mean());
    println!("stopped because: {:?}", result.termination());
    Ok(())
}
```

## Mathematical background

### Bayesian quadrature posterior

Place a zero-mean Gaussian-process prior with covariance kernel $k$ on the integrand $f$. For observation nodes $X$ with values $y = f(X)$, the integral $I = \int f(x)\,p(x)\,dx$ is itself Gaussian a posteriori, with

$$
\mathbb{E}[I \mid y] = z^\top (K + \lambda I)^{-1} y,
\qquad
\operatorname{Var}(I \mid y) = \kappa - z^\top (K + \lambda I)^{-1} z,
$$

where $K$ is the Gram matrix, $\lambda$ is the explicit jitter, $z_i = \int k(x_i, x)\,p(x)\,dx$ is the kernel mean, and $\kappa = \iint k(x, x')\,p(x)\,p(x')\,dx\,dx'$ is the prior integral variance. For the RBF kernel and a Gaussian measure both $z$ and $\kappa$ are available in closed form and are tested against independent deterministic quadrature. The inverse is never formed; both solves reuse one Cholesky factorization.

### Active-design criterion

For a candidate node $x_*$ the expected reduction in posterior integral variance is

$$
\Delta(x_*) = \frac{\left(z_* - k_*^\top A^{-1} z\right)^2}{k(x_*, x_*) + \lambda - k_*^\top A^{-1} k_*},
\qquad A = K + \lambda I,
$$

which does not depend on observed function values. The sequential design selects the first maximum over a finite candidate set, reusing a single factorization across all candidates, and stops on a variance tolerance, an evaluation budget, or candidate exhaustion.

### Projection conditioning

For an SPD system $A x = b$ with belief $x \sim \mathcal{N}(m, \Sigma)$, an exact observation along direction $s$ with $h = A^\top s$ gives the rank-one update

$$
m^+ = m + \frac{\Sigma h}{h^\top \Sigma h}\left(s^\top b - h^\top m\right),
\qquad
\Sigma^+ = \Sigma - \frac{(\Sigma h)(\Sigma h)^\top}{h^\top \Sigma h}.
$$

Three policies choose $s$: the normalized residual, the residual orthogonalized in the $A$-inner product against previous directions, and a covariance-greedy rule that maximizes the exact trace reduction $h^\top \Sigma^2 h / h^\top \Sigma h$ over a candidate set without using $b$. The first two produce equivalent posteriors at equal informative budgets; only the third keeps a calibrated interpretation under repeated prior draws, because its selections do not depend on the unknown solution. See the documents linked below.

## Validation

The integration tests are scientific studies rather than smoke tests. Each documents a claim the library makes about its own behavior.

| Suite | What it establishes |
| --- | --- |
| `scientific_validation` | Exact constant, affine, quadratic, and off-centre Gaussian integrals under a non-standard Gaussian measure; agreement with independent Simpson quadrature. |
| `posterior_calibration` | 50%, 80%, and 95% interval coverage and unit second moment of the standardized error under the assumed GP prior. |
| `misspecification_sensitivity` | Documented overconfidence for over-smooth kernels, one-sided node placement, and unresolved narrow features. |
| `numerical_stability` | Ill-conditioning from near-duplicate nodes and large length scales, and the effect of explicit jitter. |
| `active_vs_fixed` | Aggregate benefit of active placement over a fixed design at equal evaluation budget, without claiming per-function dominance. |
| `linear_solver_calibration` | Calibration under fixed projection directions, and the conservative uncertainty produced by adaptive residual selection. |
| `linear_policy_equivalence` | Residual and A-conjugate policies yield the same Gaussian posterior at equal informative budgets. |
| `covariance_greedy_calibration` | Data-independent direction selection preserves calibration. |
| `conjugate_gradient_benchmark` | Equal matrix-vector budgets: classical conjugate gradients versus the projection solvers, and the identity-prior posterior mean equals Craig's method. |

Run everything with:

```sh
cargo test --all-features
```

## Documentation

- API reference: [docs.rs/uncertain-numerics](https://docs.rs/uncertain-numerics) (or `cargo doc --open`).
- [Bayesian quadrature under misspecification](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/docs/misspecification.md)
- [Calibration of probabilistic linear-solver uncertainty](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/docs/linear-solver-calibration.md)
- [Linear-policy equivalence under exact Gaussian conditioning](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/docs/linear-policy-equivalence.md)
- [Covariance-greedy probabilistic linear solver](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/docs/covariance-greedy-linear-solver.md)
- [Probabilistic linear solvers versus conjugate gradients](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/docs/conjugate-gradient-benchmark.md)
- [Roadmap](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/ROADMAP.md) and [changelog](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/CHANGELOG.md)

## Design principles

- Expose uncertainty over the computational quantity, not only a numerical estimate.
- Keep statistical and numerical assumptions explicit and named.
- Validate algorithms against analytic reference problems wherever possible.
- Treat calibration and numerical stability as part of correctness.
- Keep the core independent of dataframe and machine-learning ecosystems.
- Add dependencies only after the numerical abstractions are stable.

## Status and scope

The crate is pre-1.0. The public API is deliberately small while the mathematical contracts are established, and minor versions may contain breaking changes that are called out in the changelog.

Current limitations worth knowing before you depend on it:

- Bayesian quadrature supports one dimension, the RBF kernel, a Gaussian measure, and a zero prior mean.
- Linear solvers operate on dense SPD systems and use dense covariance matrices, so they are meant for moderate dimensions.
- Posterior uncertainty from the adaptive residual policies is conservative under repeated prior draws; use the covariance-greedy policy when calibration matters.
- With the identity prior, the projection solvers are Craig's method in disguise: at equal matrix-vector budgets classical conjugate gradients is far more accurate. Use them for the posterior covariance, not for speed. See the benchmark note in the documentation list.

The project will not become a general machine-learning framework, Gaussian-process package, or replacement for classical numerical-integration crates.

## Minimum supported Rust version

The MSRV is Rust 1.89 (edition 2024). It is checked in CI, and raising it is treated as at least a minor version bump.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/CONTRIBUTING.md) for the development workflow and the testing standards expected for numerical claims. Security issues should follow [SECURITY.md](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/SECURITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](https://github.com/DiogoRibeiro7/uncertain-numerics/blob/main/LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
