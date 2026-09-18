# Probabilistic linear solvers versus conjugate gradients

This note records the equal-budget comparison in `tests/conjugate_gradient_benchmark.rs`, the structural result that explains it, and the numerical-stability defect the benchmark exposed.

## Question

For a dense SPD system `A x = b`, how accurate is the posterior mean of the residual-projection and A-conjugate probabilistic solvers compared with classical conjugate gradients (CG) when all three are allowed the same number of products with `A`, and why?

## Setup

All solvers start from the zero vector. The probabilistic solvers use the identity prior covariance and stop once the residual norm drops below `1e-13` times the right-hand-side norm, so nobody keeps iterating on roundoff. The error metric is the relative Euclidean error `‖m − x*‖ / ‖x*‖` against a Cholesky solution.

Matrix-vector products are counted as the algorithm needs them, not as a particular implementation recomputes them:

| Solver | Products with `A` per step | Why |
| --- | --- | --- |
| Conjugate gradients | 1 | `A p_k` |
| Residual projection | 2 | `A s_k` for the observation, `A g_k` to update the residual after the mean moves |
| A-conjugate projection | 2 | as above, with `A s_j` for earlier directions cached |
| Craig's method (CGNE) | 2 | `A p_k` and `A^T r_{k+1}` |

Three deterministic fixtures:

| Fixture | `n` | Condition number |
| --- | --- | --- |
| One-dimensional Laplacian, asymmetric right-hand side | 8 | 32.2 |
| Log-spaced diagonal from 1 to 100 plus rank-one coupling | 8 | 92.3 |
| Deterministic random `G Gᵀ / n + I` | 10 | 4.4 |

## Structural result: Craig's method in disguise

With prior covariance `Σ₀ = I` and search directions collected in `S`, write `H = A S`. Exact conditioning gives the posterior mean

```text
m_k = H (Hᵀ H)⁻¹ Sᵀ b,
```

the minimum-norm vector satisfying the Galerkin constraints `Sᵀ A m = Sᵀ b`. Since `x*` satisfies the same constraints, `Hᵀ (x* − m_k) = 0`: the error is orthogonal to `range(A S)`, so `m_k` is the Euclidean-error minimiser over `range(A S)`.

With residual directions, `range(A S) = A · K_k(A², b)`. That is exactly the search space of Craig's method (CGNE, conjugate gradients applied to `A Aᵀ y = b` with `x = Aᵀ y`), which is also the Euclidean-error minimiser over it. The two iterates therefore coincide in exact arithmetic, and the A-conjugate policy, which spans the same subspace, coincides with them too.

The test checks this numerically. The posterior mean and Craig's iterate agree to about `1e-15` relative on the Laplacian and random fixtures for every projection count. On the graded fixture the two recurrences accumulate roundoff differently and drift apart: `3e-15` after four projections, `3e-13` after five, `2e-10` after six, `2.5e-7` after seven. The test tolerance is `1e-6`.

## Results

Relative Euclidean error at equal budgets. CG is capped at `n` iterations because it has converged.

One-dimensional Laplacian (`n = 8`, condition number 32.2):

| Products with `A` | CG | Residual policy | A-conjugate policy |
| ---: | ---: | ---: | ---: |
| 2 | 3.20e-1 | 9.71e-1 | 9.71e-1 |
| 4 | 7.83e-2 | 9.33e-1 | 9.33e-1 |
| 6 | 1.55e-2 | 8.86e-1 | 8.86e-1 |
| 8 | 5.43e-16 | 8.36e-1 | 8.36e-1 |
| 12 | 5.43e-16 | 6.54e-1 | 6.54e-1 |
| 16 | 5.43e-16 | 4.09e-16 | 6.21e-16 |

Graded spectrum with rank-one coupling (`n = 8`, condition number 92.3):

| Products with `A` | CG | Residual policy | A-conjugate policy |
| ---: | ---: | ---: | ---: |
| 2 | 8.31e-1 | 9.98e-1 | 9.98e-1 |
| 4 | 5.24e-1 | 9.93e-1 | 9.93e-1 |
| 6 | 1.92e-1 | 9.78e-1 | 9.78e-1 |
| 8 | 5.01e-12 | 9.44e-1 | 9.44e-1 |
| 12 | 5.01e-12 | 7.09e-1 | 7.09e-1 |
| 16 | 5.01e-12 | 1.02e-14 | 4.17e-16 |

Deterministic random SPD (`n = 10`, condition number 4.4):

| Products with `A` | CG | Residual policy | A-conjugate policy |
| ---: | ---: | ---: | ---: |
| 2 | 1.73e-1 | 6.96e-1 | 6.96e-1 |
| 4 | 2.00e-2 | 5.33e-1 | 5.33e-1 |
| 6 | 1.41e-3 | 2.77e-1 | 2.77e-1 |
| 8 | 3.65e-5 | 1.66e-1 | 1.66e-1 |
| 10 | 1.74e-16 | 9.29e-2 | 9.29e-2 |
| 16 | 1.74e-16 | 1.76e-3 | 1.76e-3 |
| 20 | 1.74e-16 | 4.99e-16 | 2.03e-16 |

The test asserts that CG's error is no larger than either probabilistic policy's at every budget below `2n`, and that both reach the exact solution at the full budget.

## Interpretation

CG minimises the A-norm error over `K_k(A, b)` at one product per iteration, with a convergence factor governed by `√κ(A)`. The identity-prior probabilistic solver is Craig's method: it minimises the Euclidean error over `A · K_k(A², b)`, a Krylov space of `A²` whose condition number is `κ(A)²`, at two products per iteration. It pays twice per step for a space that converges at the rate of `κ(A)` rather than `√κ(A)`. The two penalties compound, and CG dominates at every budget on every fixture, at some budgets by fifteen orders of magnitude.

This is not a defect of the implementation. It is what the identity prior means: the belief is agnostic about the geometry of `A`, so the information it can extract from `k` projections is the information Craig's method extracts. What the probabilistic solver returns that CG does not is a posterior covariance, calibrated under the assumed prior when the directions are chosen independently of `b` (see `linear-solver-calibration.md` and `covariance-greedy-linear-solver.md`). Its value is uncertainty, not accuracy per product with `A`.

Recovering CG's accuracy inside the probabilistic framework requires a prior adapted to `A`. With `Σ₀ = A⁻¹` the posterior mean under residual directions is the CG iterate (Cockayne, Oates, Ipsen and Girolami, *A Bayesian conjugate gradient method*, Bayesian Analysis 14(3), 2019). That prior is not available in closed form for the systems one wants to solve, and Krylov-adapted approximations to it are the natural next step for this milestone.

The A-conjugate policy shares CG's fate for the same reason: it spans the same subspace as the residual policy, so it reaches the same accuracy at the same budget. Its benefit is a better-conditioned basis, not fewer products with `A`.

## Numerical-stability finding

The benchmark could not initially run to completion on the graded fixture. After seven residual projections the eighth update produced a covariance whose most negative eigenvalue was about `−3e-14`, beyond the `128 ε` roundoff tolerance, and `GaussianLinearBelief::new` rejected it. The rank-one downdate `Σ − (Σh)(Σh)ᵀ / (hᵀΣh)` is not self-correcting: a slightly indefinite `Σ` amplifies its own defect in the next update. Measured most negative eigenvalue after each projection on the graded fixture, in units of `ε` times the pre-update scale: −1, −2, −5, −14, −31, −69, −124, −199. At condition number 916 the breakdown came after five projections.

Four variants were measured over 8 to 32 projections at condition numbers from 10 to about 1000: the plain downdate, the Joseph form `(I − g hᵀ) Σ (I − g hᵀ)ᵀ`, and each with roundoff-scale negative eigenvalues clipped to zero afterwards. Clipping is what matters. With it, the most negative eigenvalue before clipping never exceeded a few `ε` in any run, because each update starts from an exactly positive-semidefinite matrix; the Joseph form alone reduced the drift but still reached `−1.8e-14` in one case.

`condition_on_projection` therefore symmetrises the updated covariance, rejects it if any eigenvalue is below `−128 ε` times the largest entry of the pre-update covariance, and otherwise clips negative eigenvalues to zero before storing it. This mirrors the posterior-variance policy in Bayesian quadrature: roundoff is absorbed, anything material is an error. The cost is one symmetric eigendecomposition per update, which the validation step already required.
