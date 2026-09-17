# Covariance-greedy probabilistic linear solver

## Motivation

The residual-driven solver chooses search directions from

\[
s_k \propto b-A m_k,
\]

so the realized direction depends on the unknown exact solution through `b = A x`. The rank-one Gaussian conditioning update accounts for the scalar projection observation but not for information encoded by the fact that this particular direction was selected. The calibration study therefore finds conservative posterior covariance under adaptive residual selection.

The covariance-greedy policy avoids this selection effect by choosing directions without using `b` or observed function values.

## Acquisition

For a current Gaussian solution belief

\[
x\sim\mathcal N(m,\Sigma)
\]

and candidate projection direction `s`, define

\[
h=A^\top s.
\]

Exact conditioning gives

\[
\Sigma^+=\Sigma-\frac{(\Sigma h)(\Sigma h)^\top}{h^\top\Sigma h}.
\]

Therefore the exact covariance-trace reduction is

\[
\Delta_{\mathrm{tr}}(s)
=\operatorname{tr}(\Sigma)-\operatorname{tr}(\Sigma^+)
=\frac{h^\top\Sigma^2 h}{h^\top\Sigma h}.
\]

The solver selects the finite candidate direction maximizing this quantity.

## Calibration contract

Direction selection depends only on:

- the system matrix `A`;
- the current covariance `Sigma`;
- the finite candidate-direction set;
- the deterministic tie-breaking rule.

It does not depend on the right-hand side `b`, posterior mean, or exact solution. Consequently the selected information operator is independent of the prior draw under the assumed model, and ordinary exact Gaussian conditioning remains valid.

A deterministic prior-predictive regression verifies that standardized errors for several fixed solution functionals have approximately unit second moment and nominal 95% coverage after two covariance-greedy projections.

## Stopping policy

To preserve the same interpretation, stopping is based only on data-independent quantities:

- posterior covariance-trace tolerance;
- fixed projection budget;
- candidate exhaustion;
- absence of any informative remaining candidate.

Residual-based stopping is intentionally excluded because it depends on `b` and would make the stopping time solution-dependent.
