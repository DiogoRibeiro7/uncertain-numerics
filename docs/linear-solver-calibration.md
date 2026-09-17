# Calibration of probabilistic linear-solver uncertainty

Consider an SPD system

\[
A x=b
\]

and a correctly specified Gaussian prior

\[
x\sim\mathcal N(m_0,\Sigma_0).
\]

After exact linear observations, a reported posterior

\[
x\mid\mathcal I\sim\mathcal N(m,\Sigma)
\]

can be checked through any fixed linear functional `c` with positive posterior variance. Under correct calibration,

\[
Z_c=
\frac{c^\top(x-m)}{\sqrt{c^\top\Sigma c}}
\]

should be standard normal over repeated prior draws. In particular,

\[
\mathbb E[Z_c^2]=1
\]

and a nominal 95% Gaussian interval should cover approximately 95% of draws.

## Fixed projection directions

When the projection directions are fixed independently of the unknown solution, exact Gaussian conditioning has the usual calibration interpretation. The deterministic simulation test draws exact solutions from the same Gaussian prior used by the solver, forms `b = A x`, conditions on two fixed projection directions, and checks several solution functionals.

The resulting standardized errors have approximately unit second moment and approximately nominal 95% coverage.

## Adaptive residual directions

The residual-driven solver chooses

\[
s_k=\frac{b-Am_k}{\|b-Am_k\|_2}.
\]

This direction depends on `b`, and therefore on the unknown exact solution `x`. Consequently, the selected direction itself contains information about `x`.

The current Gaussian update conditions only on the scalar equality

\[
s_k^\top A x=s_k^\top b
\]

while treating the realized direction `s_k` as if it had been fixed independently of `x`. It does not condition on the additional information conveyed by the adaptive selection event that produced `s_k`.

The simulation study therefore finds conservative uncertainty for the adaptive residual policy: standardized errors have second moment substantially below one and nominal 95% intervals over-cover.

This is not a numerical failure of the rank-one Gaussian conditioning formula. It is a distinction between conditioning on fixed linear information and performing data-dependent experimental design without accounting for the information contained in the design choice itself.

## Consequence for the API

Posterior covariance from the current adaptive residual solver should not yet be interpreted as a fully calibrated conditional distribution under repeated prior draws. It remains a useful measure of uncertainty removed by the explicitly conditioned projections, but adaptive-direction calibration requires a richer probabilistic treatment of the selection mechanism.

Future work can investigate selection-aware conditioning, randomized or externally fixed search policies, or calibration corrections before making stronger probabilistic coverage claims for adaptive linear solvers.
