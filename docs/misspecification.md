# Bayesian quadrature under misspecification

The current `BayesianQuadrature` implementation is internally calibrated when the integrand is generated from the same zero-mean RBF Gaussian-process prior used by the model. That is a necessary baseline, but it is not sufficient for arbitrary deterministic integrands.

This document records deterministic failure modes that matter for interpretation of posterior uncertainty.

## Standardized integration error

For an exact integral `I`, posterior mean `mu_I`, and posterior standard deviation `sigma_I`, define

\[
r = \frac{|I-\mu_I|}{\sigma_I}.
\]

A nominal 95% Gaussian credible interval misses the true integral when

\[
r > 1.9599639845.
\]

Large values of `r` therefore indicate overconfidence: the numerical error is large relative to the uncertainty reported by the probabilistic numerical model.

## Excessive RBF smoothness

If the RBF length scale is much larger than the length scale of the integrand, the GP prior assumes substantially more smoothness than the function actually has.

For high-frequency oscillatory functions, the posterior variance can then shrink rapidly even though the posterior mean remains biased. The result is a large standardized error and severe undercoverage.

The regression suite includes

\[
f(x)=\sin(6x)
\]

under a non-standard Gaussian integration measure. With the same observation nodes, a short RBF length scale remains comparatively conservative while an over-smooth RBF prior becomes strongly overconfident.

## Poor node placement

Posterior uncertainty also depends on where the function is observed. Nodes concentrated on one side of the high-probability region can leave important parts of the integration measure weakly informed.

The regression suite compares centred nodes with a left-heavy design for a cubic integrand. Under an over-smooth prior, one-sided placement produces a materially larger standardized integration error.

This is important because a small posterior variance is not, by itself, evidence that the numerical design adequately covers the integration measure.

## Unresolved local structure

A narrow local feature can contribute materially to an integral while being poorly represented by a smooth prior and sparse observations.

The suite includes a narrow Gaussian bump whose width is much smaller than the over-smooth RBF length scale. In that setting, the posterior variance can become extremely small relative to the actual integration error.

## Interpretation

These tests are not intended to prove that one RBF length scale is universally correct. They establish something more basic:

> Posterior computational uncertainty is conditional on the numerical model.

When kernel smoothness or node placement is badly misspecified, Bayesian quadrature can be confidently wrong.

Future work should therefore examine sensitivity to kernel hyperparameters and node design explicitly, rather than report posterior standard deviations without qualification.
