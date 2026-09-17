// Infer an integral, and its posterior uncertainty, from a few function evaluations.
//
// Run with: cargo run --example bayesian_quadrature

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
    let standardized_error = (posterior.mean() - exact).abs() / posterior.standard_deviation();

    println!("E[I | y]           = {:.6}", posterior.mean());
    println!(
        "sd[I | y]          = {:.3e}",
        posterior.standard_deviation()
    );
    println!("exact integral     = {exact:.6}");
    println!("standardized error = {standardized_error:.3}");

    // The posterior is honest about its own error here: the exact value lies well
    // inside the reported uncertainty.
    assert!((posterior.mean() - exact).abs() < 3.0 * posterior.standard_deviation());
    Ok(())
}
