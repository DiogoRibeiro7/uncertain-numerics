// Sequential active Bayesian quadrature over a finite candidate grid.
//
// Run with: cargo run --example active_quadrature

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

    let initial = quadrature.posterior(&initial_nodes, &initial_values)?;
    println!("initial posterior variance = {:.3e}", initial.variance());

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
