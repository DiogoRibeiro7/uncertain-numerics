// Probabilistic linear solvers: a Gaussian belief over x conditioned on exact projections.
//
// Run with: cargo run --example probabilistic_linear_solver

use uncertain_numerics::{
    AConjugateProjectionSolver, CovarianceGreedyProjectionSolver, GaussianLinearBelief,
    ResidualProjectionSolver, SpdLinearSystem,
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
    println!("residual policy");
    println!("  mean            = {:?}", result.belief().mean());
    println!("  stopped because = {:?}", result.termination());
    for (index, step) in result.steps().iter().enumerate() {
        println!(
            "  step {index}: residual {:.3e} -> {:.3e}, covariance trace {:.3e}",
            step.residual_norm_before(),
            step.residual_norm_after(),
            step.covariance_trace_after(),
        );
    }

    // A-conjugate policy: the same information subspace in an A-orthogonal basis.
    let conjugate = AConjugateProjectionSolver::new(1.0e-10, 0.0, 3)?;
    let result = conjugate.solve(&system, &prior)?;
    println!("A-conjugate policy");
    println!("  mean            = {:?}", result.belief().mean());
    println!("  stopped because = {:?}", result.termination());

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
    println!("covariance-greedy policy (budget of two projections)");
    println!("  mean            = {:?}", result.belief().mean());
    println!("  stopped because = {:?}", result.termination());
    for (index, step) in result.steps().iter().enumerate() {
        println!(
            "  step {index}: direction {:?}, predicted trace reduction {:.3e}, posterior trace {:.3e}",
            step.direction(),
            step.predicted_trace_reduction(),
            step.posterior_trace(),
        );
    }
    Ok(())
}
