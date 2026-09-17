use uncertain_numerics::{
    ActiveBayesianQuadrature, BayesianQuadrature, GaussianMeasure, RbfKernel,
};

#[derive(Clone, Copy)]
struct Fixture {
    function: fn(f64) -> f64,
    exact_integral: f64,
}

fn constant(_: f64) -> f64 {
    1.0
}

fn identity(x: f64) -> f64 {
    x
}

fn quadratic(x: f64) -> f64 {
    x * x
}

fn cosine_two(x: f64) -> f64 {
    (2.0 * x).cos()
}

fn cosine_four(x: f64) -> f64 {
    (4.0 * x).cos()
}

fn narrow_bump(x: f64) -> f64 {
    let centre = 0.7;
    let variance = 0.16;
    (-(x - centre) * (x - centre) / (2.0 * variance)).exp()
}

fn fixtures() -> [Fixture; 6] {
    let bump_variance = 0.16;
    let bump_centre = 0.7;
    let bump_integral = (bump_variance / (bump_variance + 1.0)).sqrt()
        * (-(bump_centre * bump_centre) / (2.0 * (bump_variance + 1.0))).exp();

    [
        Fixture {
            function: constant,
            exact_integral: 1.0,
        },
        Fixture {
            function: identity,
            exact_integral: 0.0,
        },
        Fixture {
            function: quadratic,
            exact_integral: 1.0,
        },
        Fixture {
            function: cosine_two,
            exact_integral: (-2.0_f64).exp(),
        },
        Fixture {
            function: cosine_four,
            exact_integral: (-8.0_f64).exp(),
        },
        Fixture {
            function: narrow_bump,
            exact_integral: bump_integral,
        },
    ]
}

fn quadrature() -> BayesianQuadrature {
    let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
    let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
    BayesianQuadrature::new(kernel, measure, 1.0e-10)
}

#[test]
fn active_design_beats_coarse_fixed_design_in_aggregate_at_equal_budget() {
    let quadrature = quadrature();
    let active = ActiveBayesianQuadrature::new(quadrature);

    // Both designs use exactly five evaluations. Active starts at the origin and
    // selects four more points from the same [-3, 3] domain represented by the
    // finite candidate grid. The fixed baseline spreads its five points evenly
    // across that domain.
    let initial_nodes = [0.0];
    let candidates = [
        -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5,
        -0.25, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
    ];
    let fixed_nodes = [-3.0, -1.5, 0.0, 1.5, 3.0];

    let mut active_squared_error_sum = 0.0;
    let mut fixed_squared_error_sum = 0.0;
    let mut active_variance = None;
    let mut fixed_variance = None;
    let mut fixture_count = 0.0;
    let mut fixed_wins = 0_u32;

    for fixture in fixtures() {
        let initial_values = [(fixture.function)(0.0)];
        let active_result = active
            .run(
                &initial_nodes,
                &initial_values,
                &candidates,
                4,
                0.0,
                fixture.function,
            )
            .expect("active design should be valid");
        let active_posterior = active_result.posterior();

        let fixed_values: Vec<f64> = fixed_nodes
            .iter()
            .map(|&node| (fixture.function)(node))
            .collect();
        let fixed_posterior = quadrature
            .posterior(&fixed_nodes, &fixed_values)
            .expect("fixed design should be valid");

        let active_error = (active_posterior.mean() - fixture.exact_integral).abs();
        let fixed_error = (fixed_posterior.mean() - fixture.exact_integral).abs();
        active_squared_error_sum += active_error * active_error;
        fixed_squared_error_sum += fixed_error * fixed_error;
        fixture_count += 1.0;

        if fixed_error < active_error {
            fixed_wins += 1;
        }

        active_variance = Some(active_posterior.variance());
        fixed_variance = Some(fixed_posterior.variance());
    }

    let active_mse = active_squared_error_sum / fixture_count;
    let fixed_mse = fixed_squared_error_sum / fixture_count;
    let active_variance = active_variance.expect("fixture set is non-empty");
    let fixed_variance = fixed_variance.expect("fixture set is non-empty");

    assert!(
        active_variance < fixed_variance,
        "active posterior variance {active_variance:.6e} should be below fixed variance {fixed_variance:.6e}"
    );
    assert!(
        active_mse < fixed_mse,
        "active MSE {active_mse:.6e} should be below fixed MSE {fixed_mse:.6e}"
    );
    assert!(
        fixed_wins > 0,
        "benchmark should retain at least one fixture where fixed placement is more accurate"
    );
}
