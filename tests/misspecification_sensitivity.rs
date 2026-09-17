use uncertain_numerics::{BayesianQuadrature, GaussianMeasure, RbfKernel};

const Z_95: f64 = 1.959_963_984_540_054;

fn exact_sine_integral(frequency: f64, mean: f64, variance: f64) -> f64 {
    (-0.5 * frequency * frequency * variance).exp() * (frequency * mean).sin()
}

fn exact_cubic_integral(mean: f64, variance: f64) -> f64 {
    mean * mean * mean + 3.0 * mean * variance
}

fn exact_gaussian_bump_integral(
    centre: f64,
    width: f64,
    measure_mean: f64,
    measure_variance: f64,
) -> f64 {
    let width_squared = width * width;
    let combined_variance = width_squared + measure_variance;
    let centered = measure_mean - centre;

    (width_squared / combined_variance).sqrt()
        * (-(centered * centered) / (2.0 * combined_variance)).exp()
}

fn standardized_error(
    quadrature: &BayesianQuadrature,
    nodes: &[f64],
    values: &[f64],
    exact_integral: f64,
) -> f64 {
    let posterior = quadrature
        .posterior(nodes, values)
        .expect("misspecification fixture should produce a valid posterior");
    let posterior_sd = posterior.standard_deviation();

    assert!(
        posterior_sd > 0.0,
        "posterior standard deviation must be positive"
    );
    (exact_integral - posterior.mean()).abs() / posterior_sd
}

#[test]
fn oversmooth_kernel_becomes_overconfident_for_unresolved_oscillation() {
    let measure = GaussianMeasure::new(0.4, 1.3).expect("measure parameters are valid");
    let measure_sd = measure.standard_deviation();
    let nodes = [
        measure.mean() - 2.0 * measure_sd,
        measure.mean() - measure_sd,
        measure.mean(),
        measure.mean() + measure_sd,
        measure.mean() + 2.0 * measure_sd,
    ];
    let frequency = 6.0;
    let values: Vec<f64> = nodes.iter().map(|&x| (frequency * x).sin()).collect();
    let exact = exact_sine_integral(frequency, measure.mean(), measure.variance());

    let short_kernel = RbfKernel::new(1.0, 0.3).expect("kernel parameters are valid");
    let oversmooth_kernel = RbfKernel::new(1.0, 2.0).expect("kernel parameters are valid");
    let short_quadrature = BayesianQuadrature::new(short_kernel, measure, 1.0e-12);
    let oversmooth_quadrature = BayesianQuadrature::new(oversmooth_kernel, measure, 1.0e-12);

    let short_ratio = standardized_error(&short_quadrature, &nodes, &values, exact);
    let oversmooth_ratio = standardized_error(&oversmooth_quadrature, &nodes, &values, exact);

    assert!(
        short_ratio < Z_95,
        "short length scale should not be overconfident for this fixture, got {short_ratio:.4}"
    );
    assert!(
        oversmooth_ratio > 20.0,
        "oversmooth kernel should be strongly overconfident, got {oversmooth_ratio:.4}"
    );
    assert!(
        oversmooth_ratio > 10.0 * short_ratio,
        "overconfidence should worsen materially with excessive smoothing"
    );
}

#[test]
fn poor_node_placement_can_create_false_confidence() {
    let measure = GaussianMeasure::new(0.4, 1.3).expect("measure parameters are valid");
    let measure_sd = measure.standard_deviation();
    let centred_nodes = [
        measure.mean() - 2.0 * measure_sd,
        measure.mean() - measure_sd,
        measure.mean(),
        measure.mean() + measure_sd,
        measure.mean() + 2.0 * measure_sd,
    ];
    let left_heavy_nodes = [
        measure.mean() - 3.0 * measure_sd,
        measure.mean() - 2.0 * measure_sd,
        measure.mean() - measure_sd,
        measure.mean() - 0.5 * measure_sd,
        measure.mean(),
    ];
    let kernel = RbfKernel::new(1.0, 2.0).expect("kernel parameters are valid");
    let quadrature = BayesianQuadrature::new(kernel, measure, 1.0e-12);
    let exact = exact_cubic_integral(measure.mean(), measure.variance());
    let centred_values: Vec<f64> = centred_nodes.iter().map(|&x| x * x * x).collect();
    let left_values: Vec<f64> = left_heavy_nodes.iter().map(|&x| x * x * x).collect();

    let centred_ratio = standardized_error(&quadrature, &centred_nodes, &centred_values, exact);
    let left_ratio = standardized_error(&quadrature, &left_heavy_nodes, &left_values, exact);

    assert!(
        centred_ratio > Z_95,
        "even centred nodes should reveal misspecification for this oversmooth cubic fixture"
    );
    assert!(
        left_ratio > 1.5 * centred_ratio,
        "one-sided node placement should worsen standardized error: centred={centred_ratio:.4}, left={left_ratio:.4}"
    );
}

#[test]
fn narrow_local_structure_is_missed_by_oversmooth_prior() {
    let measure = GaussianMeasure::new(0.4, 1.3).expect("measure parameters are valid");
    let measure_sd = measure.standard_deviation();
    let nodes = [
        measure.mean() - 2.0 * measure_sd,
        measure.mean() - measure_sd,
        measure.mean(),
        measure.mean() + measure_sd,
        measure.mean() + 2.0 * measure_sd,
    ];
    let centre = 1.5;
    let width = 0.15;
    let exact = exact_gaussian_bump_integral(centre, width, measure.mean(), measure.variance());
    let values: Vec<f64> = nodes
        .iter()
        .map(|&x| (-(x - centre).powi(2) / (2.0 * width * width)).exp())
        .collect();

    let local_kernel = RbfKernel::new(1.0, 0.3).expect("kernel parameters are valid");
    let oversmooth_kernel = RbfKernel::new(1.0, 2.0).expect("kernel parameters are valid");
    let local_quadrature = BayesianQuadrature::new(local_kernel, measure, 1.0e-12);
    let oversmooth_quadrature = BayesianQuadrature::new(oversmooth_kernel, measure, 1.0e-12);

    let local_ratio = standardized_error(&local_quadrature, &nodes, &values, exact);
    let oversmooth_ratio = standardized_error(&oversmooth_quadrature, &nodes, &values, exact);

    assert!(
        local_ratio < Z_95,
        "short length scale should remain conservative enough for the narrow bump, got {local_ratio:.4}"
    );
    assert!(
        oversmooth_ratio > 20.0,
        "oversmooth prior should become severely overconfident, got {oversmooth_ratio:.4}"
    );
}
