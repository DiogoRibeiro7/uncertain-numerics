use uncertain_numerics::{BayesianQuadrature, ContinuousProbabilityMeasure, GaussianMeasure, RbfKernel};

const ABS_TOLERANCE: f64 = 5.0e-3;
const DETERMINISTIC_TOLERANCE: f64 = 1.0e-10;

fn gaussian_weighted_simpson<F>(
    function: F,
    measure: GaussianMeasure,
    lower: f64,
    upper: f64,
    intervals: u32,
) -> f64
where
    F: Fn(f64) -> f64,
{
    assert!(intervals > 0);
    assert_eq!(intervals % 2, 0);

    let step = (upper - lower) / f64::from(intervals);
    let mut weighted_sum = function(lower) * measure.density(lower)
        + function(upper) * measure.density(upper);

    for index in 1..intervals {
        let x = lower + f64::from(index) * step;
        let weight = if index % 2 == 0 { 2.0 } else { 4.0 };
        weighted_sum += weight * function(x) * measure.density(x);
    }

    weighted_sum * step / 3.0
}

fn evenly_spaced_nodes(lower: f64, upper: f64, count: u32) -> Vec<f64> {
    assert!(count >= 2);

    let denominator = f64::from(count - 1);
    (0..count)
        .map(|index| lower + (upper - lower) * f64::from(index) / denominator)
        .collect()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "expected {expected:.16e}, got {actual:.16e}, tolerance {tolerance:.16e}"
    );
}

fn validation_setup() -> (BayesianQuadrature, GaussianMeasure, Vec<f64>) {
    let measure = GaussianMeasure::new(0.4, 1.3).expect("measure parameters are valid");
    let kernel = RbfKernel::new(1.2, 1.4).expect("kernel parameters are valid");
    let standard_deviation = measure.standard_deviation();
    let nodes = evenly_spaced_nodes(
        measure.mean() - 3.0 * standard_deviation,
        measure.mean() + 3.0 * standard_deviation,
        13,
    );
    let quadrature = BayesianQuadrature::new(kernel, measure, 1.0e-12);

    (quadrature, measure, nodes)
}

#[test]
fn deterministic_reference_matches_exact_gaussian_moments() {
    let (_, measure, _) = validation_setup();
    let standard_deviation = measure.standard_deviation();
    let lower = measure.mean() - 10.0 * standard_deviation;
    let upper = measure.mean() + 10.0 * standard_deviation;

    let constant = gaussian_weighted_simpson(|_| 1.0, measure, lower, upper, 20_000);
    let affine = gaussian_weighted_simpson(|x| x, measure, lower, upper, 20_000);
    let quadratic = gaussian_weighted_simpson(|x| x * x, measure, lower, upper, 20_000);

    assert_close(constant, 1.0, DETERMINISTIC_TOLERANCE);
    assert_close(affine, measure.mean(), DETERMINISTIC_TOLERANCE);
    assert_close(
        quadratic,
        measure.mean() * measure.mean() + measure.variance(),
        DETERMINISTIC_TOLERANCE,
    );
}

#[test]
fn bayesian_quadrature_matches_exact_constant_integral() {
    let (quadrature, _, nodes) = validation_setup();
    let values = vec![1.0; nodes.len()];
    let posterior = quadrature
        .posterior(&nodes, &values)
        .expect("posterior should be valid");

    assert_close(posterior.mean(), 1.0, ABS_TOLERANCE);
    assert!(posterior.variance() >= 0.0);
}

#[test]
fn bayesian_quadrature_matches_exact_affine_integral() {
    let (quadrature, measure, nodes) = validation_setup();
    let values: Vec<f64> = nodes.iter().copied().collect();
    let posterior = quadrature
        .posterior(&nodes, &values)
        .expect("posterior should be valid");

    assert_close(posterior.mean(), measure.mean(), ABS_TOLERANCE);
}

#[test]
fn bayesian_quadrature_matches_exact_quadratic_integral() {
    let (quadrature, measure, nodes) = validation_setup();
    let values: Vec<f64> = nodes.iter().map(|x| x * x).collect();
    let posterior = quadrature
        .posterior(&nodes, &values)
        .expect("posterior should be valid");
    let exact = measure.mean() * measure.mean() + measure.variance();

    assert_close(posterior.mean(), exact, ABS_TOLERANCE);
}

#[test]
fn bayesian_quadrature_matches_exact_off_centre_gaussian_integral() {
    let (quadrature, measure, nodes) = validation_setup();
    let centre = -0.7;
    let width = 0.85;
    let width_squared = width * width;
    let combined_variance = width_squared + measure.variance();
    let exact = (width_squared / combined_variance).sqrt()
        * (-(measure.mean() - centre).powi(2) / (2.0 * combined_variance)).exp();
    let values: Vec<f64> = nodes
        .iter()
        .map(|x| (-0.5 * ((x - centre) / width).powi(2)).exp())
        .collect();
    let posterior = quadrature
        .posterior(&nodes, &values)
        .expect("posterior should be valid");

    assert_close(posterior.mean(), exact, ABS_TOLERANCE);
}

#[test]
fn bayesian_quadrature_and_deterministic_reference_agree_on_all_fixtures() {
    let (quadrature, measure, nodes) = validation_setup();
    let standard_deviation = measure.standard_deviation();
    let lower = measure.mean() - 10.0 * standard_deviation;
    let upper = measure.mean() + 10.0 * standard_deviation;

    let fixtures: [fn(f64) -> f64; 3] = [|_| 1.0, |x| x, |x| x * x];

    for function in fixtures {
        let values: Vec<f64> = nodes.iter().map(|&x| function(x)).collect();
        let posterior = quadrature
            .posterior(&nodes, &values)
            .expect("posterior should be valid");
        let deterministic =
            gaussian_weighted_simpson(function, measure, lower, upper, 20_000);

        assert_close(posterior.mean(), deterministic, ABS_TOLERANCE);
    }
}
