use nalgebra::{DMatrix, DVector};
use uncertain_numerics::{
    BayesianQuadrature, GaussianMeasure, KernelIntegral, KernelMean, RbfKernel, ScalarKernel,
};

const REPLICATES_PER_CONFIGURATION: usize = 600;
const COVERAGE_TOLERANCE: f64 = 0.05;

#[derive(Debug, Clone)]
struct DeterministicNormalRng {
    state: u64,
    cached_normal: Option<f64>,
}

impl DeterministicNormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64: compact, deterministic, and sufficient for a reproducible test fixture.
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform_open_01(&mut self) -> f64 {
        // Use the upper 53 bits to construct a value strictly inside (0, 1).
        let mantissa = self.next_u64() >> 11;
        let unit = (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64);
        unit.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.cached_normal.take() {
            return value;
        }

        let u1 = self.uniform_open_01();
        let u2 = self.uniform_open_01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = core::f64::consts::TAU * u2;
        let first = radius * angle.cos();
        let second = radius * angle.sin();
        self.cached_normal = Some(second);
        first
    }
}

#[derive(Debug, Clone, Copy)]
struct NominalCoverage {
    z: f64,
    nominal: f64,
}

fn joint_covariance(
    kernel: RbfKernel,
    measure: GaussianMeasure,
    nodes: &[f64],
) -> DMatrix<f64> {
    let dimension = nodes.len() + 1;
    let mut covariance = DMatrix::zeros(dimension, dimension);

    for (row, &left) in nodes.iter().enumerate() {
        for (column, &right) in nodes.iter().enumerate() {
            covariance[(row, column)] = kernel.covariance(left, right);
        }

        let kernel_mean = kernel.kernel_mean(&measure, left);
        covariance[(row, dimension - 1)] = kernel_mean;
        covariance[(dimension - 1, row)] = kernel_mean;
    }

    covariance[(dimension - 1, dimension - 1)] = kernel.kernel_integral(&measure);
    covariance
}

fn sample_joint_gaussian(
    cholesky_factor: &DMatrix<f64>,
    rng: &mut DeterministicNormalRng,
) -> DVector<f64> {
    let standard_normal = DVector::from_iterator(
        cholesky_factor.nrows(),
        (0..cholesky_factor.nrows()).map(|_| rng.standard_normal()),
    );
    cholesky_factor * standard_normal
}

#[test]
fn posterior_intervals_are_calibrated_under_the_assumed_gp_prior() {
    let kernel = RbfKernel::new(1.4, 1.1).expect("kernel parameters are valid");
    let measure = GaussianMeasure::new(0.35, 1.25).expect("measure parameters are valid");
    let quadrature = BayesianQuadrature::new(kernel, measure, 0.0);

    let node_configurations = [
        vec![-1.5, -0.25, 0.8],
        vec![-2.0, -0.7, 0.4, 1.6],
        vec![-2.4, -1.2, -0.1, 0.9, 2.1],
    ];
    let nominal_coverages = [
        NominalCoverage {
            z: 0.674_489_750_196_081_7,
            nominal: 0.50,
        },
        NominalCoverage {
            z: 1.281_551_565_544_600_4,
            nominal: 0.80,
        },
        NominalCoverage {
            z: 1.959_963_984_540_054,
            nominal: 0.95,
        },
    ];

    let mut rng = DeterministicNormalRng::new(0x5EED_CAFE_F00D_BAAD);
    let mut covered = [0_usize; 3];
    let mut total = 0_usize;

    for nodes in node_configurations {
        let covariance = joint_covariance(kernel, measure, &nodes);
        let cholesky = covariance
            .cholesky()
            .expect("joint GP/integral covariance must be positive definite");
        let lower = cholesky.l();

        for _ in 0..REPLICATES_PER_CONFIGURATION {
            let sample = sample_joint_gaussian(&lower, &mut rng);
            let values = sample.rows(0, nodes.len());
            let true_integral = sample[nodes.len()];
            let observed_values: Vec<f64> = values.iter().copied().collect();

            let posterior = quadrature
                .posterior(&nodes, &observed_values)
                .expect("sampled observations should produce a valid posterior");
            let standard_deviation = posterior.standard_deviation();

            for (index, target) in nominal_coverages.iter().enumerate() {
                let radius = target.z * standard_deviation;
                if (true_integral - posterior.mean()).abs() <= radius {
                    covered[index] += 1;
                }
            }
            total += 1;
        }
    }

    for (index, target) in nominal_coverages.iter().enumerate() {
        let empirical = covered[index] as f64 / total as f64;
        assert!(
            (empirical - target.nominal).abs() <= COVERAGE_TOLERANCE,
            "nominal coverage {:.2} produced empirical coverage {:.4} over {total} replicates",
            target.nominal,
            empirical,
        );
    }
}

#[test]
fn standardized_integral_errors_have_unit_second_moment() {
    let kernel = RbfKernel::new(0.9, 0.85).expect("kernel parameters are valid");
    let measure = GaussianMeasure::new(-0.2, 0.8).expect("measure parameters are valid");
    let quadrature = BayesianQuadrature::new(kernel, measure, 0.0);
    let nodes = [-1.4, -0.3, 0.55, 1.45];
    let covariance = joint_covariance(kernel, measure, &nodes);
    let cholesky = covariance
        .cholesky()
        .expect("joint GP/integral covariance must be positive definite");
    let lower = cholesky.l();
    let mut rng = DeterministicNormalRng::new(0xA11C_E55E_1234_5678);

    let mut squared_standardized_error_sum = 0.0;
    let replicates = 1_200_usize;

    for _ in 0..replicates {
        let sample = sample_joint_gaussian(&lower, &mut rng);
        let values: Vec<f64> = sample.rows(0, nodes.len()).iter().copied().collect();
        let true_integral = sample[nodes.len()];
        let posterior = quadrature
            .posterior(&nodes, &values)
            .expect("sampled observations should produce a valid posterior");
        let standardized_error =
            (true_integral - posterior.mean()) / posterior.standard_deviation();
        squared_standardized_error_sum += standardized_error * standardized_error;
    }

    let second_moment = squared_standardized_error_sum / replicates as f64;
    assert!(
        (second_moment - 1.0).abs() <= 0.12,
        "standardized-error second moment should be near one, got {second_moment:.6}"
    );
}
