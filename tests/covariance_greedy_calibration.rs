use nalgebra::{DMatrix, DVector};
use uncertain_numerics::{
    CovarianceGreedyProjectionSolver, GaussianLinearBelief, SpdLinearSystem,
};

const DIMENSION: usize = 4;
const REPLICATES: usize = 1_200;
const Z_95: f64 = 1.959_963_984_540_054;

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
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn uniform_open_01(&mut self) -> f64 {
        let upper_bits = u32::try_from(self.next_u64() >> 32).expect("upper bits fit in u32");
        (f64::from(upper_bits) + 0.5) / (f64::from(u32::MAX) + 1.0)
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
        self.cached_normal = Some(radius * angle.sin());
        first
    }
}

fn matrix() -> [f64; DIMENSION * DIMENSION] {
    [
        4.0, 1.0, 0.3, 0.0,
        1.0, 3.0, 0.5, 0.2,
        0.3, 0.5, 2.5, 0.4,
        0.0, 0.2, 0.4, 1.8,
    ]
}

fn prior_covariance() -> [f64; DIMENSION * DIMENSION] {
    [
        2.0, 0.3, 0.0, 0.0,
        0.3, 1.5, 0.2, 0.0,
        0.0, 0.2, 1.0, 0.1,
        0.0, 0.0, 0.1, 0.7,
    ]
}

fn prior_belief() -> GaussianLinearBelief {
    GaussianLinearBelief::new(&[0.0; DIMENSION], &prior_covariance(), DIMENSION)
        .expect("correlated anisotropic Gaussian prior is valid")
}

fn candidates() -> Vec<Vec<f64>> {
    vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0],
    ]
}

fn matvec(matrix: &[f64], vector: &[f64]) -> Vec<f64> {
    (0..DIMENSION)
        .map(|row| {
            (0..DIMENSION)
                .map(|column| matrix[row * DIMENSION + column] * vector[column])
                .sum()
        })
        .collect()
}

fn sample_prior(rng: &mut DeterministicNormalRng) -> Vec<f64> {
    let covariance = DMatrix::from_row_slice(DIMENSION, DIMENSION, &prior_covariance());
    let lower = covariance
        .cholesky()
        .expect("prior covariance is positive definite")
        .l();
    let standard_normal = DVector::from_iterator(
        DIMENSION,
        (0..DIMENSION).map(|_| rng.standard_normal()),
    );
    (lower * standard_normal).iter().copied().collect()
}

fn functional_variance(c: &[f64], covariance: &[f64]) -> f64 {
    (0..DIMENSION)
        .map(|row| {
            (0..DIMENSION)
                .map(|column| c[row] * covariance[row * DIMENSION + column] * c[column])
                .sum::<f64>()
        })
        .sum()
}

fn functional_error(c: &[f64], truth: &[f64], mean: &[f64]) -> f64 {
    c.iter()
        .zip(truth.iter().zip(mean))
        .map(|(weight, (truth_value, mean_value))| weight * (truth_value - mean_value))
        .sum()
}

#[test]
fn covariance_greedy_selection_remains_calibrated_under_the_assumed_prior() {
    let matrix = matrix();
    let solver = CovarianceGreedyProjectionSolver::new(0.0, 2).expect("solver is valid");
    let functionals = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, -0.3, 0.7, 0.2],
    ];
    let mut rng = DeterministicNormalRng::new(0xC0A7_1B5E_CAFE_BABE);
    let mut squared_z_sum = [0.0; 3];
    let mut covered = [0_usize; 3];

    let reference_system =
        SpdLinearSystem::new(&matrix, &[0.0; DIMENSION], DIMENSION).expect("matrix is SPD");
    let reference = solver
        .solve(&reference_system, &prior_belief(), &candidates())
        .expect("reference solve is valid");
    let reference_directions: Vec<Vec<f64>> = reference
        .steps()
        .iter()
        .map(|step| step.direction().to_vec())
        .collect();

    assert_eq!(
        reference_directions,
        vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ]
    );

    for _ in 0..REPLICATES {
        let truth = sample_prior(&mut rng);
        let rhs = matvec(&matrix, &truth);
        let system = SpdLinearSystem::new(&matrix, &rhs, DIMENSION).expect("system is SPD");
        let result = solver
            .solve(&system, &prior_belief(), &candidates())
            .expect("covariance-greedy solve should succeed");
        let directions: Vec<Vec<f64>> = result
            .steps()
            .iter()
            .map(|step| step.direction().to_vec())
            .collect();
        assert_eq!(directions, reference_directions);

        for (index, c) in functionals.iter().enumerate() {
            let variance = functional_variance(c, result.belief().covariance());
            assert!(variance > 0.0);
            let error = functional_error(c, &truth, result.belief().mean());
            let z = error / variance.sqrt();
            squared_z_sum[index] += z * z;
            if z.abs() <= Z_95 {
                covered[index] += 1;
            }
        }
    }

    let replicates = f64::from(u32::try_from(REPLICATES).expect("replicate count fits in u32"));
    for index in 0..functionals.len() {
        let second_moment = squared_z_sum[index] / replicates;
        let coverage = f64::from(u32::try_from(covered[index]).expect("coverage fits in u32"))
            / replicates;
        assert!(
            (second_moment - 1.0).abs() <= 0.16,
            "functional {index}: expected unit second moment, got {second_moment:.4}"
        );
        assert!(
            (coverage - 0.95).abs() <= 0.045,
            "functional {index}: expected about 95% coverage, got {coverage:.4}"
        );
    }
}
