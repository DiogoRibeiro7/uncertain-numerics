use uncertain_numerics::{
    GaussianLinearBelief, ResidualProjectionSolver, SpdLinearSystem,
};

const DIMENSION: usize = 4;
const REPLICATES: usize = 1_500;
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

fn identity_belief() -> GaussianLinearBelief {
    let mut covariance = vec![0.0; DIMENSION * DIMENSION];
    for index in 0..DIMENSION {
        covariance[index * DIMENSION + index] = 1.0;
    }
    GaussianLinearBelief::new(&[0.0; DIMENSION], &covariance, DIMENSION)
        .expect("identity Gaussian prior is valid")
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
fn fixed_projection_conditioning_is_calibrated_under_the_assumed_prior() {
    let matrix = matrix();
    let directions = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
    let functionals = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.5, -0.3, 0.7, 0.2],
    ];
    let mut rng = DeterministicNormalRng::new(0x51A7_1C5E_CAFE_BABE);
    let mut squared_z_sum = [0.0; 3];
    let mut covered = [0_usize; 3];

    for _ in 0..REPLICATES {
        let truth: Vec<f64> = (0..DIMENSION).map(|_| rng.standard_normal()).collect();
        let rhs = matvec(&matrix, &truth);
        let system = SpdLinearSystem::new(&matrix, &rhs, DIMENSION).expect("system is SPD");
        let mut belief = identity_belief();
        for direction in directions {
            belief = belief
                .condition_on_projection(&system, &direction)
                .expect("fixed projection is informative");
        }

        for (index, c) in functionals.iter().enumerate() {
            let variance = functional_variance(c, belief.covariance());
            assert!(variance > 0.0);
            let error = functional_error(c, &truth, belief.mean());
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
            (second_moment - 1.0).abs() <= 0.15,
            "functional {index}: expected unit second moment, got {second_moment:.4}"
        );
        assert!(
            (coverage - 0.95).abs() <= 0.04,
            "functional {index}: expected about 95% coverage, got {coverage:.4}"
        );
    }
}

#[test]
fn adaptive_residual_selection_makes_naive_posterior_uncertainty_conservative() {
    let matrix = matrix();
    let functional = [0.0, 0.0, 1.0, 0.0];
    let solver = ResidualProjectionSolver::new(0.0, 0.0, 2).expect("solver is valid");
    let mut rng = DeterministicNormalRng::new(0xADA9_71VE_CAFE_F00D_u64);
    let mut squared_z_sum = 0.0;
    let mut covered = 0_usize;
    let mut usable = 0_usize;

    for _ in 0..REPLICATES {
        let truth: Vec<f64> = (0..DIMENSION).map(|_| rng.standard_normal()).collect();
        let rhs = matvec(&matrix, &truth);
        let system = SpdLinearSystem::new(&matrix, &rhs, DIMENSION).expect("system is SPD");
        let result = solver
            .solve(&system, &identity_belief())
            .expect("adaptive solve should succeed");
        let belief = result.belief();
        let variance = functional_variance(&functional, belief.covariance());
        if variance <= 1.0e-12 {
            continue;
        }
        let error = functional_error(&functional, &truth, belief.mean());
        let z = error / variance.sqrt();
        squared_z_sum += z * z;
        if z.abs() <= Z_95 {
            covered += 1;
        }
        usable += 1;
    }

    assert!(usable > REPLICATES / 2);
    let usable_f64 = f64::from(u32::try_from(usable).expect("usable count fits in u32"));
    let second_moment = squared_z_sum / usable_f64;
    let coverage = f64::from(u32::try_from(covered).expect("coverage count fits in u32"))
        / usable_f64;

    assert!(
        second_moment < 0.4,
        "adaptive selection should expose conservative covariance here; got second moment {second_moment:.4}"
    );
    assert!(
        coverage > 0.985,
        "adaptive selection should over-cover under the naive covariance; got {coverage:.4}"
    );
}
