//! Equal-matrix-vector-budget comparison of the probabilistic linear solvers
//! against classical conjugate gradients, plus the structural result that
//! explains it: with an identity prior covariance, the residual-projection
//! posterior mean is Craig's method (CGNE) in disguise.
//!
//! Matrix-vector accounting counts the products with `A` an algorithm needs,
//! not what a particular implementation happens to recompute:
//!
//! - conjugate gradients: one product per iteration (`A p_k`);
//! - residual projection: two per projection, `A s_k` for the observation and
//!   `A g_k` to update the residual after the mean moves;
//! - Craig's method: two per iteration (`A p_k` and `A^T r_{k+1}`).
//!
//! All solvers start from the zero vector, so the initial residual is free.

use nalgebra::{DMatrix, DVector};
use uncertain_numerics::{
    AConjugateProjectionSolver, GaussianLinearBelief, ResidualProjectionSolver, SpdLinearSystem,
};

/// The posterior mean and Craig's iterate coincide in exact arithmetic. The two
/// recurrences accumulate roundoff differently, and at condition number ~100
/// they drift apart at the 1e-7 level by the seventh projection.
const EQUIVALENCE_TOLERANCE: f64 = 1.0e-6;
const EXACTNESS_TOLERANCE: f64 = 1.0e-8;
/// Every solver stops once the residual norm falls below this multiple of the
/// right-hand-side norm, so none of them keeps iterating on roundoff noise.
const RELATIVE_RESIDUAL_STOP: f64 = 1.0e-13;

fn residual_stop(rhs: &DVector<f64>) -> f64 {
    RELATIVE_RESIDUAL_STOP * rhs.norm()
}

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

struct Fixture {
    name: &'static str,
    dimension: usize,
    matrix: DMatrix<f64>,
    rhs: DVector<f64>,
}

impl Fixture {
    fn system(&self) -> SpdLinearSystem {
        let mut row_major = Vec::with_capacity(self.dimension * self.dimension);
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                row_major.push(self.matrix[(row, column)]);
            }
        }
        SpdLinearSystem::new(&row_major, self.rhs.as_slice(), self.dimension)
            .expect("fixture is SPD")
    }

    fn exact_solution(&self) -> DVector<f64> {
        self.matrix
            .clone()
            .cholesky()
            .expect("fixture is SPD")
            .solve(&self.rhs)
    }

    fn condition_number(&self) -> f64 {
        let eigenvalues = self.matrix.clone().symmetric_eigen().eigenvalues;
        eigenvalues.max() / eigenvalues.min()
    }
}

fn laplacian_fixture(dimension: usize) -> Fixture {
    let mut matrix = DMatrix::zeros(dimension, dimension);
    for index in 0..dimension {
        matrix[(index, index)] = 2.0;
        if index + 1 < dimension {
            matrix[(index, index + 1)] = -1.0;
            matrix[(index + 1, index)] = -1.0;
        }
    }
    // An asymmetric right-hand side keeps the Krylov space full-dimensional; a
    // constant one would let every solver finish in n/2 steps by symmetry.
    let rhs = DVector::from_fn(dimension, |index, _| {
        1.0 + 0.25 * f64::from(u32::try_from(index).expect("small dimension"))
    });
    Fixture {
        name: "one-dimensional Laplacian",
        dimension,
        matrix,
        rhs,
    }
}

fn graded_fixture(dimension: usize) -> Fixture {
    // Log-spaced diagonal from 1 to 100 plus a rank-one coupling keeps the
    // matrix SPD while giving it a known spectral spread.
    let mut matrix = DMatrix::from_element(dimension, dimension, 0.1);
    let mut exponent = 0.0;
    let step = 2.0 / f64::from(u32::try_from(dimension - 1).expect("small dimension"));
    for index in 0..dimension {
        matrix[(index, index)] += 10.0_f64.powf(exponent);
        exponent += step;
    }
    let rhs = DVector::from_iterator(
        dimension,
        (0..dimension).map(|index| if index % 2 == 0 { 1.0 } else { -1.0 }),
    );
    Fixture {
        name: "graded spectrum with rank-one coupling",
        dimension,
        matrix,
        rhs,
    }
}

fn random_fixture(dimension: usize, seed: u64) -> Fixture {
    let mut rng = DeterministicNormalRng::new(seed);
    let factor = DMatrix::from_fn(dimension, dimension, |_, _| rng.standard_normal());
    let scale = f64::from(u32::try_from(dimension).expect("small dimension"));
    let matrix = &factor * factor.transpose() / scale + DMatrix::identity(dimension, dimension);
    let rhs = DVector::from_fn(dimension, |_, _| rng.standard_normal());
    Fixture {
        name: "deterministic random SPD",
        dimension,
        matrix,
        rhs,
    }
}

fn fixtures() -> Vec<Fixture> {
    vec![
        laplacian_fixture(8),
        graded_fixture(8),
        random_fixture(10, 0x5EED_0000_0000_0001),
    ]
}

fn identity_belief(dimension: usize) -> GaussianLinearBelief {
    let identity = DMatrix::<f64>::identity(dimension, dimension);
    let mut row_major = Vec::with_capacity(dimension * dimension);
    for row in 0..dimension {
        for column in 0..dimension {
            row_major.push(identity[(row, column)]);
        }
    }
    GaussianLinearBelief::new(&vec![0.0; dimension], &row_major, dimension)
        .expect("identity belief is valid")
}

/// Classical conjugate gradients from the zero vector; one product with `A`
/// per iteration.
fn conjugate_gradient(
    matrix: &DMatrix<f64>,
    rhs: &DVector<f64>,
    iterations: usize,
) -> DVector<f64> {
    let stop = residual_stop(rhs);
    let mut solution = DVector::zeros(rhs.len());
    let mut residual = rhs.clone();
    let mut direction = residual.clone();
    let mut residual_norm_squared = residual.dot(&residual);
    for _ in 0..iterations {
        if residual_norm_squared.sqrt() <= stop {
            break;
        }
        let matrix_direction = matrix * &direction;
        let step_length = residual_norm_squared / direction.dot(&matrix_direction);
        solution += step_length * &direction;
        residual -= step_length * &matrix_direction;
        let next_norm_squared = residual.dot(&residual);
        direction = &residual + (next_norm_squared / residual_norm_squared) * &direction;
        residual_norm_squared = next_norm_squared;
    }
    solution
}

/// Craig's method (CGNE) from the zero vector: conjugate gradients on
/// `A A^T y = b`, `x = A^T y`, which minimises the Euclidean error over
/// `A^T K_k(A A^T, b)`. Two products with `A` per iteration.
fn craig_method(matrix: &DMatrix<f64>, rhs: &DVector<f64>, iterations: usize) -> DVector<f64> {
    let stop = residual_stop(rhs);
    let mut solution = DVector::zeros(rhs.len());
    let mut residual = rhs.clone();
    let mut direction = matrix.transpose() * &residual;
    let mut residual_norm_squared = residual.dot(&residual);
    for _ in 0..iterations {
        if residual_norm_squared.sqrt() <= stop {
            break;
        }
        let step_length = residual_norm_squared / direction.dot(&direction);
        solution += step_length * &direction;
        residual -= step_length * (matrix * &direction);
        let next_norm_squared = residual.dot(&residual);
        direction = matrix.transpose() * &residual
            + (next_norm_squared / residual_norm_squared) * &direction;
        residual_norm_squared = next_norm_squared;
    }
    solution
}

fn residual_policy_mean(system: &SpdLinearSystem, projections: usize) -> DVector<f64> {
    let stop = residual_stop(&DVector::from_column_slice(system.rhs()));
    let solver = ResidualProjectionSolver::new(stop, 0.0, projections).expect("valid solver");
    let result = solver
        .solve(system, &identity_belief(system.dimension()))
        .unwrap_or_else(|error| {
            panic!("residual policy failed after at most {projections} projections: {error}")
        });
    DVector::from_column_slice(result.belief().mean())
}

fn conjugate_policy_mean(system: &SpdLinearSystem, projections: usize) -> DVector<f64> {
    let stop = residual_stop(&DVector::from_column_slice(system.rhs()));
    let solver = AConjugateProjectionSolver::new(stop, 0.0, projections).expect("valid solver");
    let result = solver
        .solve(system, &identity_belief(system.dimension()))
        .unwrap_or_else(|error| {
            panic!("A-conjugate policy failed after at most {projections} projections: {error}")
        });
    DVector::from_column_slice(result.belief().mean())
}

fn relative_error(estimate: &DVector<f64>, exact: &DVector<f64>) -> f64 {
    (estimate - exact).norm() / exact.norm()
}

#[test]
fn residual_projection_with_identity_prior_reproduces_craig_iterates() {
    for fixture in fixtures() {
        let system = fixture.system();
        let exact = fixture.exact_solution();
        for projections in 1..fixture.dimension {
            let posterior_mean = residual_policy_mean(&system, projections);
            let craig = craig_method(&fixture.matrix, &fixture.rhs, projections);
            let discrepancy = (&posterior_mean - &craig).norm() / exact.norm();
            println!(
                "{:>40} k={projections}: |mean - craig|/|x*| = {discrepancy:.2e}",
                fixture.name
            );
            assert!(
                discrepancy <= EQUIVALENCE_TOLERANCE,
                "{}: posterior mean after {projections} projections differs from Craig's method by {discrepancy:.3e}",
                fixture.name
            );
        }
    }
}

#[test]
fn conjugate_gradients_dominate_at_equal_matrix_vector_budgets() {
    for fixture in fixtures() {
        let system = fixture.system();
        let exact = fixture.exact_solution();
        println!(
            "{} (n = {}, condition number {:.1})",
            fixture.name,
            fixture.dimension,
            fixture.condition_number()
        );
        println!(
            "  matvecs | CG iterations, rel. error | projections, rel. error (residual / A-conjugate)"
        );
        for projections in 1..=fixture.dimension {
            let budget = 2 * projections;
            let cg =
                conjugate_gradient(&fixture.matrix, &fixture.rhs, budget.min(fixture.dimension));
            let residual = residual_policy_mean(&system, projections);
            let conjugate = conjugate_policy_mean(&system, projections);
            let cg_error = relative_error(&cg, &exact);
            let residual_error = relative_error(&residual, &exact);
            let conjugate_error = relative_error(&conjugate, &exact);
            println!(
                "  {budget:>7} | {:>2}, {cg_error:.3e}           | {projections:>2}, {residual_error:.3e} / {conjugate_error:.3e}",
                budget.min(fixture.dimension)
            );

            if projections < fixture.dimension {
                assert!(
                    cg_error <= residual_error,
                    "{}: at {budget} matrix-vector products CG error {cg_error:.3e} exceeds residual-policy error {residual_error:.3e}",
                    fixture.name
                );
                assert!(
                    cg_error <= conjugate_error,
                    "{}: at {budget} matrix-vector products CG error {cg_error:.3e} exceeds A-conjugate-policy error {conjugate_error:.3e}",
                    fixture.name
                );
            } else {
                assert!(
                    cg_error <= EXACTNESS_TOLERANCE,
                    "{}: CG error {cg_error:.3e}",
                    fixture.name
                );
                assert!(
                    residual_error <= EXACTNESS_TOLERANCE,
                    "{}: residual policy error {residual_error:.3e}",
                    fixture.name
                );
            }
        }
    }
}

#[test]
fn both_solvers_are_exact_with_a_full_budget() {
    for fixture in fixtures() {
        let system = fixture.system();
        let exact = fixture.exact_solution();
        let cg = conjugate_gradient(&fixture.matrix, &fixture.rhs, fixture.dimension);
        let residual = residual_policy_mean(&system, fixture.dimension);
        let cg_error = relative_error(&cg, &exact);
        let residual_error = relative_error(&residual, &exact);
        println!(
            "{:>40}: full-budget errors CG {cg_error:.2e}, residual policy {residual_error:.2e}",
            fixture.name
        );
        assert!(
            cg_error <= EXACTNESS_TOLERANCE,
            "{}: CG error {cg_error:.3e}",
            fixture.name
        );
        assert!(
            residual_error <= EXACTNESS_TOLERANCE,
            "{}: residual policy error {residual_error:.3e}",
            fixture.name
        );
    }
}
