use uncertain_numerics::{
    AConjugateProjectionSolver, GaussianLinearBelief, ResidualProjectionSolver, SpdLinearSystem,
};

fn identity_belief(dimension: usize) -> GaussianLinearBelief {
    let mut covariance = vec![0.0; dimension * dimension];
    for index in 0..dimension {
        covariance[index * dimension + index] = 1.0;
    }
    GaussianLinearBelief::new(&vec![0.0; dimension], &covariance, dimension)
        .expect("identity belief is valid")
}

fn assert_vectors_close(left: &[f64], right: &[f64], tolerance: f64) {
    assert_eq!(left.len(), right.len());
    for (left_value, right_value) in left.iter().zip(right) {
        assert!(
            (left_value - right_value).abs() <= tolerance,
            "expected {left_value:.16e} ~= {right_value:.16e}"
        );
    }
}

#[test]
fn equal_projection_budgets_produce_equivalent_gaussian_posteriors() {
    let fixtures = [
        (
            vec![4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0],
            vec![1.0, 2.0, -1.0],
            3,
        ),
        (
            vec![1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 10.0],
            vec![1.0, 2.0, -1.0],
            3,
        ),
        (
            vec![
                6.0, 2.0, 1.0, 0.0,
                2.0, 5.0, 2.0, 1.0,
                1.0, 2.0, 4.0, 1.0,
                0.0, 1.0, 1.0, 3.0,
            ],
            vec![1.0, 1.0, 1.0, 1.0],
            4,
        ),
    ];

    for (matrix, rhs, dimension) in fixtures {
        let system = SpdLinearSystem::new(&matrix, &rhs, dimension).expect("system is SPD");
        let belief = identity_belief(dimension);

        for budget in 1..dimension {
            let residual_solver =
                ResidualProjectionSolver::new(0.0, 0.0, budget).expect("valid solver");
            let conjugate_solver =
                AConjugateProjectionSolver::new(0.0, 0.0, budget).expect("valid solver");

            let residual_result = residual_solver
                .solve(&system, &belief)
                .expect("residual solve should succeed");
            let conjugate_result = conjugate_solver
                .solve(&system, &belief)
                .expect("conjugate solve should succeed");

            assert_eq!(residual_result.steps().len(), budget);
            assert_eq!(conjugate_result.steps().len(), budget);

            assert_vectors_close(
                residual_result.belief().mean(),
                conjugate_result.belief().mean(),
                5.0e-11,
            );
            assert_vectors_close(
                residual_result.belief().covariance(),
                conjugate_result.belief().covariance(),
                5.0e-11,
            );
        }
    }
}

#[test]
fn search_bases_can_differ_even_when_posteriors_match() {
    let system = SpdLinearSystem::new(
        &[4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0],
        &[1.0, 2.0, -1.0],
        3,
    )
    .expect("system is SPD");
    let belief = identity_belief(3);

    let residual_result = ResidualProjectionSolver::new(0.0, 0.0, 2)
        .expect("valid solver")
        .solve(&system, &belief)
        .expect("residual solve should succeed");
    let conjugate_result = AConjugateProjectionSolver::new(0.0, 0.0, 2)
        .expect("valid solver")
        .solve(&system, &belief)
        .expect("conjugate solve should succeed");

    let residual_second = residual_result.steps()[1].search_direction();
    let conjugate_second = conjugate_result.steps()[1].search_direction();
    let direction_difference = residual_second
        .iter()
        .zip(conjugate_second)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt();

    assert!(direction_difference > 1.0e-3);
    assert_vectors_close(
        residual_result.belief().mean(),
        conjugate_result.belief().mean(),
        5.0e-11,
    );
    assert_vectors_close(
        residual_result.belief().covariance(),
        conjugate_result.belief().covariance(),
        5.0e-11,
    );
}
