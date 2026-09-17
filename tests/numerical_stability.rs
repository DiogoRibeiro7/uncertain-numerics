use nalgebra::{DMatrix, SymmetricEigen};
use uncertain_numerics::{BayesianQuadrature, GaussianMeasure, RbfKernel, ScalarKernel};

fn gram_matrix(kernel: RbfKernel, nodes: &[f64], jitter: f64) -> DMatrix<f64> {
    let dimension = nodes.len();
    let mut matrix = DMatrix::zeros(dimension, dimension);

    for (row, &left) in nodes.iter().enumerate() {
        for (column, &right) in nodes.iter().enumerate() {
            matrix[(row, column)] = kernel.covariance(left, right);
        }
        matrix[(row, row)] += jitter;
    }

    matrix
}

fn spectral_condition_number(matrix: &DMatrix<f64>) -> f64 {
    let eigenvalues = SymmetricEigen::new(matrix.clone()).eigenvalues;
    let largest = eigenvalues.max();
    let smallest = eigenvalues.min();

    largest / smallest
}

#[test]
fn near_duplicate_nodes_create_severe_ill_conditioning() {
    let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
    let nodes = [0.0, 1.0e-6, 1.0];
    let matrix = gram_matrix(kernel, &nodes, 0.0);
    let condition_number = spectral_condition_number(&matrix);

    assert!(
        condition_number > 1.0e12,
        "near-duplicate nodes should create severe ill-conditioning, got {condition_number:.6e}"
    );
}

#[test]
fn explicit_jitter_regularizes_near_duplicate_nodes() {
    let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
    let nodes = [0.0, 1.0e-6, 1.0];
    let unregularized = spectral_condition_number(&gram_matrix(kernel, &nodes, 0.0));
    let regularized = spectral_condition_number(&gram_matrix(kernel, &nodes, 1.0e-8));

    assert!(regularized < unregularized);
    assert!(
        regularized < 1.0e9,
        "1e-8 jitter should reduce this fixture below 1e9 condition number, got {regularized:.6e}"
    );
}

#[test]
fn very_large_length_scale_drives_gram_matrix_toward_rank_one() {
    let nodes = [-1.0, 0.0, 1.0];
    let moderate = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
    let extremely_smooth = RbfKernel::new(1.0, 1.0e8).expect("kernel parameters are valid");

    let moderate_condition = spectral_condition_number(&gram_matrix(moderate, &nodes, 0.0));
    let smooth_condition = spectral_condition_number(&gram_matrix(extremely_smooth, &nodes, 0.0));

    assert!(smooth_condition > moderate_condition * 1.0e10);
}

#[test]
fn posterior_remains_finite_across_reasonable_jitter_sweep_for_duplicate_nodes() {
    let kernel = RbfKernel::new(1.0, 1.0).expect("kernel parameters are valid");
    let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
    let nodes = [0.0, 0.0, 1.0];
    let values = [1.0, 1.0, (-0.5_f64).exp()];

    for jitter in [1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6] {
        let posterior = BayesianQuadrature::new(kernel, measure, jitter)
            .posterior(&nodes, &values)
            .expect("positive jitter should regularize duplicate-node fixture");

        assert!(posterior.mean().is_finite());
        assert!(posterior.variance().is_finite());
        assert!(posterior.variance() >= 0.0);
    }
}

#[test]
fn large_length_scale_requires_regularization_for_stable_posterior() {
    let kernel = RbfKernel::new(1.0, 1.0e8).expect("kernel parameters are valid");
    let measure = GaussianMeasure::new(0.0, 1.0).expect("measure parameters are valid");
    let nodes = [-1.0, 0.0, 1.0];
    let values = [1.0, 0.0, 1.0];

    let regularized = BayesianQuadrature::new(kernel, measure, 1.0e-8)
        .posterior(&nodes, &values)
        .expect("explicit jitter should stabilize the nearly rank-one Gram matrix");

    assert!(regularized.mean().is_finite());
    assert!(regularized.variance().is_finite());
    assert!(regularized.variance() >= 0.0);
}
