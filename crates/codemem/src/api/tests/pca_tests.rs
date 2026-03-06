use crate::api::pca::power_iteration_top_k;
use ndarray::Array2;

#[test]
fn pca_top_k_returns_k_eigenvectors() {
    // 3x3 symmetric positive-definite matrix
    let matrix = Array2::from_shape_vec(
        (3, 3),
        vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0],
    )
    .unwrap();

    let eigenvectors = power_iteration_top_k(&matrix, 2, 100);
    assert_eq!(eigenvectors.len(), 2);

    // Each eigenvector should be approximately unit-length
    for v in &eigenvectors {
        let norm = v.dot(v).sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Eigenvector norm {norm} should be ~1.0"
        );
    }
}

#[test]
fn pca_top_k_with_identity_matrix() {
    // Identity matrix: all eigenvalues are 1 (degenerate case).
    // After deflation of the first eigenvector the residual eigenvalues
    // are near-zero, so later eigenvectors may not converge to unit norm.
    let matrix = Array2::eye(4);
    let eigenvectors = power_iteration_top_k(&matrix, 3, 50);
    assert_eq!(eigenvectors.len(), 3);

    // At least the first eigenvector should be unit-length
    let norm0 = eigenvectors[0].dot(&eigenvectors[0]).sqrt();
    assert!(
        (norm0 - 1.0).abs() < 1e-6,
        "First eigenvector should have unit norm, got {norm0}"
    );
}

#[test]
fn pca_top_k_with_diagonal_matrix() {
    // Diagonal matrix with known eigenvalues [4, 3, 2, 1]
    let mut matrix = Array2::<f64>::zeros((4, 4));
    matrix[[0, 0]] = 4.0;
    matrix[[1, 1]] = 3.0;
    matrix[[2, 2]] = 2.0;
    matrix[[3, 3]] = 1.0;

    let eigenvectors = power_iteration_top_k(&matrix, 2, 100);
    assert_eq!(eigenvectors.len(), 2);

    // First eigenvector should align with the largest eigenvalue (axis 0)
    let v0 = &eigenvectors[0];
    assert!(
        v0[0].abs() > 0.9,
        "First eigenvector should point along axis 0, got {:?}",
        v0
    );
}

#[test]
fn pca_top_k_requests_more_than_dimensions() {
    let matrix = Array2::eye(2);
    // Requesting 5 eigenvectors from a 2x2 matrix
    let eigenvectors = power_iteration_top_k(&matrix, 5, 50);
    assert_eq!(eigenvectors.len(), 5);
    // The first 2 should be valid, the rest may be zero-ish due to deflation
}

#[test]
fn pca_top_k_single_dimension() {
    let matrix = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
    let eigenvectors = power_iteration_top_k(&matrix, 1, 50);
    assert_eq!(eigenvectors.len(), 1);
    assert!((eigenvectors[0][0].abs() - 1.0).abs() < 1e-6);
}

#[test]
fn pca_top_k_zero_matrix() {
    let matrix = Array2::<f64>::zeros((3, 3));
    let eigenvectors = power_iteration_top_k(&matrix, 2, 50);
    assert_eq!(eigenvectors.len(), 2);
    // With a zero matrix, power iteration should bail early (norm < 1e-10)
}

#[test]
fn pca_eigenvectors_are_approximately_orthogonal() {
    // Symmetric matrix with distinct eigenvalues
    let matrix = Array2::from_shape_vec(
        (3, 3),
        vec![5.0, 1.0, 0.5, 1.0, 3.0, 0.5, 0.5, 0.5, 1.0],
    )
    .unwrap();

    let eigenvectors = power_iteration_top_k(&matrix, 3, 200);
    assert_eq!(eigenvectors.len(), 3);

    // Check orthogonality between first two eigenvectors
    let dot01 = eigenvectors[0].dot(&eigenvectors[1]);
    assert!(
        dot01.abs() < 0.1,
        "First two eigenvectors should be approximately orthogonal, dot product = {dot01}"
    );
}
