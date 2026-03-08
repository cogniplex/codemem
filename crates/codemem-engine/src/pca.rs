use ndarray::{Array1, Array2};

/// Power iteration with deflation to extract the top-k eigenvectors
/// from a symmetric matrix.
pub fn power_iteration_top_k(
    matrix: &Array2<f64>,
    k: usize,
    iterations: usize,
) -> Vec<Array1<f64>> {
    let dim = matrix.nrows();
    let mut deflated = matrix.clone();
    let mut eigenvectors = Vec::with_capacity(k);

    for _ in 0..k {
        let mut v = Array1::<f64>::ones(dim);
        for i in 0..dim {
            if i % 2 == 0 {
                v[i] = 1.0;
            } else {
                v[i] = -1.0;
            }
        }

        for _ in 0..iterations {
            let mv = deflated.dot(&v);
            let norm = mv.dot(&mv).sqrt();
            if norm < 1e-10 {
                break;
            }
            v = mv / norm;
        }

        let eigenvalue = v.dot(&deflated.dot(&v));
        let outer = outer_product(&v, &v);
        deflated = deflated - eigenvalue * &outer;

        eigenvectors.push(v);
    }

    eigenvectors
}

fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let n = a.len();
    let m = b.len();
    let mut result = Array2::<f64>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn top_k_returns_k_eigenvectors() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0])
                .unwrap();

        let eigenvectors = power_iteration_top_k(&matrix, 2, 100);
        assert_eq!(eigenvectors.len(), 2);

        for v in &eigenvectors {
            let norm = v.dot(v).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Eigenvector norm {norm} should be ~1.0"
            );
        }
    }

    #[test]
    fn top_k_with_identity_matrix() {
        let matrix = Array2::eye(4);
        let eigenvectors = power_iteration_top_k(&matrix, 3, 50);
        assert_eq!(eigenvectors.len(), 3);

        let norm0 = eigenvectors[0].dot(&eigenvectors[0]).sqrt();
        assert!(
            (norm0 - 1.0).abs() < 1e-6,
            "First eigenvector should have unit norm, got {norm0}"
        );
    }

    #[test]
    fn top_k_with_diagonal_matrix() {
        let mut matrix = Array2::<f64>::zeros((4, 4));
        matrix[[0, 0]] = 4.0;
        matrix[[1, 1]] = 3.0;
        matrix[[2, 2]] = 2.0;
        matrix[[3, 3]] = 1.0;

        let eigenvectors = power_iteration_top_k(&matrix, 2, 100);
        assert_eq!(eigenvectors.len(), 2);

        let v0 = &eigenvectors[0];
        assert!(
            v0[0].abs() > 0.9,
            "First eigenvector should point along axis 0, got {:?}",
            v0
        );
    }

    #[test]
    fn top_k_requests_more_than_dimensions() {
        let matrix = Array2::eye(2);
        let eigenvectors = power_iteration_top_k(&matrix, 5, 50);
        assert_eq!(eigenvectors.len(), 5);
    }

    #[test]
    fn top_k_single_dimension() {
        let matrix = Array2::from_shape_vec((1, 1), vec![5.0]).unwrap();
        let eigenvectors = power_iteration_top_k(&matrix, 1, 50);
        assert_eq!(eigenvectors.len(), 1);
        assert!((eigenvectors[0][0].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn top_k_zero_matrix() {
        let matrix = Array2::<f64>::zeros((3, 3));
        let eigenvectors = power_iteration_top_k(&matrix, 2, 50);
        assert_eq!(eigenvectors.len(), 2);
    }

    #[test]
    fn eigenvectors_are_approximately_orthogonal() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![5.0, 1.0, 0.5, 1.0, 3.0, 0.5, 0.5, 0.5, 1.0])
                .unwrap();

        let eigenvectors = power_iteration_top_k(&matrix, 3, 200);
        assert_eq!(eigenvectors.len(), 3);

        let dot01 = eigenvectors[0].dot(&eigenvectors[1]);
        assert!(
            dot01.abs() < 0.1,
            "First two eigenvectors should be approximately orthogonal, dot product = {dot01}"
        );
    }
}
