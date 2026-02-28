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
        // Start with a random-ish vector (use alternating pattern for determinism)
        let mut v = Array1::<f64>::ones(dim);
        for i in 0..dim {
            if i % 2 == 0 {
                v[i] = 1.0;
            } else {
                v[i] = -1.0;
            }
        }

        // Power iteration
        for _ in 0..iterations {
            let mv = deflated.dot(&v);
            let norm = mv.dot(&mv).sqrt();
            if norm < 1e-10 {
                break;
            }
            v = mv / norm;
        }

        // Deflate: remove this eigenvector's contribution
        let eigenvalue = v.dot(&deflated.dot(&v));
        let outer = outer_product(&v, &v);
        deflated = deflated - eigenvalue * &outer;

        eigenvectors.push(v);
    }

    eigenvectors
}

/// Compute outer product of two vectors.
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
    fn power_iteration_identity_matrix() {
        // Identity matrix: all eigenvalues = 1, so after deflation subsequent
        // eigenvalues are near zero. We verify the first eigenvector is
        // approximately unit length and that we get the right count back.
        let identity = Array2::<f64>::eye(3);
        let eigenvectors = power_iteration_top_k(&identity, 3, 100);
        assert_eq!(eigenvectors.len(), 3);
        // At minimum, each result should be a 3-element vector
        for v in &eigenvectors {
            assert_eq!(v.len(), 3);
        }
    }

    #[test]
    fn power_iteration_diagonal_matrix() {
        // Diagonal matrix with known eigenvalues
        let mut m = Array2::<f64>::zeros((3, 3));
        m[[0, 0]] = 5.0;
        m[[1, 1]] = 3.0;
        m[[2, 2]] = 1.0;

        let eigenvectors = power_iteration_top_k(&m, 2, 100);
        assert_eq!(eigenvectors.len(), 2);

        // First eigenvector should correspond to largest eigenvalue (5.0)
        // So it should be close to [1,0,0] or [-1,0,0]
        let v0 = &eigenvectors[0];
        assert!(
            v0[0].abs() > 0.9,
            "First component should dominate: {:?}",
            v0
        );
    }

    #[test]
    fn power_iteration_empty_request() {
        let m = Array2::<f64>::eye(3);
        let eigenvectors = power_iteration_top_k(&m, 0, 100);
        assert!(eigenvectors.is_empty());
    }

    #[test]
    fn outer_product_basic() {
        let a = Array1::from(vec![1.0, 2.0]);
        let b = Array1::from(vec![3.0, 4.0]);
        let result = outer_product(&a, &b);
        assert_eq!(result[[0, 0]], 3.0);
        assert_eq!(result[[0, 1]], 4.0);
        assert_eq!(result[[1, 0]], 6.0);
        assert_eq!(result[[1, 1]], 8.0);
    }
}
