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
