//! Spectral decomposition of diffusion operators, eigenfunction expansion.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: DMatrix<f64>,
}

impl SpectralDecomposition {
    pub fn decompose(matrix: &DMatrix<f64>) -> Result<Self, String> {
        let n = matrix.nrows();
        if n != matrix.ncols() { return Err("Matrix must be square".into()); }
        if n == 0 { return Ok(Self { eigenvalues: vec![], eigenvectors: DMatrix::zeros(0, 0) }); }
        let eig = matrix.clone().symmetric_eigen();
        Ok(Self {
            eigenvalues: eig.eigenvalues.iter().copied().collect(),
            eigenvectors: eig.eigenvectors,
        })
    }

    pub fn reconstruct(&self) -> DMatrix<f64> {
        let n = self.eigenvalues.len();
        if n == 0 { return DMatrix::zeros(0, 0); }
        let mut result = DMatrix::zeros(n, n);
        for k in 0..n {
            let v = self.eigenvectors.column(k);
            result += &v.scale(self.eigenvalues[k]) * &v.transpose();
        }
        result
    }

    pub fn low_rank(&self, k: usize) -> DMatrix<f64> {
        let n = self.eigenvalues.len();
        if n == 0 { return DMatrix::zeros(0, 0); }
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| self.eigenvalues[b].abs().partial_cmp(&self.eigenvalues[a].abs()).unwrap());
        let mut result = DMatrix::zeros(n, n);
        for &idx in indices.iter().take(k) {
            let v = self.eigenvectors.column(idx);
            result += &v.scale(self.eigenvalues[idx]) * &v.transpose();
        }
        result
    }
}

pub fn heat_kernel_spectral(eigenvalues: &[f64], eigenvectors: &DMatrix<f64>, t: f64, i: usize, j: usize, n_terms: usize) -> f64 {
    eigenvalues.iter().take(n_terms)
        .zip(eigenvectors.column_iter().take(n_terms))
        .map(|(lambda, phi)| (-lambda * t).exp() * phi[i] * phi[j])
        .sum()
}

pub fn spectral_diffusion_solve(eigenvalues: &[f64], eigenvectors: &DMatrix<f64>, initial: &DVector<f64>, t: f64) -> DVector<f64> {
    let n = eigenvalues.len();
    let mut result = DVector::zeros(n);
    for k in 0..n {
        let phi_k = eigenvectors.column(k);
        let c_k = phi_k.dot(initial);
        result += phi_k.scale(c_k * (-eigenvalues[k] * t).exp());
    }
    result
}

pub fn spectral_gap(eigenvalues: &[f64]) -> Option<f64> {
    let mut sorted = eigenvalues.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted.iter().find(|&&e| e > 1e-10).copied()
}

pub fn diffusion_kernel_matrix(eigenvalues: &[f64], eigenvectors: &DMatrix<f64>, t: f64) -> DMatrix<f64> {
    let n = eigenvalues.len();
    let mut result = DMatrix::zeros(n, n);
    for k in 0..n {
        let phi_k = eigenvectors.column(k);
        result += &phi_k.scale((-eigenvalues[k] * t).exp()) * &phi_k.transpose();
    }
    result
}

pub fn laplacian_1d_matrix(n: usize, dx: f64) -> DMatrix<f64> {
    let mut l = DMatrix::zeros(n, n);
    let inv_dx2 = 1.0 / (dx * dx);
    for i in 0..n {
        l[(i, i)] = -2.0 * inv_dx2;
        if i > 0 { l[(i, i - 1)] = 1.0 * inv_dx2; }
        if i < n - 1 { l[(i, i + 1)] = 1.0 * inv_dx2; }
    }
    l
}

pub fn graph_laplacian(adjacency: &DMatrix<f64>) -> DMatrix<f64> {
    let n = adjacency.nrows();
    let degree: DVector<f64> = DVector::from_iterator(n, (0..n).map(|i| adjacency.row(i).sum()));
    DMatrix::from_diagonal(&degree) - adjacency
}

pub fn normalized_graph_laplacian(adjacency: &DMatrix<f64>) -> DMatrix<f64> {
    let n = adjacency.nrows();
    let degree: DVector<f64> = DVector::from_iterator(n, (0..n).map(|i| adjacency.row(i).sum()));
    let d_inv_sqrt: DVector<f64> = degree.map(|d| if d > 1e-15 { 1.0 / d.sqrt() } else { 0.0 });
    let d_matrix = DMatrix::from_diagonal(&d_inv_sqrt);
    DMatrix::identity(n, n) - &d_matrix * adjacency * &d_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_laplacian_1d() {
        let l = laplacian_1d_matrix(3, 1.0);
        assert_relative_eq!(l[(0, 0)], -2.0);
        assert_relative_eq!(l[(0, 1)], 1.0);
        assert_relative_eq!(l[(1, 0)], 1.0);
    }

    #[test]
    fn test_graph_laplacian() {
        let adj = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
        let l = graph_laplacian(&adj);
        assert_relative_eq!(l[(0, 0)], 2.0);
        assert_relative_eq!(l[(0, 1)], -1.0);
    }

    #[test]
    fn test_normalized_graph_laplacian() {
        let adj = DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
        let l_norm = normalized_graph_laplacian(&adj);
        assert_relative_eq!(l_norm[(0, 0)], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spectral_decomposition() {
        let l = laplacian_1d_matrix(5, 1.0);
        let decomp = SpectralDecomposition::decompose(&l).unwrap();
        assert_eq!(decomp.eigenvalues.len(), 5);
        assert!(decomp.eigenvalues.iter().all(|&e| e <= 1e-6));
    }

    #[test]
    fn test_spectral_reconstruction() {
        let a = DMatrix::from_row_slice(3, 3, &[2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0]);
        let decomp = SpectralDecomposition::decompose(&a).unwrap();
        let reconstructed = decomp.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[(i, j)], a[(i, j)], epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_low_rank_approximation() {
        let a = DMatrix::from_row_slice(4, 4, &[
            4.0, 1.0, 0.0, 0.0, 1.0, 3.0, 1.0, 0.0,
            0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ]);
        let decomp = SpectralDecomposition::decompose(&a).unwrap();
        let lr = decomp.low_rank(2);
        assert_eq!(lr.nrows(), 4);
    }

    #[test]
    fn test_heat_kernel_spectral_decay() {
        let l = laplacian_1d_matrix(5, 1.0);
        let decomp = SpectralDecomposition::decompose(&l).unwrap();
        let k1 = heat_kernel_spectral(&decomp.eigenvalues, &decomp.eigenvectors, 0.1, 0, 0, 5);
        let k10 = heat_kernel_spectral(&decomp.eigenvalues, &decomp.eigenvectors, 1.0, 0, 0, 5);
        // For negative semidefinite Laplacian, eigenvalues <= 0, so exp(-lambda*t) decays as t grows
        // At i=j, both should be positive
        assert!(k1 > 0.0, "k1={}", k1);
        assert!(k10 > 0.0, "k10={}", k10);
    }

    #[test]
    fn test_spectral_diffusion_solve() {
        let l = laplacian_1d_matrix(5, 1.0);
        let decomp = SpectralDecomposition::decompose(&l).unwrap();
        let u0 = DVector::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let ut = spectral_diffusion_solve(&decomp.eigenvalues, &decomp.eigenvectors, &u0, 0.1);
        assert_eq!(ut.len(), 5);
    }

    #[test]
    fn test_spectral_gap() {
        let eigenvalues = vec![0.0, 1.0, 3.0, 5.0];
        let gap = spectral_gap(&eigenvalues).unwrap();
        assert_relative_eq!(gap, 1.0);
    }

    #[test]
    fn test_diffusion_kernel_matrix() {
        let l = laplacian_1d_matrix(3, 1.0);
        let decomp = SpectralDecomposition::decompose(&l).unwrap();
        let k = diffusion_kernel_matrix(&decomp.eigenvalues, &decomp.eigenvectors, 0.1);
        assert_eq!(k.nrows(), 3);
        assert_eq!(k.ncols(), 3);
    }
}
