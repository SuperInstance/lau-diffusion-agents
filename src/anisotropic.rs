//! Direction-dependent diffusion, diffusion tensors on manifolds.

use nalgebra::{Matrix3, Vector3};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionTensor {
    pub tensor: Matrix3<f64>,
}

impl DiffusionTensor {
    pub fn isotropic(scalar: f64) -> Self {
        Self { tensor: Matrix3::identity().scale(scalar) }
    }

    pub fn from_eigen(eigenvalues: &Vector3<f64>, eigenvectors: &Matrix3<f64>) -> Self {
        let d = Matrix3::from_diagonal(eigenvalues);
        Self { tensor: eigenvectors * d * eigenvectors.transpose() }
    }

    pub fn axis_aligned(dx: f64, dy: f64, dz: f64) -> Self {
        Self { tensor: Matrix3::from_diagonal(&Vector3::new(dx, dy, dz)) }
    }

    pub fn fractional_anisotropy(&self) -> f64 {
        let eig = self.tensor.symmetric_eigen();
        let vals = eig.eigenvalues;
        let mean = (vals[0] + vals[1] + vals[2]) / 3.0;
        let numerator = (vals[0] - mean).powi(2) + (vals[1] - mean).powi(2) + (vals[2] - mean).powi(2);
        let denominator = vals[0].powi(2) + vals[1].powi(2) + vals[2].powi(2);
        if denominator < 1e-15 { return 0.0; }
        (3.0 / 2.0 * numerator / denominator).sqrt()
    }

    pub fn mean_diffusivity(&self) -> f64 {
        (self.tensor[(0, 0)] + self.tensor[(1, 1)] + self.tensor[(2, 2)]) / 3.0
    }

    pub fn radial_diffusivity(&self) -> f64 {
        let eig = self.tensor.symmetric_eigen();
        let vals = eig.eigenvalues;
        let max_idx = if vals[0] >= vals[1] && vals[0] >= vals[2] { 0 }
                      else if vals[1] >= vals[2] { 1 } else { 2 };
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..3 { if i != max_idx { sum += vals[i]; count += 1; } }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }

    pub fn apply(&self, gradient: &Vector3<f64>) -> Vector3<f64> {
        &self.tensor * gradient
    }

    pub fn trace(&self) -> f64 {
        self.tensor[(0, 0)] + self.tensor[(1, 1)] + self.tensor[(2, 2)]
    }
}

pub struct AnisotropicDiffusion2D {
    pub n_x: usize,
    pub n_y: usize,
    pub dx: f64,
    pub dy: f64,
}

impl AnisotropicDiffusion2D {
    pub fn new(n_x: usize, n_y: usize, dx: f64, dy: f64) -> Self { Self { n_x, n_y, dx, dy } }

    pub fn step_perona_malik(&self, u: &[f64], dt: f64, kappa: f64) -> Vec<f64> {
        let nx = self.n_x; let ny = self.n_y;
        let mut u_new = u.to_vec();
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let idx = i * ny + j;
                let grad_x = (u[idx + ny] - u[idx - ny]) / (2.0 * self.dx);
                let grad_y = (u[idx + 1] - u[idx - 1]) / (2.0 * self.dy);
                let grad_sq = grad_x * grad_x + grad_y * grad_y;
                let g = 1.0 / (1.0 + grad_sq / (kappa * kappa));
                let laplacian_x = (u[idx + ny] - 2.0 * u[idx] + u[idx - ny]) / (self.dx * self.dx);
                let laplacian_y = (u[idx + 1] - 2.0 * u[idx] + u[idx - 1]) / (self.dy * self.dy);
                u_new[idx] = u[idx] + dt * g * (laplacian_x + laplacian_y);
            }
        }
        u_new
    }

    pub fn evolve_perona_malik(&self, u0: &[f64], dt: f64, kappa: f64, n_steps: usize) -> Vec<f64> {
        let mut u = u0.to_vec();
        for _ in 0..n_steps { u = self.step_perona_malik(&u, dt, kappa); }
        u
    }

    pub fn step_tensor(&self, u: &[f64], dxx: &[f64], dxy: &[f64], dyy: &[f64], dt: f64) -> Vec<f64> {
        let nx = self.n_x; let ny = self.n_y;
        let mut u_new = u.to_vec();
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let idx = i * ny + j;
                let ux = (u[idx + ny] - u[idx - ny]) / (2.0 * self.dx);
                let uy = (u[idx + 1] - u[idx - 1]) / (2.0 * self.dy);
                let div_x = (dxx[idx + ny] * ux + dxy[idx + ny] * uy - dxx[idx - ny] * ux - dxy[idx - ny] * uy) / (2.0 * self.dx);
                let div_y = (dxy[idx + 1] * ux + dyy[idx + 1] * uy - dxy[idx - 1] * ux - dyy[idx - 1] * uy) / (2.0 * self.dy);
                u_new[idx] = u[idx] + dt * (div_x + div_y);
            }
        }
        u_new
    }
}

pub fn structure_tensor(u: &[f64], nx: usize, ny: usize, dx: f64, dy: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dxx = vec![0.0; nx * ny];
    let mut dxy = vec![0.0; nx * ny];
    let mut dyy = vec![0.0; nx * ny];
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            let idx = i * ny + j;
            let ux = (u[idx + ny] - u[idx - ny]) / (2.0 * dx);
            let uy = (u[idx + 1] - u[idx - 1]) / (2.0 * dy);
            dxx[idx] = ux * ux;
            dxy[idx] = ux * uy;
            dyy[idx] = uy * uy;
        }
    }
    (dxx, dxy, dyy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_isotropic_tensor() {
        let t = DiffusionTensor::isotropic(2.0);
        assert_relative_eq!(t.tensor[(0, 0)], 2.0);
        assert_relative_eq!(t.trace(), 6.0);
    }

    #[test]
    fn test_axis_aligned_tensor() {
        let t = DiffusionTensor::axis_aligned(1.0, 2.0, 3.0);
        assert_relative_eq!(t.mean_diffusivity(), 2.0);
    }

    #[test]
    fn test_fractional_anisotropy_isotropic() {
        let t = DiffusionTensor::isotropic(1.0);
        assert_relative_eq!(t.fractional_anisotropy(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fractional_anisotropy_anisotropic() {
        let t = DiffusionTensor::axis_aligned(3.0, 1.0, 1.0);
        let fa = t.fractional_anisotropy();
        assert!(fa > 0.0 && fa <= 1.0, "FA={}", fa);
    }

    #[test]
    fn test_diffusion_tensor_apply() {
        let t = DiffusionTensor::isotropic(2.0);
        let g = Vector3::new(1.0, 0.0, 0.0);
        let result = t.apply(&g);
        assert_relative_eq!(result[0], 2.0);
    }

    #[test]
    fn test_radial_diffusivity() {
        let t = DiffusionTensor::axis_aligned(3.0, 1.0, 1.0);
        assert_relative_eq!(t.radial_diffusivity(), 1.0);
    }

    #[test]
    fn test_perona_malik() {
        let diff = AnisotropicDiffusion2D::new(10, 10, 0.1, 0.1);
        let u0: Vec<f64> = (0..100).map(|i| {
            let x = (i / 10) as f64 / 10.0;
            (-((x - 0.5) * 10.0).powi(2)).exp()
        }).collect();
        let result = diff.evolve_perona_malik(&u0, 0.001, 0.1, 10);
        assert!(result.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_structure_tensor() {
        let u: Vec<f64> = (0..100).map(|i| { let x = (i / 10) as f64; let y = (i % 10) as f64; x + y }).collect();
        let (dxx, _, _) = structure_tensor(&u, 10, 10, 1.0, 1.0);
        assert_eq!(dxx.len(), 100);
    }

    #[test]
    fn test_tensor_step() {
        let diff = AnisotropicDiffusion2D::new(10, 10, 0.1, 0.1);
        let u0 = vec![0.5; 100];
        let dxx = vec![1.0; 100]; let dxy = vec![0.0; 100]; let dyy = vec![1.0; 100];
        let result = diff.step_tensor(&u0, &dxx, &dxy, &dyy, 0.001);
        for &v in &result { assert_relative_eq!(v, 0.5, epsilon = 0.01); }
    }

    #[test]
    fn test_mean_diffusivity() {
        let t = DiffusionTensor::axis_aligned(1.0, 2.0, 3.0);
        assert_relative_eq!(t.mean_diffusivity(), 2.0);
    }
}
