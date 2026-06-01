//! Heat kernel on Riemannian manifolds, Gaussian diffusion, fundamental solutions.

use nalgebra::{DVector, DMatrix};

pub fn heat_kernel_euclidean(dim: usize, t: f64, x: &DVector<f64>, y: &DVector<f64>) -> f64 {
    assert!(t > 0.0, "Time must be positive");
    let diff = x - y;
    let dist_sq = diff.iter().map(|v| v * v).sum::<f64>();
    let norm = (4.0 * std::f64::consts::PI * t).powf(-(dim as f64) / 2.0);
    norm * (-dist_sq / (4.0 * t)).exp()
}

pub fn heat_kernel_circle(t: f64, x: f64, y: f64, r: f64, n_terms: usize) -> f64 {
    assert!(t > 0.0);
    let mut sum = 0.0;
    let circumference = 2.0 * std::f64::consts::PI * r;
    for k in -(n_terms as i32)..=(n_terms as i32) {
        let shift = k as f64 * circumference;
        let d = x - y + shift;
        sum += (-d * d / (4.0 * t)).exp();
    }
    sum / (4.0 * std::f64::consts::PI * t).sqrt()
}

pub fn heat_kernel_sphere(t: f64, theta1: f64, phi1: f64, theta2: f64, phi2: f64, r: f64, max_l: usize) -> f64 {
    assert!(t > 0.0);
    let cos_gamma = theta1.cos() * theta2.cos() + theta1.sin() * theta2.sin() * (phi1 - phi2).cos();
    let cos_gamma = cos_gamma.clamp(-1.0, 1.0);
    let mut sum = 0.0;
    for l in 0..=max_l {
        let cl = (2.0 * l as f64 + 1.0) / (4.0 * std::f64::consts::PI);
        let pl = legendre_polynomial(l, cos_gamma);
        sum += cl * pl * (-(l as f64) * (l as f64 + 1.0) * t / (r * r)).exp();
    }
    sum / (r * r)
}

pub fn legendre_polynomial(l: usize, x: f64) -> f64 {
    match l {
        0 => 1.0,
        1 => x,
        _ => {
            let mut p_prev = 1.0;
            let mut p_curr = x;
            for n in 1..l {
                let p_next = ((2 * n + 1) as f64 * x * p_curr - n as f64 * p_prev) / (n + 1) as f64;
                p_prev = p_curr;
                p_curr = p_next;
            }
            p_curr
        }
    }
}

pub fn gaussian_kernel(x: &DVector<f64>, mu: &DVector<f64>, sigma_inv: &DMatrix<f64>, det_sigma: f64) -> f64 {
    let diff = x - mu;
    let dim = x.nrows() as f64;
    let exponent: f64 = -0.5 * (sigma_inv * &diff).dot(&diff);
    let norm = 1.0 / ((2.0 * std::f64::consts::PI).powf(dim / 2.0) * det_sigma.sqrt());
    norm * exponent.exp()
}

pub fn heat_equation_1d(u0: &[f64], alpha: f64, dx: f64, dt: f64, n_steps: usize) -> Vec<f64> {
    let n = u0.len();
    let r = alpha * dt / (dx * dx);
    assert!(r < 0.5, "CFL condition violated: r={} >= 0.5", r);
    let mut u = u0.to_vec();
    let mut u_new = u.clone();

    for _ in 0..n_steps {
        for i in 1..n - 1 {
            u_new[i] = u[i] + r * (u[i + 1] - 2.0 * u[i] + u[i - 1]);
        }
        u_new[0] = u_new[1];
        u_new[n - 1] = u_new[n - 2];
        std::mem::swap(&mut u, &mut u_new);
    }
    u
}

pub fn graph_heat_kernel(laplacian: &DMatrix<f64>, t: f64, n_terms: usize) -> DMatrix<f64> {
    let n = laplacian.nrows();
    let mut result = DMatrix::identity(n, n);
    let mut power = laplacian.clone();
    let mut factorial = 1.0;

    for k in 1..=n_terms {
        factorial *= k as f64;
        let coeff = (-t).powi(k as i32) / factorial;
        result += &power.scale(coeff);
        power = &power * laplacian;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_heat_kernel_euclidean_point() {
        let x = DVector::from_vec(vec![0.0]);
        let y = DVector::from_vec(vec![0.0]);
        let k = heat_kernel_euclidean(1, 1.0, &x, &y);
        assert_relative_eq!(k, 1.0 / (4.0 * std::f64::consts::PI).sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_heat_kernel_euclidean_2d() {
        let x = DVector::from_vec(vec![0.0, 0.0]);
        let y = DVector::from_vec(vec![0.0, 0.0]);
        let k = heat_kernel_euclidean(2, 1.0, &x, &y);
        assert_relative_eq!(k, 1.0 / (4.0 * std::f64::consts::PI), epsilon = 1e-10);
    }

    #[test]
    fn test_heat_kernel_symmetry() {
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let y = DVector::from_vec(vec![3.0, 4.0]);
        let k1 = heat_kernel_euclidean(2, 1.0, &x, &y);
        let k2 = heat_kernel_euclidean(2, 1.0, &y, &x);
        assert_relative_eq!(k1, k2, epsilon = 1e-12);
    }

    #[test]
    fn test_heat_kernel_decays_with_distance() {
        let origin = DVector::from_vec(vec![0.0]);
        let near = DVector::from_vec(vec![1.0]);
        let far = DVector::from_vec(vec![5.0]);
        let k_near = heat_kernel_euclidean(1, 1.0, &origin, &near);
        let k_far = heat_kernel_euclidean(1, 1.0, &origin, &far);
        assert!(k_near > k_far);
    }

    #[test]
    fn test_legendre_polynomials() {
        assert_relative_eq!(legendre_polynomial(0, 0.5), 1.0);
        assert_relative_eq!(legendre_polynomial(1, 0.5), 0.5);
        assert_relative_eq!(legendre_polynomial(2, 0.5), -0.125, epsilon = 1e-10);
    }

    #[test]
    fn test_heat_equation_1d_decay() {
        let n = 50;
        let u0: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / (n - 1) as f64;
            (-((x - 0.5) * 10.0).powi(2)).exp()
        }).collect();
        let result = heat_equation_1d(&u0, 0.1, 0.02, 0.0001, 100);
        let max_val = result.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let initial_max = u0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val < initial_max);
    }

    #[test]
    fn test_heat_equation_1d_preserves_mass() {
        let n = 20;
        let u0: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / (n - 1) as f64;
            (-((x - 0.5) * 8.0).powi(2)).exp()
        }).collect();
        let initial_mass: f64 = u0.iter().sum();
        let result = heat_equation_1d(&u0, 0.01, 0.1, 0.001, 10);
        let final_mass: f64 = result.iter().sum();
        assert_relative_eq!(initial_mass, final_mass, epsilon = 0.5);
    }

    #[test]
    fn test_gaussian_kernel_normalized() {
        let mu = DVector::from_vec(vec![0.0, 0.0]);
        let sigma = DMatrix::identity(2, 2);
        let sigma_inv = sigma.clone();
        let mut sum = 0.0;
        let step = 0.5;
        for x in (-40..=40).map(|i| i as f64 * step) {
            for y in (-40..=40).map(|i| i as f64 * step) {
                let p = DVector::from_vec(vec![x, y]);
                sum += gaussian_kernel(&p, &mu, &sigma_inv, 1.0) * step * step;
            }
        }
        assert_relative_eq!(sum, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_heat_kernel_circle_positive() {
        let k = heat_kernel_circle(0.1, 0.0, 1.0, 1.0, 5);
        assert!(k > 0.0);
    }

    #[test]
    fn test_heat_kernel_sphere_positive() {
        let k = heat_kernel_sphere(0.1, 0.0, 0.0, 0.5, 0.5, 1.0, 5);
        assert!(k > 0.0);
    }
}
