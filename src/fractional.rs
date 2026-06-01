//! Fractional diffusion (anomalous), Levy flights, alpha-stable distributions.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableParams {
    pub alpha: f64,
    pub beta: f64,
    pub scale: f64,
    pub location: f64,
}

impl Default for StableParams {
    fn default() -> Self { Self { alpha: 2.0, beta: 0.0, scale: 1.0, location: 0.0 } }
}

pub fn stable_random(params: &StableParams) -> f64 {
    let mut rng = rand::thread_rng();
    if (params.alpha - 2.0).abs() < 1e-10 {
        let u: f64 = rng.gen_range(-1.0..1.0);
        return params.location + params.scale * u * std::f64::consts::SQRT_2;
    }
    let u: f64 = rng.gen_range(-std::f64::consts::FRAC_PI_2..std::f64::consts::FRAC_PI_2);
    let w: f64 = -rng.gen_range(0.01..1.0_f64).ln();
    let alpha = params.alpha;
    let zeta = if params.beta.abs() < 1e-10 { 0.0 }
        else { params.beta * (std::f64::consts::FRAC_PI_2 * alpha / 2.0).tan() };
    let term1 = (1.0 + zeta * zeta).powf(1.0 / (2.0 * alpha));
    let term2 = (u + zeta).sin();
    let cos_u = (u + zeta).cos().max(1e-15);
    let term3 = (cos_u).powf((1.0 - alpha) / alpha);
    let x = term1 * term2 * term3 * w.powf((1.0 - alpha) / alpha);
    params.location + params.scale * x
}

pub fn stable_samples(params: &StableParams, n: usize) -> Vec<f64> {
    (0..n).map(|_| stable_random(params)).collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevyFlight {
    pub alpha: f64,
    pub positions: Vec<DVector<f64>>,
    pub step_sizes: Vec<f64>,
}

impl LevyFlight {
    pub fn simulate(dimension: usize, alpha: f64, n_steps: usize, scale: f64) -> Self {
        let params = StableParams { alpha, beta: 0.0, scale, location: 0.0 };
        let mut rng = rand::thread_rng();
        let mut positions = Vec::with_capacity(n_steps + 1);
        let mut step_sizes = Vec::with_capacity(n_steps);
        let mut pos = DVector::zeros(dimension);
        positions.push(pos.clone());

        for _ in 0..n_steps {
            let step_length = stable_random(&params).abs();
            step_sizes.push(step_length);
            let mut direction = DVector::zeros(dimension);
            for j in 0..dimension {
                let u: f64 = rng.gen_range(-1.0..1.0);
                direction[j] = u;
            }
            let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                direction = direction.scale(step_length / norm);
            }
            pos += &direction;
            positions.push(pos.clone());
        }
        LevyFlight { alpha, positions, step_sizes }
    }

    pub fn mean_square_displacement(&self) -> Vec<f64> {
        let origin = &self.positions[0];
        self.positions.iter().map(|p| { let d = p - origin; d.iter().map(|x| x * x).sum::<f64>() }).collect()
    }
}

pub struct FractionalDiffusion1D { pub n_points: usize, pub alpha: f64 }

impl FractionalDiffusion1D {
    pub fn new(n_points: usize, alpha: f64) -> Self {
        assert!((0.0..=2.0).contains(&alpha), "Alpha must be in (0, 2]");
        Self { n_points, alpha }
    }

    pub fn gl_coefficients(&self, n: usize) -> Vec<f64> {
        let mut coeffs = vec![0.0; n + 1];
        coeffs[0] = 1.0;
        for k in 1..=n {
            coeffs[k] = coeffs[k - 1] * (1.0 - (self.alpha + 1.0) / k as f64);
        }
        coeffs
    }

    pub fn solve(&self, u0: &[f64], dx: f64, dt: f64, n_steps: usize) -> Vec<f64> {
        let n = self.n_points;
        let mut u = u0.to_vec();
        let coeffs = self.gl_coefficients(n);
        let r = dt / dx.powf(self.alpha);

        for _ in 0..n_steps {
            let mut u_new = u.clone();
            for i in 1..n - 1 {
                let mut sum = 0.0;
                for k in 0..=i { sum += coeffs[k] * u[i - k]; }
                u_new[i] = u[i] - r * sum;
                u_new[i] = u_new[i].max(0.0);
            }
            u_new[0] = 0.0;
            u_new[n - 1] = 0.0;
            u = u_new;
        }
        u
    }
}

pub fn stable_characteristic_function(t: f64, params: &StableParams) -> num_complex::Complex64 {
    use num_complex::Complex64;
    let exp_part = (-params.scale * t.abs()).exp();
    Complex64::new(exp_part, 0.0) * Complex64::new(0.0, params.location * t).exp()
}

pub fn anomalous_exponent(msd: &[f64], dt: f64) -> f64 {
    if msd.len() < 3 { return 1.0; }
    let n = msd.len();
    let log_t: Vec<f64> = (0..n).map(|i| ((i as f64 * dt).max(1e-10)).ln()).collect();
    let log_msd: Vec<f64> = msd.iter().map(|&m| m.max(1e-10).ln()).collect();
    let n_f = n as f64;
    let sum_x: f64 = log_t.iter().sum();
    let sum_y: f64 = log_msd.iter().sum();
    let sum_xy: f64 = log_t.iter().zip(log_msd.iter()).map(|(x, y)| x * y).sum();
    let sum_xx: f64 = log_t.iter().map(|x| x * x).sum();
    (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stable_params_default() {
        let p = StableParams::default();
        assert_eq!(p.alpha, 2.0);
        assert_eq!(p.beta, 0.0);
    }

    #[test]
    fn test_stable_gaussian() {
        let params = StableParams { alpha: 2.0, ..Default::default() };
        let samples = stable_samples(&params, 1000);
        assert_eq!(samples.len(), 1000);
        let mean = samples.iter().sum::<f64>() / 1000.0;
        assert!(mean.abs() < 1.0, "Mean too far from 0: {}", mean);
    }

    #[test]
    fn test_stable_heavy_tail() {
        let params = StableParams { alpha: 1.5, beta: 0.0, scale: 1.0, location: 0.0 };
        let samples = stable_samples(&params, 1000);
        let max_abs = samples.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(max_abs > 1.0, "Expected heavy tails, max_abs={}", max_abs);
    }

    #[test]
    fn test_levy_flight() {
        let flight = LevyFlight::simulate(2, 1.5, 100, 1.0);
        assert_eq!(flight.positions.len(), 101);
        assert_eq!(flight.step_sizes.len(), 100);
    }

    #[test]
    fn test_levy_flight_msd() {
        let flight = LevyFlight::simulate(1, 1.5, 200, 1.0);
        let msd = flight.mean_square_displacement();
        assert_eq!(msd.len(), 201);
        assert!(msd[100] > msd[10]);
    }

    #[test]
    fn test_gl_coefficients() {
        let fd = FractionalDiffusion1D::new(50, 1.5);
        let coeffs = fd.gl_coefficients(5);
        assert_relative_eq!(coeffs[0], 1.0);
        assert!(coeffs[1] < 0.0);
    }

    #[test]
    fn test_fractional_diffusion_solve() {
        let fd = FractionalDiffusion1D::new(50, 1.5);
        let u0: Vec<f64> = (0..50).map(|i| {
            let x = i as f64 / 49.0;
            (-((x - 0.5) * 10.0).powi(2)).exp()
        }).collect();
        let result = fd.solve(&u0, 0.02, 0.0001, 10);
        assert_eq!(result.len(), 50);
        assert!(result.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_characteristic_function_magnitude() {
        let params = StableParams { alpha: 2.0, ..Default::default() };
        let phi = stable_characteristic_function(0.0, &params);
        assert_relative_eq!(phi.norm(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_anomalous_exponent_range() {
        let msd: Vec<f64> = (1..=100).map(|i| (i as f64).powf(1.5)).collect();
        let gamma = anomalous_exponent(&msd, 1.0);
        assert!(gamma > 0.0, "gamma={}", gamma);
    }

    #[test]
    fn test_fractional_diffusion_alpha_bounds() {
        let fd = FractionalDiffusion1D::new(10, 1.0);
        assert_eq!(fd.alpha, 1.0);
    }
}
