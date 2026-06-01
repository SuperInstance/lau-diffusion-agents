//! Fokker-Planck equation, drift-diffusion decomposition, stationary distributions.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftDiffusionDecomposition {
    pub estimated_drift: f64,
    pub estimated_diffusion: f64,
}

impl DriftDiffusionDecomposition {
    pub fn estimate(observations: &[f64], dt: f64) -> Self {
        let n = observations.len();
        if n < 2 {
            return Self { estimated_drift: 0.0, estimated_diffusion: 0.0 };
        }
        let mut sum_drift = 0.0;
        let mut sum_diff = 0.0;
        for i in 1..n {
            let dx = observations[i] - observations[i - 1];
            sum_drift += dx / dt;
            sum_diff += dx * dx / dt;
        }
        let estimated_drift = sum_drift / (n - 1) as f64;
        let estimated_diffusion = (sum_diff / (n - 1) as f64 - estimated_drift * estimated_drift * dt) / 2.0;
        DriftDiffusionDecomposition {
            estimated_drift,
            estimated_diffusion: estimated_diffusion.abs(),
        }
    }
}

pub struct FokkerPlanck1D {
    pub n_points: usize,
    pub x_min: f64,
    pub x_max: f64,
    pub dx: f64,
}

impl FokkerPlanck1D {
    pub fn new(n_points: usize, x_min: f64, x_max: f64) -> Self {
        let dx = (x_max - x_min) / (n_points - 1) as f64;
        Self { n_points, x_min, x_max, dx }
    }

    pub fn grid(&self) -> Vec<f64> {
        (0..self.n_points).map(|i| self.x_min + i as f64 * self.dx).collect()
    }

    pub fn evolve<F, G>(&self, p0: &[f64], drift_fn: F, diffusion_fn: G, dt: f64, n_steps: usize) -> Vec<f64>
    where F: Fn(f64) -> f64, G: Fn(f64) -> f64,
    {
        let n = self.n_points;
        let dx = self.dx;
        let r = dt / (dx * dx);
        assert!(r < 0.5, "Stability condition violated");
        let grid = self.grid();
        let mut p = p0.to_vec();

        for _ in 0..n_steps {
            let mut p_new = p.clone();
            for i in 1..n - 1 {
                let x_m = grid[i] - 0.5 * dx;
                let x_p = grid[i] + 0.5 * dx;
                let d_m = diffusion_fn(x_m);
                let d_p = diffusion_fn(x_p);
                let diff_flux = (d_p * (p[i + 1] - p[i]) - d_m * (p[i] - p[i - 1])) / (dx * dx);
                let a_m = drift_fn(x_m);
                let a_p = drift_fn(x_p);
                let drift_flux = -(a_p * (p[i + 1] + p[i]) * 0.5 - a_m * (p[i] + p[i - 1]) * 0.5) / dx;
                p_new[i] = p[i] + dt * (diff_flux + drift_flux);
                p_new[i] = p_new[i].max(0.0);
            }
            p_new[0] = p_new[1];
            p_new[n - 1] = p_new[n - 2];
            p = p_new;
        }
        p
    }

    pub fn stationary_ou(&self, theta: f64, mu: f64, sigma: f64) -> Vec<f64> {
        let var = sigma * sigma / (2.0 * theta);
        let grid = self.grid();
        let norm = 1.0 / (2.0 * std::f64::consts::PI * var).sqrt();
        grid.iter().map(|&x| norm * (-(x - mu).powi(2) / (2.0 * var)).exp()).collect()
    }
}

pub fn entropy_production_rate(drift: &[f64], diffusion: f64, density: &[f64], dx: f64) -> f64 {
    let n = density.len();
    let mut rate = 0.0;
    for i in 1..n - 1 {
        let grad_p = (density[i + 1] - density[i - 1]) / (2.0 * dx);
        let j = drift[i] * density[i] - diffusion * grad_p;
        if density[i] > 1e-15 {
            rate += j * j / (diffusion * density[i]) * dx;
        }
    }
    rate
}

pub fn probability_current(drift: &[f64], diffusion: &[f64], density: &[f64], dx: f64) -> Vec<f64> {
    let n = density.len();
    let mut current = vec![0.0; n];
    for i in 1..n - 1 {
        let grad_p = (density[i + 1] - density[i - 1]) / (2.0 * dx);
        current[i] = drift[i] * density[i] - diffusion[i] * grad_p;
    }
    current[0] = current[1];
    current[n - 1] = current[n - 2];
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fp1d_grid() {
        let fp = FokkerPlanck1D::new(101, -5.0, 5.0);
        assert_eq!(fp.grid().len(), 101);
        assert_relative_eq!(fp.grid()[0], -5.0);
        assert_relative_eq!(fp.grid()[100], 5.0);
    }

    #[test]
    fn test_fp1d_ou_evolution() {
        let fp = FokkerPlanck1D::new(101, -5.0, 5.0);
        let grid = fp.grid();
        let p0: Vec<f64> = grid.iter().map(|&x| (-(x - 2.0).powi(2) / 0.1).exp()).collect();
        let result = fp.evolve(&p0, |x| -(x), |_| 1.0, 0.001, 500);
        assert!(result.iter().all(|&v| v >= 0.0));
        let mass: f64 = result.iter().sum::<f64>() * fp.dx;
        assert!(mass > 0.5, "Mass = {}", mass);
    }

    #[test]
    fn test_fp1d_stationary_ou() {
        let fp = FokkerPlanck1D::new(201, -10.0, 10.0);
        let p_stat = fp.stationary_ou(1.0, 0.0, 1.0);
        let mass: f64 = p_stat.iter().sum::<f64>() * fp.dx;
        assert_relative_eq!(mass, 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_drift_diffusion_estimation() {
        let mut obs = vec![0.0];
        let theta = 1.0; let mu = 2.0; let sigma = 0.5; let dt: f64 = 0.001;
        let mut rng = rand::thread_rng();
        for _ in 0..10000 {
            let x = *obs.last().unwrap();
            let z: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
            let dx = theta * (mu - x) * dt + sigma * z * dt.sqrt() * std::f64::consts::SQRT_2;
            obs.push(x + dx);
        }
        let decomp = DriftDiffusionDecomposition::estimate(&obs, dt);
        assert!(decomp.estimated_diffusion > 0.0);
    }

    #[test]
    fn test_drift_diffusion_short() {
        let decomp = DriftDiffusionDecomposition::estimate(&[1.0], 0.1);
        assert_eq!(decomp.estimated_drift, 0.0);
        assert_eq!(decomp.estimated_diffusion, 0.0);
    }

    #[test]
    fn test_probability_current_zero_drift() {
        let n = 11;
        let density: Vec<f64> = (0..n).map(|i| {
            let x = i as f64 / 10.0;
            (-((x - 0.5) * 5.0).powi(2)).exp()
        }).collect();
        let drift = vec![0.0; n];
        let diffusion = vec![1.0; n];
        let current = probability_current(&drift, &diffusion, &density, 0.1);
        assert!(current.iter().any(|&c| c.abs() > 1e-10));
    }

    #[test]
    fn test_entropy_production_nonneg() {
        let n = 51;
        let drift: Vec<f64> = (0..n).map(|i| -0.1 * (i as f64 - 25.0)).collect();
        let density: Vec<f64> = (0..n).map(|i| (-((i as f64 - 25.0) / 5.0).powi(2)).exp()).collect();
        let rate = entropy_production_rate(&drift, 1.0, &density, 0.1);
        assert!(rate >= 0.0);
    }

    #[test]
    fn test_evolve_preserves_positivity() {
        let fp = FokkerPlanck1D::new(51, -5.0, 5.0);
        let grid = fp.grid();
        let p0: Vec<f64> = grid.iter().map(|&x| (-x.powi(2)).exp()).collect();
        let result = fp.evolve(&p0, |_| 0.0, |_| 1.0, 0.0005, 100);
        assert!(result.iter().all(|&v| v >= 0.0));
    }
}
