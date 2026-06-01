//! Langevin dynamics, Hamiltonian Monte Carlo for agent sampling.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use rand::Rng;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangevinConfig {
    pub dimension: usize,
    pub dt: f64,
    pub friction: f64,
    pub temperature: f64,
    pub n_steps: usize,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self { dimension: 1, dt: 0.001, friction: 1.0, temperature: 1.0, n_steps: 1000 }
    }
}

pub trait PotentialEnergy: Send + Sync {
    fn value(&self, x: &DVector<f64>) -> f64;
    fn gradient(&self, x: &DVector<f64>) -> DVector<f64>;
}

#[derive(Debug, Clone)]
pub struct QuadraticPotential { pub a: nalgebra::DMatrix<f64> }

impl QuadraticPotential {
    pub fn new(a: nalgebra::DMatrix<f64>) -> Self { Self { a } }
    pub fn identity(dim: usize) -> Self { Self { a: nalgebra::DMatrix::identity(dim, dim) } }
}

impl PotentialEnergy for QuadraticPotential {
    fn value(&self, x: &DVector<f64>) -> f64 { 0.5 * (&self.a * x).dot(x) }
    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> { &self.a * x }
}

pub struct DoubleWellPotential;

impl PotentialEnergy for DoubleWellPotential {
    fn value(&self, x: &DVector<f64>) -> f64 {
        x.iter().map(|&xi| (xi * xi - 1.0).powi(2)).sum()
    }
    fn gradient(&self, x: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(x.iter().map(|&xi| 4.0 * xi * (xi * xi - 1.0)).collect())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LangevinResult {
    pub trajectory: Vec<DVector<f64>>,
    pub energies: Vec<f64>,
    pub config: LangevinConfig,
}

pub fn overdamped_langevin(config: &LangevinConfig, x0: &DVector<f64>, potential: &dyn PotentialEnergy) -> LangevinResult {
    let mut rng = rand::thread_rng();
    let mut x = x0.clone();
    let mut trajectory = Vec::with_capacity(config.n_steps);
    let mut energies = Vec::with_capacity(config.n_steps);
    let noise_scale = (2.0 * config.temperature * config.dt / config.friction).sqrt();

    for _ in 0..config.n_steps {
        let grad = potential.gradient(&x);
        let drift = -grad / config.friction * config.dt;
        let mut noise = DVector::zeros(config.dimension);
        for j in 0..config.dimension {
            let z: f64 = rng.gen_range(-1.0..1.0);
            noise[j] = noise_scale * z * std::f64::consts::SQRT_2;
        }
        x += drift + noise;
        trajectory.push(x.clone());
        energies.push(potential.value(&x));
    }
    LangevinResult { trajectory, energies, config: config.clone() }
}

pub fn underdamped_langevin(
    config: &LangevinConfig, x0: &DVector<f64>, v0: &DVector<f64>, potential: &dyn PotentialEnergy,
) -> (Vec<DVector<f64>>, Vec<DVector<f64>>) {
    let mut rng = rand::thread_rng();
    let mut x = x0.clone();
    let mut v = v0.clone();
    let mut positions = Vec::with_capacity(config.n_steps);
    let mut velocities = Vec::with_capacity(config.n_steps);
    let noise_scale = (2.0 * config.friction * config.temperature * config.dt).sqrt();

    for _ in 0..config.n_steps {
        let grad = potential.gradient(&x);
        let mut noise = DVector::zeros(config.dimension);
        for j in 0..config.dimension {
            let z: f64 = rng.gen_range(-1.0..1.0);
            noise[j] = noise_scale * z * std::f64::consts::SQRT_2;
        }
        v += (-&grad - &v.scale(config.friction)) * config.dt + noise;
        x += &v * config.dt;
        positions.push(x.clone());
        velocities.push(v.clone());
    }
    (positions, velocities)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMCConfig {
    pub dimension: usize,
    pub step_size: f64,
    pub n_leapfrog: usize,
    pub n_samples: usize,
    pub temperature: f64,
}

impl Default for HMCConfig {
    fn default() -> Self {
        Self { dimension: 1, step_size: 0.1, n_leapfrog: 10, n_samples: 1000, temperature: 1.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMCResult {
    pub samples: Vec<DVector<f64>>,
    pub acceptance_rate: f64,
    pub hamiltonians: Vec<f64>,
}

pub fn hmc_sample(config: &HMCConfig, x0: &DVector<f64>, potential: &dyn PotentialEnergy) -> HMCResult {
    let mut rng = rand::thread_rng();
    let mut x = x0.clone();
    let mut samples = Vec::with_capacity(config.n_samples);
    let mut accepted = 0usize;
    let mut hamiltonians = Vec::new();

    for _ in 0..config.n_samples {
        let mut p = DVector::zeros(config.dimension);
        for j in 0..config.dimension {
            let z: f64 = rng.gen_range(-1.0..1.0);
            p[j] = z * (config.temperature).sqrt() * std::f64::consts::SQRT_2;
        }
        let current_h = potential.value(&x) + 0.5 * p.dot(&p);
        let mut x_new = x.clone();
        let mut p_new = p.clone();
        p_new -= potential.gradient(&x_new).scale(config.step_size / 2.0);
        for _ in 1..config.n_leapfrog {
            x_new += &p_new * config.step_size;
            p_new -= potential.gradient(&x_new).scale(config.step_size);
        }
        x_new += &p_new * config.step_size;
        p_new -= potential.gradient(&x_new).scale(config.step_size / 2.0);
        let proposed_h = potential.value(&x_new) + 0.5 * p_new.dot(&p_new);
        let delta_h = proposed_h - current_h;
        let u: f64 = rng.gen_range(0.0..1.0);
        if delta_h < 0.0 || u < (-delta_h / config.temperature).exp() {
            x = x_new;
            accepted += 1;
        }
        samples.push(x.clone());
        hamiltonians.push(potential.value(&x));
    }
    HMCResult { samples, acceptance_rate: accepted as f64 / config.n_samples as f64, hamiltonians }
}

pub fn kinetic_temperature(velocities: &[DVector<f64>]) -> f64 {
    if velocities.is_empty() { return 0.0; }
    let dim = velocities[0].nrows();
    let ke: f64 = velocities.iter().map(|v| v.dot(v) / 2.0).sum::<f64>() / velocities.len() as f64;
    ke / (dim as f64 / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quadratic_potential() {
        let pot = QuadraticPotential::identity(2);
        let x = DVector::from_vec(vec![1.0, 2.0]);
        assert_relative_eq!(pot.value(&x), 2.5);
        let grad = pot.gradient(&x);
        assert_relative_eq!(grad[0], 1.0);
        assert_relative_eq!(grad[1], 2.0);
    }

    #[test]
    fn test_double_well() {
        let pot = DoubleWellPotential;
        let x_min = DVector::from_vec(vec![1.0]);
        assert_relative_eq!(pot.value(&x_min), 0.0);
        let x_saddle = DVector::from_vec(vec![0.0]);
        assert_relative_eq!(pot.value(&x_saddle), 1.0);
    }

    #[test]
    fn test_overdamped_langevin_runs() {
        let config = LangevinConfig { dimension: 2, dt: 0.001, friction: 1.0, temperature: 1.0, n_steps: 100 };
        let x0 = DVector::zeros(2);
        let pot = QuadraticPotential::identity(2);
        let result = overdamped_langevin(&config, &x0, &pot);
        assert_eq!(result.trajectory.len(), 100);
        assert_eq!(result.energies.len(), 100);
    }

    #[test]
    fn test_underdamped_langevin_runs() {
        let config = LangevinConfig { dimension: 1, dt: 0.001, friction: 1.0, temperature: 1.0, n_steps: 200 };
        let x0 = DVector::zeros(1);
        let v0 = DVector::zeros(1);
        let pot = QuadraticPotential::identity(1);
        let (positions, velocities) = underdamped_langevin(&config, &x0, &v0, &pot);
        assert_eq!(positions.len(), 200);
        assert_eq!(velocities.len(), 200);
    }

    #[test]
    fn test_hmc_sampling_quadratic() {
        let config = HMCConfig { dimension: 1, step_size: 0.1, n_leapfrog: 10, n_samples: 500, temperature: 1.0 };
        let x0 = DVector::zeros(1);
        let pot = QuadraticPotential::identity(1);
        let result = hmc_sample(&config, &x0, &pot);
        assert_eq!(result.samples.len(), 500);
        assert!(result.acceptance_rate > 0.3, "Acceptance rate too low: {}", result.acceptance_rate);
    }

    #[test]
    fn test_hmc_double_well() {
        let config = HMCConfig { dimension: 1, step_size: 0.05, n_leapfrog: 20, n_samples: 500, temperature: 1.0 };
        let x0 = DVector::from_vec(vec![1.0]);
        let result = hmc_sample(&config, &x0, &DoubleWellPotential);
        assert!(result.acceptance_rate > 0.1);
    }

    #[test]
    fn test_kinetic_temperature() {
        let v = DVector::from_vec(vec![1.0]);
        let vels = vec![v];
        let t = kinetic_temperature(&vels);
        assert_relative_eq!(t, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kinetic_temperature_empty() {
        let t = kinetic_temperature(&[]);
        assert_relative_eq!(t, 0.0);
    }

    #[test]
    fn test_langevin_config_default() {
        let cfg = LangevinConfig::default();
        assert_eq!(cfg.dimension, 1);
        assert_eq!(cfg.friction, 1.0);
    }

    #[test]
    fn test_hmc_config_default() {
        let cfg = HMCConfig::default();
        assert_eq!(cfg.n_leapfrog, 10);
    }
}
