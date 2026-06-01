//! Unified API — AgentDiffusion struct combining all modules.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};
use rand::Rng;

use crate::brownian::BrownianConfig;
use crate::fokker_planck::DriftDiffusionDecomposition;
use crate::langevin::{HMCConfig, HMCResult, PotentialEnergy};
use crate::fractional::LevyFlight;
use crate::spectral_diffusion::{SpectralDecomposition, graph_laplacian};
use crate::transport_diffusion::DiscreteDistribution;
use crate::anisotropic::DiffusionTensor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDiffusionConfig {
    pub n_agents: usize,
    pub dimension: usize,
    pub dt: f64,
    pub total_time: f64,
    pub diffusion_coeff: f64,
    pub drift_coeff: f64,
}

impl Default for AgentDiffusionConfig {
    fn default() -> Self {
        Self { n_agents: 100, dimension: 2, dt: 0.01, total_time: 1.0, diffusion_coeff: 1.0, drift_coeff: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub positions: Vec<Vec<f64>>,
    pub time: f64,
}

pub struct AgentDiffusion {
    pub config: AgentDiffusionConfig,
    pub state: AgentState,
}

impl AgentDiffusion {
    pub fn new(config: AgentDiffusionConfig) -> Self {
        let positions = vec![vec![0.0; config.dimension]; config.n_agents];
        Self { config, state: AgentState { positions, time: 0.0 } }
    }

    pub fn new_random(config: AgentDiffusionConfig, spread: f64) -> Self {
        let mut rng = rand::thread_rng();
        let positions = (0..config.n_agents)
            .map(|_| (0..config.dimension).map(|_| rng.gen_range(-spread..spread)).collect())
            .collect();
        Self { config, state: AgentState { positions, time: 0.0 } }
    }

    pub fn brownian_step(&mut self) {
        let mut rng = rand::thread_rng();
        let sqrt_dt = self.config.dt.sqrt();
        for pos in &mut self.state.positions {
            for j in 0..self.config.dimension {
                let z: f64 = rng.gen_range(-1.0..1.0);
                pos[j] += self.config.drift_coeff * self.config.dt
                    + self.config.diffusion_coeff * sqrt_dt * z * std::f64::consts::SQRT_2;
            }
        }
        self.state.time += self.config.dt;
    }

    pub fn run_brownian(&mut self) {
        let n_steps = (self.config.total_time / self.config.dt) as usize;
        for _ in 0..n_steps { self.brownian_step(); }
    }

    pub fn heat_kernel(&self, i: usize, j: usize, t: f64) -> f64 {
        let dim = self.config.dimension;
        let dist_sq: f64 = self.state.positions[i].iter()
            .zip(self.state.positions[j].iter())
            .map(|(a, b)| (a - b).powi(2)).sum();
        let norm = (4.0 * std::f64::consts::PI * t).powf(-(dim as f64) / 2.0);
        norm * (-dist_sq / (4.0 * t)).exp()
    }

    pub fn estimate_drift_diffusion(&self, trajectories: &[Vec<Vec<f64>>]) -> Vec<DriftDiffusionDecomposition> {
        trajectories.iter().map(|traj| {
            let obs: Vec<f64> = traj.iter().map(|p| p[0]).collect();
            DriftDiffusionDecomposition::estimate(&obs, self.config.dt)
        }).collect()
    }

    pub fn hmc_sample(&self, hmc_config: &HMCConfig, potential: &dyn PotentialEnergy) -> HMCResult {
        let x0 = DVector::zeros(self.config.dimension);
        crate::langevin::hmc_sample(hmc_config, &x0, potential)
    }

    pub fn levy_flight(&self, _agent_idx: usize, alpha: f64, n_steps: usize) -> LevyFlight {
        LevyFlight::simulate(self.config.dimension, alpha, n_steps, self.config.diffusion_coeff)
    }

    pub fn spectral_decomposition(&self, adjacency: &DMatrix<f64>) -> Result<SpectralDecomposition, String> {
        let lap = graph_laplacian(adjacency);
        SpectralDecomposition::decompose(&lap)
    }

    pub fn position_distribution(&self, dim: usize, n_bins: usize, range: (f64, f64)) -> DiscreteDistribution {
        let bin_width = (range.1 - range.0) / n_bins as f64;
        let mut counts = vec![0.0; n_bins];
        for pos in &self.state.positions {
            if dim < pos.len() {
                let idx = ((pos[dim] - range.0) / bin_width).floor() as usize;
                if idx < n_bins { counts[idx] += 1.0; }
            }
        }
        let support: Vec<f64> = (0..n_bins).map(|i| range.0 + (i as f64 + 0.5) * bin_width).collect();
        DiscreteDistribution::normalized(support, counts)
    }

    pub fn covariance_tensor(&self) -> DiffusionTensor {
        let n = self.state.positions.len() as f64;
        let dim = self.config.dimension.min(3);
        let mut mean = nalgebra::Vector3::zeros();
        for pos in &self.state.positions {
            for d in 0..dim { mean[d] += pos[d]; }
        }
        mean /= n;
        let mut cov = nalgebra::Matrix3::zeros();
        for pos in &self.state.positions {
            for i in 0..dim {
                for j in 0..dim {
                    cov[(i, j)] += (pos[i] - mean[i]) * (pos[j] - mean[j]);
                }
            }
        }
        cov /= (n - 1.0).max(1.0);
        DiffusionTensor { tensor: cov }
    }

    pub fn mean_position(&self) -> Vec<f64> {
        let n = self.state.positions.len();
        if n == 0 { return vec![0.0; self.config.dimension]; }
        let mut mean = vec![0.0; self.config.dimension];
        for pos in &self.state.positions {
            for (i, &v) in pos.iter().enumerate() { mean[i] += v; }
        }
        for m in &mut mean { *m /= n as f64; }
        mean
    }

    pub fn mean_squared_displacement(&self) -> f64 {
        self.state.positions.iter()
            .map(|p| p.iter().map(|x| x * x).sum::<f64>())
            .sum::<f64>() / self.state.positions.len() as f64
    }

    pub fn reset(&mut self) {
        for pos in &mut self.state.positions { pos.fill(0.0); }
        self.state.time = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_diffusion_new() {
        let config = AgentDiffusionConfig::default();
        let ad = AgentDiffusion::new(config);
        assert_eq!(ad.state.positions.len(), 100);
        assert_relative_eq!(ad.state.time, 0.0);
    }

    #[test]
    fn test_agent_diffusion_default_config() {
        let config = AgentDiffusionConfig::default();
        assert_eq!(config.n_agents, 100);
        assert_eq!(config.dimension, 2);
    }

    #[test]
    fn test_brownian_step() {
        let config = AgentDiffusionConfig { n_agents: 10, ..Default::default() };
        let mut ad = AgentDiffusion::new(config);
        ad.brownian_step();
        assert_relative_eq!(ad.state.time, 0.01);
        assert!(ad.state.positions.iter().any(|p| p.iter().any(|&x| x != 0.0)));
    }

    #[test]
    fn test_run_brownian() {
        let config = AgentDiffusionConfig { n_agents: 10, total_time: 0.1, dt: 0.01, ..Default::default() };
        let mut ad = AgentDiffusion::new(config);
        ad.run_brownian();
        assert!(ad.state.time > 0.09);
        assert!(ad.mean_squared_displacement() > 0.0);
    }

    #[test]
    fn test_heat_kernel_agents() {
        let config = AgentDiffusionConfig { n_agents: 2, dimension: 1, ..Default::default() };
        let mut ad = AgentDiffusion::new(config);
        ad.state.positions[0] = vec![0.0];
        ad.state.positions[1] = vec![1.0];
        let k = ad.heat_kernel(0, 1, 1.0);
        assert!(k > 0.0);
    }

    #[test]
    fn test_position_distribution() {
        let config = AgentDiffusionConfig { n_agents: 100, dimension: 1, ..Default::default() };
        let ad = AgentDiffusion::new_random(config, 1.0);
        let dist = ad.position_distribution(0, 10, (-5.0, 5.0));
        let sum: f64 = dist.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_mean_position() {
        let config = AgentDiffusionConfig { n_agents: 3, dimension: 2, ..Default::default() };
        let mut ad = AgentDiffusion::new(config);
        ad.state.positions[0] = vec![1.0, 0.0];
        ad.state.positions[1] = vec![0.0, 1.0];
        ad.state.positions[2] = vec![-1.0, -1.0];
        let mean = ad.mean_position();
        assert_relative_eq!(mean[0], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let config = AgentDiffusionConfig { n_agents: 5, ..Default::default() };
        let mut ad = AgentDiffusion::new(config);
        ad.run_brownian();
        ad.reset();
        for pos in &ad.state.positions { assert!(pos.iter().all(|&x| x == 0.0)); }
    }

    #[test]
    fn test_covariance_tensor() {
        let config = AgentDiffusionConfig { n_agents: 50, dimension: 2, ..Default::default() };
        let ad = AgentDiffusion::new_random(config, 2.0);
        let tensor = ad.covariance_tensor();
        assert!(tensor.mean_diffusivity() > 0.0);
    }

    #[test]
    fn test_spectral_decomposition() {
        let config = AgentDiffusionConfig { n_agents: 5, dimension: 1, ..Default::default() };
        let ad = AgentDiffusion::new(config);
        let adj = DMatrix::from_element(5, 5, 0.2);
        let decomp = ad.spectral_decomposition(&adj).unwrap();
        assert_eq!(decomp.eigenvalues.len(), 5);
    }

    #[test]
    fn test_levy_flight() {
        let config = AgentDiffusionConfig { n_agents: 5, dimension: 2, ..Default::default() };
        let ad = AgentDiffusion::new(config);
        let flight = ad.levy_flight(0, 1.5, 50);
        assert_eq!(flight.positions.len(), 51);
    }

    #[test]
    fn test_mean_squared_displacement() {
        let config = AgentDiffusionConfig { n_agents: 100, dimension: 2, total_time: 0.5, ..Default::default() };
        let mut ad = AgentDiffusion::new(config);
        ad.run_brownian();
        assert!(ad.mean_squared_displacement() > 0.0);
    }
}
