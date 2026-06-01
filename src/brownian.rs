//! Brownian motion on manifolds, Wiener process, stochastic calculus, Ito's lemma.

use nalgebra::DVector;
use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrownianConfig {
    pub dimension: usize,
    pub dt: f64,
    pub sigma: f64,
    pub drift: f64,
}

impl Default for BrownianConfig {
    fn default() -> Self {
        Self { dimension: 1, dt: 0.01, sigma: 1.0, drift: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WienerStep {
    pub time: f64,
    pub position: DVector<f64>,
    pub increment: DVector<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrownianPath {
    pub steps: Vec<WienerStep>,
    pub config: BrownianConfig,
}

impl BrownianPath {
    pub fn simulate(config: &BrownianConfig, n_steps: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut steps = Vec::with_capacity(n_steps);
        let mut position = DVector::zeros(config.dimension);
        let sqrt_dt = config.dt.sqrt();

        for i in 0..n_steps {
            let t = i as f64 * config.dt;
            let mut inc = DVector::zeros(config.dimension);
            for j in 0..config.dimension {
                let z: f64 = rng.gen_range(-1.0..1.0);
                inc[j] = config.sigma * sqrt_dt * z * std::f64::consts::SQRT_2
                    + config.drift * config.dt;
            }
            position += &inc;
            steps.push(WienerStep { time: t, position: position.clone(), increment: inc });
        }
        BrownianPath { steps, config: config.clone() }
    }

    pub fn final_position(&self) -> Option<&DVector<f64>> {
        self.steps.last().map(|s| &s.position)
    }

    pub fn quadratic_variation(&self) -> f64 {
        self.steps.iter().map(|s| s.increment.iter().map(|x| x * x).sum::<f64>()).sum()
    }
}

pub fn ito_lemma(f_value: f64, f_t: f64, f_w: f64, f_ww: f64, dt: f64, dw: f64) -> f64 {
    f_value + f_t * dt + f_w * dw + 0.5 * f_ww * dw * dw
}

pub fn ito_integral(f_values: &[f64], increments: &[f64]) -> f64 {
    f_values.iter().zip(increments.iter()).map(|(f, dw)| f * dw).sum()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricBrownian {
    pub initial_value: f64,
    pub mu: f64,
    pub sigma: f64,
    pub path: Vec<(f64, f64)>,
}

impl GeometricBrownian {
    pub fn simulate(s0: f64, mu: f64, sigma: f64, dt: f64, n_steps: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut s = s0;
        path.push((0.0, s));
        let sqrt_dt = dt.sqrt();

        for i in 1..=n_steps {
            let z: f64 = rng.gen_range(-1.0..1.0);
            let dw = z * sqrt_dt * std::f64::consts::SQRT_2;
            s = s * ((mu - 0.5 * sigma * sigma) * dt + sigma * dw).exp();
            path.push((i as f64 * dt, s));
        }
        GeometricBrownian { initial_value: s0, mu, sigma, path }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrnsteinUhlenbeck {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub path: Vec<(f64, f64)>,
}

impl OrnsteinUhlenbeck {
    pub fn simulate(x0: f64, theta: f64, mu: f64, sigma: f64, dt: f64, n_steps: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut path = Vec::with_capacity(n_steps + 1);
        let mut x = x0;
        path.push((0.0, x));
        let sqrt_dt = dt.sqrt();

        for i in 1..=n_steps {
            let z: f64 = rng.gen_range(-1.0..1.0);
            let dw = z * sqrt_dt * std::f64::consts::SQRT_2;
            x = x + theta * (mu - x) * dt + sigma * dw;
            path.push((i as f64 * dt, x));
        }
        OrnsteinUhlenbeck { theta, mu, sigma, path }
    }

    pub fn stationary_variance(&self) -> f64 {
        self.sigma * self.sigma / (2.0 * self.theta)
    }
}

pub fn reflecting_brownian(x0: f64, sigma: f64, dt: f64, n_steps: usize, lower: f64, upper: f64) -> Vec<(f64, f64)> {
    let mut rng = rand::thread_rng();
    let mut path = Vec::with_capacity(n_steps + 1);
    let mut x = x0;
    let sqrt_dt = dt.sqrt();
    path.push((0.0, x));

    for i in 1..=n_steps {
        let z: f64 = rng.gen_range(-1.0..1.0);
        x += sigma * z * sqrt_dt * std::f64::consts::SQRT_2;
        while x < lower || x > upper {
            if x < lower { x = 2.0 * lower - x; }
            if x > upper { x = 2.0 * upper - x; }
        }
        path.push((i as f64 * dt, x));
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_brownian_default_config() {
        let cfg = BrownianConfig::default();
        assert_eq!(cfg.dimension, 1);
        assert_eq!(cfg.sigma, 1.0);
        assert_eq!(cfg.drift, 0.0);
    }

    #[test]
    fn test_brownian_simulation() {
        let cfg = BrownianConfig { dimension: 2, dt: 0.01, sigma: 1.0, drift: 0.0 };
        let path = BrownianPath::simulate(&cfg, 100);
        assert_eq!(path.steps.len(), 100);
        assert!(path.final_position().is_some());
    }

    #[test]
    fn test_quadratic_variation() {
        let cfg = BrownianConfig { dimension: 1, dt: 0.001, sigma: 1.0, drift: 0.0 };
        let path = BrownianPath::simulate(&cfg, 10000);
        let qv = path.quadratic_variation();
        let t = 10000.0 * 0.001;
        assert!(qv > 0.0 && (qv - t).abs() / t < 1.0, "QV={} vs T={}", qv, t);
    }

    #[test]
    fn test_ito_lemma_constant() {
        let result = ito_lemma(0.0, 0.0, 1.0, 0.0, 0.01, 0.1);
        assert_relative_eq!(result, 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_ito_lemma_quadratic() {
        let w = 1.0;
        let dw = 0.05;
        let result = ito_lemma(w * w, 1.0, 2.0 * w, 2.0, 0.01, dw);
        let expected = w * w + 0.01 + 2.0 * w * dw + 0.5 * 2.0 * dw * dw;
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_ito_integral() {
        let f = vec![1.0, 2.0, 3.0];
        let dw = vec![0.1, -0.2, 0.3];
        let result = ito_integral(&f, &dw);
        assert_relative_eq!(result, 1.0 * 0.1 + 2.0 * (-0.2) + 3.0 * 0.3);
    }

    #[test]
    fn test_geometric_brownian() {
        let gbm = GeometricBrownian::simulate(100.0, 0.05, 0.2, 0.01, 1000);
        assert_eq!(gbm.path.len(), 1001);
        assert_relative_eq!(gbm.path[0].1, 100.0);
        assert!(gbm.path.last().unwrap().1 > 0.0);
    }

    #[test]
    fn test_ou_stationary_variance() {
        let ou = OrnsteinUhlenbeck { theta: 1.0, mu: 0.0, sigma: 1.0, path: vec![] };
        let var = ou.stationary_variance();
        assert_relative_eq!(var, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_ou_simulation() {
        let ou = OrnsteinUhlenbeck::simulate(5.0, 1.0, 0.0, 1.0, 0.01, 500);
        assert_eq!(ou.path.len(), 501);
        assert_relative_eq!(ou.path[0].1, 5.0);
    }

    #[test]
    fn test_reflecting_brownian_bounds() {
        let path = reflecting_brownian(0.5, 1.0, 0.01, 1000, 0.0, 1.0);
        for (_, x) in &path {
            assert!(*x >= -1e-10 && *x <= 1.0 + 1e-10, "x={}", x);
        }
    }
}
