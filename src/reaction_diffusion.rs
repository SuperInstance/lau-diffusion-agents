//! Turing patterns in agent fleets, activator-inhibitor dynamics.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionDiffusionParams {
    pub da: f64,
    pub di: f64,
    pub f: f64,
    pub k: f64,
    pub dt: f64,
    pub dx: f64,
}

impl Default for ReactionDiffusionParams {
    fn default() -> Self {
        Self { da: 1.0, di: 0.5, f: 0.055, k: 0.062, dt: 1.0, dx: 1.0 }
    }
}

pub struct GrayScott {
    pub params: ReactionDiffusionParams,
    pub n: usize,
    pub u: Vec<f64>,
    pub v: Vec<f64>,
}

impl GrayScott {
    pub fn new(params: ReactionDiffusionParams, n: usize) -> Self {
        let u = vec![1.0; n * n];
        let v = vec![0.0; n * n];
        Self { params, n, u, v }
    }

    pub fn seed_center(&mut self, radius: usize, u_val: f64, v_val: f64) {
        let center = self.n / 2;
        for i in (center - radius)..=(center + radius) {
            for j in (center - radius)..=(center + radius) {
                if i < self.n && j < self.n {
                    self.u[i * self.n + j] = u_val;
                    self.v[i * self.n + j] = v_val;
                }
            }
        }
    }

    fn laplacian(&self, field: &[f64], i: usize, j: usize) -> f64 {
        let n = self.n;
        let dx2 = self.params.dx * self.params.dx;
        let ip = if i + 1 < n { i + 1 } else { i };
        let im = if i > 0 { i - 1 } else { 0 };
        let jp = if j + 1 < n { j + 1 } else { j };
        let jm = if j > 0 { j - 1 } else { 0 };
        (field[ip * n + j] + field[im * n + j] + field[i * n + jp] + field[i * n + jm]
            - 4.0 * field[i * n + j]) / dx2
    }

    pub fn step(&mut self) {
        let dt = self.params.dt;
        let f = self.params.f;
        let k = self.params.k;
        let da = self.params.da;
        let di = self.params.di;
        let mut u_new = self.u.clone();
        let mut v_new = self.v.clone();

        for i in 0..self.n {
            for j in 0..self.n {
                let idx = i * self.n + j;
                let u = self.u[idx];
                let v = self.v[idx];
                let uvv = u * v * v;
                u_new[idx] = u + dt * (da * self.laplacian(&self.u, i, j) - uvv + f * (1.0 - u));
                v_new[idx] = v + dt * (di * self.laplacian(&self.v, i, j) + uvv - (f + k) * v);
            }
        }
        self.u = u_new;
        self.v = v_new;
    }

    pub fn evolve(&mut self, n_steps: usize) {
        for _ in 0..n_steps { self.step(); }
    }
}

pub struct FitzHughNagumo {
    pub n: usize,
    pub a: f64, pub b: f64, pub epsilon: f64,
    pub du: f64, pub dv: f64, pub dt: f64, pub dx: f64,
    pub u: Vec<f64>, pub v: Vec<f64>,
}

impl FitzHughNagumo {
    pub fn new(n: usize, a: f64, b: f64, epsilon: f64, du: f64, dv: f64, dt: f64, dx: f64) -> Self {
        Self { n, a, b, epsilon, du, dv, dt, dx, u: vec![0.0; n], v: vec![0.0; n] }
    }

    pub fn set_initial(&mut self, u: Vec<f64>, v: Vec<f64>) { self.u = u; self.v = v; }

    fn laplacian_1d(&self, field: &[f64], i: usize) -> f64 {
        let n = self.n;
        let dx2 = self.dx * self.dx;
        let ip = (i + 1).min(n - 1);
        let im = if i > 0 { i - 1 } else { 0 };
        (field[ip] + field[im] - 2.0 * field[i]) / dx2
    }

    pub fn step(&mut self) {
        let mut u_new = self.u.clone();
        let mut v_new = self.v.clone();
        for i in 0..self.n {
            let u = self.u[i]; let v = self.v[i];
            u_new[i] = u + self.dt * (u - u.powi(3) / 3.0 - v + self.du * self.laplacian_1d(&self.u, i));
            v_new[i] = v + self.dt * self.epsilon * (u + self.a - self.b * v + self.dv * self.laplacian_1d(&self.v, i));
        }
        self.u = u_new; self.v = v_new;
    }

    pub fn evolve(&mut self, n_steps: usize) { for _ in 0..n_steps { self.step(); } }
}

pub fn check_turing_instability(f_u: f64, f_v: f64, g_u: f64, g_v: f64, da: f64, di: f64) -> bool {
    let trace = f_u + g_v;
    let det = f_u * g_v - f_v * g_u;
    let stable_no_diff = trace < 0.0 && det > 0.0;
    let turing = da * g_v + di * f_u > 0.0 &&
        (da * g_v + di * f_u).powi(2) - 4.0 * da * di * det > 0.0;
    stable_no_diff && turing
}

pub fn turing_wavelength(f_u: f64, f_v: f64, g_u: f64, g_v: f64, da: f64, di: f64) -> f64 {
    let k_c_sq = ((di * f_u + da * g_v) / (2.0 * da * di)).max(0.0);
    if k_c_sq > 0.0 { 2.0 * std::f64::consts::PI / k_c_sq.sqrt() } else { f64::INFINITY }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_gray_scott_init() {
        let gs = GrayScott::new(ReactionDiffusionParams::default(), 10);
        assert_eq!(gs.u.len(), 100);
        assert!(gs.u.iter().all(|&u| u == 1.0));
        assert!(gs.v.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gray_scott_seed() {
        let mut gs = GrayScott::new(ReactionDiffusionParams::default(), 20);
        gs.seed_center(3, 0.5, 0.25);
        assert!(gs.v[10 * 20 + 10] > 0.0);
    }

    #[test]
    fn test_gray_scott_step() {
        let mut gs = GrayScott::new(ReactionDiffusionParams::default(), 10);
        gs.seed_center(2, 0.5, 0.25);
        gs.step();
        assert!(gs.u.iter().all(|&u| u.is_finite()));
        assert!(gs.v.iter().all(|&v| v.is_finite()));
    }

    #[test]
    fn test_gray_scott_evolve() {
        let params = ReactionDiffusionParams { dt: 0.5, ..Default::default() };
        let mut gs = GrayScott::new(params, 10);
        gs.seed_center(2, 0.5, 0.25);
        gs.evolve(10);
        assert!(gs.u.iter().all(|&u| u.is_finite()));
    }

    #[test]
    fn test_fitzhugh_nagumo() {
        let mut fhn = FitzHughNagumo::new(50, 0.7, 0.8, 0.08, 1.0, 5.0, 0.01, 1.0);
        let u: Vec<f64> = (0..50).map(|i| if i == 25 { 1.0 } else { 0.0 }).collect();
        fhn.set_initial(u, vec![0.0; 50]);
        fhn.evolve(100);
        assert!(fhn.u.iter().all(|&u| u.is_finite()));
    }

    #[test]
    fn test_turing_instability() {
        // f_u=0.5, f_v=-1.0, g_u=1.0, g_v=-1.5
        // trace = 0.5 + (-1.5) = -1.0 < 0 ✓
        // det = 0.5*(-1.5) - (-1.0)*1.0 = -0.75 + 1.0 = 0.25 > 0 ✓
        // Turing: di*f_u + da*g_v = 10*0.5 + 1*(-1.5) = 3.5 > 0 ✓
        assert!(check_turing_instability(0.5, -1.0, 1.0, -1.5, 1.0, 10.0));
    }

    #[test]
    fn test_turing_stable() {
        assert!(!check_turing_instability(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0));
    }

    #[test]
    fn test_turing_wavelength_finite() {
        let lambda = turing_wavelength(1.0, -1.0, 2.0, -2.5, 1.0, 10.0);
        assert!(lambda.is_finite() && lambda > 0.0);
    }

    #[test]
    fn test_reaction_diffusion_params_default() {
        let p = ReactionDiffusionParams::default();
        assert_relative_eq!(p.f, 0.055);
        assert_relative_eq!(p.k, 0.062);
    }
}
