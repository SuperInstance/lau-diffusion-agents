//! Wasserstein gradient flow, Sinkhorn divergence.

use nalgebra::{DVector, DMatrix};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteDistribution {
    pub support: Vec<f64>,
    pub probabilities: Vec<f64>,
}

impl DiscreteDistribution {
    pub fn new(support: Vec<f64>, probabilities: Vec<f64>) -> Result<Self, String> {
        let sum: f64 = probabilities.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err(format!("Probabilities must sum to 1.0, got {}", sum));
        }
        Ok(Self { support, probabilities })
    }

    pub fn normalized(support: Vec<f64>, mut probabilities: Vec<f64>) -> Self {
        let sum: f64 = probabilities.iter().sum();
        if sum > 0.0 { for p in &mut probabilities { *p /= sum; } }
        Self { support, probabilities }
    }

    pub fn mean(&self) -> f64 {
        self.support.iter().zip(self.probabilities.iter()).map(|(x, p)| x * p).sum()
    }

    pub fn variance(&self) -> f64 {
        let mu = self.mean();
        self.support.iter().zip(self.probabilities.iter()).map(|(x, p)| p * (x - mu).powi(2)).sum()
    }

    pub fn entropy(&self) -> f64 {
        -self.probabilities.iter().filter(|&&p| p > 1e-15).map(|&p| p * p.ln()).sum::<f64>()
    }
}

pub fn cost_matrix(a: &DiscreteDistribution, b: &DiscreteDistribution, p: f64) -> DMatrix<f64> {
    let n = a.support.len();
    let m = b.support.len();
    let mut c = DMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            c[(i, j)] = (a.support[i] - b.support[j]).abs().powf(p);
        }
    }
    c
}

pub fn wasserstein_1d(a: &DiscreteDistribution, b: &DiscreteDistribution) -> f64 {
    let mut a_pairs: Vec<_> = a.support.iter().zip(a.probabilities.iter()).collect();
    let mut b_pairs: Vec<_> = b.support.iter().zip(b.probabilities.iter()).collect();
    a_pairs.sort_by(|x, y| x.0.partial_cmp(y.0).unwrap());
    b_pairs.sort_by(|x, y| x.0.partial_cmp(y.0).unwrap());

    let mut all_points: Vec<f64> = a.support.iter().chain(b.support.iter()).copied().collect();
    all_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_points.dedup();

    let mut w1 = 0.0;
    for window in all_points.windows(2) {
        let dx = window[1] - window[0];
        let mid = (window[0] + window[1]) / 2.0;
        let fa = cdf_at(&a_pairs, mid);
        let fb = cdf_at(&b_pairs, mid);
        w1 += (fa - fb).abs() * dx;
    }
    w1
}

fn cdf_at(sorted_pairs: &[(&f64, &f64)], x: f64) -> f64 {
    sorted_pairs.iter().filter(|(xi, _)| **xi <= x).map(|(_, p)| **p).sum()
}

pub fn sinkhorn(a: &DVector<f64>, b: &DVector<f64>, cost: &DMatrix<f64>, reg: f64, max_iter: usize, tol: f64) -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let n = a.nrows();
    let m = b.nrows();
    let mut u = DVector::from_element(n, 1.0 / n as f64);
    let mut v = DVector::from_element(m, 1.0 / m as f64);
    let k = cost.map(|c| (-c / reg).exp());

    for _ in 0..max_iter {
        let kv = &k * &v;
        for i in 0..n { if kv[i].abs() > 1e-15 { u[i] = a[i] / kv[i]; } }
        let ktu = &k.transpose() * &u;
        for j in 0..m { if ktu[j].abs() > 1e-15 { v[j] = b[j] / ktu[j]; } }
        let ktu_new = &k.transpose() * &u;
        let err: f64 = (&ktu_new - b).iter().map(|x| x.abs()).sum();
        if err < tol { break; }
    }

    let mut plan = DMatrix::zeros(n, m);
    for i in 0..n { for j in 0..m { plan[(i, j)] = u[i] * k[(i, j)] * v[j]; } }
    (plan, u, v)
}

pub fn kl_divergence(p: &DVector<f64>, q: &DVector<f64>) -> f64 {
    p.iter().zip(q.iter())
        .filter(|(&pi, _)| pi > 1e-15)
        .map(|(&pi, &qi)| pi * (pi / qi.max(1e-15)).ln())
        .sum()
}

pub fn js_divergence(p: &DVector<f64>, q: &DVector<f64>) -> f64 {
    let m = (p + q).scale(0.5);
    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

pub fn wasserstein_gradient_step(positions: &mut [f64], target: &DiscreteDistribution, step_size: f64) {
    let n = positions.len();
    positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for i in 0..n {
        let quantile = (i as f64 + 0.5) / n as f64;
        let target_val = quantile_function(target, quantile);
        positions[i] += step_size * (target_val - positions[i]);
    }
}

pub fn quantile_function(dist: &DiscreteDistribution, q: f64) -> f64 {
    let mut pairs: Vec<_> = dist.support.iter().zip(dist.probabilities.iter()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());
    let mut cumsum = 0.0;
    for (&x, &p) in &pairs {
        cumsum += p;
        if cumsum >= q { return x; }
    }
    *pairs.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_discrete_distribution_normalized() {
        let d = DiscreteDistribution::normalized(vec![0.0, 1.0, 2.0], vec![1.0, 2.0, 3.0]);
        let sum: f64 = d.probabilities.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_discrete_distribution_mean() {
        let d = DiscreteDistribution::normalized(vec![0.0, 1.0], vec![1.0, 1.0]);
        assert_relative_eq!(d.mean(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_discrete_distribution_variance() {
        let d = DiscreteDistribution::normalized(vec![0.0, 2.0], vec![1.0, 1.0]);
        assert_relative_eq!(d.variance(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_entropy() {
        let d = DiscreteDistribution::normalized(vec![0.0, 1.0], vec![1.0, 1.0]);
        assert_relative_eq!(d.entropy(), 2.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_wasserstein_1d_positive() {
        let a = DiscreteDistribution::normalized(vec![0.0, 1.0], vec![1.0, 1.0]);
        let b = DiscreteDistribution::normalized(vec![1.0, 2.0], vec![1.0, 1.0]);
        let w1 = wasserstein_1d(&a, &b);
        assert!(w1 > 0.0);
    }

    #[test]
    fn test_wasserstein_same() {
        let a = DiscreteDistribution::normalized(vec![0.0, 1.0], vec![1.0, 1.0]);
        let w1 = wasserstein_1d(&a, &a);
        assert_relative_eq!(w1, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_sinkhorn() {
        let a = DVector::from_vec(vec![0.5, 0.5]);
        let b = DVector::from_vec(vec![0.5, 0.5]);
        let cost = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, 1.0, 0.0]);
        let (plan, _, _) = sinkhorn(&a, &b, &cost, 0.1, 100, 1e-6);
        assert_eq!(plan.nrows(), 2);
    }

    #[test]
    fn test_kl_divergence_same() {
        let p = DVector::from_vec(vec![0.5, 0.5]);
        let kl = kl_divergence(&p, &p);
        assert_relative_eq!(kl, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kl_divergence_positive() {
        let p = DVector::from_vec(vec![1.0, 0.0]);
        let q = DVector::from_vec(vec![0.5, 0.5]);
        let kl = kl_divergence(&p, &q);
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_js_symmetry() {
        let p = DVector::from_vec(vec![0.8, 0.2]);
        let q = DVector::from_vec(vec![0.3, 0.7]);
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        assert_relative_eq!(js_pq, js_qp, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_function() {
        let d = DiscreteDistribution::normalized(vec![0.0, 1.0, 2.0], vec![1.0, 1.0, 1.0]);
        let q25 = quantile_function(&d, 0.25);
        assert!(q25 >= 0.0);
    }

    #[test]
    fn test_wasserstein_gradient_step() {
        let mut positions = vec![0.0, 0.1, 0.2];
        let target = DiscreteDistribution::normalized(vec![5.0, 6.0, 7.0], vec![1.0, 1.0, 1.0]);
        wasserstein_gradient_step(&mut positions, &target, 0.5);
        assert!(positions[0] > 0.0);
    }
}
