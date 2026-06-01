# lau-diffusion-agents

> Diffusion processes on agent interaction manifolds: Brownian motion, heat kernels, Fokker-Planck, Langevin dynamics, Lévy flights, and Turing patterns.

## What This Does

This crate implements the full spectrum of diffusion processes for multi-agent systems. It covers Brownian motion and Wiener processes (including geometric Brownian and Ornstein-Uhlenbeck), Itô calculus, heat kernels on Euclidean space, circles, and spheres, the Fokker-Planck equation for probability density evolution, Langevin dynamics and Hamiltonian Monte Carlo for sampling, fractional diffusion and Lévy flights for anomalous transport, reaction-diffusion systems (Gray-Scott, FitzHugh-Nagumo) for Turing patterns, spectral decomposition of diffusion operators, Wasserstein gradient flows and Sinkhorn divergence, anisotropic diffusion tensors with Perona-Malik edge-preserving diffusion, and a unified `AgentDiffusion` API for simulating populations of diffusing agents.

Use this when you need to model how agents, particles, or information spreads through space — from standard Brownian diffusion to anomalous Lévy flights to pattern-forming reaction-diffusion systems.

## The Key Idea

A diffusion process describes how something spreads over time. Mathematically, it's governed by the heat equation ∂u/∂t = Δu (or its stochastic counterpart, Brownian motion). This crate implements the deterministic side (PDE solvers, heat kernels, Fokker-Planck) and the stochastic side (Brownian motion, Langevin dynamics, Lévy flights), connected through the Einstein relation D = σ²/2. For agents, diffusion models exploration, belief propagation, and collective pattern formation.

## Install

```bash
cargo add lau-diffusion-agents
```

## Quick Start

```rust
use lau_diffusion_agents::*;
use nalgebra::DVector;

fn main() {
    // Simulate 100 agents diffusing in 2D
    let config = AgentDiffusionConfig {
        n_agents: 100,
        dimension: 2,
        dt: 0.01,
        total_time: 1.0,
        diffusion_coeff: 1.0,
        drift_coeff: 0.0,
    };
    let mut agents = AgentDiffusion::new_random(config, 1.0);
    agents.run_brownian();
    println!("Mean position: {:?}", agents.mean_position());
    println!("MSD: {:.4}", agents.mean_squared_displacement());

    // Heat kernel: how much does agent i influence agent j?
    let k = agents.heat_kernel(0, 1, 1.0);

    // Estimate drift and diffusion from observed trajectories
    let decomp = agents.estimate_drift_diffusion(&trajectories);

    // Anisotropic diffusion tensor
    let tensor = agents.covariance_tensor();
    println!("FA: {:.4}", tensor.fractional_anisotropy());
}
```

## API Reference

### Brownian Motion

#### `BrownianPath`
Simulated Wiener process path.

```rust
let config = BrownianConfig { dimension: 2, dt: 0.01, sigma: 1.0, drift: 0.0 };
let path = BrownianPath::simulate(&config, 1000);
path.final_position();
path.quadratic_variation();
```

#### `GeometricBrownian`
dS = μS dt + σS dW (stock price model).

```rust
let gb = GeometricBrownian::simulate(s0, mu, sigma, dt, n_steps);
gb.path;  // Vec<(f64, f64)>
```

#### `OrnsteinUhlenbeck`
dx = θ(μ - x)dt + σdW (mean-reverting process).

```rust
let ou = OrnsteinUhlenbeck::simulate(x0, theta, mu, sigma, dt, n_steps);
ou.stationary_variance();  // σ²/(2θ)
```

#### Itô Calculus

```rust
// Itô's lemma: df = (∂f/∂t + ∂f/∂W · dW + ½ ∂²f/∂W² dt)
ito_lemma(f_value, f_t, f_w, f_ww, dt, dw);

// Itô integral: ∫ f dW
ito_integral(&f_values, &increments);

// Reflecting Brownian in [lower, upper]
reflecting_brownian(x0, sigma, dt, n_steps, lower, upper);
```

### Heat Kernel

```rust
// Euclidean: K(x,y,t) = (4πt)^{-d/2} exp(-|x-y|²/4t)
heat_kernel_euclidean(dim, t, &x, &y);

// On a circle of radius r
heat_kernel_circle(t, x, y, r, n_terms);

// On a sphere of radius r (spherical harmonics)
heat_kernel_sphere(t, theta1, phi1, theta2, phi2, r, max_l);

// Gaussian kernel (multivariate)
gaussian_kernel(&x, &mu, &sigma_inv, det_sigma);

// 1D heat equation solver (finite differences)
heat_equation_1d(&u0, alpha, dx, dt, n_steps);

// Graph heat kernel: exp(-tL)
graph_heat_kernel(&laplacian, t, n_terms);
```

### Fokker-Planck Equation

#### `FokkerPlanck1D`
Solve ∂p/∂t = -∂/∂x[a(x)p] + ∂²/∂x²[D(x)p].

```rust
let fp = FokkerPlanck1D::new(101, -5.0, 5.0);
let result = fp.evolve(&p0, |x| -x, |_| 1.0, 0.001, 500);

// Ornstein-Uhlenbeck stationary distribution
fp.stationary_ou(theta, mu, sigma);
```

#### Drift-Diffusion Estimation

```rust
let decomp = DriftDiffusionDecomposition::estimate(&observations, dt);
decomp.estimated_drift;
decomp.estimated_diffusion;
```

#### Diagnostics

```rust
entropy_production_rate(&drift, diffusion, &density, dx);
probability_current(&drift, &diffusion, &density, dx);
```

### Langevin Dynamics

#### Overdamped Langevin

```rust
let config = LangevinConfig { dimension: 2, dt: 0.001, friction: 1.0, temperature: 1.0, n_steps: 10000 };
let result = overdamped_langevin(&config, &x0, &potential);
// result.trajectory, result.energies
```

#### Underdamped Langevin

```rust
let (positions, velocities) = underdamped_langevin(&config, &x0, &v0, &potential);
```

#### Hamiltonian Monte Carlo

```rust
let hmc_config = HMCConfig { dimension: 2, step_size: 0.1, n_leapfrog: 10, n_samples: 1000, temperature: 1.0 };
let result = hmc_sample(&hmc_config, &x0, &potential);
// result.samples, result.acceptance_rate, result.hamiltonians
```

#### Built-in Potentials

```rust
QuadraticPotential::new(matrix);      // U(x) = ½ x'Ax
QuadraticPotential::identity(dim);    // U(x) = ½|x|²
DoubleWellPotential;                  // U(x) = Σ(xᵢ²-1)²
```

### Fractional Diffusion & Lévy Flights

#### `LevyFlight`
Lévy flights with α-stable step sizes.

```rust
let flight = LevyFlight::simulate(dimension, alpha, n_steps, scale);
flight.positions;
flight.step_sizes;
flight.mean_square_displacement();
```

#### α-Stable Random Variables

```rust
let params = StableParams { alpha: 1.5, beta: 0.0, scale: 1.0, location: 0.0 };
stable_random(&params);
stable_samples(&params, 1000);
```

#### `FractionalDiffusion1D`

```rust
let fd = FractionalDiffusion1D::new(n_points, alpha);  // α ∈ (0, 2]
fd.gl_coefficients(n);  // Grünwald-Letnikov coefficients
fd.solve(&u0, dx, dt, n_steps);
```

### Reaction-Diffusion

#### `GrayScott`
Turing pattern formation (activator-inhibitor).

```rust
let mut gs = GrayScott::new(ReactionDiffusionParams::default(), 50);
gs.seed_center(5, 0.5, 0.25);
gs.evolve(1000);
// gs.u, gs.v — concentration fields
```

#### `FitzHughNagumo`
Excitable media (reaction-diffusion on a line).

```rust
let mut fhn = FitzHughNagumo::new(n, a, b, epsilon, du, dv, dt, dx);
fhn.set_initial(u0, v0);
fhn.evolve(n_steps);
```

### Spectral Diffusion

```rust
// Eigen decomposition of diffusion operators
let spec = SpectralDecomposition::decompose(&matrix)?;
spec.eigenvalues;
spec.eigenvectors;
spec.reconstruct();
spec.low_rank(k);

// Heat kernel from spectral decomposition
heat_kernel_spectral(&eigenvalues, &eigenvectors, t, i, j, n_terms);

// Solve diffusion equation spectrally
spectral_diffusion_solve(&eigenvalues, &eigenvectors, &initial, t);

// Spectral gap (mixing time)
spectral_gap(&eigenvalues);

// Graph Laplacian
let lap = graph_laplacian(&adjacency);
let norm_lap = normalized_graph_laplacian(&adjacency);

// 1D Laplacian matrix
laplacian_1d_matrix(n, dx);
```

### Transport Diffusion

#### `DiscreteDistribution`

```rust
let dist = DiscreteDistribution::normalized(support, weights);
dist.mean();
dist.variance();
dist.entropy();
```

#### Optimal Transport

```rust
// 1D Wasserstein distance
wasserstein_1d(&dist_a, &dist_b);

// Sinkhorn algorithm for regularized OT
let (plan, u, v) = sinkhorn(&a, &b, &cost, regularization, max_iter, tolerance);

// KL divergence
kl_divergence(&p, &q);
```

### Anisotropic Diffusion

#### `DiffusionTensor`
3×3 diffusion tensor for direction-dependent diffusion.

```rust
let dt = DiffusionTensor::isotropic(1.0);
let dt = DiffusionTensor::axis_aligned(dx, dy, dz);
let dt = DiffusionTensor::from_eigen(&eigenvalues, &eigenvectors);

dt.fractional_anisotropy();  // FA ∈ [0, 1]
dt.mean_diffusivity();
dt.radial_diffusivity();
dt.trace();
dt.apply(&gradient);
```

#### Perona-Malik Edge-Preserving Diffusion

```rust
let ad = AnisotropicDiffusion2D::new(nx, ny, dx, dy);
let result = ad.evolve_perona_malik(&u0, dt, kappa, n_steps);
```

### Agent Integration

#### `AgentDiffusion`
Unified API for population-level diffusion simulation.

```rust
let mut agents = AgentDiffusion::new_random(config, spread);
agents.brownian_step();
agents.run_brownian();
agents.heat_kernel(i, j, t);
agents.mean_position();
agents.mean_squared_displacement();
agents.covariance_tensor();
agents.position_distribution(dim, n_bins, range);
agents.estimate_drift_diffusion(&trajectories);
agents.hmc_sample(&hmc_config, &potential);
agents.levy_flight(agent_idx, alpha, n_steps);
agents.spectral_decomposition(&adjacency);
agents.reset();
```

## How It Works

**Brownian motion** uses Euler-Maruyama discretization: x_{n+1} = x_n + drift·dt + σ·√dt·Z where Z is uniform on [-1,1] scaled by √2. Quadratic variation converges to T as expected.

**Heat kernels** are computed analytically: the Euclidean kernel uses the Gaussian formula (4πt)^{-d/2} exp(-|x-y|²/4t), the circle kernel sums over periodic images, and the sphere kernel uses Legendre polynomial expansions. The 1D heat equation uses explicit finite differences with CFL stability condition r < ½.

**Fokker-Planck** is solved via conservative finite differences: the flux J = a(x)p - D(x)∂p/∂x is discretized on a staggered grid, ensuring mass conservation.

**Langevin dynamics** uses Euler-Maruyama for overdamped (dx = -∇U/γ dt + √(2T/γ) dW) and velocity Verlet for underdamped. HMC uses leapfrog integration with Metropolis acceptance.

**Lévy flights** generate step sizes from α-stable distributions using the Chambers-Mallows-Stuck method. Fractional diffusion uses Grünwald-Letnikov coefficients for the fractional Laplacian.

**Gray-Scott** evolves the activator-inhibitor PDE system with explicit Euler on a 2D grid. FitzHugh-Nagumo uses a similar approach for excitable media.

**Spectral decomposition** diagonalizes the graph Laplacian L = D - A and uses the eigenbasis for heat kernel computation and diffusion solving.

**Anisotropic diffusion** represents direction-dependent diffusion as a 3×3 tensor. Perona-Malik uses an edge-stopping function g(|∇u|²) = 1/(1 + |∇u|²/κ²) to preserve edges while smoothing.

## The Math

### Brownian Motion (Wiener Process)

W(0) = 0, independent increments, W(t) - W(s) ~ N(0, t-s).

### Itô's Lemma

For f(W, t):

$$df = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial W} dW + \frac{1}{2} \frac{\partial^2 f}{\partial W^2} dt$$

### Heat Equation

$$\frac{\partial u}{\partial t} = \alpha \Delta u$$

Fundamental solution (heat kernel):

$$K(x, y, t) = \frac{1}{(4\pi t)^{d/2}} \exp\left(-\frac{|x-y|^2}{4t}\right)$$

### Fokker-Planck Equation

$$\frac{\partial p}{\partial t} = -\frac{\partial}{\partial x}[a(x) p] + \frac{\partial^2}{\partial x^2}[D(x) p]$$

### Langevin Equation (Overdamped)

$$dx = -\nabla U(x) \, dt + \sqrt{2T} \, dW$$

Stationary distribution: p(x) ∝ exp(-U(x)/T).

### Hamiltonian Monte Carlo

Leapfrog integration of Hamilton's equations with Metropolis acceptance:

$$H(x, p) = U(x) + \frac{1}{2}|p|^2, \quad p(x) \propto \exp(-H)$$

### Fractional Diffusion

$$\frac{\partial u}{\partial t} = -(-\Delta)^{\alpha/2} u, \quad \alpha \in (0, 2]$$

For α = 2: standard diffusion. For α < 2: anomalous (superdiffusion via Lévy flights).

### Gray-Scott Model

$$\frac{\partial u}{\partial t} = D_A \Delta u - uv^2 + f(1-u)$$
$$\frac{\partial v}{\partial t} = D_I \Delta v + uv^2 - (f+k)v$$

### Wasserstein Distance (1D)

$$W_1(\mu, \nu) = \int_{-\infty}^{\infty} |F_\mu(x) - F_\nu(x)| \, dx$$

### Sinkhorn Algorithm

Regularized optimal transport via iterative scaling:

$$K = \exp(-C/\epsilon), \quad u^{(n+1)} = a / (Kv^{(n)}), \quad v^{(n+1)} = b / (K^T u^{(n+1)})$$

### Fractional Anisotropy

$$FA = \sqrt{\frac{3}{2}} \frac{\sqrt{(\lambda_1 - \bar{\lambda})^2 + (\lambda_2 - \bar{\lambda})^2 + (\lambda_3 - \bar{\lambda})^2}}{\sqrt{\lambda_1^2 + \lambda_2^2 + \lambda_3^2}}$$

## License

MIT
