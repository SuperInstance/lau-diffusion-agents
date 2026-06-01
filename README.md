# lau-diffusion-agents

> Diffusion processes for agents — Brownian motion, stochastic calculus, Fokker–Planck, Langevin dynamics, anomalous diffusion, Turing patterns, spectral methods, optimal transport, and anisotropic diffusion.

A Rust crate implementing the core objects of **stochastic diffusion theory**: Wiener processes and Itô calculus, geometric Brownian motion and Ornstein–Uhlenbeck processes, Fokker–Planck PDE solvers with drift-diffusion decomposition, Langevin and Hamiltonian Monte Carlo sampling, fractional diffusion and Lévy flights, reaction-diffusion systems (Gray–Scott, FitzHugh–Nagumo), spectral graph diffusion, Wasserstein optimal transport, and anisotropic diffusion tensors with Perona–Malik edge-preserving smoothing.

A unified `AgentDiffusion` API ties all modules together for multi-agent particle systems.

Every result is verified by **101 unit tests**.

---

## What This Does

- **Brownian motion**: Wiener process simulation, quadratic variation, Itô's lemma, Itô integral, geometric Brownian motion (GBM), Ornstein–Uhlenbeck (OU) process, reflecting Brownian motion
- **Heat kernel**: Euclidean heat kernel, heat kernel on circles and spheres, Gaussian kernel, 1D heat equation (finite differences), graph heat kernel (matrix exponential)
- **Fokker–Planck**: 1D PDE solver (upwind + central differences), drift-diffusion decomposition from observations, stationary OU distribution, probability current, entropy production rate
- **Langevin dynamics**: Overdamped Langevin, underdamped Langevin, Hamiltonian Monte Carlo (HMC) with leapfrog integrator, quadratic and double-well potentials, kinetic temperature
- **Fractional diffusion**: α-stable distributions via Chambers–Mallows–Stuck method, Lévy flights, Grünwald–Letnikov fractional derivatives, fractional diffusion PDE, anomalous exponent estimation, characteristic function
- **Reaction-diffusion**: Gray–Scott model (Turing patterns), FitzHugh–Nagumo model (excitable media), Turing instability analysis
- **Spectral diffusion**: 1D Laplacian, graph Laplacian (unnormalized + normalized), spectral decomposition via SVD, heat kernel from eigenvalues, spectral gap, diffusion kernel matrix, low-rank approximation
- **Transport diffusion**: Discrete distributions, Wasserstein-1 distance, Sinkhorn algorithm for regularized optimal transport, KL divergence, Jensen–Shannon divergence, quantile functions, Wasserstein gradient flow
- **Anisotropic diffusion**: 3×3 diffusion tensors, fractional anisotropy, mean/radial diffusivity, Perona–Malik edge-preserving diffusion, tensor-valued diffusion, structure tensor computation

---

## Key Idea

**Diffusion is the universal smoothing mechanism.** Whether it's particles spreading in a fluid, information propagating across a network, probability distributions relaxing to equilibrium, or agents exploring a state space — the mathematics is the same:

```
∂u/∂t = D ∇²u
```

This crate implements diffusion in all its forms:

1. **Microscopic** (SDE level): Brownian motion dX = σ dW, Itô's lemma, Langevin dynamics
2. **Mesoscopic** (PDF level): Fokker–Planck equation ∂p/∂t = −∇·(ap) + ∇²(Dp)
3. **Macroscopic** (PDE level): Heat equation, reaction-diffusion, anisotropic diffusion
4. **Geometric** (manifold level): Heat kernel on circles, spheres, and graphs
5. **Optimal transport** (distribution level): Wasserstein distance, Sinkhorn, gradient flows

The `AgentDiffusion` struct treats diffusion as a **population-level process**: N agents undergoing Brownian motion in ℝ^d, with methods for mean squared displacement, position distributions, covariance tensors, spectral analysis, and Lévy flights.

---

## Install

```toml
[dependencies]
lau-diffusion-agents = "0.1"
```

Or clone directly:

```bash
git clone https://github.com/SuperInstance/lau-diffusion-agents.git
cargo build
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` 0.33 | Linear algebra (vectors, matrices, SVD, eigendecomposition) |
| `num-complex` 0.4 | Complex numbers for characteristic functions |
| `serde` 1 | Serialization of all structures |
| `rand` 0.8 | Random number generation |
| `rand_distr` 0.4 | Statistical distributions |

---

## Quick Start

### Brownian motion and Itô calculus

```rust
use lau_diffusion_agents::brownian::{BrownianConfig, BrownianPath, ito_lemma, GeometricBrownian};

// Simulate a 2D Brownian path
let config = BrownianConfig { dimension: 2, dt: 0.01, sigma: 1.0, drift: 0.0 };
let path = BrownianPath::simulate(&config, 1000);
println!("Final position: {:?}", path.final_position());
println!("Quadratic variation: {}", path.quadratic_variation());

// Itô's lemma: df = f_t dt + f_W dW + ½ f_{WW} dW²
let f_new = ito_lemma(f_value, f_t, f_w, f_ww, dt, dw);

// Geometric Brownian motion (stock price model)
let gbm = GeometricBrownian::simulate(100.0, 0.05, 0.2, 0.01, 1000);
```

### Ornstein–Uhlenbeck process

```rust
use lau_diffusion_agents::brownian::OrnsteinUhlenbeck;

// Mean-reverting process: dX = θ(μ − X)dt + σ dW
let ou = OrnsteinUhlenbeck::simulate(5.0, 1.0, 0.0, 1.0, 0.01, 500);
println!("Stationary variance: {}", ou.stationary_variance()); // σ²/(2θ) = 0.5
```

### Fokker–Planck equation

```rust
use lau_diffusion_agents::fokker_planck::{FokkerPlanck1D, DriftDiffusionDecomposition};

// Solve ∂p/∂t = −∂(a·p)/∂x + ∂²(D·p)/∂x²
let fp = FokkerPlanck1D::new(101, -5.0, 5.0);
let result = fp.evolve(&p0, |x| -x, |_| 1.0, 0.001, 500);

// Estimate drift and diffusion from observations
let decomp = DriftDiffusionDecomposition::estimate(&observations, dt);
println!("drift={}, diffusion={}", decomp.estimated_drift, decomp.estimated_diffusion);
```

### Langevin dynamics and HMC

```rust
use lau_diffusion_agents::langevin::{
    overdamped_langevin, hmc_sample, LangevinConfig, HMCConfig,
    QuadraticPotential, DoubleWellPotential,
};

// Overdamped Langevin: dX = −∇V/γ dt + √(2T/γ) dW
let config = LangevinConfig { dimension: 2, dt: 0.001, friction: 1.0, temperature: 1.0, n_steps: 1000 };
let result = overdamped_langevin(&config, &x0, &QuadraticPotential::identity(2));

// Hamiltonian Monte Carlo sampling
let hmc_config = HMCConfig { dimension: 1, step_size: 0.1, n_leapfrog: 10, n_samples: 500, temperature: 1.0 };
let hmc_result = hmc_sample(&hmc_config, &x0, &DoubleWellPotential);
println!("Acceptance rate: {}", hmc_result.acceptance_rate);
```

### Multi-agent diffusion

```rust
use lau_diffusion_agents::AgentDiffusion;
use lau_diffusion_agents::agent_diffusion::AgentDiffusionConfig;

let config = AgentDiffusionConfig {
    n_agents: 1000, dimension: 2, dt: 0.01,
    total_time: 1.0, diffusion_coeff: 1.0, drift_coeff: 0.0,
};
let mut agents = AgentDiffusion::new_random(config, 1.0);
agents.run_brownian();
println!("MSD: {}", agents.mean_squared_displacement());
```

---

## API Reference

### `brownian` — Brownian Motion and Itô Calculus

| Type / Function | Description |
|----------------|-------------|
| `BrownianConfig` | Configuration: dimension, dt, σ, drift. Implements `Default` |
| `WienerStep` | Single step: time, position, increment |
| `BrownianPath` | Simulated path. Methods: `simulate(config, n_steps)`, `final_position()`, `quadratic_variation()` |
| `ito_lemma(f, f_t, f_w, f_ww, dt, dw)` | Itô's lemma: df = f_t dt + f_w dW + ½ f_ww dW² |
| `ito_integral(f_vals, increments)` | Itô integral: Σ fᵢ dWᵢ |
| `GeometricBrownian` | GBM: dS = μS dt + σS dW. Methods: `simulate(s0, μ, σ, dt, n_steps)`, `path` |
| `OrnsteinUhlenbeck` | OU: dX = θ(μ−X)dt + σ dW. Methods: `simulate(x0, θ, μ, σ, dt, n)`, `stationary_variance()` → σ²/(2θ) |
| `reflecting_brownian(x0, σ, dt, n, lower, upper)` | Brownian motion reflected at boundaries |

### `heat_kernel` — Heat Kernel on Manifolds

| Type / Function | Description |
|----------------|-------------|
| `heat_kernel_euclidean(dim, t, x, y)` | K(x,y,t) = (4πt)^(−d/2) exp(−|x−y|²/4t) |
| `heat_kernel_circle(t, x, y, r, n_terms)` | Heat kernel on S¹(r) via periodic images |
| `heat_kernel_sphere(t, θ₁, φ₁, θ₂, φ₂, r, max_l)` | Heat kernel on S²(r) via spherical harmonics and Legendre polynomials |
| `legendre_polynomial(l, x)` | Legendre polynomial P_l(x) by recurrence |
| `gaussian_kernel(x, μ, Σ⁻¹, det Σ)` | Multivariate Gaussian N(μ, Σ) |
| `heat_equation_1d(u0, α, dx, dt, n_steps)` | Explicit finite differences with CFL check |
| `graph_heat_kernel(L, t, n_terms)` | exp(−tL) via Taylor series |

### `fokker_planck` — Fokker–Planck Equation

| Type / Function | Description |
|----------------|-------------|
| `DriftDiffusionDecomposition` | Estimated drift and diffusion from observations. Methods: `estimate(obs, dt)` |
| `FokkerPlanck1D` | 1D PDE solver. Methods: `new(n, x_min, x_max)`, `grid()`, `evolve(p0, drift_fn, diff_fn, dt, steps)`, `stationary_ou(θ, μ, σ)` |
| `entropy_production_rate(drift, diff, density, dx)` | Entropy production ∫ j²/(Dp) dx ≥ 0 |
| `probability_current(drift, diff, density, dx)` | Current j = ap − D∇p |

### `langevin` — Langevin Dynamics and HMC

| Type / Function | Description |
|----------------|-------------|
| `LangevinConfig` | Configuration: dimension, dt, friction, temperature, n_steps |
| `PotentialEnergy` (trait) | `value(x)`, `gradient(x)` |
| `QuadraticPotential` | V(x) = ½ x^T A x. Methods: `new(A)`, `identity(dim)` |
| `DoubleWellPotential` | V(x) = Σ (xᵢ² − 1)² |
| `LangevinResult` | Trajectory + energies + config |
| `overdamped_langevin(config, x0, potential)` | dX = −∇V/γ dt + √(2T/γ) dW |
| `underdamped_langevin(config, x0, v0, potential)` | Full Langevin with momentum |
| `HMCConfig` | HMC parameters: step_size, n_leapfrog, n_samples, temperature |
| `HMCResult` | Samples + acceptance rate + Hamiltonians |
| `hmc_sample(config, x0, potential)` | Hamiltonian Monte Carlo with leapfrog integrator |
| `kinetic_temperature(velocities)` | T_kin = ⟨½mv²⟩ / (d/2) |

### `fractional` — Anomalous Diffusion and Lévy Flights

| Type / Function | Description |
|----------------|-------------|
| `StableParams` | α-stable parameters: α ∈ (0,2], β, scale, location |
| `stable_random(params)` | Single α-stable sample (Chambers–Mallows–Stuck method) |
| `stable_samples(params, n)` | n α-stable samples |
| `LevyFlight` | Lévy flight simulation. Methods: `simulate(dim, α, n_steps, scale)`, `mean_square_displacement()` |
| `FractionalDiffusion1D` | Fractional PDE solver. Methods: `new(n, α)`, `gl_coefficients(n)` (Grünwald–Letnikov), `solve(u0, dx, dt, steps)` |
| `stable_characteristic_function(t, params)` | Characteristic function of α-stable distribution |
| `anomalous_exponent(msd, dt)` | Fit MSD ~ t^γ via log-log regression |

### `reaction_diffusion` — Turing Patterns

| Type / Function | Description |
|----------------|-------------|
| `ReactionDiffusionParams` | Parameters: Da, Di, f, k, dt, dx |
| `GrayScott` | Gray–Scott model. Methods: `new(params, n)`, `seed_center(r, u, v)`, `step()`, `evolve(n_steps)` |
| `FitzHughNagumo` | FHN model. Methods: `new(n, a, b, ε, Du, Dv, dt, dx)`, `set_initial(u, v)`, `step()`, `evolve(n_steps)` |
| `check_turing_instability(fu, fv, gu, gv, Da, Di)` | Check if reaction-diffusion system exhibits Turing instability |

### `spectral_diffusion` — Spectral Graph Diffusion

| Type / Function | Description |
|----------------|-------------|
| `SpectralDecomposition` | Eigenvalues + eigenvectors. Methods: `decompose(M)`, `reconstruct()`, `low_rank(k)`, `eigenvalues`, `eigenvectors` |
| `laplacian_1d_matrix(n, dx)` | 1D discrete Laplacian |
| `graph_laplacian(adj)` | Unnormalized graph Laplacian L = D − A |
| `normalized_graph_laplacian(adj)` | Symmetric normalized L = I − D^(−½)AD^(−½) |
| `heat_kernel_spectral(evals, evecs, t, i, j, n_terms)` | K_t(i,j) = Σ exp(−λₖt) φₖ(i)φₖ(j) |
| `spectral_diffusion_solve(evals, evecs, u0, t)` | u(t) = Σ exp(−λₖt) ⟨φₖ, u₀⟩ φₖ |
| `spectral_gap(eigenvalues)` | λ₁ − λ₀ (algebraic connectivity) |
| `diffusion_kernel_matrix(evals, evecs, t)` | Full diffusion kernel matrix |

### `transport_diffusion` — Optimal Transport

| Type / Function | Description |
|----------------|-------------|
| `DiscreteDistribution` | Finite-support distribution. Methods: `new(support, probs)`, `normalized(support, probs)`, `mean()`, `variance()`, `entropy()` |
| `cost_matrix(a, b, p)` | C_ij = |aᵢ − bⱼ|^p |
| `wasserstein_1d(a, b)` | W₁ distance via CDFs |
| `sinkhorn(a, b, cost, reg, max_iter, tol)` | Sinkhorn algorithm → (transport plan, u, v) |
| `kl_divergence(p, q)` | D_KL(p‖q) = Σ pᵢ ln(pᵢ/qᵢ) |
| `js_divergence(p, q)` | D_JS(p‖q) = ½ D_KL(p‖m) + ½ D_KL(q‖m) |
| `wasserstein_gradient_step(positions, target, step_size)` | One step of Wasserstein gradient flow |
| `quantile_function(dist, q)` | Inverse CDF |

### `anisotropic` — Anisotropic Diffusion

| Type / Function | Description |
|----------------|-------------|
| `DiffusionTensor` | 3×3 symmetric positive-definite tensor. Methods: `isotropic(d)`, `from_eigen(λ, V)`, `axis_aligned(dx, dy, dz)`, `fractional_anisotropy()`, `mean_diffusivity()`, `radial_diffusivity()`, `apply(gradient)`, `trace()` |
| `AnisotropicDiffusion2D` | 2D anisotropic solver. Methods: `new(nx, ny, dx, dy)`, `step_perona_malik(u, dt, κ)`, `evolve_perona_malik(u0, dt, κ, n)`, `step_tensor(u, Dxx, Dxy, Dyy, dt)` |
| `structure_tensor(u, nx, ny, dx, dy)` | Compute structure tensor field (Jxx, Jxy, Jyy) from image |

### `agent_diffusion` — Unified Agent API

| Type / Function | Description |
|----------------|-------------|
| `AgentDiffusionConfig` | Configuration: n_agents, dimension, dt, total_time, diffusion_coeff, drift_coeff |
| `AgentState` | Agent positions at time t |
| `AgentDiffusion` | Unified API. Methods: `new(config)`, `new_random(config, spread)`, `brownian_step()`, `run_brownian()`, `heat_kernel(i, j, t)`, `position_distribution(dim, bins, range)`, `mean_position()`, `mean_squared_displacement()`, `covariance_tensor()`, `spectral_decomposition(adj)`, `levy_flight(agent_idx, alpha, n_steps)`, `reset()` |

---

## How It Works

### Architecture

```
brownian ──→ agent_diffusion (unified API)
   │              │
   ├── fokker_planck ──→ transport_diffusion (Wasserstein, Sinkhorn)
   ├── heat_kernel ──→ spectral_diffusion (graph Laplacian, eigenvalues)
   ├── langevin ──→ HMC sampling
   ├── fractional ──→ Lévy flights, anomalous diffusion
   └── reaction_diffusion ──→ Gray-Scott, FitzHugh-Nagumo
                                     │
anisotropic ──→ diffusion tensors ──→ structure tensors
```

### Numerical Methods

- **Brownian motion**: Euler–Maruyama discretization dX = σ√dt · Z where Z ~ Uniform[−1,1] (scaled to match variance)
- **Fokker–Planck**: Upwind scheme for drift + central differences for diffusion, with CFL stability check (r = dt/dx² < 0.5)
- **Heat equation**: Explicit forward Euler with Neumann boundary conditions
- **HMC**: Leapfrog integrator (symplectic, time-reversible) with Metropolis–Hastings acceptance
- **Fractional diffusion**: Grünwald–Letnikov coefficients for the fractional Laplacian
- **Sinkhorn**: Entropy-regularized optimal transport via iterative proportional fitting
- **Spectral**: SVD-based eigendecomposition for symmetric matrices, matrix exponential via Taylor series
- **Anisotropic**: Perona–Malik diffusivity g(∇u) = 1/(1 + |∇u|²/κ²) for edge-preserving smoothing

### Random Number Generation

All stochastic processes use `rand::thread_rng()` with uniform samples on [−1,1] scaled by √2 to match the variance of N(0,1). This is sufficient for the simulation purposes of this crate; for production Monte Carlo, consider replacing with a proper Gaussian sampler.

---

## The Math

### Brownian Motion and the Wiener Process

A **Wiener process** W(t) satisfies: W(0) = 0, independent increments, W(t) − W(s) ~ N(0, t−s), continuous paths. Discretized: ΔW = σ√Δt · Z.

**Quadratic variation**: [W,W]_T = T, computed as Σ (ΔWᵢ)². This is the fundamental result that distinguishes stochastic calculus from classical calculus.

### Itô's Lemma

For f(W,t): df = (∂f/∂t) dt + (∂f/∂W) dW + ½ (∂²f/∂W²) dW², where dW² = dt.

The extra term ½ f_{WW} dt is the **Itô correction** and is what makes stochastic calculus different from ordinary calculus. For f(W) = W², this gives d(W²) = 2W dW + dt, not just 2W dW.

### Fokker–Planck Equation

For an SDE dX = a(X)dt + b(X)dW, the PDF p(x,t) satisfies:

```
∂p/∂t = −∂(a·p)/∂x + ½ ∂²(b²·p)/∂x²
```

This is the **forward Kolmogorov equation**. The crate solves it numerically with upwind drift and central diffusion on a uniform grid.

**Entropy production rate**: ġ = ∫ j²/(D·p) dx ≥ 0, where j = ap − D∇p is the probability current.

### Langevin Dynamics

**Overdamped** (high friction): dX = −∇V/γ dt + √(2T/γ) dW. The stationary distribution is the Boltzmann distribution p ∝ exp(−V/T).

**Underdamped** (full): dX = v dt, dv = (−∇V − γv)dt + √(2γT) dW.

**Hamiltonian Monte Carlo**: Introduce momentum p ~ N(0,T), evolve Hamilton's equations H(x,p) = V(x) + p²/2 via leapfrog, accept/reject by Metropolis criterion exp(−ΔH/T). This gives efficient exploration of complex distributions.

### Geometric Brownian Motion

dS = μS dt + σS dW. By Itô's lemma: S(t) = S₀ exp((μ − σ²/2)t + σW(t)).

Used to model stock prices (Black–Scholes), population growth with noise, and any multiplicative noise process.

### Ornstein–Uhlenbeck Process

dX = θ(μ − X)dt + σ dW. Mean-reverting with rate θ to level μ. Stationary distribution: N(μ, σ²/(2θ)).

The stationary variance σ²/(2θ) is the **fluctuation-dissipation relation**: thermal fluctuations (σ) balanced by dissipation (θ).

### α-Stable Distributions and Lévy Flights

A symmetric α-stable distribution S(α, 0, σ, 0) has characteristic function φ(t) = exp(−σ|t|^α). For α = 2, this is Gaussian; for α < 2, the distribution has **heavy tails** P(|X| > x) ~ x^(−α).

**Lévy flights**: Random walks with step sizes drawn from an α-stable distribution. The MSD diverges for α < 2, leading to **superdiffusion** with anomalous exponent γ > 1 (for MSD ~ t^γ with γ computed from finite-time data).

### Fractional Diffusion

The fractional Laplacian (−Δ)^(α/2) is discretized via **Grünwald–Letnikov coefficients**:

```
w₀ = 1,  wₖ = wₖ₋₁ · (1 − (α+1)/k)
```

For α = 2, this reduces to the standard Laplacian. For α < 2, the operator is **non-local**, capturing long-range interactions.

### Heat Kernel

The **heat kernel** K(x,y,t) is the fundamental solution of the heat equation ∂u/∂t = Δu:

- **Euclidean**: K(x,y,t) = (4πt)^(−d/2) exp(−|x−y|²/4t)
- **Circle S¹(r)**: K(x,y,t) = Σ_k (4πt)^(−½) exp(−(x−y+2πkr)²/4t) (periodic images)
- **Sphere S²(r)**: K = Σ_l (2l+1)/(4πr²) P_l(cos γ) exp(−l(l+1)t/r²) (spherical harmonics)

On graphs, K_t = exp(−tL) where L is the graph Laplacian.

### Reaction-Diffusion and Turing Instability

Two species u, v with different diffusion rates Da, Di and nonlinear reaction terms can produce **Turing patterns** — spatial structure emerging from homogeneous initial conditions.

**Turing instability** requires: (1) the system is stable without diffusion (trace < 0, det > 0), and (2) diffusion destabilizes it (Da·gv + Di·fu > 0 with discriminant positive).

The **Gray–Scott** model: ∂u/∂t = DaΔu − uv² + f(1−u), ∂v/∂t = DiΔv + uv² − (f+k)v.

The **FitzHugh–Nagumo** model: ∂u/∂t = u − u³/3 − v + DuΔu, ∂v/∂t = ε(u + a − bv) + DvΔv.

### Optimal Transport and Wasserstein Distance

The **Wasserstein-1 distance** between distributions μ, ν on ℝ:

```
W₁(μ, ν) = ∫ |F_μ(x) − F_ν(x)| dx
```

where F is the CDF. Computed via sorting and numerical integration.

**Sinkhorn algorithm**: Approximate optimal transport with entropy regularization. Given marginals a, b and cost C, find the optimal plan P minimizing ⟨P,C⟩ + ε H(P) subject to P·1 = a, P^T·1 = b. Solved by iterating u ← a/(Kv), v ← b/(K^T u) where K = exp(−C/ε).

**Wasserstein gradient flow**: Particles evolve toward a target distribution by moving toward quantile-matched positions.

### Anisotropic Diffusion

Instead of scalar diffusivity D, use a **diffusion tensor** D (3×3 SPD matrix). Key metrics:
- **Fractional anisotropy**: FA = √(3/2) · √(Σ(λᵢ − λ̄)²) / √(Σλᵢ²) ∈ [0,1]
- **Mean diffusivity**: MD = trace(D)/3
- **Radial diffusivity**: RD = mean of eigenvalues perpendicular to principal direction

**Perona–Malik**: Edge-preserving diffusion with conductance g(|∇u|) = 1/(1 + |∇u|²/κ²). Smooths homogeneous regions while preserving edges.

---

## License

MIT
