# lau-diffusion-agents

Diffusion processes on agent interaction manifolds — the mathematical backbone of how information, improvement, and capability spread through agent fleets.

## Modules

| Module | Description |
|--------|-------------|
| `brownian` | Brownian motion, Wiener process, Ito calculus |
| `heat_kernel` | Heat kernel on Riemannian manifolds |
| `fokker_planck` | Fokker-Planck equation, drift-diffusion decomposition |
| `langevin` | Langevin dynamics, Hamiltonian Monte Carlo |
| `fractional` | Fractional diffusion, Levy flights, alpha-stable distributions |
| `reaction_diffusion` | Turing patterns, activator-inhibitor dynamics |
| `spectral_diffusion` | Spectral decomposition of diffusion operators |
| `transport_diffusion` | Wasserstein gradient flow, Sinkhorn divergence |
| `anisotropic` | Direction-dependent diffusion, diffusion tensors |
| `agent_diffusion` | Unified `AgentDiffusion` API |

## Usage

```rust
use lau_diffusion_agents::AgentDiffusion;

let mut ad = AgentDiffusion::new(Default::default());
ad.run_brownian();
let msd = ad.mean_squared_displacement();
println!("Mean squared displacement: {}", msd);
```

## Dependencies

- `nalgebra` — linear algebra
- `num-complex` — complex numbers
- `serde` — serialization
