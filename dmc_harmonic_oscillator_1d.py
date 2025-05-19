import numpy as np
import matplotlib.pyplot as plt

def potential(x):
    """Harmonic oscillator potential: V(x) = 0.5 * x^2"""
    return 0.5 * x**2

def diffusion_monte_carlo(
    n_walkers=500, 
    n_steps=1000, 
    dt=0.01, 
    alpha=1.0, 
    trial_energy=None, 
    branching=True
):
    # Initialize walkers at origin
    walkers = np.zeros(n_walkers)
    weights = np.ones(n_walkers)
    if trial_energy is None:
        trial_energy = 0.5 # Ground state energy for 1D harmonic oscillator

    energy_history = []

    for step in range(n_steps):
        # Diffuse walkers
        walkers += np.sqrt(dt) * np.random.randn(n_walkers)

        # Calculate local potential energy
        v = potential(walkers)

        # Branching: compute multiplicities and weights
        dE = v - trial_energy
        if branching:
            # Birth/death (integer branching)
            multiplicities = np.exp(-dt * dE)
            n_copies = np.floor(multiplicities + np.random.rand(n_walkers)).astype(int)
            walkers = np.repeat(walkers, n_copies)
            n_walkers = len(walkers)
            if n_walkers == 0:
                raise RuntimeError("All walkers died. Try smaller dt or fewer steps.")
        else:
            # Weighted walkers
            weights *= np.exp(-dt * dE)
            total_weight = np.sum(weights)
            weights /= total_weight
            # Optional: Resample if effective # of walkers drops too low (not implemented)

        # Update trial energy to control walker population
        trial_energy = trial_energy + alpha * (1.0 - n_walkers / 500)

        # Estimate ground-state energy as average local energy
        energy_estimate = np.mean(v if branching else v * weights)
        energy_history.append(energy_estimate)

        # Optional: prune to keep the walker number manageable
        if n_walkers > 1000:
            idx = np.random.choice(n_walkers, 500, replace=False)
            walkers = walkers[idx]
            n_walkers = 500

    return walkers, energy_history

if __name__ == "__main__":
    n_walkers = 500
    n_steps = 1000
    walkers, energy_history = diffusion_monte_carlo(n_walkers=n_walkers, n_steps=n_steps, dt=0.01)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(energy_history)
    plt.xlabel("Step")
    plt.ylabel("Energy estimate")
    plt.title("DMC Energy Estimate")

    plt.subplot(1,2,2)
    plt.hist(walkers, bins=50, density=True, alpha=0.7, label="DMC")
    # Plot analytical ground state wavefunction squared (Gaussian)
    x = np.linspace(-4,4,200)
    psi_0_sq = np.exp(-x**2)
    psi_0_sq /= np.trapz(psi_0_sq, x)
    plt.plot(x, psi_0_sq, label="|ψ₀(x)|²", color="black")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.title("DMC Position Distribution")
    plt.legend()
    plt.tight_layout()
    plt.show()