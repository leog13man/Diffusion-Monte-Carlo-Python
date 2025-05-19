import numpy as np

# Parameters
n_walkers = 500         # Number of walkers
n_steps = 5000          # Number of time steps
dt = 0.01               # Time step
alpha = 1.0             # Trial wavefunction exponent
equil_steps = 500       # Steps to equilibrate before energy averaging

def potential(r):
    """Harmonic oscillator potential: V(r) = 0.5 * r^2"""
    return 0.5 * np.sum(r**2, axis=1)

def drift_force(r):
    """Drift force: F = 2*alpha*r"""
    return -2.0 * alpha * r

def trial_wavefunction(r):
    """Gaussian trial wavefunction"""
    r2 = np.sum(r**2, axis=1)
    return np.exp(-alpha * r2 / 2)

# Initialize walkers at the origin
walkers = np.zeros((n_walkers, 3))
weights = np.ones(n_walkers)
energies = []

ref_energy = 1.5  # Initial guess (ground state for 3D harmonic oscillator)

for step in range(n_steps):
    # Drift-diffusion step
    drift = drift_force(walkers) * dt / 2
    random = np.random.normal(0, np.sqrt(dt), size=(n_walkers, 3))
    new_walkers = walkers + drift + random

    # Metropolis-Hastings importance sampling
    psi_new = trial_wavefunction(new_walkers)
    psi_old = trial_wavefunction(walkers)
    greens_ratio = np.exp(-np.sum((walkers - new_walkers - drift_force(new_walkers) * dt / 2)**2, axis=1) / (2 * dt)
                          + np.sum((new_walkers - walkers - drift_force(walkers) * dt / 2)**2, axis=1) / (2 * dt))
    accept_prob = (psi_new**2 / psi_old**2) * greens_ratio
    accept = np.random.rand(n_walkers) < accept_prob
    walkers[accept] = new_walkers[accept]

    # Local energy
    r2 = np.sum(walkers**2, axis=1)
    eloc = 1.5 + 0.5 * r2 * (1 - alpha**2)
    mean_eloc = np.mean(eloc)

    # Branching
    w = np.exp(-dt * (eloc - ref_energy))
    weights *= w
    # Normalize weights and resample
    weights /= np.mean(weights)
    survivors = []
    new_weights = []
    for i in range(n_walkers):
        n_copies = int(weights[i] + np.random.rand())
        for _ in range(n_copies):
            survivors.append(walkers[i])
            new_weights.append(1.0)
    if len(survivors) == 0:
        raise RuntimeError("All walkers died. Try increasing n_walkers or reducing dt.")
    # Resample to maintain total walker count
    if len(survivors) > n_walkers:
        inds = np.random.choice(len(survivors), n_walkers, replace=False)
    else:
        inds = np.random.choice(len(survivors), n_walkers, replace=True)
    walkers = np.array([survivors[i] for i in inds])
    weights = np.ones(n_walkers)

    # Update reference energy to control walker population
    ref_energy = mean_eloc - (np.log(len(survivors)/n_walkers) / dt)

    # Collect energy after equilibration
    if step > equil_steps:
        energies.append(ref_energy)

    if step % 500 == 0:
        print(f"Step {step}, Ref Energy {ref_energy:.6f}, Walkers {len(walkers)}")

print("DMC finished.")
print(f"Ground state energy estimate: {np.mean(energies):.6f} ± {np.std(energies)/np.sqrt(len(energies)):.6f}")