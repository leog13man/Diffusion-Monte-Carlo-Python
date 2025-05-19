import numpy as np

# Parameters
n_walkers = 1000      # Number of walkers
n_steps = 1000        # Number of DMC steps
dt = 0.01             # Time step
alpha = 1.0           # Importance sampling parameter (from trial wf)
equil_steps = 100     # Steps to equilibrate before averaging

# Potential energy for hydrogen atom
def V(r):
    return -1.0 / r if r > 1e-12 else -1e6  # Prevent division by zero

# Trial wavefunction and quantum force
def psi_T(r):
    return np.exp(-alpha * r)

def quantum_force(x):
    r = np.linalg.norm(x)
    return -alpha * x / r if r > 1e-12 else np.zeros(3)

# Initialize walkers at random positions
walkers = np.random.normal(0, 1, size=(n_walkers, 3))
weights = np.ones(n_walkers)

# Reference energy (trial energy)
E_ref = 0.0

energies = []

for step in range(n_steps):
    new_walkers = []
    local_energies = []

    for i in range(n_walkers):
        x = walkers[i]
        r = np.linalg.norm(x)
        # Drift-diffusion move
        drift = quantum_force(x) * dt
        diffusion = np.sqrt(dt) * np.random.normal(0, 1, 3)
        x_new = x + drift + diffusion

        r_new = np.linalg.norm(x_new)
        # Metropolis-Hastings acceptance (Green's function)
        psi_ratio = (psi_T(r_new) / psi_T(r))**2
        if np.random.rand() < psi_ratio:
            x = x_new
            r = r_new

        # Local energy
        E_loc = -0.5 * (alpha**2 - 2 * alpha / r) - 1 / r if r > 1e-12 else 1e6
        local_energies.append(E_loc)

        # Branching: weight for walker
        w = np.exp(-dt * (E_loc - E_ref))
        n_copies = int(w + np.random.rand())
        for _ in range(n_copies):
            new_walkers.append(x)

    walkers = np.array(new_walkers)
    if len(walkers) == 0:
        # All walkers died; repopulate
        walkers = np.random.normal(0, 1, size=(n_walkers, 3))
    else:
        # Population control: resample to keep walker count fixed
        if len(walkers) > n_walkers:
            walkers = walkers[np.random.choice(len(walkers), n_walkers, replace=False)]
        elif len(walkers) < n_walkers:
            extra = walkers[np.random.choice(len(walkers), n_walkers - len(walkers), replace=True)]
            walkers = np.concatenate([walkers, extra], axis=0)

    # Update reference energy
    E_avg = np.mean(local_energies)
    E_ref = E_avg - (1.0 / dt) * np.log(len(walkers) / n_walkers)

    if step > equil_steps:
        energies.append(E_ref)

    if step % 100 == 0:
        print(f"Step {step}, E_ref = {E_ref:.5f}")

print(f"\nDMC ground state energy: {np.mean(energies):.5f} ± {np.std(energies)/np.sqrt(len(energies)):.5f} (Hartree)")