import numpy as np

# Constants
hbar = 1.0  # Planck's constant (atomic units)
mass_O = 15.999  # Oxygen mass (amu)
mass_H = 1.008   # Hydrogen mass (amu)
amu_to_au = 1822.888486  # 1 amu in atomic units

# Water geometry (TIP4P, flexible)
r_OH = 0.9572  # Angstrom
angle_HOH = 104.52 * np.pi / 180  # radians

def tip4p_potential(positions):
    """
    Flexible TIP4P potential energy for a cluster of water molecules.
    positions: shape (N, 3, 3) for N water molecules, 3 atoms (O, H, H), 3D coords (in Angstrom).
    Returns total potential energy in atomic units.
    """
    # Parameters (atomic units)
    bohr = 0.52917721092
    epsilon = 0.0002951349  # Hartree
    sigma = 3.1589 / bohr   # bohr
    qM = 1.1128             # |e|
    gamma = 0.73612
    Dr = 0.1850012          # Hartree
    alphar = 2.287 * bohr   # bohr^-1
    req = 0.9419 / bohr     # bohr
    ktheta = 0.139998       # Hartree/rad^2
    thetaeq = 1.87448361664 # radians

    N = positions.shape[0]
    # Convert positions from Angstrom to bohr
    pos = positions / bohr  # shape (N, 3, 3)

    # --- Intramolecular energy ---
    U_intra = 0.0
    for i in range(N):
        O = pos[i, 0]
        H1 = pos[i, 1]
        H2 = pos[i, 2]
        r1vec = H1 - O
        r2vec = H2 - O
        r1 = np.linalg.norm(r1vec)
        r2 = np.linalg.norm(r2vec)
        r1dotr2 = np.dot(r1vec, r2vec)
        rtilde = r1dotr2 / (r1 * r2)
        theta = np.arccos(rtilde)
        # Morse-like stretch
        U1 = (Dr * alphar**2 * (r1 - req)**2) * (1.0 - alphar*(r1-req)*(1.0 - (7.0/12.0)*alphar*(r1-req)))
        U2 = (Dr * alphar**2 * (r2 - req)**2) * (1.0 - alphar*(r2-req)*(1.0 - (7.0/12.0)*alphar*(r2-req)))
        Uth = (ktheta/2.0) * (theta - thetaeq)**2
        U_intra += U1 + U2 + Uth

    # --- Intermolecular energy ---
    # Build charge positions (M-site)
    charges = []
    charge_pos = []
    for i in range(N):
        O = pos[i, 0]
        H1 = pos[i, 1]
        H2 = pos[i, 2]
        # M-site
        M = gamma * O + 0.5 * (1 - gamma) * (H1 + H2)
        charges.extend([-qM, 0.5*qM, 0.5*qM])
        charge_pos.extend([M, H1, H2])
    charges = np.array(charges)
    charge_pos = np.array(charge_pos)  # shape (3*N, 3)

    # Coulomb energy
    U_coul = 0.0
    n_atoms = 3 * N
    for i in range(n_atoms - 1):
        for j in range(i + 1, n_atoms):
            # Only intermolecular pairs
            if i // 3 != j // 3:
                rij = np.linalg.norm(charge_pos[i] - charge_pos[j])
                U_coul += charges[i] * charges[j] / rij

    # Lennard-Jones between Oxygens only
    U_LJ = 0.0
    for i in range(N - 1):
        for j in range(i + 1, N):
            Oi = pos[i, 0]
            Oj = pos[j, 0]
            rij = np.linalg.norm(Oi - Oj)
            sir6 = (sigma / rij)**6
            sir12 = sir6**2
            U_LJ += 4 * epsilon * (sir12 - sir6)

    return U_intra + U_coul + U_LJ

def initialize_cluster(N):
    """
    Initialize N water molecules in a cluster.
    Returns positions: shape (N, 3, 3)
    """
    positions = np.zeros((N, 3, 3))
    for i in range(N):
        # Place O atom randomly in a box
        O = np.random.uniform(-2, 2, 3)
        # Place H atoms
        theta = angle_HOH / 2
        H1 = O + np.array([r_OH, 0, 0])
        H2 = O + np.array([
            r_OH * np.cos(angle_HOH),
            r_OH * np.sin(angle_HOH),
            0
        ])
        positions[i, 0] = O
        positions[i, 1] = H1
        positions[i, 2] = H2
    return positions

def propagate_walkers(walkers, dt, mass):
    """
    Propagate walkers using a simple diffusion step.
    walkers: shape (n_walkers, N, 3, 3)
    """
    sigma = np.sqrt(dt / mass)
    noise = np.random.normal(0, sigma, walkers.shape)
    return walkers + noise

def dmc(N=2, n_walkers=100, n_steps=1000, dt=1e-3):
    """
    Diffusion Monte Carlo for a water cluster of N molecules.
    """
    # Mass for each atom (O, H, H)
    masses = np.array([mass_O, mass_H, mass_H]) * amu_to_au
    mass = np.mean(masses)  # Approximate average mass

    # Initialize walkers
    walkers = np.array([initialize_cluster(N) for _ in range(n_walkers)])
    weights = np.ones(n_walkers)
    ref_energy = 0.0
    energies = []

    for step in range(n_steps):
        # Propagate
        walkers = propagate_walkers(walkers, dt, mass)

        # Compute potential energies
        potentials = np.array([tip4p_potential(w) for w in walkers])

        # Branching weights
        dE = potentials - ref_energy
        weights *= np.exp(-dt * dE)

        # Normalize weights
        weights /= np.mean(weights)

        # Resample walkers (simple multinomial resampling)
        indices = np.random.choice(np.arange(n_walkers), size=n_walkers, p=weights/np.sum(weights))
        walkers = walkers[indices]
        weights = np.ones(n_walkers)

        # Update reference energy
        ref_energy = np.mean(potentials) - (1.0/dt) * np.log(np.sum(weights)/n_walkers)
        energies.append(ref_energy)

        if step % 100 == 0:
            print(f"Step {step}, Ref Energy: {ref_energy:.6f}")

    print(f"Final DMC Energy: {np.mean(energies[int(n_steps/2):]):.6f} Hartree")
    return energies

if __name__ == "__main__":
    energies = dmc(N=2, n_walkers=200, n_steps=2000, dt=1e-3)