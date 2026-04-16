import numpy as np
import matplotlib.pyplot as plt
from force_exterieure import force_double_well, potential_double_well


def first_passage_times(n_traj, dt, D, gamma, a, b, x_threshold=0.0, max_steps=500000, seed=0):
    """
    Simule n_traj particules dans V(x) = a*(x^2 - b^2)^2, partant de x = -b.
    Retourne le temps de premier passage en x = x_threshold (sommet de barriere).
    Les particules qui n'ont pas traverse avant max_steps sont ignorees.
    """
    rng = np.random.default_rng(seed)
    noise_scale = np.sqrt(2.0 * D * dt)

    x = np.full(n_traj, -b)
    passage_times = np.full(n_traj, np.nan)
    active = np.ones(n_traj, dtype=bool)

    for step in range(max_steps):
        if not np.any(active):
            break

        x_active = x[active]
        drift = (force_double_well(x_active, a, b) / gamma) * dt
        noise = noise_scale * rng.standard_normal(np.sum(active))
        x[active] = x_active + drift + noise

        # Detection du passage
        crossed = active & (x >= x_threshold)
        passage_times[crossed] = (step + 1) * dt
        active[crossed] = False

    n_valid = np.sum(~np.isnan(passage_times))
    print(f"  D={D:.3f}, a={a:.3f} | {n_valid}/{n_traj} particules ont traverse "
          f"({100*n_valid/n_traj:.1f}%) | <t> = {np.nanmean(passage_times):.2f}")

    return passage_times


def run_eyring_barrier_analysis(D=0.3, gamma=1.0, b=1.2, n_traj=3000, dt=0.005):
    """Fait varier la hauteur de barriere DeltaV = a*b^4 a D fixe."""
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    delta_V = a_values * b**4

    mean_times = []
    print("=== Variation de la barriere (D fixe) ===")
    for i, a in enumerate(a_values):
        fpt = first_passage_times(n_traj, dt, D, gamma, a, b, seed=i)
        mean_times.append(np.nanmean(fpt))

    mean_times = np.array(mean_times)
    log_t = np.log(mean_times)

    # Fit lineaire ln<t> = alpha * DeltaV + cst
    coeffs = np.polyfit(delta_V, log_t, 1)
    slope, intercept = coeffs
    print(f"\nPente mesuree : {slope:.4f}")
    print(f"Pente theorique (Arrhenius) : 1/(D*gamma) = {1/(D*gamma):.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(delta_V, log_t, 'o', ms=7, label="Simulation")
    ax.plot(delta_V, np.polyval(coeffs, delta_V), '--', lw=2,
            label=f"Fit : pente = {slope:.3f}\nThéorie : {1/(D*gamma):.3f}")
    ax.set_xlabel(r"$\Delta V = a b^4$")
    ax.set_ylabel(r"$\ln\langle t \rangle$")
    ax.set_title(r"Loi d'Arrhenius : $\ln\langle t\rangle$ vs $\Delta V$ ($D$ fixé)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("arrhenius_barrier.png", dpi=150)
    plt.show()


def run_eyring_D_analysis(a=0.4, gamma=1.0, b=1.2, n_traj=3000, dt=0.005):
    """Fait varier D a barriere fixe."""
    D_values = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
    delta_V = a * b**4

    mean_times = []
    print("\n=== Variation de D (barriere fixe) ===")
    for i, D in enumerate(D_values):
        fpt = first_passage_times(n_traj, dt, D, gamma, a, b, seed=i+20)
        mean_times.append(np.nanmean(fpt))

    mean_times = np.array(mean_times)
    inv_D = 1.0 / D_values
    log_t = np.log(mean_times)

    coeffs = np.polyfit(inv_D, log_t, 1)
    slope, intercept = coeffs
    print(f"\nPente mesuree : {slope:.4f}")
    print(f"Pente theorique (Arrhenius) : DeltaV/gamma = {delta_V/gamma:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(inv_D, log_t, 'o', ms=7, label="Simulation")
    ax.plot(inv_D, np.polyval(coeffs, inv_D), '--', lw=2,
            label=f"Fit : pente = {slope:.3f}\nThéorie : {delta_V/gamma:.3f}")
    ax.set_xlabel(r"$1/D$")
    ax.set_ylabel(r"$\ln\langle t \rangle$")
    ax.set_title(r"Loi d'Arrhenius : $\ln\langle t\rangle$ vs $1/D$ ($\Delta V$ fixé)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("arrhenius_D.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_eyring_barrier_analysis()
    run_eyring_D_analysis()