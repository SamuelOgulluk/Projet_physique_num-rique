import numpy as np
import matplotlib.pyplot as plt
from force_exterieure import force_double_well


def first_passage_time(dt, D, gamma, a, b, x_start=-1.2, x_threshold=0.0,
                       max_steps=500000, rng=None):
    """
    Simule une particule dans V(x) = a*(x^2 - b^2)^2 par l'algorithme d'Euler.
    Retourne le temps de premier passage en x >= x_threshold, ou np.nan si
    la particule n'a pas traverse avant max_steps pas.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = x_start
    noise_scale = np.sqrt(2.0 * D * dt)

    for step in range(max_steps):
        force = force_double_well(np.array([x]), a, b)[0]
        x = x + (force / gamma) * dt + noise_scale * rng.standard_normal()
        if x >= x_threshold:
            return (step + 1) * dt

    return np.nan   # n'a pas traverse


def mean_passage_time(n_traj, dt, D, gamma, a, b, seed=0):
    """
    Lance n_traj simulations et retourne le temps moyen de premier passage.
    """
    rng = np.random.default_rng(seed)
    times = [first_passage_time(dt, D, gamma, a, b, rng=rng)
             for _ in range(n_traj)]
    times = np.array(times)

    n_valid = np.sum(~np.isnan(times))
    t_mean  = np.nanmean(times)
    print(f"  D={D:.3f}, a={a:.3f} | {n_valid}/{n_traj} ont traverse "
          f"({100*n_valid/n_traj:.0f}%) | <t> = {t_mean:.2f}")
    return t_mean


def run_eyring_analysis(gamma=1.0, b=1.2, n_traj=500, dt=0.005):
    """
    Verifie la loi d'Arrhenius <t> ~ exp(DeltaV / (D*gamma)) sous deux angles :
      - ln<t> vs DeltaV  a D fixe  (pente theorique = 1/(D*gamma))
      - ln<t> vs 1/D     a DeltaV fixe (pente theorique = DeltaV/gamma)
    """
    # --- Paramètres des deux études ---
    D_fixe   = 0.3
    a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    a_fixe   = 0.4
    D_values = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])

    delta_V_barrier = a_values * b**4   # hauteur de barriere pour chaque a
    delta_V_fixed   = a_fixe  * b**4   # hauteur de barriere fixe

    # --- Variation de la barriere (D fixe) ---
    print("=== Variation de la barriere (D fixe) ===")
    log_t_barrier = []
    for i, a in enumerate(a_values):
        t_mean = mean_passage_time(n_traj, dt, D_fixe, gamma, a, b, seed=i)
        log_t_barrier.append(np.log(t_mean))
    log_t_barrier = np.array(log_t_barrier)

    slope1, intercept1 = np.polyfit(delta_V_barrier, log_t_barrier, 1)
    print(f"\n  Pente mesuree  : {slope1:.3f}")
    print(f"  Pente theorique 1/(D*gamma) = {1/(D_fixe*gamma):.3f}")

    # ---- Variation de D (barriere fixe) ----
    print("\n=== Variation de D (barriere fixe) ===")
    log_t_D = []
    for i, D in enumerate(D_values):
        t_mean = mean_passage_time(n_traj, dt, D, gamma, a_fixe, b, seed=i+20)
        log_t_D.append(np.log(t_mean))
    log_t_D = np.array(log_t_D)

    inv_D = 1.0 / D_values
    slope2, intercept2 = np.polyfit(inv_D, log_t_D, 1)
    print(f"\n  Pente mesuree  : {slope2:.3f}")
    print(f"  Pente theorique DeltaV/gamma = {delta_V_fixed/gamma:.3f}")

    # --- Figure  ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(delta_V_barrier, log_t_barrier, 'o', ms=7, label="Simulation")
    ax1.plot(delta_V_barrier, slope1 * delta_V_barrier + intercept1, '--', lw=2,
             label=f"Fit : pente = {slope1:.3f}\nThéorie : {1/(D_fixe*gamma):.3f}")
    ax1.set_xlabel(r"$\Delta V = a\, b^4$")
    ax1.set_ylabel(r"$\ln\langle t \rangle$")
    ax1.set_title(r"$\ln\langle t\rangle$ vs $\Delta V$  ($D$ fixé)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(inv_D, log_t_D, 'o', ms=7, label="Simulation")
    ax2.plot(inv_D, slope2 * inv_D + intercept2, '--', lw=2,
             label=f"Fit : pente = {slope2:.3f}\nThéorie : {delta_V_fixed/gamma:.3f}")
    ax2.set_xlabel(r"$1/D$")
    ax2.set_ylabel(r"$\ln\langle t \rangle$")
    ax2.set_title(r"$\ln\langle t\rangle$ vs $1/D$  ($\Delta V$ fixé)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.suptitle("Loi d'Arrhenius — temps de premier passage", fontsize=14)
    plt.tight_layout()
    plt.savefig("arrhenius.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_eyring_analysis()