import numpy as np
import matplotlib.pyplot as plt

PLOT_LINEWIDTH = 2.2
TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12

def simulate_2d_correlated_particle(n_trajectories, n_steps, dt, D, correlation, seed=0):
    """
    Simule r(t) = (x(t), y(t)) avec un bruit corrélé entre X et Y.
    """
    rng = np.random.default_rng(seed)
    
    # Construction de la matrice de covariance
    variance = 2.0 * D * dt
    cov_matrix = variance * np.array([
        [1.0, correlation],
        [correlation, 1.0]
    ])
    
    # Tirage des incréments corrélés
    increments_raw = rng.multivariate_normal(
        mean=[0.0, 0.0], 
        cov=cov_matrix, 
        size=(n_trajectories, n_steps)
    )
    
    # On transpose pour retrouver la structure optimale (n_trajectories, 2, n_steps)
    increments = increments_raw.transpose(0, 2, 1)
    
    r = np.zeros((n_trajectories, 2, n_steps + 1))
    r[:, :, 1:] = np.cumsum(increments, axis=2)
    
    t = np.arange(n_steps + 1) * dt
    return t, r

def plot_correlation_comparison(n_trajectories=3000, n_steps=1000, dt=0.01, D=0.5):
    """
    Simule et trace l'état final du nuage de particules pour comparer
    un mouvement classique (c=0) et un mouvement fortement corrélé (c=0.85).
    """
    # Simulation sans corrélation 
    t, r_uncorr = simulate_2d_correlated_particle(n_trajectories, n_steps, dt, D, correlation=0.0, seed=42)
    
    # Simulation avec forte corrélation positive
    t, r_corr = simulate_2d_correlated_particle(n_trajectories, n_steps, dt, D, correlation=0.85, seed=42)
    
    # extraction des positions finales
    x_final_uncorr = r_uncorr[:, 0, -1]
    y_final_uncorr = r_uncorr[:, 1, -1]
    
    x_final_corr = r_corr[:, 0, -1]
    y_final_corr = r_corr[:, 1, -1]

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    
    # Nuage non corrélé
    axes[0].scatter(x_final_uncorr, y_final_uncorr, alpha=0.15, s=5, color='blue')
    axes[0].set_title("Sans corrélation (c = 0)", fontsize=TITLE_FONTSIZE)
    axes[0].set_xlabel("Position X finale", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel("Position Y finale", fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect('equal', adjustable='box')
    
    # Nuage corrélé
    axes[1].scatter(x_final_corr, y_final_corr, alpha=0.15, s=5, color='red')
    axes[1].set_title("Forte corrélation (c = 0.85)", fontsize=TITLE_FONTSIZE)
    axes[1].set_xlabel("Position X finale", fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')

    plt.suptitle(f"Comparaison de la distribution spatiale après {n_steps} pas", fontsize=TITLE_FONTSIZE + 2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_correlation_comparison()