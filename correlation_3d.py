import numpy as np
import matplotlib.pyplot as plt

PLOT_LINEWIDTH = 2.2
TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12

def simulate_3d_correlated_particle(n_trajectories, n_steps, dt, D, rho, seed=0):
    """
    Simule r(t) = (x(t), y(t), z(t)) avec un bruit thermique corrélé.
    """
    rng = np.random.default_rng(seed)
    
    # Construction de la matrice de covariance 3x3
    variance = 2.0 * D * dt
    cov_matrix = variance * np.array([
        [1.0, rho, rho],
        [rho, 1.0, rho],
        [rho, rho, 1.0]
    ])
    
    # Tirage des incréments corrélés en 3D
    increments_raw = rng.multivariate_normal(
        mean=[0.0, 0.0, 0.0], 
        cov=cov_matrix, 
        size=(n_trajectories, n_steps)
    )
    
    # On transpose pour avoir la forme (n_trajectories, 3, n_steps)
    increments = increments_raw.transpose(0, 2, 1)
    
    r = np.zeros((n_trajectories, 3, n_steps + 1))
    r[:, :, 1:] = np.cumsum(increments, axis=2)
    
    t = np.arange(n_steps + 1) * dt
    return t, r

def plot_correlation_comparison_3d(n_trajectories=3000, n_steps=1000, dt=0.01, D=0.5):
    """
    Simule et trace l'état final du nuage de particules en 3D pour comparer.
    """
    # Sans corrélation (rho = 0)
    t, r_uncorr = simulate_3d_correlated_particle(n_trajectories, n_steps, dt, D, rho=0.0, seed=42)
    
    # Avec forte corrélation positive (rho = 0.8)
    t, r_corr = simulate_3d_correlated_particle(n_trajectories, n_steps, dt, D, rho=0.8, seed=42)
    
    # extraction des positions finales sur X, Y et Z
    x_uncorr, y_uncorr, z_uncorr = r_uncorr[:, 0, -1], r_uncorr[:, 1, -1], r_uncorr[:, 2, -1]
    x_corr, y_corr, z_corr = r_corr[:, 0, -1], r_corr[:, 1, -1], r_corr[:, 2, -1]

    # Figures
    fig = plt.figure(figsize=(14, 6))
    
    # Nuage non corrélé
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(x_uncorr, y_uncorr, z_uncorr, alpha=0.15, s=4, color='blue')
    ax1.set_title(r"Sans corrélation ($\rho = 0$)", fontsize=TITLE_FONTSIZE)
    ax1.set_xlabel("Position X", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_ylabel("Position Y", fontsize=AXIS_LABEL_FONTSIZE)
    ax1.set_zlabel("Position Z", fontsize=AXIS_LABEL_FONTSIZE)
    
    # Nuage corrélé
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(x_corr, y_corr, z_corr, alpha=0.15, s=4, color='red')
    ax2.set_title(r"Forte corrélation ($\rho = 0.8$)", fontsize=TITLE_FONTSIZE)
    ax2.set_xlabel("Position X", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_ylabel("Position Y", fontsize=AXIS_LABEL_FONTSIZE)
    ax2.set_zlabel("Position Z", fontsize=AXIS_LABEL_FONTSIZE)

    plt.suptitle(f"Comparaison de la distribution spatiale 3D après {n_steps} pas", fontsize=TITLE_FONTSIZE + 2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_correlation_comparison_3d()