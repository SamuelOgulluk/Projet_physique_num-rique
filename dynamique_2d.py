import numpy as np
import matplotlib.pyplot as plt

PLOT_LINEWIDTH = 2.2
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 14
AXIS_LABEL_FONTSIZE = 12

def simulate_2d_free_particle(n_trajectories, n_steps, dt, D, seed=0):
    """
    Simule r(t) = (x(t), y(t)) pour une particule libre en 2D.
    dx = sqrt(2D) dWx  et  dy = sqrt(2D) dWy
    """
    rng = np.random.default_rng(seed)
    
    increments = np.sqrt(2.0 * D * dt) * rng.standard_normal((n_trajectories, 2, n_steps))
    
    r = np.zeros((n_trajectories, 2, n_steps + 1))
    
    r[:, :, 1:] = np.cumsum(increments, axis=2)
    
    t = np.arange(n_steps + 1) * dt
    return t, r

def plot_2d_trajectories(t, r, n_plot=5):
    """
    Trace les trajectoires (x, y) de quelques particules.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    
    for i in range(min(n_plot, r.shape[0])):
        x = r[i, 0, :]
        y = r[i, 1, :]
        ax.plot(x, y, lw=1.5, alpha=0.8, label=f"Particule {i+1}")
        ax.plot(x[0], y[0], 'go', ms=6) # Vert = Départ
        ax.plot(x[-1], y[-1], 'ro', ms=6) # Rouge = Arrivée

    ax.set_title("Mouvement Brownien en 2D (Particules libres)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Position X", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Position Y", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.axis('equal') 
    plt.tight_layout()
    plt.show()

def plot_2d_variance(t, r, D):
    """
    Vérifie que le déplacement quadratique moyen <r^2> = 4Dt en 2D.
    """

    x = r[:, 0, :]
    y = r[:, 1, :]
    

    r_squared = x**2 + y**2
    
    # Moyenne sur toutes les trajectoires
    mean_r_squared = np.mean(r_squared, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean_r_squared, lw=PLOT_LINEWIDTH, label="Simulation $\\langle r^2 \\rangle$")
    ax.plot(t, 4 * D * t, '--', lw=PLOT_LINEWIDTH, label=f"Théorie $4Dt$ (avec D={D})")
    
    ax.set_title("Déplacement Quadratique Moyen (MSD) en 2D", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Temps t", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("$\\langle r^2(t) \\rangle$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.show()


def plot_2d_mean(t, r):
    """
    Vérifie que la position moyenne <x(t)> et <y(t)> reste nulle.
    """
    x = r[:, 0, :]
    y = r[:, 1, :]
    
    # Moyenne sur l'ensemble des trajectoires (axis=0) pour chaque instant
    mean_x = np.mean(x, axis=0)
    mean_y = np.mean(y, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, mean_x, lw=PLOT_LINEWIDTH, label="Simulation <x(t)>")
    ax.plot(t, mean_y, lw=PLOT_LINEWIDTH, label="Simulation <y(t)>")
    
    # La ligne théorique à 0
    ax.plot(t, np.zeros_like(t), 'k--', lw=PLOT_LINEWIDTH, alpha=0.7, label="Théorie (0)")
    
    ax.set_title("Moyenne des positions en 2D", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Temps t", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Position moyenne", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.show()



def plot_2d_std(t, r, D):
    """
    Calcule et trace la racine du déplacement quadratique moyen (RMS) en 2D.
    """
    x = r[:, 0, :]
    y = r[:, 1, :]
    
    # Calcul de la distance au carré r^2
    r_squared = x**2 + y**2
    
    # Racine de la moyenne spatiale sur toutes les trajectoires
    rms_r = np.sqrt(np.mean(r_squared, axis=0))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, rms_r, lw=PLOT_LINEWIDTH, label="Simulation $\\sqrt{\\langle r^2 \\rangle}$")
    
    # Ligne théorique : racine(4Dt) en 2D
    ax.plot(t, np.sqrt(4 * D * t), 'k--', lw=PLOT_LINEWIDTH, alpha=0.7, label="Théorie $\\sqrt{4Dt}$")
    
    ax.set_title("Écart-type de la distance (RMS) en 2D", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Temps t", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("$\\sqrt{\\langle r^2(t) \\rangle}$", fontsize=AXIS_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.show()



def run_2d_analysis(D=0.5, n_trajectories=5000, n_steps=2000, dt=0.01):
    print(f"--- Lancement de l'analyse 2D (D={D}) ---")
    t, r = simulate_2d_free_particle(n_trajectories, n_steps, dt, D)
    
    plot_2d_trajectories(t, r, n_plot=3)
    plot_2d_mean(t, r)
    plot_2d_variance(t, r, D)
    plot_2d_std(t, r, D)
    

if __name__ == "__main__":
    run_2d_analysis()