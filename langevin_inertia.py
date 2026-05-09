import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from force_exterieure import force_double_well


def potential_harmonic(x, k):
    """Potentiel harmonique V(x) = (k/2)*x^2"""
    return 0.5 * k * x**2


def force_harmonic(x, k):
    """Force: F = -dV/dx = -k*x"""
    return -k * x


PLOT_LINEWIDTH = 2.2
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 13
TICK_LABEL_FONTSIZE = 11

def simulate_langevin_with_inertia_free(n_trajectories, n_steps, dt, D, m=1.0, gamma=1.0, seed=0):
    """Simule particules libres avec inertie: dx = v dt, dv = -(gamma/m)v dt + (gamma/m)√(2D) dW"""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trajectories, n_steps + 1))
    v = np.zeros((n_trajectories, n_steps + 1))
    
    noise_scale = np.sqrt(2.0 * D * gamma / m * dt)
    exp_factor = np.exp(-gamma / m * dt)
    
    for step in range(n_steps):
        noise = rng.standard_normal(n_trajectories)
        v[:, step + 1] = exp_factor * v[:, step] + noise_scale * noise
        x[:, step + 1] = x[:, step] + v[:, step] * dt
    
    t = np.arange(n_steps + 1) * dt
    return t, x, v


def simulate_langevin_with_inertia_harmonic(n_trajectories, n_steps, dt, D, k, m=1.0, gamma=1.0, seed=0):
    """Simule particules avec inertie dans potentiel harmonique V(x) = (k/2)x^2"""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trajectories, n_steps + 1))
    v = np.zeros((n_trajectories, n_steps + 1))
    
    noise_scale = np.sqrt(2.0 * D * gamma / m * dt)
    exp_factor = np.exp(-gamma / m * dt)
    
    for step in range(n_steps):
        noise = rng.standard_normal(n_trajectories)
        accel = -(k / m) * x[:, step]
        v[:, step + 1] = exp_factor * v[:, step] + (accel * dt) + noise_scale * noise
        x[:, step + 1] = x[:, step] + v[:, step] * dt
    
    t = np.arange(n_steps + 1) * dt
    return t, x, v


def simulate_langevin_with_inertia_double_well(n_trajectories, n_steps, dt, D, a, b, m=1.0, gamma=1.0, seed=0):
    """Simule particules avec inertie dans potentiel double puits"""
    rng = np.random.default_rng(seed)
    x = np.zeros((n_trajectories, n_steps + 1))
    v = np.zeros((n_trajectories, n_steps + 1))
    x[:, 0] = -b
    
    noise_scale = np.sqrt(2.0 * D * gamma / m * dt)
    exp_factor = np.exp(-gamma / m * dt)
    
    for step in range(n_steps):
        noise = rng.standard_normal(n_trajectories)
        force = force_double_well(x[:, step], a, b)
        accel = force / m
        v[:, step + 1] = exp_factor * v[:, step] + (accel * dt) + noise_scale * noise
        x[:, step + 1] = x[:, step] + v[:, step] * dt
    
    t = np.arange(n_steps + 1) * dt
    return t, x, v


def compute_trajectory_stats(x, v):
    """Calcule propriétés statistiques des trajectoires"""
    x_mean = np.mean(x, axis=0)
    x_var = np.var(x, axis=0)
    x_std = np.std(x, axis=0)
    v_mean = np.mean(v, axis=0)
    v_var = np.var(v, axis=0)
    v_std = np.std(v, axis=0)
    ke = 0.5 * v**2
    mean_ke = np.mean(ke, axis=0)
    
    return {'x_mean': x_mean,'x_var': x_var,'x_std': x_std,'v_mean': v_mean,'v_var': v_var,'v_std': v_std,'mean_ke': mean_ke}


def fit_diffusion_coefficient(t, x_var):
    """Extrait coefficient de diffusion par ajustement linéaire"""
    t_fit = t[len(t)//2:]
    var_fit = x_var[len(x_var)//2:]
    coeffs = np.polyfit(t_fit, var_fit, 1)
    return coeffs[0], coeffs[1]


def run_free_particle_inertia_analysis(D_values=[0.25, 0.5, 1.0], n_trajectories=10000, n_steps=1000, dt=0.01,m=1.0, gamma=1.0):
    """Étude 1: Particule libre avec inertie"""
    print("\n" + "="*70)
    print("Particule libre avec inertie")
    print("="*70)
    
    results = {}
    
    for D in D_values:
        print(f"\nSimulation D = {D}")
        t, x, v = simulate_langevin_with_inertia_free(n_trajectories=n_trajectories,n_steps=n_steps,dt=dt,D=D,m=m,gamma=gamma,seed=42)
        
        stats = compute_trajectory_stats(x, v)
        results[D] = {
            't': t,
            'x': x,
            'v': v,
            'stats': stats
        }
        
        slope, intercept = fit_diffusion_coefficient(t, stats['x_var'])
        D_eff = slope / 2
        
        print(f"  Variance position en t_final: {stats['x_var'][-1]:.4f}")
        print(f"  Écart-type position en t_final: {stats['x_std'][-1]:.4f}")
        print(f"  Variance vitesse (régime stationnaire): {stats['v_var'][-1]:.4f}")
        print(f"  Énergie cinétique moyenne (régime stationnaire): {stats['mean_ke'][-1]:.4f}")
        print(f"  Coefficient diffusion extrait: D_eff = {D_eff:.4f}")
        print(f"  D entré: {D:.4f}")
    
    plot_free_particle_inertia(results)
    
    return results


def run_velocity_statistics_analysis(D_values=[0.25, 0.5, 1.0],n_trajectories=20000,n_steps=2000,dt=0.01,m=1.0,gamma=1.0):
    """Statistique des vitesses et distribution Maxwell-Boltzmann"""
    print("\n" + "="*70)
    print("Statistique des vitesses et Maxwell-Boltzmann")
    print("="*70)
    
    results = {}
    
    for D in D_values:
        print(f"\nAnalyse D = {D}")
        t, x, v = simulate_langevin_with_inertia_free(n_trajectories=n_trajectories,n_steps=n_steps,dt=dt,D=D,m=m,gamma=gamma,seed=42)
        
        v_ss = v[:, n_steps//2:].flatten()
        v_mean = np.mean(v_ss)
        v_std = np.std(v_ss)
        v_var = np.var(v_ss)
        ke_ss = 0.5 * v_ss**2
        mean_ke = np.mean(ke_ss)
        expected_v_var = 2 * D / m
        
        results[D] = {'v_steady_state': v_ss,'v_mean': v_mean,'v_std': v_std,'v_var': v_var,'mean_ke': mean_ke,'expected_v_var': expected_v_var,}
        
        print(f"  Moyenne vitesse (≈0): {v_mean:.6f}")
        print(f"  Écart-type vitesse: {v_std:.4f}")
        print(f"  Variance vitesse: {v_var:.4f}")
        print(f"  Variance vitesse attendue (équipartition): {expected_v_var:.4f}")
        print(f"  Rapport (v_var/attendu): {v_var / expected_v_var:.4f}")
        print(f"  Énergie cinétique moyenne: {mean_ke:.4f}")
        print(f"  Énergie attendue (=D/2): {D/2:.4f}")
    
    plot_velocity_statistics(results)
    
    return results


def run_harmonic_potential_inertia_analysis(D_values=[0.5, 1.0],n_trajectories=10000,n_steps=2000,dt=0.01,k=1.0,m=1.0,gamma=1.0):
    """Potentiel harmonique avec inertie"""
    print("\n" + "="*70)
    print("Potentiel harmonique avec inertie")
    print("="*70)
    
    results = {}
    
    for D in D_values:
        print(f"\nSimulation avec D = {D}, k = {k}")
        t, x, v = simulate_langevin_with_inertia_harmonic(n_trajectories=n_trajectories,n_steps=n_steps,dt=dt,D=D,k=k,m=m,gamma=gamma,seed=42)
        
        stats = compute_trajectory_stats(x, v)
        x_ss = x[:, n_steps//2:].flatten()
        v_ss = v[:, n_steps//2:].flatten()
        expected_x_var = D / k
        
        results[D] = {'t': t,'x': x,'v': v,'x_ss': x_ss,'v_ss': v_ss,'x_var': np.var(x_ss),'v_var': np.var(v_ss),'expected_x_var': expected_x_var,'stats': stats,}

        print(f"  Variance position (régime stationnaire): {np.var(x_ss):.4f}")
        print(f"  Variance position attendue: {expected_x_var:.4f}")
        print(f"  Rapport: {np.var(x_ss) / expected_x_var:.4f}")
        print(f"  Variance vitesse (régime stationnaire): {np.var(v_ss):.4f}")
    
    plot_harmonic_with_inertia(results)
    
    return results


def plot_free_particle_inertia(results):
    """Trace variance position et velocity pour particules libres"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (D, res), color in zip(results.items(), colors):
        t = res['t']
        stats = res['stats']
        
        axes[0, 0].plot(t, stats['x_var'], label=f'D={D}', color=color, linewidth=PLOT_LINEWIDTH)
        axes[0, 1].plot(t, stats['v_var'], label=f'D={D}', color=color, linewidth=PLOT_LINEWIDTH)
        axes[1, 0].plot(t, stats['mean_ke'], label=f'D={D}', color=color, linewidth=PLOT_LINEWIDTH)
    
    axes[0, 0].set_xlabel('Temps', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 0].set_ylabel('Variance position', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 0].set_title('Variance position vs temps (avec inertie)', fontsize=TITLE_FONTSIZE)
    axes[0, 0].legend(fontsize=LEGEND_FONTSIZE)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Temps', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 1].set_ylabel('Variance vitesse', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 1].set_title('Variance vitesse vs temps', fontsize=TITLE_FONTSIZE)
    axes[0, 1].legend(fontsize=LEGEND_FONTSIZE)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temps', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 0].set_ylabel('Énergie cinétique moyenne', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 0].set_title('Énergie cinétique vs temps', fontsize=TITLE_FONTSIZE)
    axes[1, 0].legend(fontsize=LEGEND_FONTSIZE)
    axes[1, 0].grid(True, alpha=0.3)
    
    D_first = list(results.keys())[0]
    x = results[D_first]['x'][:50]
    t = results[D_first]['t']
    for traj in x:
        axes[1, 1].plot(t, traj, alpha=0.3, linewidth=0.5)
    axes[1, 1].set_xlabel('Temps', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 1].set_ylabel('Position', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 1].set_title(f'Trajectoires exemples (D={D_first})', fontsize=TITLE_FONTSIZE)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('langevin_inertia_free_particle.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sauvegardé: langevin_inertia_free_particle.png")


def plot_velocity_statistics(results):
    """Trace distribution vitesses vs Maxwell-Boltzmann"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (D, res), color in zip(results.items(), colors):
        v_ss = res['v_steady_state']
        v_std = res['v_std']
        
        # Histogramme
        axes[0].hist(v_ss, bins=100, alpha=0.5, label=f'D={D}', color=color, density=True)
        
        v_range = np.linspace(-5*v_std, 5*v_std, 1000)
        maxwell_boltzmann = norm.pdf(v_range, loc=0, scale=v_std)
        axes[0].plot(v_range, maxwell_boltzmann, '--', color=color, linewidth=2)
    
    axes[0].set_xlabel('Vitesse', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_ylabel('Densité de probabilité', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0].set_title('Distribution vitesses vs Maxwell-Boltzmann', fontsize=TITLE_FONTSIZE)
    axes[0].legend(fontsize=LEGEND_FONTSIZE)
    axes[0].grid(True, alpha=0.3)
    
    D_vals = list(results.keys())
    measured_vars = [results[D]['v_var'] for D in D_vals]
    expected_vars = [results[D]['expected_v_var'] for D in D_vals]
    
    x_pos = np.arange(len(D_vals))
    width = 0.35
    
    axes[1].bar(x_pos - width/2, measured_vars, width, label='Mesurée', alpha=0.8)
    axes[1].bar(x_pos + width/2, expected_vars, width, label='Théorique (équipartition)', alpha=0.8)
    
    axes[1].set_xlabel('Paramètre D', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_ylabel('Variance vitesse', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1].set_title('Test équipartition énergie', fontsize=TITLE_FONTSIZE)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f'{D}' for D in D_vals])
    axes[1].legend(fontsize=LEGEND_FONTSIZE)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('velocity_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sauvegardé: velocity_statistics.png")


def plot_harmonic_with_inertia(results):
    """Trace position et velocity avec potentiel harmonique"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    for (D, res), color in zip(results.items(), colors):
        axes[0, 0].hist(res['x_ss'], bins=50, alpha=0.5, label=f'D={D}', 
                       color=color, density=True)
        
        axes[0, 1].hist(res['v_ss'], bins=50, alpha=0.5, label=f'D={D}', 
                       color=color, density=True)
        
        axes[1, 0].plot(res['t'], res['stats']['x_var'], label=f'D={D}', 
                       color=color, linewidth=PLOT_LINEWIDTH)
        
        axes[1, 1].plot(res['t'], res['stats']['v_var'], label=f'D={D}', 
                       color=color, linewidth=PLOT_LINEWIDTH)
    
    axes[0, 0].set_xlabel('Position', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 0].set_ylabel('Densité de probabilité', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 0].set_title('Distribution position (potentiel harmonique)', fontsize=TITLE_FONTSIZE)
    axes[0, 0].legend(fontsize=LEGEND_FONTSIZE)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Vitesse', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 1].set_ylabel('Densité de probabilité', fontsize=AXIS_LABEL_FONTSIZE)
    axes[0, 1].set_title('Distribution vitesses', fontsize=TITLE_FONTSIZE)
    axes[0, 1].legend(fontsize=LEGEND_FONTSIZE)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temps', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 0].set_ylabel('Variance position', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 0].set_title('Variance position vs temps', fontsize=TITLE_FONTSIZE)
    axes[1, 0].legend(fontsize=LEGEND_FONTSIZE)
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Temps', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 1].set_ylabel('Variance vitesse', fontsize=AXIS_LABEL_FONTSIZE)
    axes[1, 1].set_title('Variance vitesse vs temps', fontsize=TITLE_FONTSIZE)
    axes[1, 1].legend(fontsize=LEGEND_FONTSIZE)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('harmonic_potential_inertia.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Sauvegardé: harmonic_potential_inertia.png")
