import numpy as np
import matplotlib.pyplot as plt


def simulate_harmonic_potential(n_trajectories, n_steps, dt, D, k, gamma, seed=0):
	"""Simule x(t) pour une particule soumise a un potentiel harmonique en 1D."""
	rng = np.random.default_rng(seed)
	x = np.zeros((n_trajectories, n_steps + 1))
	noise_scale = np.sqrt(2.0 * D * dt)

	for i in range(n_steps):
		noise = noise_scale * rng.standard_normal(n_trajectories)
		drift = -(k / gamma) * x[:, i] * dt
		x[:, i + 1] = x[:, i] + drift + noise

	t = np.arange(n_steps + 1) * dt
	return t, x


def analyze_harmonic_cases(D_values, gamma_values, n_trajectories, n_steps, dt, k, seed=0):
	harmonic_results = {}
	for D in D_values:
		for gamma in gamma_values:
			t, x = simulate_harmonic_potential(
				n_trajectories=n_trajectories,
				n_steps=n_steps,
				dt=dt,
				D=D,
				k=k,
				gamma=gamma,
				seed=seed,
			)
			x_final = x[:, -1]
			harmonic_results[(D, gamma)] = {
				"t": t,
				"var_t": np.var(x, axis=0),
				"x_final": x_final,
				"mean_final": np.mean(x_final),
				"var_final": np.var(x_final),
			}
	return harmonic_results


def plot_harmonic_variances(harmonic_results, k):
	fig, ax = plt.subplots(figsize=(9, 5))

	for (D, gamma), res in harmonic_results.items():
		t = res["t"]
		var_t = res["var_t"]
		var_theory = (D * gamma / k) * (1 - np.exp(-2 * k * t / gamma))

		ax.plot(t, var_t, label=f"num D={D}, gamma={gamma}")
		ax.plot(t, var_theory, "--", alpha=0.75, label=f"th D={D}, gamma={gamma}")

	ax.set_title("Potentiel harmonique: variances temporelles")
	ax.set_xlabel("Temps t")
	ax.set_ylabel("Var[x(t)]")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8, ncol=2)
	plt.tight_layout()
	plt.show()


def plot_harmonic_gaussians(harmonic_results, bins=80):
	fig, ax = plt.subplots(figsize=(9, 5))

	for (D, gamma), res in harmonic_results.items():
		x_final = res["x_final"]
		mean_sim = res["mean_final"]
		var_sim = max(res["var_final"], 1e-12)

		ax.hist(
			x_final,
			bins=bins,
			density=True,
			histtype="step",
			lw=1.3,
			alpha=0.9,
			label=f"hist D={D}, gamma={gamma}",
		)

		x_grid = np.linspace(x_final.min(), x_final.max(), 400)
		pdf_fit = (1.0 / np.sqrt(2 * np.pi * var_sim)) * np.exp(-((x_grid - mean_sim) ** 2) / (2 * var_sim))
		ax.plot(x_grid, pdf_fit, lw=2, label=f"gauss D={D}, gamma={gamma}")

	ax.set_title("Potentiel harmonique: distributions finales et fits gaussiens")
	ax.set_xlabel("Position x")
	ax.set_ylabel("P(x)")
	ax.grid(True, alpha=0.3)
	ax.legend(fontsize=8, ncol=2)
	plt.tight_layout()
	plt.show()


def simulate_with_force(n_trajectories, n_steps, dt, D, gamma, force_fn, x0=0.0, seed=0, reflect_min=None):
	"""Euler pour dx = (f(x)/gamma) dt + sqrt(2D) dW. Option de reflet a reflect_min"""
	rng = np.random.default_rng(seed)
	x = np.zeros((n_trajectories, n_steps + 1))
	x[:, 0] = x0
	noise_scale = np.sqrt(2.0 * D * dt)

	for i in range(n_steps):
		noise = noise_scale * rng.standard_normal(n_trajectories)
		drift = (force_fn(x[:, i]) / gamma) * dt
		x_next = x[:, i] + drift + noise

		if reflect_min is not None:
			mask = x_next < reflect_min
			x_next[mask] = 2.0 * reflect_min - x_next[mask]

		x[:, i + 1] = x_next

	t = np.arange(n_steps + 1) * dt
	return t, x


def force_gravity(x, mg):
	return -mg * np.ones_like(x)


def potential_gravity(x, mg):
	return mg * x


def force_single_well(x, a):
	return -4.0 * a * x**3


def potential_single_well(x, a):
	return a * x**4


def force_double_well(x, a, b):
	return -4.0 * a * x * (x**2 - b**2)


def potential_double_well(x, a, b):
	return a * (x**2 - b**2) ** 2


def plot_distribution_evolution(t, x, title, time_indices=None, bins=90):
	if time_indices is None:
		time_indices = [len(t) // 10, len(t) // 3, 2 * len(t) // 3, len(t) - 1]

	fig, ax = plt.subplots(figsize=(9, 5))
	for idx in time_indices:
		xs = x[:, idx]
		ax.hist(xs, bins=bins, density=True, histtype="step", lw=1.8, alpha=0.9, label=f"t={t[idx]:.2f}")

	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("P(x,t)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.show()


def boltzmann_pdf_on_grid(x_grid, potential_fn, gamma, D):
	energy = potential_fn(x_grid)
	weight = np.exp(-(energy - np.min(energy)) / (gamma * D))
	norm = np.trapz(weight, x_grid)
	if norm <= 0:
		return np.zeros_like(x_grid)
	return weight / norm


def plot_stationary_vs_boltzmann(x_final, potential_fn, D, gamma, title, bins=90, x_min=None):
	x_left = np.percentile(x_final, 0.5) if x_min is None else x_min
	x_right = np.percentile(x_final, 99.5)
	x_grid = np.linspace(x_left, x_right, 600)

	p_boltz = boltzmann_pdf_on_grid(x_grid, potential_fn, gamma, D)
	hist, edges = np.histogram(x_final, bins=bins, density=True, range=(x_left, x_right))
	centers = 0.5 * (edges[:-1] + edges[1:])
	p_boltz_centers = np.interp(centers, x_grid, p_boltz)

	dx = centers[1] - centers[0] if len(centers) > 1 else 1.0
	l2_error = np.sqrt(np.sum((hist - p_boltz_centers) ** 2) * dx)

	fig, ax = plt.subplots(figsize=(9, 5))
	ax.hist(x_final, bins=bins, density=True, alpha=0.6, label="Simulation stationnaire")
	ax.plot(x_grid, p_boltz, "r-", lw=2.2, label="Boltzmann predite")
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.set_ylabel("P_st(x)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.show()

	return l2_error


def run_external_force_analysis(D_values, n_trajectories=20000, n_steps=1000, dt=0.01):
	k = 1.0
	gamma_values = [1.0, 2.0]
	harmonic_results = analyze_harmonic_cases(
		D_values=D_values,
		gamma_values=gamma_values,
		n_trajectories=n_trajectories,
		n_steps=n_steps,
		dt=dt,
		k=k,
		seed=42,
	)
	plot_harmonic_variances(harmonic_results, k)
	plot_harmonic_gaussians(harmonic_results, bins=80)

	for (D, gamma), res in harmonic_results.items():
		print(f"Resultat experimental -> D={D}, gamma={gamma} | Variance mesuree = {res['var_final']:.4f}")

	print("\n=== Evolution de P(x,t) sous force exterieure et test de Boltzmann ===")
	D_force = 0.5
	gamma_force = 1.0
	n_steps_force = 4000
	mg = 1.0
	a_single = 0.25
	a_double = 0.4
	b_double = 1.2

	t_g, x_g = simulate_with_force(
		n_trajectories=n_trajectories,
		n_steps=n_steps_force,
		dt=dt,
		D=D_force,
		gamma=gamma_force,
		force_fn=lambda x: force_gravity(x, mg),
		x0=2.0,
		seed=7,
		reflect_min=0.0,
	)
	plot_distribution_evolution(t_g, x_g, "Pesanteur + fond reflechissant: evolution de P(z,t)")
	err_g = plot_stationary_vs_boltzmann(
		x_final=x_g[:, -1],
		potential_fn=lambda z: potential_gravity(z, mg),
		D=D_force,
		gamma=gamma_force,
		title="Pesanteur: stationnaire simulee vs Boltzmann",
		x_min=0.0,
	)
	lambda_th = gamma_force * D_force / mg
	mean_z = np.mean(x_g[:, -1])
	print(f"Pesanteur -> erreur L2(sim, Boltzmann) = {err_g:.4e}")
	print(f"Pesanteur -> longueur caracteristique theorique lambda = gamma*D/mg = {lambda_th:.4f}")
	print(f"Pesanteur -> moyenne stationnaire simulee <z> = {mean_z:.4f} (attendu ~ lambda)")

	t_s, x_s = simulate_with_force(
		n_trajectories=n_trajectories,
		n_steps=n_steps_force,
		dt=dt,
		D=D_force,
		gamma=gamma_force,
		force_fn=lambda x: force_single_well(x, a_single),
		x0=0.1,
		seed=11,
	)
	plot_distribution_evolution(t_s, x_s, "Puits simple V(x)=a x^4: evolution de P(x,t)")
	err_s = plot_stationary_vs_boltzmann(
		x_final=x_s[:, -1],
		potential_fn=lambda x: potential_single_well(x, a_single),
		D=D_force,
		gamma=gamma_force,
		title="Puits simple: stationnaire simulee vs Boltzmann",
	)
	print(f"Puits simple -> erreur L2(sim, Boltzmann) = {err_s:.4e}")

	t_d, x_d = simulate_with_force(
		n_trajectories=n_trajectories,
		n_steps=n_steps_force,
		dt=dt,
		D=D_force,
		gamma=gamma_force,
		force_fn=lambda x: force_double_well(x, a_double, b_double),
		x0=0.0,
		seed=19,
	)
	plot_distribution_evolution(t_d, x_d, "Double puits: evolution de P(x,t)")
	err_d = plot_stationary_vs_boltzmann(
		x_final=x_d[:, -1],
		potential_fn=lambda x: potential_double_well(x, a_double, b_double),
		D=D_force,
		gamma=gamma_force,
		title="Double puits: stationnaire simulee vs Boltzmann",
	)
	print(f"Double puits -> erreur L2(sim, Boltzmann) = {err_d:.4e}")
