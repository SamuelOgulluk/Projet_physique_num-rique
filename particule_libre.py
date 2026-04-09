import numpy as np
import matplotlib.pyplot as plt


def simulate_free_particle(n_trajectories, n_steps, dt, D, seed=0):
	"""Simule x(t) pour une particule libre en 1D: dx = sqrt(2D) dW."""
	rng = np.random.default_rng(seed)
	increments = np.sqrt(2.0 * D * dt) * rng.standard_normal((n_trajectories, n_steps))
	x = np.zeros((n_trajectories, n_steps + 1))
	x[:, 1:] = np.cumsum(increments, axis=1)
	t = np.arange(n_steps + 1) * dt
	return t, x


def compute_stats(x: np.ndarray):
	mean_t = np.mean(x, axis=0)
	var_t = np.var(x, axis=0)
	std_t = np.std(x, axis=0)
	return mean_t, var_t, std_t


def analyze_D_values(D_values, n_trajectories=20000, n_steps=1000, dt=0.01):
	all_results = {}
	for D in D_values:
		t, x = simulate_free_particle(n_trajectories, n_steps, dt, D, seed=42)
		mean_t, var_t, std_t = compute_stats(x)
		all_results[D] = {
			"t": t,
			"x": x,
			"mean": mean_t,
			"var": var_t,
			"std": std_t,
		}
	return all_results


def print_summary(results):
	print("=== Resume statistique (particule libre, diffusion 1D) ===")
	for D, res in results.items():
		t = res["t"]
		mean_t = res["mean"]
		var_t = res["var"]

		coeffs = np.polyfit(t, var_t, 1)
		slope, intercept = coeffs[0], coeffs[1]

		print(f"\n D = {D}")
		print(f"  moyenne de x(t_fin) = {mean_t[-1]:.4e} (attendu: 0)")
		print(f"  Var(x(t_fin)) = {var_t[-1]:.4e} (attendu: 2Dt = {2 * D * t[-1]:.4e})")
		print(f"  Pente fit Var(t) = a t + b : a = {slope:.4e}, b = {intercept:.4e}")
		print(f"  Rapport pente/(2D) = {slope / (2 * D):.4f} (attendu: 1)")


def plot_time_statistics(results):
	fig, axes = plt.subplots(1, 3, figsize=(16, 4))
	ax_mean, ax_var, ax_std = axes

	for D, res in results.items():
		t = res["t"]
		mean_t = res["mean"]
		var_t = res["var"]
		std_t = res["std"]

		ax_mean.plot(t, mean_t, label=f"D={D}")
		ax_var.plot(t, var_t, label=f"num D={D}")
		ax_var.plot(t, 2 * D * t, "--", alpha=0.7, label=f"th 2Dt, D={D}")
		ax_std.plot(t, std_t, label=f"num D={D}")
		ax_std.plot(t, np.sqrt(2 * D * t), "--", alpha=0.7, label=f"th sqrt(2Dt), D={D}")

	ax_mean.set_title("Moyenne <x(t)>")
	ax_mean.set_xlabel("t")
	ax_mean.set_ylabel("<x>")
	ax_mean.grid(True, alpha=0.3)
	ax_mean.legend(fontsize=8)

	ax_var.set_title("Variance Var[x(t)]")
	ax_var.set_xlabel("t")
	ax_var.set_ylabel("Var[x]")
	ax_var.grid(True, alpha=0.3)
	ax_var.legend(fontsize=8)

	ax_std.set_title("Ecart-type sigma_x(t)")
	ax_std.set_xlabel("t")
	ax_std.set_ylabel("sigma_x")
	ax_std.grid(True, alpha=0.3)
	ax_std.legend(fontsize=8)

	plt.tight_layout()
	plt.show()


def plot_distribution_at_time(results, t_index=None, bins=80):
	nD = len(results)
	fig, axes = plt.subplots(1, nD, figsize=(6 * nD, 4), squeeze=False)

	for i, (D, res) in enumerate(results.items()):
		t = res["t"]
		x = res["x"]
		idx = len(t) // 2 if t_index is None else t_index

		t0 = t[idx]
		xs = x[:, idx]
		mu = 0.0
		sigma2 = 2 * D * t0

		ax = axes[0, i]
		ax.hist(xs, bins=bins, density=True, alpha=0.65, label="Simulation")

		x_grid = np.linspace(xs.min(), xs.max(), 400)
		pdf = (
			(1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x_grid - mu) ** 2) / (2 * sigma2))
			if sigma2 > 0
			else np.zeros_like(x_grid)
		)
		ax.plot(x_grid, pdf, "r-", lw=2, label="Theorie gaussienne")

		ax.set_title(f"Distribution a t={t0:.2f}, D={D}")
		ax.set_xlabel("x")
		ax.set_ylabel("P(x,t)")
		ax.grid(True, alpha=0.3)
		ax.legend(fontsize=8)

	plt.tight_layout()
	plt.show()


def run_free_particle_analysis(D_values, n_trajectories=20000, n_steps=1000, dt=0.01):
	results = analyze_D_values(D_values, n_trajectories=n_trajectories, n_steps=n_steps, dt=dt)
	print_summary(results)
	plot_time_statistics(results)
	plot_distribution_at_time(results, t_index=800)
