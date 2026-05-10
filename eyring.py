import numpy as np
import matplotlib.pyplot as plt

from force_exterieure import force_double_well


def first_passage_time(
	dt,
	D,
	gamma,
	a,
	b,
	x_start=-1.2,
	x_threshold=0.0,
	max_steps=500000,
	rng=None,
):
	"""Simule un temps de premier passage dans un double puits."""
	if rng is None:
		rng = np.random.default_rng()

	x = x_start
	noise_scale = np.sqrt(2.0 * D * dt)

	for step in range(max_steps):
		force = force_double_well(np.array([x]), a, b)[0]
		x = x + (force / gamma) * dt + noise_scale * rng.standard_normal()
		if x >= x_threshold:
			return (step + 1) * dt

	return np.nan


def mean_passage_time(n_traj, dt, D, gamma, a, b, seed=0):
	"""Retourne le temps moyen de premier passage sur n_traj trajectoires."""
	rng = np.random.default_rng(seed)
	times = [first_passage_time(dt, D, gamma, a, b, rng=rng) for _ in range(n_traj)]
	times = np.array(times, dtype=float)

	n_valid = np.sum(~np.isnan(times))
	t_mean = np.nanmean(times)
	print(
		f"  D={D:.3f}, a={a:.3f} | {n_valid}/{n_traj} ont traverse "
		f"({100 * n_valid / n_traj:.0f}%) | <t> = {t_mean:.2f}"
	)
	return t_mean


def run_eyring_analysis(gamma=1.0, b=1.2, n_traj=250, dt=0.01, max_steps=100000):
	"""Vue d'ensemble: temps moyens de premier passage pour quelques valeurs."""
	D_fixe = 0.3
	a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
	delta_v = a_values * b**4

	print("=== Variation de la barriere (D fixe) ===")
	mean_times = []
	for index, a in enumerate(a_values):
		mean_times.append(
			mean_passage_time(n_traj, dt, D_fixe, gamma, a, b, seed=index)
		)

	mean_times = np.array(mean_times)
	log_t = np.log(mean_times)
	slope, intercept = np.polyfit(delta_v, log_t, 1)

	print(f"\n  Pente mesuree : {slope:.3f}")
	print(f"  Pente theorique 1/(D*gamma) = {1 / (D_fixe * gamma):.3f}")

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.plot(delta_v, log_t, "o", ms=7, label="Simulation")
	ax.plot(delta_v, slope * delta_v + intercept, "--", lw=2, label=f"Fit : pente = {slope:.3f}")
	ax.set_xlabel(r"$\Delta V = a b^4$")
	ax.set_ylabel(r"$\ln\langle t \rangle$")
	ax.set_title(r"Loi d'Arrhenius : $\ln\langle t\rangle$ vs $\Delta V$ ($D$ fixe)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.savefig("arrhenius.png", dpi=150)
	plt.show()


def run_eyring_barrier_analysis(gamma=1.0, b=1.2, n_traj=250, dt=0.01, max_steps=100000):
	"""Fait varier la barriere a D fixe."""
	D_fixe = 0.3
	a_values = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
	delta_v = a_values * b**4

	print("\n=== Variation de la barriere (D fixe) ===")
	mean_times = []
	for index, a in enumerate(a_values):
		mean_times.append(
			mean_passage_time(n_traj, dt, D_fixe, gamma, a, b, seed=index)
		)

	mean_times = np.array(mean_times)
	log_t = np.log(mean_times)
	slope, intercept = np.polyfit(delta_v, log_t, 1)

	print(f"\n  Pente mesuree : {slope:.3f}")
	print(f"  Pente theorique 1/(D*gamma) = {1 / (D_fixe * gamma):.3f}")

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.plot(delta_v, log_t, "o", ms=7, label="Simulation")
	ax.plot(delta_v, slope * delta_v + intercept, "--", lw=2, label=f"Fit : pente = {slope:.3f}")
	ax.set_xlabel(r"$\Delta V = a b^4$")
	ax.set_ylabel(r"$\ln\langle t \rangle$")
	ax.set_title(r"Loi d'Arrhenius : $\ln\langle t\rangle$ vs $\Delta V$ ($D$ fixe)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.savefig("arrhenius_barrier.png", dpi=150)
	plt.show()


def run_eyring_D_analysis(a=0.4, gamma=1.0, b=1.2, n_traj=300, dt=0.01, max_steps=100000):
	"""Fait varier D a barriere fixe."""
	D_values = np.array([0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
	delta_v_fixed = a * b**4

	print("\n=== Variation de D (barriere fixe) ===")
	mean_times = []
	for index, D in enumerate(D_values):
		print(f"  Simulation en cours pour D={D:.3f}...")
		mean_times.append(
			mean_passage_time(n_traj, dt, D, gamma, a, b, seed=index + 20)
		)

	mean_times = np.array(mean_times)
	log_t = np.log(mean_times)
	inv_D = 1.0 / D_values
	slope, intercept = np.polyfit(inv_D, log_t, 1)

	print(f"\n  Pente mesuree : {slope:.3f}")
	print(f"  Pente theorique DeltaV/gamma = {delta_v_fixed / gamma:.3f}")

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.plot(inv_D, log_t, "o", ms=7, label="Simulation")
	ax.plot(inv_D, slope * inv_D + intercept, "--", lw=2, label=f"Fit : pente = {slope:.3f}")
	ax.set_xlabel(r"$1/D$")
	ax.set_ylabel(r"$\ln\langle t \rangle$")
	ax.set_title(r"Loi d'Arrhenius : $\ln\langle t\rangle$ vs $1/D$ ($\Delta V$ fixe)")
	ax.grid(True, alpha=0.3)
	ax.legend()
	plt.tight_layout()
	plt.savefig("arrhenius_D.png", dpi=150)
	plt.show()


if __name__ == "__main__":
	run_eyring_analysis()