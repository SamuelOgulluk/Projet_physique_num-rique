from particule_libre import run_free_particle_analysis
from force_exterieure import run_external_force_analysis
from eyring import run_eyring_barrier_analysis, run_eyring_D_analysis
from dynamique_2d import run_2d_analysis
from dynamique_3d import run_3d_analysis

def main():
	D_values = [0.25, 0.5, 1.0]
	n_trajectories = 20000
	n_steps = 1000
	dt = 0.01

	"""print("Particule libre")
	run_free_particle_analysis(
		D_values=D_values,
		n_trajectories=n_trajectories,
		n_steps=n_steps,
		dt=dt,
	)

	print("\nForce exterieure")
	run_external_force_analysis(
		D_values=D_values,
		n_trajectories=n_trajectories,
		n_steps=n_steps,
		dt=dt,
	)

	print("\nTemps de franchissement - loi d'Arrhenius")
	run_eyring_barrier_analysis()
	run_eyring_D_analysis()"""

	print("\nDynamique en 2D")
	run_2d_analysis(D=0.5, n_trajectories=10000, n_steps=1000, dt=0.01)

	print("\nDynamique en 3D")
	run_3d_analysis(D=0.5, n_trajectories=10000, n_steps=1000, dt=0.01)

if __name__ == "__main__":
	main()