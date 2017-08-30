import importlib
import collections

from multiprocessing import Pool
import copy

import cProfile
import pstats

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib
#import mpl_toolkits.axes_grid1

import cosmology
import utils

pi = np.pi

def run_sim(config):
    particles = cosmology.ParticleDistribution(**config["N-body config"])
    np.random.seed(config["seed"])
    particles.create_initial_conditions_CDM(utils.interpolated_powerspectrum_from_file("data/ps_lin_z=0.txt"), config["a_min"])

    snapshots = {"snapshots" : cosmology.ParticleSnapshots(particles,
                                                        **config["snapshot config"])}
            
    l = np.linspace(config["a_min"]**particles.beta, config["a_max"]**particles.beta, config["n_timestep"], endpoint=True)
    snapshot_l = np.concatenate((l[0:1], 1/(1+config["snapshot z"])**particles.beta))
    l, snapshots["timestep_idx"] = cosmology.insert_snapshot_timesteps(l, snapshot_l)
    particles.run(l, [snapshots], verbose=config["verbose"])
    #cProfile.run("""particles.run(l, [snapshots], verbose=False)""", "output/profiling_stats")

if __name__ == "__main__":
    n_particle = 10000000
    L = 5000.0
    n_grid = n_particle//10
    n_timestep = 500
    a_min = 0.015
    a_max = 1.0

    snapshot_z = np.array([4, 3, 2, 1, 0])
    n_snapshot = snapshot_z.size + 1 

    downsample_density = 1#n_grid//1000
    downsample_particles = 1#n_particle//50000

    k_min = 2*pi/L
    k_max = pi/L*n_grid
    n_k_bin = None

    Omega_m = 0.3
    Omega_L = 1.0-Omega_m
    beta = 0.5

    config_direct_LCDM_dl = {"n_particle" : n_particle, "n_grid" : n_grid, "box_size" : L,
                            "integrator" : cosmology.LEAPFROG_INTEGRATOR,
                            "time_unit" : cosmology.DLAMBDA_TIMESTEPS,
                            "leapfrog_type" : cosmology.DKD_LEAPFROG,
                            "force_calculation" : cosmology.DIRECT_FORCE_CALCULATION,
                            "pm_type" : cosmology.FFT_PM,
                            "mesh_interpolation" : cosmology.CIC_MESH_INTERPOLATION,
                            "leapfrog_kick_type" : cosmology.FLRW_KICK,
                            "density_scaling" : cosmology.DENSITY_SCALING_3D,
                            "h" : 0.7, "Omega_m" : Omega_m, "Omega_L" : Omega_L,
                            "beta" : beta}

    snapshot_config = dict(run_id="",
                        n_snapshot=n_snapshot,
                        phase_space=True, particle_downsampling=downsample_particles,
                        density=True, density_downsampling=downsample_density,
                        power_spectrum=True, n_k_bin=n_k_bin, k_min=k_min, k_max=k_max, logspaced=True, linear_powerspectrum=True,
                        phases=False,
                        halos=False,
                        keep_in_memory=False,
                        output_directory="/data2/tilman/1dcosmo/output/snapshots/",
                        phase_space_filename_format="run{run_id}_L{L:.0f}_n{n_particle:.1e}_z{z:.3f}_phase_space",
                        density_filename_format="run{run_id}_L{L:.0f}_n{n_particle:.1e}_z{z:.3f}_density",
                        power_spectrum_filename_format="run{run_id}_L{L:.0f}_n{n_particle:.1e}_z{z:.3f}_power_spectrum",
                        file_format="npz")

    n_sims = 100
    start_sim_idx = 100
    n_cpu = 6

    np.random.seed(42)
    seeds = np.unique(np.random.randint(low=634980, high=983245714, size=100000))

    configs = []
    for i in range(start_sim_idx, start_sim_idx+n_sims):
        ss_config = copy.deepcopy(snapshot_config)
        ss_config["run_id"] = "{}".format(seeds[i])
        configs.append({"seed" : seeds[i], 
                        "N-body config" : config_direct_LCDM_dl, 
                        "snapshot config" : ss_config,
                        "a_min" : a_min,
                        "a_max" : a_max,
                        "n_timestep" : n_timestep,
                        "snapshot z" : snapshot_z,
                        "verbose" : True})

    print("Running {} simulations.".format(n_sims))
    print("Using {} CPUs.".format(n_cpu))

    pool =  Pool(n_cpu, maxtasksperchild=1)
    pool.imap_unordered(run_sim, configs)
    pool.close()
    pool.join()

