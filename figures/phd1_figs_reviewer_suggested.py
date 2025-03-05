# %% Import, global variables, etc
import sys
sys.path.append("..")
sys.path.append(".")
#sys.path.append("..\simulations")


from dMRItools import waveforms
from dMRItools import simulate_pulse_sequences
import numpy as np
import os
import matplotlib.pyplot as plt

from dMRItools import bval_calc_tools

from fitting_algorithms import ivim_fit_method_biexp
from dipy.core.gradients import gradient_table

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

simulation_files_folder = r"..\simulation_data\figs_reviewer_suggested"
folder_best_case = "optimal_allCrushers_onlyCrushWhenNeeded_sequence"
folder_worst_case = "allCrushers_sequence"

path_best = os.path.join(simulation_files_folder, folder_best_case)
path_worst = os.path.join(simulation_files_folder, folder_worst_case)

lw = 1
alpha_area = 0.3
alpha_lines = 1

# %% Functions

def read_simulation_files_and_average(path_to_folder, correction="uncorrected", angles="xyz", xyres=[1e-3], zres=[1e-3]):
    bvalues_actual_vs_resolution = []

    # Loop over all the voxel sizes
    for idx in range(len(xyres)):
        # Build the filename string
        fname = f"{correction}_{angles}_xy{xyres[idx]}_z{zres[idx]}"

        filename_bvalues_actual = fname + "_bvalues_actual.npy"
        filename_bvalues_nominal = fname + "_bvalues_nominal.npy"
        filename_rotation_angles = fname + "_uvecs.npy"

        # Load the arrays
        bvalues_actual = np.load(os.path.join(path_to_folder, filename_bvalues_actual))
        bvalues_nominal = np.load(os.path.join(path_to_folder, filename_bvalues_nominal))
        rotation_angles = np.load(os.path.join(path_to_folder, filename_rotation_angles))

        bvalues_actual_vs_resolution.append(bvalues_actual)

    bvalues_actual_vs_resolution = np.asarray(bvalues_actual_vs_resolution)
    return bvalues_nominal, bvalues_actual_vs_resolution.T

# %% b0 vs isotropic resolution

resolutions = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3])

bvalues_nominal_worst, bvalues_actual_worst = read_simulation_files_and_average(path_worst, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)
bvalues_nominal_best, bvalues_actual_best = read_simulation_files_and_average(path_best, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)

# Get the b0's for each isotropic resolution
# b0 is independent of diffusion direction so we only show for one of them
#b0_actual_xy_worst = bvalues_actual_worst[0, -1, :]
#b0_actual_xy_best = bvalues_actual_best[0, -1, :]

b0_actual_z_worst = bvalues_actual_worst[-1, -1, :]
b0_actual_z_best = bvalues_actual_best[-1, -1, :]

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(resolutions*1e3, b0_actual_z_worst, label="Large cross-terms", color="black", ls="-", lw=lw)
ax.plot(resolutions*1e3, b0_actual_z_best, label="Minimal cross-terms", color="black", ls="--", lw=lw)

ax.set_xlim(1.0, 4.0)
ax.set_ylim(0)

ax.set_xlabel("Isotropic resolution [mm]")
ax.set_ylabel("Actual b0 [s/mm$^2$]")

ax.legend(frameon=False)


# %%
