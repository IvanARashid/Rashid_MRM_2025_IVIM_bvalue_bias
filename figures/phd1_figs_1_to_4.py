# %% Imports, etc

import sys
sys.path.append("..")
sys.path.append("..\simulations")


from dMRItools import waveforms
from dMRItools import simulate_pulse_sequences
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches
import mpl_toolkits.mplot3d

from dMRItools import bval_calc_tools

from fitting_algorithms import ivim_fit_method_biexp
from dipy.core.gradients import gradient_table

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13


nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

save_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations"
fig_save_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\figures"

folder_best_case = "optimal_allCrushers_onlyCrushWhenNeeded_sequence"
folder_worst_case = "allCrushers_sequence"

path_best = os.path.join(save_path, folder_best_case)
path_worst = os.path.join(save_path, folder_worst_case)


lw = 1
alpha_area = 0.3
alpha_lines = 1

# %% Functions

class Arrow3D(matplotlib.patches.FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        matplotlib.patches.FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        self.do_3d_projection = self.draw

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = mpl_toolkits.mplot3d.proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        matplotlib.patches.FancyArrowPatch.draw(self, renderer)

def read_simulation_log_old(filename, nominal_bvalues=nominal_bvalues, ax=False, alpha=0.2, label=None, color="red"):
    # Read the file
    worst_bvalues, worst_angles, bvals_nom = bval_calc_tools.find_worst_case_waveforms(save_path, filename, plot=False)
    #print(worst_bvalues[:,:])

    # Get the b-value arrays
    #lower_b = np.hstack((worst_bvalues[0][-1], worst_bvalues[0][0:-1]))
    #upper_b = np.hstack((worst_bvalues[1][-1], worst_bvalues[1][0:-1]))
    #nominal_bvalues = np.array(nominal_bvalues)
    #nominal_b = np.hstack((nominal_bvalues[-1], nominal_bvalues[0:-1]))

    lower_b = np.flip(worst_bvalues[0])
    upper_b = np.flip(worst_bvalues[1])
    nominal_bvalues = np.array(nominal_bvalues)
    nominal_b = np.flip(nominal_bvalues)

    # Simulate signals
    Dstar = 0.03
    signals_lower = bval_calc_tools.ivim_signal(lower_b, Dstar=Dstar)
    signals_upper = bval_calc_tools.ivim_signal(upper_b, Dstar=Dstar)

    if ax:
        ax.plot(nominal_b, signals_lower, ls="")
        ax.plot(nominal_b, signals_upper, ls="")
        ax.fill_between(x=nominal_b, y1=signals_upper, y2=signals_lower, alpha=alpha, color=color, label=label)

    return signals_lower, signals_upper

def read_simulation_log(filename, nominal_bvalues=nominal_bvalues, ax=False, alpha=0.2, label=None, color="red"):
    # Read the file
    worst_bvalues, worst_angles, bvals_nom = simulate_pulse_sequences.find_worst_case_waveforms_uvecs(save_path, filename, plot=False)
    #print(worst_bvalues[:,:])

    # Get the b-value arrays
    #lower_b = np.hstack((worst_bvalues[0][-1], worst_bvalues[0][0:-1]))
    #upper_b = np.hstack((worst_bvalues[1][-1], worst_bvalues[1][0:-1]))
    #nominal_bvalues = np.array(nominal_bvalues)
    #nominal_b = np.hstack((nominal_bvalues[-1], nominal_bvalues[0:-1]))

    lower_b = np.flip(worst_bvalues[0])
    upper_b = np.flip(worst_bvalues[1])
    nominal_bvalues = np.array(nominal_bvalues)
    nominal_b = np.flip(nominal_bvalues)

    # Simulate signals
    Dstar = 0.02
    signals_lower = simulate_pulse_sequences.ivim_signal(lower_b, Dstar=Dstar)
    signals_upper = simulate_pulse_sequences.ivim_signal(upper_b, Dstar=Dstar)

    if ax:
        ax.plot(nominal_b, signals_lower, ls="")
        ax.plot(nominal_b, signals_upper, ls="")
        ax.fill_between(x=nominal_b, y1=signals_upper, y2=signals_lower, alpha=alpha, color=color, label=label)

    return signals_lower, signals_upper

def read_all_generic_simulation_logs(path, correction="uncorrected", angles="ndir1000", xyres=[1e-3], zres=[1e-3], nom_b0=False, return_angles=False):
    """
    Reads simulation logs for specified simulation run. 
    
    Provide xy-res and z-res as lists of voxel sizes in SI units. These should be of equal length, i.e. one entry per file.

    Worst cases are found in each file. Parameter estimates are performed.

    Function returns array of worst-case parameter estimates.
    """
    f = np.zeros((len(xyres), 3))
    Dstar = np.zeros((len(xyres), 3))
    D = np.zeros((len(xyres), 3))
    minimum_bvalues = np.zeros((len(xyres), 4))
    maximum_bvalues = np.zeros((len(xyres), 4))

    # Loop over all the voxel sizes
    for idx in range(len(xyres)):
        # Build the filename string
        fname = f"{correction}_{angles}_xy{xyres[idx]}_z{zres[idx]}"
        
        # Get the worst bvalues, worst angles, and nominal bvalues
        worst_bvalues, worst_angles, bvals_nom = simulate_pulse_sequences.find_worst_case_waveforms(path, fname, plot=False)

        # Get the minimum and maximum b50, 200, and 800
        minimum_bvalues[idx, 0] = worst_bvalues[0, np.where(bvals_nom==0)[0][0]]
        minimum_bvalues[idx, 1] = worst_bvalues[0, np.where(bvals_nom==50)[0][0]]
        minimum_bvalues[idx, 2] = worst_bvalues[0, np.where(bvals_nom==200)[0][0]]
        minimum_bvalues[idx, 3] = worst_bvalues[0, np.where(bvals_nom==800)[0][0]]

        maximum_bvalues[idx, 0] = worst_bvalues[1, np.where(bvals_nom==0)[0][0]]
        maximum_bvalues[idx, 1] = worst_bvalues[1, np.where(bvals_nom==50)[0][0]]
        maximum_bvalues[idx, 2] = worst_bvalues[1, np.where(bvals_nom==200)[0][0]]
        maximum_bvalues[idx, 3] = worst_bvalues[1, np.where(bvals_nom==800)[0][0]]


        # Calculate the signals 
        signals_nominal = simulate_pulse_sequences.ivim_signal(bvals_nom, f=0.1, Dstar=20e-3, D=1e-3)
        signals_lower = simulate_pulse_sequences.ivim_signal(worst_bvalues[0,:], f=0.1, Dstar=20e-3, D=1e-3)
        signals_upper = simulate_pulse_sequences.ivim_signal(worst_bvalues[1,:], f=0.1, Dstar=20e-3, D=1e-3)

        ### Perform parameter estimations
        # Construct gradient table
        bvec = np.zeros((bvals_nom.size, 3))
        bvec[:,2] = 1
        factor = 1000
        gtab_nominal = gradient_table(bvals_nom/factor, bvec, b0_threshold=0)
        model_nominal = ivim_fit_method_biexp.IvimModelBiExp(gtab_nominal, rescale_units=True)

        # Perform parameter estimation
        estimates_nominal = model_nominal.fit(signals_nominal)
        estimates_lower = model_nominal.fit(signals_lower)
        estimates_upper = model_nominal.fit(signals_upper)

        f[idx, 0] = estimates_lower.perfusion_fraction
        f[idx, 1] = estimates_nominal.perfusion_fraction
        f[idx, 2] = estimates_upper.perfusion_fraction

        Dstar[idx, 0] = estimates_lower.D_star
        Dstar[idx, 1] = estimates_nominal.D_star
        Dstar[idx, 2] = estimates_upper.D_star

        D[idx, 0] = estimates_lower.D
        D[idx, 1] = estimates_nominal.D
        D[idx, 2] = estimates_upper.D

    if return_angles:
        return f, Dstar, D, np.flip(bvals_nom), minimum_bvalues, maximum_bvalues, worst_angles
    else:
        return f, Dstar, D, np.flip(bvals_nom), minimum_bvalues, maximum_bvalues

def read_all_generic_simulation_logs_uvecs(path, correction="uncorrected", angles="ndir1000", xyres=[1e-3], zres=[1e-3], nom_b0=False, return_angles=False):
    """
    Reads simulation logs for specified simulation run. 
    
    Provide xy-res and z-res as lists of voxel sizes in SI units. These should be of equal length, i.e. one entry per file.

    Worst cases are found in each file. Parameter estimates are performed.

    Function returns array of worst-case parameter estimates.
    """
    f = np.zeros((len(xyres), 3))
    Dstar = np.zeros((len(xyres), 3))
    D = np.zeros((len(xyres), 3))
    minimum_bvalues = np.zeros((len(xyres), 4))
    maximum_bvalues = np.zeros((len(xyres), 4))

    # Loop over all the voxel sizes
    for idx in range(len(xyres)):
        # Build the filename string
        fname = f"{correction}_{angles}_xy{xyres[idx]}_z{zres[idx]}"
        
        # Get the worst bvalues, worst angles, and nominal bvalues
        worst_bvalues, worst_angles, bvals_nom = simulate_pulse_sequences.find_worst_case_waveforms_uvecs(path, fname, plot=False)

        # Get the minimum and maximum b50, 200, and 800
        minimum_bvalues[idx, 0] = worst_bvalues[0, np.where(bvals_nom==0)[0][0]]
        minimum_bvalues[idx, 1] = worst_bvalues[0, np.where(bvals_nom==50)[0][0]]
        minimum_bvalues[idx, 2] = worst_bvalues[0, np.where(bvals_nom==200)[0][0]]
        minimum_bvalues[idx, 3] = worst_bvalues[0, np.where(bvals_nom==800)[0][0]]

        maximum_bvalues[idx, 0] = worst_bvalues[1, np.where(bvals_nom==0)[0][0]]
        maximum_bvalues[idx, 1] = worst_bvalues[1, np.where(bvals_nom==50)[0][0]]
        maximum_bvalues[idx, 2] = worst_bvalues[1, np.where(bvals_nom==200)[0][0]]
        maximum_bvalues[idx, 3] = worst_bvalues[1, np.where(bvals_nom==800)[0][0]]


        # Calculate the signals 
        signals_nominal = simulate_pulse_sequences.ivim_signal(bvals_nom, f=0.1, Dstar=20e-3, D=1e-3)
        signals_lower = simulate_pulse_sequences.ivim_signal(worst_bvalues[0,:], f=0.1, Dstar=20e-3, D=1e-3)
        signals_upper = simulate_pulse_sequences.ivim_signal(worst_bvalues[1,:], f=0.1, Dstar=20e-3, D=1e-3)

        ### Perform parameter estimations
        # Construct gradient table
        bvec = np.zeros((bvals_nom.size, 3))
        bvec[:,2] = 1
        factor = 1000
        gtab_nominal = gradient_table(bvals_nom/factor, bvec, b0_threshold=0)
        model_nominal = ivim_fit_method_biexp.IvimModelBiExp(gtab_nominal, rescale_units=True)

        # Perform parameter estimation
        estimates_nominal = model_nominal.fit(signals_nominal)
        estimates_lower = model_nominal.fit(signals_lower)
        estimates_upper = model_nominal.fit(signals_upper)

        f[idx, 0] = estimates_lower.perfusion_fraction
        f[idx, 1] = estimates_nominal.perfusion_fraction
        f[idx, 2] = estimates_upper.perfusion_fraction

        Dstar[idx, 0] = estimates_lower.D_star
        Dstar[idx, 1] = estimates_nominal.D_star
        Dstar[idx, 2] = estimates_upper.D_star

        D[idx, 0] = estimates_lower.D
        D[idx, 1] = estimates_nominal.D
        D[idx, 2] = estimates_upper.D

    if return_angles:
        return f, Dstar, D, np.flip(bvals_nom), minimum_bvalues, maximum_bvalues, worst_angles
    else:
        return f, Dstar, D, np.flip(bvals_nom), minimum_bvalues, maximum_bvalues

def powder_average_all_simulation_logs(path, correction="uncorrected", angles="xyz", xyres=[1e-3], zres=[1e-3]):
    f = np.zeros(len(xyres))
    Dstar = np.zeros(len(xyres))
    D = np.zeros(len(xyres))

    for idx in range(len(xyres)):
        # Build the filename string
        fname = f"{correction}_{angles}_xy{xyres[idx]}_z{zres[idx]}"

        # Get powdered average signals using actual bvals
        powder_averaged_signals = bval_calc_tools.powder_average_signals_from_file(os.path.join(path, fname))

        # Get nominal bvals
        bvals_nom = np.load(os.path.join(path, fname+"_bvalues_nominal.npy"))
        bvals_nom = np.flip(bvals_nom)
        bvals_nom = bvals_nom

        ### Perform parameter estimations
        # Construct gradient table
        bvec = np.zeros((bvals_nom.size, 3))
        bvec[:,2] = 1
        factor = 1000
        gtab_nominal = gradient_table(bvals_nom/factor, bvec, b0_threshold=0)
        model_nominal = ivim_fit_method_biexp.IvimModelBiExp(gtab_nominal, rescale_units=True)

        # Perform parameter estimation
        estimates = model_nominal.fit(powder_averaged_signals)

        f[idx] = estimates.perfusion_fraction
        Dstar[idx] = estimates.D_star
        D[idx] = estimates.D

    return f, Dstar, D


def powder_average_all_simulation_logs_new(path, correction="uncorrected", angles="xyz", xyres=[1e-3], zres=[1e-3]):
    f = np.zeros(len(xyres))
    Dstar = np.zeros(len(xyres))
    D = np.zeros(len(xyres))

    for idx in range(len(xyres)):
        # Build the filename string
        reported_fname = f"{correction}_{angles}_xy{xyres[idx]}_z{zres[idx]}_bvalues_actual.npy"
        actual_fname = f"crossterm_corrected_{angles}_xy{xyres[idx]}_z{zres[idx]}"

        # Get true powder averaged signals using actual bvals
        #powder_averaged_signals = bval_calc_tools.powder_average_signals_from_file(os.path.join(path, actual_fname))
        signals = bval_calc_tools.signals_from_file(os.path.join(path, actual_fname))
        signals = signals.flatten()

        # Get reported bvals
        bvals_reported = np.load(os.path.join(path, reported_fname))
        bvals_reported = np.flip(bvals_reported)
        bvals_reported = bvals_reported.flatten()

        ### Perform parameter estimations
        # Construct gradient table
        bvec = np.zeros((bvals_reported.size, 3))
        bvec[:,2] = 1
        factor = 1000
        gtab_nominal = gradient_table(bvals_reported/factor, bvec, b0_threshold=0)
        model_nominal = ivim_fit_method_biexp.IvimModelBiExp(gtab_nominal, rescale_units=True)

        # Perform parameter estimation
        #estimates = model_nominal.fit(powder_averaged_signals)
        estimates = model_nominal.fit(signals)

        f[idx] = estimates.perfusion_fraction
        Dstar[idx] = estimates.D_star
        D[idx] = estimates.D

    return f, Dstar, D





# %% Figure 1


uncorrected_optimal = r"20241101\optimal_allCrushers_onlyCrushWhenNeeded_sequence\uncorrected_froeling_200_xy0.001_z0.001"
#imaging_optimal = r"optimal_allCrushers_onlyCrushWhenNeeded_sequence\imaging_corrected_ndir1000_xy0.001_z0.001"
cross_terms_optimal = r"20241101\optimal_allCrushers_onlyCrushWhenNeeded_sequence\crossterm_corrected_froeling_200_xy0.001_z0.001"

uncorrected_bad = r"20241101\allCrushers_sequence\uncorrected_froeling_200_xy0.001_z0.001"
#imaging_bad = r"allCrushers_sequence\imaging_corrected_ndir1000_xy0.001_z0.001"
cross_terms_bad = r"20241101\allCrushers_sequence\crossterm_corrected_froeling_200_xy0.001_z0.001"

powder_average_xyz_optimal = r"20241101\optimal_allCrushers_onlyCrushWhenNeeded_sequence\crossterm_corrected_xyz_xy0.001_z0.001"
powder_average_xyz_bad = r"20241101\allCrushers_sequence\crossterm_corrected_xyz_xy0.001_z0.001"

powder_average_xyz_antipodal_optimal = r"20241101\optimal_allCrushers_onlyCrushWhenNeeded_sequence\crossterm_corrected_xyz_antipodal_xy0.001_z0.001"
powder_average_xyz_antipodal_bad = r"20241101\allCrushers_sequence\crossterm_corrected_xyz_antipodal_xy0.001_z0.001"

#powder_average_xyz_antipodal_optimal = r"optimal_allCrushers_onlyCrushWhenNeeded_sequence\uncorrected_xyz_antipodal_xy0.001_z0.001"
#powder_average_xyz_antipodal_bad = r"allCrushers_sequence\uncorrected_xyz_antipodal_xy0.001_z0.001"

#powder_average_xy_z_optimal = r"optimal_allCrushers_onlyCrushWhenNeeded_sequence\uncorrected_xy-z_xy0.001_z0.001"
#powder_average_xy_z_bad = r"allCrushers_sequence\uncorrected_xy-z_xy0.001_z0.001"

#powder_average_filip_optimal = r"optimal_allCrushers_onlyCrushWhenNeeded_sequence\uncorrected_uvec_elstat_6_filip_xy0.001_z0.001"
#powder_average_filip_bad = r"allCrushers_sequence\uncorrected_uvec_elstat_6_filip_xy0.001_z0.001"

#powder_average_GE_6_optimal = r"optimal_allCrushers_onlyCrushWhenNeeded_sequence\uncorrected_GE_6_xy0.001_z0.001"
#powder_average_GE_6_bad = r"allCrushers_sequence\uncorrected_GE_6_xy0.001_z0.001"

#powder_average_GE_16_optimal = r"optimal_allCrushers_onlyCrushWhenNeeded_sequence\uncorrected_GE_16_xy0.001_z0.001"
#powder_average_GE_16_bad = r"allCrushers_sequence\uncorrected_GE_16_xy0.001_z0.001"

#worst_bvalues, worst_angles = simulate_pulse_sequences.find_worst_case_waveforms(save_path, file_imaging, plot=False)
#lower_b = np.hstack((worst_bvalues[0][-1], worst_bvalues[0][0:-1]))
#upper_b = np.hstack((worst_bvalues[1][-1], worst_bvalues[1][0:-1]))
#nominal_bvalues = np.array(nominal_bvalues)
#nominal_b = np.hstack((nominal_bvalues[-1], nominal_bvalues[0:-1]))

#Dstar = 0.03
#signals_lower = simulate_pulse_sequences.ivim_signal(lower_b, Dstar=Dstar)
#signals_upper = simulate_pulse_sequences.ivim_signal(upper_b, Dstar=Dstar)
#signals_nominal = simulate_pulse_sequences.ivim_signal(nominal_b, Dstar=Dstar)
#nominal_b = np.hstack((nominal_bvalues[-1], nominal_bvalues[0:-1]))
nominal_b = np.flip(np.array(nominal_bvalues))

fig, axs = plt.subplots(nrows=2, ncols=2, sharey=False, figsize=(9,5), height_ratios=[1,3])
lw = 1

def fig1_subplot(uncorrected_fname, fig, ax_column, plot_title, sequence_design, labels):
    ax = axs[1, ax_column]

    ax.plot(nominal_b, simulate_pulse_sequences.ivim_signal(nominal_b, Dstar=0.02), color="black", label=labels[0], linewidth=1.5, ls=(0,(2,5)))
    uncorrected_lower, uncorrected_upper = read_simulation_log(uncorrected_fname, ax=None, label="Uncorrected", alpha=0.2)
    #imaging_corrected_lower, imaging_corrected_upper = read_simulation_log(imaging_fname, ax=None, label="Corrected for imaging", color="yellow", alpha=0.2)
    #crossterm_corrected_lower, crossterm_corrected_upper = read_simulation_log(cross_terms_fname, ax=None, label="Corrected for cross terms", color="green", alpha=0.4)
    #powder_averaged_signals = bval_calc_tools.powder_average_signals_from_file(os.path.join(save_path, uncorrected_fname))

    ax.plot(nominal_b, uncorrected_lower, ls="")
    ax.plot(nominal_b, uncorrected_upper, ls="")
    #ax.plot(nominal_b, imaging_corrected_lower, ls="")
    #ax.plot(nominal_b, imaging_corrected_upper, ls="")
    #ax.plot(nominal_b, crossterm_corrected_lower, ls="")
    #ax.plot(nominal_b, crossterm_corrected_upper, ls="")
    #ax.plot(nominal_b, powder_averaged_signals, ls="-", color="tab:blue", label=labels[2], alpha=1, lw=1)
    ax.set_yscale("log")

    #formatter = matplotlib.ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),1)))).format(y))
    formatter = matplotlib.ticker.ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_yticks([1.0, 0.8, 0.6, 0.4])

    # Uncorrected
    ax.fill_between(x=nominal_b, y1=uncorrected_upper, y2=uncorrected_lower, alpha=.4, ls="", lw=0, label=labels[1])
    # Imaging corrected
    #ax.fill_between(x=nominal_b, y1=imaging_corrected_upper, y2=imaging_corrected_lower, alpha=.4, ls="", lw=0, label=labels[2])
    # Cross-term corrected
    #ax.fill_between(x=nominal_b, y1=crossterm_corrected_upper, y2=crossterm_corrected_lower, alpha=.4, ls="", lw=0, label=labels[3])

    #ax.fill_between(x=nominal_b, y1=uncorrected_upper, y2=uncorrected_lower, alpha=1, ls="-", color="tab:blue", facecolor="None", lw=lw)
    #ax.fill_between(x=nominal_b, y1=imaging_corrected_upper, y2=imaging_corrected_lower, alpha=1, ls="-", color="tab:orange", facecolor="None", lw=lw)
    #ax.fill_between(x=nominal_b, y1=crossterm_corrected_upper, y2=crossterm_corrected_lower, alpha=1, ls="-", color="tab:green", facecolor="None", lw=lw)

    # Fill between imaging and uncorrected
    #axs[0].fill_between(x=nominal_b, y1=uncorrected_lower, y2=imaging_corrected_lower, alpha=0.2, color="red", ls="")

    #axs[0].fill_between(x=nominal_b, y1=signals_upper, y2=signals_lower, alpha=0.2, color="red", label="Uncorrected")

    # Create an inset
    axins = ax.inset_axes([0.11, 0.1, 0.3, 0.4], xlim=(-10, 80), ylim=(0.83,1.02))
    axins.plot(nominal_b, simulate_pulse_sequences.ivim_signal(nominal_b, Dstar=0.02), color="black", linewidth=1.5, ls=(0, (2,5)))
    #axins.plot(nominal_b, uncorrected_lower, ls="")
    #axins.plot(nominal_b, uncorrected_upper, ls="")
    #axins.plot(nominal_b, powder_averaged_signals, ls="-", color="tab:blue", alpha=1, lw=1)
    #axins.plot(nominal_b, imaging_corrected_lower, ls="")
    #axins.plot(nominal_b, imaging_corrected_upper, ls="")
    #axins.plot(nominal_b, crossterm_corrected_lower, ls="")
    #axins.plot(nominal_b, crossterm_corrected_upper, ls="")

    # Uncorrected
    axins.fill_between(x=nominal_b, y1=uncorrected_upper, y2=uncorrected_lower, alpha=.3, ls="", lw=0, facecolor="tab:blue")
    # Imaging corrected
    #axins.fill_between(x=nominal_b, y1=imaging_corrected_upper, y2=imaging_corrected_lower, alpha=.3, ls="", lw=0, facecolor="tab:orange")
    # Cross-term corrected
    #axins.fill_between(x=nominal_b, y1=crossterm_corrected_upper, y2=crossterm_corrected_lower, alpha=.3, ls="", lw=0, facecolor="tab:green")

    #axins.fill_between(x=nominal_b, y1=uncorrected_upper, y2=uncorrected_lower, alpha=1, ls="-", color="tab:blue", facecolor="None", lw=lw)
    #axins.fill_between(x=nominal_b, y1=imaging_corrected_upper, y2=imaging_corrected_lower, alpha=1, ls="-", color="tab:orange", facecolor="None", lw=lw)
    #axins.fill_between(x=nominal_b, y1=crossterm_corrected_upper, y2=crossterm_corrected_lower, alpha=1, ls="-", color="tab:green", facecolor="None", lw=lw)

    if plot_title == "Well-designed sequence":
        axins.annotate("Crushers on", (30, 0.99), fontsize=11)
        axins.annotate(text="", xy=(27, 1), xytext=(10, 0.97), arrowprops=dict(arrowstyle="<-"))

    ax.indicate_inset_zoom(axins)
    
    ax.set_xlabel("Nominal b-value [s/mm$^\mathdefault{2}$]")
    ax.set_xlim(-25, 800)

def fig1_sequence(ax_column, sequence_design, plot_title, legend_flag=False):
    sequence_ax = axs[0, ax_column]
    ### Plot sequence design

    # Initialize sequence
    slice_select_trap = waveforms.trapezoid(delta=5e-3, slew_rate=1e6, amplitude=10e-3, gradient_update_rate=4e-6, return_time_axis=False)
    slice_select_zeros = np.zeros(slice_select_trap.shape)
    gwf_excitation = np.vstack((slice_select_zeros, slice_select_zeros, slice_select_trap))
    gwf_excitation = np.transpose(gwf_excitation)
    sequence = waveforms.Sequence_base(gwf_excitation, 4e-6)

    # Generate diffusion gradients
    trapezoid = waveforms.trapezoid(delta=20e-3, slew_rate=1e6, amplitude=30e-3, gradient_update_rate=4e-6, return_time_axis=False)
    trapezoid_pre = np.concatenate((trapezoid, np.zeros(int(10e-3/4e-6))))
    trapezoid_post = np.concatenate((np.zeros(int(3e-3/4e-6)), trapezoid))
    gwf = waveforms.Gwf(dt=4e-6, duration180=7e-3, pre180=trapezoid_pre, post180=trapezoid_post)
    gwf.set_b_by_scaling_amplitude(800e6)
    gwf.get_pre180_and_post180()

    # Generate generic sequence
    if sequence_design == "optimal":
        sequence.generic_sequence(1e-3, 1e-3, 1e-3, gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=True, all_crushers=True, crushers=True, only_crush_when_needed=True, nominal_bvalue=800e6)
    else:
        sequence.generic_sequence(1e-3, 1e-3, 1e-3, gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=False, all_crushers=True, crushers=True, only_crush_when_needed=False, nominal_bvalue=800e6)

    sequence.get_rf(optimize=True, start_time=sequence.t_180)
    sequence.get_optimal_TE(start_time=3e-3)
    sequence.set_b_by_scaling_amplitude(800e6, include_imaging=False, include_cross_terms=False)
    #print(f"Sequence b-value: {sequence.get_b(start_time=3e-3)*1e-6:.1f} s/mm2")
    #sequence.plot_gwf()

    axins_sequence = sequence_ax #ax.inset_axes([0.53, 0.58, 0.45, 0.4], ylim=(-33e-3, 33e-3))
    gwf_array = sequence.gwf[:40000,:]
    gwf_array[21000:35000, :] = 0.013


    if sequence_design == "bad":
        g_90 = gwf_array[:2000]
        pre180 = gwf_array[2000:8000]
        g_180 = gwf_array[8000:14000]
        post180 = gwf_array[14000:21000]
        g_readout = gwf_array[21000:]
    elif sequence_design == "optimal":
        g_90 = gwf_array[:3000]
        pre180 = gwf_array[3000:8000]
        g_180 = gwf_array[8000:12500]
        post180 = gwf_array[12500:20000]
        g_readout = gwf_array[20000:]

    gwf_img = np.concatenate([g_90, pre180*0, g_180, post180*0, g_readout*0])
    gwf_diff = gwf_array - gwf_img
    gwf_diff[20000:] = 0
    gwf_readout = gwf_array - gwf_img - gwf_diff

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_img[:,0], facecolor="grey", alpha=.4, label="")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_img[:,0], color="grey", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_img[:,1], facecolor="black", alpha=.4, label="Freq./phase")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_img[:,1], color="black", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_img[:,2], facecolor="tab:red", alpha=.4, label="Slice")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_img[:,2], color="tab:red", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_diff[:,0])), y1=gwf_diff[:,0], facecolor="tab:blue", alpha=.4, label="Diffusion encoding")
    axins_sequence.fill_between(x=range(len(gwf_diff[:,0])), y1=gwf_diff[:,0], color="tab:blue", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_readout[:,0])), y1=gwf_readout[:,0], facecolor="white", alpha=.4)
    axins_sequence.fill_between(x=range(len(gwf_readout[:,0])), y1=gwf_readout[:,0], color="black", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.plot(range(len(gwf_array[:,2])), [0 for x in range(len(gwf_array[:,2]))], color="black", lw=1)
    axins_sequence.spines[:].set_visible(False)
    axins_sequence.tick_params(labelleft=False, left=False)
    axins_sequence.tick_params(labelbottom=False, bottom=False)

    if legend_flag:
        axins_sequence.legend(frameon=False, loc=(.022, -.1), fontsize=11, ncols=2)

    axins_sequence.annotate("Read-out", (24200,.002), fontsize=11)

    if sequence_design == "optimal":
        axins_sequence.annotate("Crushers only when needed", (15000,-15e-3), fontsize=11)
        #axins_sequence.arrow(14500, -10e-3, -2500, 8e-3, width=.5e-3, color="black", shape="full", length_includes_head=True)
        axins_sequence.annotate(text="", xy=(14800, -10e-3), xytext=(12000, -2e-3), arrowprops=dict(arrowstyle="<-"))


        axins_sequence.annotate("Immediate rewinding", (7000,-33e-3), fontsize=11)
        #axins_sequence.arrow(6500, -25e-3, -3800, 10e-3, width=.5e-3, color="black", shape="full", length_includes_head=True, head_length=800, head_width=3e-3)
        axins_sequence.annotate(text="", xy=(6500, -25e-3), xytext=(2500, -15e-3), arrowprops=dict(arrowstyle="<-"))


    axins_sequence.set_ylim(-33e-3, 33e-3)

    axins_sequence.set_title(plot_title)

def plot_powder_average(filename, ax, label, ls="--"):
    powder_averaged_signals = bval_calc_tools.powder_average_signals_from_file(os.path.join(save_path, filename))

    ax.plot(nominal_b, powder_averaged_signals, color="tab:blue", label=label, ls=ls, lw=1)

    ax.get_children()[-2].plot(nominal_b, powder_averaged_signals, color="tab:blue", ls=ls, lw=1)
    

labels1 = ["Nominal", "Actual signal range", "Actual powder average"]
labels2 = [None for i in range(4)]
fig1_subplot(cross_terms_bad, fig=fig, ax_column=0, plot_title="Poor sequence design", sequence_design="bad", labels=labels1)
fig1_subplot(cross_terms_optimal, fig=fig, ax_column=1, plot_title="Well-designed sequence", sequence_design="optimal", labels=labels2)

plot_powder_average(powder_average_xyz_bad, axs[1,0], "[x, y, z] average",(0, (5,5)))
#plot_powder_average(powder_average_xy_z_bad, axs[1,0], "xy-z", "dotted")
#plot_powder_average(powder_average_GE_6_bad, axs[1,0], "6 dir", "dotted")
#plot_powder_average(powder_average_filip_bad, axs[1,0], "6 dir Filip", "dotted")
#plot_powder_average(powder_average_GE_16_bad, axs[1,0], "16 dir", "-")
plot_powder_average(powder_average_xyz_antipodal_bad, axs[1,0], "[x, y, z, -x, -y, -z] average", "-")

plot_powder_average(powder_average_xyz_optimal, axs[1,1], "",(0, (5,5)))
#plot_powder_average(powder_average_xy_z_optimal, axs[1,1], "xy-z", "")
#plot_powder_average(powder_average_GE_6_optimal, axs[1,1], "", "dotted")
#plot_powder_average(powder_average_filip_optimal, axs[1,1], "6 dir Filip", "dotted")
#plot_powder_average(powder_average_GE_16_optimal, axs[1,1], "", "-")
plot_powder_average(powder_average_xyz_antipodal_optimal, axs[1,1], "", "-")

axs[1,0].set_ylabel("Signal")
fig.legend(frameon=False, ncols=5, loc=(0.03, 0))

fig1_sequence(0, "bad", plot_title="Sequence with large cross-terms", legend_flag=True)
fig1_sequence(1, "optimal", plot_title="Sequence with minimal cross-terms")

fig.tight_layout()
#fig.savefig(os.path.join(fig_save_path, "fig1.pdf"), bbox_inches="tight")
# %% Figure 2

xy = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3])
#z = np.array([1e-3, 2e-3, 3e-3, 4e-3])
z = xy

path_worst = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\20241101\allCrushers_sequence"
f_u_worst, Dstar_u_worst, D_u_worst, bvals_nom_u_worst, minimum_bvalues_u_worst_z, maximum_bvalues_u_worst_z, angles_worst_u = read_all_generic_simulation_logs_uvecs(path_worst, correction="uncorrected", angles="froeling_200", xyres=xy, zres=z, return_angles=True)
f_i_worst, Dstar_i_worst, D_i_worst, bvals_nom_i_worst, minimum_bvalues_i_worst_z, maximum_bvalues_i_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="imaging_corrected", angles="froeling_200", xyres=xy, zres=z)
f_c_worst, Dstar_c_worst, D_c_worst, bvals_nom_c_worst, minimum_bvalues_c_worst_z, maximum_bvalues_c_worst_z, angles_worst_c = read_all_generic_simulation_logs_uvecs(path_worst, correction="crossterm_corrected", angles="froeling_200", xyres=xy, zres=z, return_angles=True)

path_best = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\20241101\optimal_allCrushers_onlyCrushWhenNeeded_sequence"
f_u_best, Dstar_u_best, D_u_best, bvals_nom_u_best, minimum_bvalues_u_best_z, maximum_bvalues_u_best_z, angles_best_u = read_all_generic_simulation_logs_uvecs(path_best, correction="uncorrected", angles="froeling_200", xyres=xy, zres=z, return_angles=True)
f_i_best, Dstar_i_best, D_i_best, bvals_nom_i_best, minimum_bvalues_i_best_z, maximum_bvalues_i_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="imaging_corrected", angles="froeling_200", xyres=xy, zres=z)
f_c_best, Dstar_c_best, D_c_best, bvals_nom_c_best, minimum_bvalues_c_best_z, maximum_bvalues_c_best_z, angles_best_c = read_all_generic_simulation_logs_uvecs(path_best, correction="crossterm_corrected", angles="froeling_200", xyres=xy, zres=z, return_angles=True)

#f_u_worst, Dstar_u_worst, D_u_worst, bvals_nom_u_worst, minimum_bvalues_u_worst_z, maximum_bvalues_u_worst_z, angles_worst_u = read_all_generic_simulation_logs(path_worst, correction="uncorrected", ndir=1000, xyres=xy, zres=z, return_angles=True)
#f_i_worst, Dstar_i_worst, D_i_worst, bvals_nom_i_worst, minimum_bvalues_i_worst_z, maximum_bvalues_i_worst_z = read_all_generic_simulation_logs(path_worst, correction="imaging_corrected", ndir=1000, xyres=xy, zres=z)
#f_c_worst, Dstar_c_worst, D_c_worst, bvals_nom_c_worst, minimum_bvalues_c_worst_z, maximum_bvalues_c_worst_z = read_all_generic_simulation_logs(path_worst, correction="crossterm_corrected", ndir=1000, xyres=xy, zres=z)

#f_u_best, Dstar_u_best, D_u_best, bvals_nom_u_best, minimum_bvalues_u_best_z, maximum_bvalues_u_best_z, angles_best_u = read_all_generic_simulation_logs(path_best, correction="uncorrected", ndir=1000, xyres=xy, zres=z, return_angles=True)
#f_i_best, Dstar_i_best, D_i_best, bvals_nom_i_best, minimum_bvalues_i_best_z, maximum_bvalues_i_best_z = read_all_generic_simulation_logs(path_best, correction="imaging_corrected", ndir=1000, xyres=xy, zres=z)
#f_c_best, Dstar_c_best, D_c_best, bvals_nom_c_best, minimum_bvalues_c_best_z, maximum_bvalues_c_best_z = read_all_generic_simulation_logs(path_best, correction="crossterm_corrected", ndir=1000, xyres=xy, zres=z)
################# In-plane resolution
# xy res
nominal_bvals = np.array([0.0000001, 50, 200, 800])
minimum_bvalues_relative_u_best_z = (minimum_bvalues_u_best_z - nominal_bvals)/nominal_bvals
minimum_bvalues_relative_i_best_z = (minimum_bvalues_i_best_z - nominal_bvals)/nominal_bvals
minimum_bvalues_relative_c_best_z = (minimum_bvalues_c_best_z - nominal_bvals)/nominal_bvals

maximum_bvalues_relative_u_best_z = (maximum_bvalues_u_best_z - nominal_bvals)/nominal_bvals
maximum_bvalues_relative_i_best_z = (maximum_bvalues_i_best_z - nominal_bvals)/nominal_bvals
maximum_bvalues_relative_c_best_z = (maximum_bvalues_c_best_z - nominal_bvals)/nominal_bvals

minimum_bvalues_relative_u_worst_z = (minimum_bvalues_u_worst_z - nominal_bvals)/nominal_bvals
minimum_bvalues_relative_i_worst_z = (minimum_bvalues_i_worst_z - nominal_bvals)/nominal_bvals
minimum_bvalues_relative_c_worst_z = (minimum_bvalues_c_worst_z - nominal_bvals)/nominal_bvals

maximum_bvalues_relative_u_worst_z = (maximum_bvalues_u_worst_z - nominal_bvals)/nominal_bvals
maximum_bvalues_relative_i_worst_z = (maximum_bvalues_i_worst_z - nominal_bvals)/nominal_bvals
maximum_bvalues_relative_c_worst_z = (maximum_bvalues_c_worst_z - nominal_bvals)/nominal_bvals
#z = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 2.75e-3, 3e-3, 3.25e-3, 3.5e-3, 3.75e-3, 4e-3]) #xy res

def get_encoding_array(angles, bval):
    trapezoid = waveforms.trapezoid(delta=23e-3, slew_rate=1e6, amplitude=10e-3, gradient_update_rate=4e-6, return_time_axis=False)
    gwf = waveforms.Gwf(4e-6, duration180=6e-3, pre180=trapezoid, post180=trapezoid)
    gwf.rotate_encoding_with_uvec(angles)
    gwf.set_b_by_scaling_amplitude(bval*1e6)

    return gwf

def plot_waveform_inset(angles, bval, ax, legend_flag=False):
    """
    Creates an inset plot with rotated and scaled diffusion waveforms.
    """
    axins_sequence = ax.inset_axes([0.53, 0.78, 0.45, 0.2], ylim=(-1e-3, 33e-3))
    gwf = get_encoding_array(angles[1, :, 0], bval)
    gwf_array = gwf.gwf
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,0], facecolor="grey", alpha=.4, label="x")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,0], color="grey", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,1], facecolor="black", alpha=.4, label="y")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,1], color="black", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,2], facecolor="tab:red", alpha=.4, label="z")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,2], color="tab:red", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.plot(range(len(gwf_array[:,2])), [0 for x in range(len(gwf_array[:,2]))], color="black", lw=1)
    axins_sequence.spines[:].set_visible(False)
    axins_sequence.tick_params(labelleft=False, left=False)
    axins_sequence.tick_params(labelbottom=False, bottom=False)
    if legend_flag:
        axins_sequence.legend(frameon=False, loc=(-.5, -0.5), fontsize=11, ncols=1)

    axins_sequence = ax.inset_axes([0.53, 0.01, 0.45, 0.2], ylim=(-33e-3, 1e-3))
    gwf = get_encoding_array(angles[0, :, 0], bval)
    gwf_array = gwf.gwf
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,0], facecolor="grey", alpha=.4, label="x")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,0], color="grey", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,1], facecolor="black", alpha=.4, label="y")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,1], color="black", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,2], facecolor="tab:red", alpha=.4, label="z")
    axins_sequence.fill_between(x=range(len(gwf_array[:,2])), y1=gwf_array[:,2], color="tab:red", facecolor="None", alpha=1, ls="-", lw=.8)

    axins_sequence.plot(range(len(gwf_array[:,2])), [0 for x in range(len(gwf_array[:,2]))], color="black", lw=1)
    axins_sequence.spines[:].set_visible(False)
    axins_sequence.tick_params(labelleft=False, left=False)
    axins_sequence.tick_params(labelbottom=False, bottom=False)

def plot_3d_uvec_inset_old(fig, position, angles):
    """
    Creates a 3d inset plot of the uvec.
    """
    gwf_max = get_encoding_array(angles[1,:,0], 200)
    gwf_min = get_encoding_array(angles[0,:,0], 200)

    # Get uvec by taking the square root of the diagonal of the B-tensor, and normalizing the vector
    B_max = gwf_max.get_b(return_b_tensor=True)
    B_min = gwf_min.get_b(return_b_tensor=True)

    B_max_diagonal = np.sqrt(B_max[:3])
    B_min_diagonal = np.sqrt(B_min[:3])
    uvec_max = B_max_diagonal/np.linalg.norm(B_max_diagonal)
    uvec_min = B_min_diagonal/np.linalg.norm(B_min_diagonal)*-1

    #uvec_max = np.max(gwf_max.gwf, axis=0)/np.linalg.norm(np.max(gwf_max.gwf, axis=0))
    #uvec_min = np.min(gwf_min.gwf, axis=0)/np.linalg.norm(np.min(gwf_min.gwf, axis=0))

    ax = fig.add_axes(position, projection=("3d"))
    ax.set_proj_type("persp")
    ax.view_init(10,70+180)
    ax.view_init(10,140+180)
    # x line
    ax.plot([-1, 1], [0,0], [0,0],  color="black", ls="-", lw=.5)
    ax.plot([0,0], [-1,1], [0,0], color="black", ls="-", lw=.5)
    ax.plot([0,0], [0,0], [-1,1], color="black", ls="-", lw=.5)


    # Plot the uvecs
    #ax.plot([0, uvec_max[0]], [0, uvec_max[1]], [0, uvec_max[2]], color="tab:blue", ls="-")
    #ax.plot([0, uvec_min[0]], [0, uvec_min[1]], [0, uvec_min[2]], color="tab:orange", ls="-")
    ax.quiver(0, 0, 0, *uvec_max, color="tab:red")
    ax.quiver(0, 0, 0, *uvec_min, color="black", alpha=.6)

    ax.text(-1, -.7, 1.4, f"Max pos deviation $\mathbf g_\mathrm d$ dir\n[{uvec_max[0]:.2f}, {uvec_max[1]:.2f}, {uvec_max[2]:.2f}]", color="tab:red", fontsize=10)
    ax.text(-1.1, -.7, -1.7, f"Max neg deviation $\mathbf g_\mathrm d$ dir\n[{uvec_min[0]:.2f}, {uvec_min[1]:.2f}, {uvec_min[2]:.2f}]", color="black", alpha=.6, fontsize=10)

    ax.text(1.1, 0, -.1, "x")
    ax.text(0, -1.3, -.15, "y")
    ax.text(0, -.1, 1.1, "z")

    #arrow_max = Arrow3D([0, uvec_max[0]], [0, uvec_max[1]], [0, uvec_max[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="tab:blue")
    #ax.add_artist(arrow_max)
    #ax.annotate(text="", xy=(0, 0), xytext=(uvec_max[0], uvec_max[1]), arrowprops=dict(arrowstyle="<-"))


    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        

    ax.grid(False)
    ax.set_box_aspect(aspect=(1,1,1))

def plot_3d_uvec_inset(fig, position, angles):
    """
    Creates a 3d inset plot of the uvec.
    """
    gwf_max = get_encoding_array(angles[1,:,10], 200)
    gwf_min = get_encoding_array(angles[0,:,10], 200)

    # Get uvec by taking the square root of the diagonal of the B-tensor, and normalizing the vector
    B_max = gwf_max.get_b(return_b_tensor=True)
    B_min = gwf_min.get_b(return_b_tensor=True)

    B_max_diagonal = np.sqrt(B_max[:3])
    B_min_diagonal = np.sqrt(B_min[:3])
    uvec_max = B_max_diagonal/np.linalg.norm(B_max_diagonal)
    uvec_min = B_min_diagonal/np.linalg.norm(B_min_diagonal)*-1

    #uvec_max = np.max(gwf_max.gwf, axis=0)/np.linalg.norm(np.max(gwf_max.gwf, axis=0))
    #uvec_min = np.min(gwf_min.gwf, axis=0)/np.linalg.norm(np.min(gwf_min.gwf, axis=0))

    uvec_max = angles[1,:,10]
    uvec_min = angles[0,:,10]

    ax = fig.add_axes(position, projection=("3d"))
    ax.set_proj_type("persp")
    ax.view_init(10,70+180)
    ax.view_init(10,140+180)
    # x line
    ax.plot([-1, 1], [0,0], [0,0],  color="black", ls="-", lw=.5)
    ax.plot([0,0], [-1,1], [0,0], color="black", ls="-", lw=.5)
    ax.plot([0,0], [0,0], [-1,1], color="black", ls="-", lw=.5)


    # Plot the uvecs
    #ax.plot([0, uvec_max[0]], [0, uvec_max[1]], [0, uvec_max[2]], color="tab:blue", ls="-")
    #ax.plot([0, uvec_min[0]], [0, uvec_min[1]], [0, uvec_min[2]], color="tab:orange", ls="-")
    ax.quiver(0, 0, 0, *uvec_max, color="tab:red")
    ax.quiver(0, 0, 0, *uvec_min, color="black", alpha=.6)

    ax.text(-1, -.7, 1.4, f"Max pos deviation $\mathbf g_\mathrm d$ dir\n[{uvec_max[0]:.2f}, {uvec_max[1]:.2f}, {uvec_max[2]:.2f}]", color="tab:red", fontsize=10)
    ax.text(-1.1, -.7, -1.7, f"Max neg deviation $\mathbf g_\mathrm d$ dir\n[{uvec_min[0]:.2f}, {uvec_min[1]:.2f}, {uvec_min[2]:.2f}]", color="black", alpha=.6, fontsize=10)

    ax.text(1.1, 0, -.1, "x")
    ax.text(0, -1.3, -.15, "y")
    ax.text(0, -.1, 1.1, "z")

    #arrow_max = Arrow3D([0, uvec_max[0]], [0, uvec_max[1]], [0, uvec_max[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="tab:blue")
    #ax.add_artist(arrow_max)
    #ax.annotate(text="", xy=(0, 0), xytext=(uvec_max[0], uvec_max[1]), arrowprops=dict(arrowstyle="<-"))


    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Transparent spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Transparent panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        

    ax.grid(False)
    ax.set_box_aspect(aspect=(1,1,1))

def fig_2_inset(ax, x_axis, colors, xlim, ylim, position=[0.3, 0.46, 0.6, 0.5], areas=None):
    axins = ax.inset_axes(position, xlim=xlim, ylim=ylim)

    #for line_idx in range(len(pwd_avg_lines)):
        #axins.fill_between(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=1, ls=linestyles[line_idx])
    axins.plot(z*1e3, [0 for i in range(len(z))], color="black", ls=(0, (2,5)), lw=1.5)
    
    print(colors)

    if areas:
        for area_idx in range(0, len(areas), 2):
            axins.fill_between(x=x_axis, y1=areas[area_idx], y2=areas[area_idx+1], facecolor=colors[area_idx], alpha=alpha_area, ls="")
            axins.fill_between(x=x_axis, y1=areas[area_idx], y2=areas[area_idx+1], color=colors[area_idx], facecolor="None", alpha=alpha_lines, ls="-", lw=lw)


    ax.indicate_inset_zoom(axins)

### Bad
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(10,6), sharey=True, sharex=True)
# 50 Nominal
axs[0,0].plot(z*1e3, [0 for i in range(len(z))], color="black", label="Nominal", ls=(0, (2,5)), lw=1.5)
# 50 Uncorrected
#axs[0,0].fill_between(z*1e3, y1=minimum_bvalues_relative_u_worst_z[:, 1]*100, y2=maximum_bvalues_relative_u_worst_z[:, 1]*100, label="Uncorrected", facecolor="tab:blue", ls="", alpha=alpha_area)
#axs[0,0].fill_between(z*1e3, y1=minimum_bvalues_relative_u_worst_z[:, 1]*100, y2=maximum_bvalues_relative_u_worst_z[:, 1]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 50 Imaging
axs[0,0].fill_between(z*1e3, y1=minimum_bvalues_relative_i_worst_z[:, 1]*100, y2=maximum_bvalues_relative_i_worst_z[:, 1]*100, label="Accounting for imaging", facecolor="tab:orange", ls="", alpha=alpha_area)
axs[0,0].fill_between(z*1e3, y1=minimum_bvalues_relative_i_worst_z[:, 1]*100, y2=maximum_bvalues_relative_i_worst_z[:, 1]*100, color="tab:orange", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 50 Cross-term
axs[0,0].fill_between(z*1e3, y1=minimum_bvalues_relative_c_worst_z[:, 1]*100, y2=maximum_bvalues_relative_c_worst_z[:, 1]*100, label="Accounting for imaging and cross-terms", facecolor="tab:blue", ls="", alpha=alpha_area)
axs[0,0].fill_between(z*1e3, y1=minimum_bvalues_relative_c_worst_z[:, 1]*100, y2=maximum_bvalues_relative_c_worst_z[:, 1]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# Settings
axs[0,0].set_title("Nominal b = 50 s/mm$^2$")
axs[0,0].set_ylabel("Relative b-value deviation [\%]")
# Waveforms
#plot_waveform_inset(angles_worst_u, 50, axs[0,0], legend_flag=True)

# 200 Nominal
axs[0,1].plot(z*1e3, [0 for i in range(len(z))], color="black", ls=(0, (2,5)), lw=1.5)
# 200 Uncorrected
#axs[0,1].fill_between(z*1e3, y1=minimum_bvalues_relative_u_worst_z[:, 2]*100, y2=maximum_bvalues_relative_u_worst_z[:, 2]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
#axs[0,1].fill_between(z*1e3, y1=minimum_bvalues_relative_u_worst_z[:, 2]*100, y2=maximum_bvalues_relative_u_worst_z[:, 2]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 200 Imaging
axs[0,1].fill_between(z*1e3, y1=minimum_bvalues_relative_i_worst_z[:, 2]*100, y2=maximum_bvalues_relative_i_worst_z[:, 2]*100, facecolor="tab:orange", ls="", alpha=alpha_area)
axs[0,1].fill_between(z*1e3, y1=minimum_bvalues_relative_i_worst_z[:, 2]*100, y2=maximum_bvalues_relative_i_worst_z[:, 2]*100, color="tab:orange", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 200 Cross-term
axs[0,1].fill_between(z*1e3, y1=minimum_bvalues_relative_c_worst_z[:, 2]*100, y2=maximum_bvalues_relative_c_worst_z[:, 2]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
axs[0,1].fill_between(z*1e3, y1=minimum_bvalues_relative_c_worst_z[:, 2]*100, y2=maximum_bvalues_relative_c_worst_z[:, 2]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# Settings
axs[0,1].set_title("Nominal b = 200 s/mm$^2$")
#plot_waveform_inset(angles_worst_u, 200, axs[0,1])

# 800 Nominal
axs[0,2].plot(z*1e3, [0 for i in range(len(z))], color="black", ls=(0, (2,5)), lw=1.5)
# 800 Uncorrected
#axs[0,2].fill_between(z*1e3, y1=minimum_bvalues_relative_u_worst_z[:, 3]*100, y2=maximum_bvalues_relative_u_worst_z[:, 3]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
#axs[0,2].fill_between(z*1e3, y1=minimum_bvalues_relative_u_worst_z[:, 3]*100, y2=maximum_bvalues_relative_u_worst_z[:, 3]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 800 Imaging
axs[0,2].fill_between(z*1e3, y1=minimum_bvalues_relative_i_worst_z[:, 3]*100, y2=maximum_bvalues_relative_i_worst_z[:, 3]*100, facecolor="tab:orange", ls="", alpha=alpha_area)
axs[0,2].fill_between(z*1e3, y1=minimum_bvalues_relative_i_worst_z[:, 3]*100, y2=maximum_bvalues_relative_i_worst_z[:, 3]*100, color="tab:orange", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 800 Cross-term
axs[0,2].fill_between(z*1e3, y1=minimum_bvalues_relative_c_worst_z[:, 3]*100, y2=maximum_bvalues_relative_c_worst_z[:, 3]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
axs[0,2].fill_between(z*1e3, y1=minimum_bvalues_relative_c_worst_z[:, 3]*100, y2=maximum_bvalues_relative_c_worst_z[:, 3]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# Settings
axs[0,2].set_title("Nominal b = 800 s/mm$^2$")
#plot_waveform_inset(angles_worst_u, 800, axs[0,2])
#axs[0,2].annotate("Maximum error\ndiffusion direction", (1.1, 83))
#axs[0,2].annotate("Minimum error\ndiffusion direction", (1.1, -63))


### Optimal
#xy = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 2.75e-3, 3e-3, 3.25e-3, 3.5e-3, 3.75e-3, 4e-3])
# 50 Nominal
axs[1,0].plot(z*1e3, [0 for i in range(len(z))], color="black", ls=(0, (2,5)), lw=1.5)
# 50 Uncorrected
#axs[1,0].fill_between(z*1e3, y1=minimum_bvalues_relative_u_best_z[:, 1]*100, y2=maximum_bvalues_relative_u_best_z[:, 1]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
#axs[1,0].fill_between(z*1e3, y1=minimum_bvalues_relative_u_best_z[:, 1]*100, y2=maximum_bvalues_relative_u_best_z[:, 1]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 50 Imaging
axs[1,0].fill_between(z*1e3, y1=minimum_bvalues_relative_i_best_z[:, 1]*100, y2=maximum_bvalues_relative_i_best_z[:, 1]*100, facecolor="tab:orange", ls="", alpha=alpha_area)
axs[1,0].fill_between(z*1e3, y1=minimum_bvalues_relative_i_best_z[:, 1]*100, y2=maximum_bvalues_relative_i_best_z[:, 1]*100, color="tab:orange", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 50 Cross-term
axs[1,0].fill_between(z*1e3, y1=minimum_bvalues_relative_c_best_z[:, 1]*100, y2=maximum_bvalues_relative_c_best_z[:, 1]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
axs[1,0].fill_between(z*1e3, y1=minimum_bvalues_relative_c_best_z[:, 1]*100, y2=maximum_bvalues_relative_c_best_z[:, 1]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# Settings
axs[1,0].set_ylabel("Relative b-value deviation [\%]")
axs[1,0].set_xlabel("Isotropic resolution [mm]")
#plot_waveform_inset(angles_best_u, 50, axs[1,0])
colors = ["tab:orange", "tab:orange", "tab:blue", "tab:blue"]
fig_2_inset(axs[1,0], z*1e3, colors, [1, 4], [-10, 10], areas=[minimum_bvalues_relative_i_best_z[:,1]*100, maximum_bvalues_relative_i_best_z[:,1]*100, minimum_bvalues_relative_c_best_z[:,1]*100, maximum_bvalues_relative_c_best_z[:,1]*100])


# 200 Nominal
axs[1,1].plot(z*1e3, [0 for i in range(len(z))], color="black", ls=(0, (2,5)), lw=1.5)
# 200 Uncorrected
#axs[1,1].fill_between(z*1e3, y1=minimum_bvalues_relative_u_best_z[:, 2]*100, y2=maximum_bvalues_relative_u_best_z[:, 2]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
#axs[1,1].fill_between(z*1e3, y1=minimum_bvalues_relative_u_best_z[:, 2]*100, y2=maximum_bvalues_relative_u_best_z[:, 2]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 200 Imaging
axs[1,1].fill_between(z*1e3, y1=minimum_bvalues_relative_i_best_z[:, 2]*100, y2=maximum_bvalues_relative_i_best_z[:, 2]*100, facecolor="tab:orange", ls="", alpha=alpha_area)
axs[1,1].fill_between(z*1e3, y1=minimum_bvalues_relative_i_best_z[:, 2]*100, y2=maximum_bvalues_relative_i_best_z[:, 2]*100, color="tab:orange", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 200 Cross-term
axs[1,1].fill_between(z*1e3, y1=minimum_bvalues_relative_c_best_z[:, 2]*100, y2=maximum_bvalues_relative_c_best_z[:, 2]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
axs[1,1].fill_between(z*1e3, y1=minimum_bvalues_relative_c_best_z[:, 2]*100, y2=maximum_bvalues_relative_c_best_z[:, 2]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# Settings
axs[1,1].set_xlabel("Isotropic resolution [mm]")
#plot_waveform_inset(angles_best_u, 200, axs[1,1])
fig_2_inset(axs[1,1], z*1e3, colors, [1, 4], [-10, 10], areas=[minimum_bvalues_relative_i_best_z[:,2]*100, maximum_bvalues_relative_i_best_z[:,2]*100, minimum_bvalues_relative_c_best_z[:,2]*100, maximum_bvalues_relative_c_best_z[:,2]*100])

# 800 Nominal
axs[1,2].plot(z*1e3, [0 for i in range(len(z))], color="black", ls=(0,(2,5)), lw=1.5)
# 800 Uncorrected
#axs[1,2].fill_between(z*1e3, y1=minimum_bvalues_relative_u_best_z[:, 3]*100, y2=maximum_bvalues_relative_u_best_z[:, 3]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
#axs[1,2].fill_between(z*1e3, y1=minimum_bvalues_relative_u_best_z[:, 3]*100, y2=maximum_bvalues_relative_u_best_z[:, 3]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 800 Imaging
axs[1,2].fill_between(z*1e3, y1=minimum_bvalues_relative_i_best_z[:, 3]*100, y2=maximum_bvalues_relative_i_best_z[:, 3]*100, facecolor="tab:orange", ls="", alpha=alpha_area)
axs[1,2].fill_between(z*1e3, y1=minimum_bvalues_relative_i_best_z[:, 3]*100, y2=maximum_bvalues_relative_i_best_z[:, 3]*100, color="tab:orange", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# 800 Cross-term
axs[1,2].fill_between(z*1e3, y1=minimum_bvalues_relative_c_best_z[:, 3]*100, y2=maximum_bvalues_relative_c_best_z[:, 3]*100, facecolor="tab:blue", ls="", alpha=alpha_area)
axs[1,2].fill_between(z*1e3, y1=minimum_bvalues_relative_c_best_z[:, 3]*100, y2=maximum_bvalues_relative_c_best_z[:, 3]*100, color="tab:blue", facecolor="None", ls="-", alpha=alpha_lines, lw=lw)
# Settings
axs[1,2].set_xlabel("Isotropic resolution [mm]")
#plot_waveform_inset(angles_best_u, 800, axs[1,2])
fig_2_inset(axs[1,2], z*1e3, colors, [1, 4], [-10, 10], areas=[minimum_bvalues_relative_i_best_z[:,3]*100, maximum_bvalues_relative_i_best_z[:,3]*100, minimum_bvalues_relative_c_best_z[:,3]*100, maximum_bvalues_relative_c_best_z[:,3]*100])

axs[0,0].set_xlim([1, 4])

#plot_3d_uvec_inset(axs[0,2], 1)
plot_3d_uvec_inset(fig, position=[.98, .43, 0.2, 0.5], angles=angles_worst_c)
plot_3d_uvec_inset(fig, position=[.98, -0.01, 0.2, 0.5], angles=angles_best_c)

fig.legend(frameon=False, ncols=4, loc=(.1, -.005))

fig.tight_layout()
fig.text(0.99, 0.89, "Large cross-terms", fontsize=16)
fig.text(0.99, .46, "Minimal cross-terms", fontsize=16)
fig.suptitle("Relative b-value deviation vs. isotropic resolution", x=.53, y=1.02)
#fig.savefig(os.path.join(fig_save_path, "fig2.pdf"), bbox_inches="tight")
# %% Figure 3

folder_best_case = "optimal_allCrushers_onlyCrushWhenNeeded_sequence"
folder_worst_case = "allCrushers_sequence"

z = np.array([2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 10e-3])
xy = np.array([2e-3 for i in range(len(z))])

#path_best = os.path.join(save_path, folder_best_case)
#path_worst = os.path.join(save_path, folder_worst_case)

path_test = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\test\optimal_allCrushers_onlyCrushWhenNeeded_sequence"

#nom_b0 = np.array([3.])
# 1000 dir sets
#f_u_worst, Dstar_u_worst, D_u_worst, bvals_nom_u_worst, minimum_bvalues_u_worst_z, maximum_bvalues_u_worst_z = read_all_generic_simulation_logs(path_worst, correction="uncorrected", angles="ndir1000", xyres=xy, zres=z)
#f_i_worst, Dstar_i_worst, D_i_worst, bvals_nom_i_worst, minimum_bvalues_i_worst_z, maximum_bvalues_i_worst_z = read_all_generic_simulation_logs(path_worst, correction="imaging_corrected", angles="ndir1000", xyres=xy, zres=z)
#f_c_worst, Dstar_c_worst, D_c_worst, bvals_nom_c_worst, minimum_bvalues_c_worst_z, maximum_bvalues_c_worst_z = read_all_generic_simulation_logs(path_worst, correction="crossterm_corrected", angles="ndir1000", xyres=xy, zres=z)

#f_u_best, Dstar_u_best, D_u_best, bvals_nom_u_best, minimum_bvalues_u_best_z, maximum_bvalues_u_best_z = read_all_generic_simulation_logs(path_best, correction="uncorrected", angles="ndir1000", xyres=xy, zres=z)
#f_i_best, Dstar_i_best, D_i_best, bvals_nom_i_best, minimum_bvalues_i_best_z, maximum_bvalues_i_best_z = read_all_generic_simulation_logs(path_best, correction="imaging_corrected", angles="ndir1000", xyres=xy, zres=z)
#f_c_best, Dstar_c_best, D_c_best, bvals_nom_c_best, minimum_bvalues_c_best_z, maximum_bvalues_c_best_z = read_all_generic_simulation_logs(path_best, correction="crossterm_corrected", angles="ndir1000", xyres=xy, zres=z)


### Powder averages
# Worst
#f_u_xyz_worst, Dstar_u_xyz_worst, D_u_xyz_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
#f_i_xyz_worst, Dstar_i_xyz_worst, D_i_xyz_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
#f_c_xyz_worst, Dstar_c_xyz_worst, D_c_xyz_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

#f_u_xyz_antipodal_worst, Dstar_u_xyz_antipodal_worst, D_u_xyz_antipodal_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_i_xyz_antipodal_worst, Dstar_i_xyz_antipodal_worst, D_i_xyz_antipodal_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_c_xyz_antipodal_worst, Dstar_c_xyz_antipodal_worst, D_c_xyz_antipodal_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 

#f_u_GE_6_worst, Dstar_u_GE_6_worst, D_u_GE_6_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="GE_6", xyres=xy, zres=z) 
#f_i_GE_6_worst, Dstar_i_GE_6_worst, D_i_GE_6_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="GE_6", xyres=xy, zres=z) 
#f_c_GE_6_worst, Dstar_c_GE_6_worst, D_c_GE_6_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="GE_6", xyres=xy, zres=z) 

#f_u_GE_16_worst, Dstar_u_GE_16_worst, D_u_GE_16_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="GE_16", xyres=xy, zres=z) 
#f_i_GE_16_worst, Dstar_i_GE_16_worst, D_i_GE_16_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="GE_16", xyres=xy, zres=z) 
#f_c_GE_16_worst, Dstar_c_GE_16_worst, D_c_GE_16_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="GE_16", xyres=xy, zres=z) 
# Best
#f_u_xyz_best, Dstar_u_xyz_best, D_u_xyz_best = powder_average_all_simulation_logs(path_best, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
#f_i_xyz_best, Dstar_i_xyz_best, D_i_xyz_best = powder_average_all_simulation_logs(path_best, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
#f_c_xyz_best, Dstar_c_xyz_best, D_c_xyz_best = powder_average_all_simulation_logs(path_best, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

#f_u_xyz_antipodal_best, Dstar_u_xyz_antipodal_best, D_u_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_test, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_i_xyz_antipodal_best, Dstar_i_xyz_antipodal_best, D_i_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_test, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_c_xyz_antipodal_best, Dstar_c_xyz_antipodal_best, D_c_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_test, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 

f_u_xyz_antipodal_best_test, Dstar_u_xyz_antipodal_best_test, D_u_xyz_antipodal_best_test = powder_average_all_simulation_logs(path_test, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 

#f_u_GE_6_best, Dstar_u_GE_6_best, D_u_GE_6_best = powder_average_all_simulation_logs(path_best, correction="uncorrected", angles="GE_6", xyres=xy, zres=z) 
#f_i_GE_6_best, Dstar_i_GE_6_best, D_i_GE_6_best = powder_average_all_simulation_logs(path_best, correction="imaging_corrected", angles="GE_6", xyres=xy, zres=z) 
#f_c_GE_6_best, Dstar_c_GE_6_best, D_c_GE_6_best = powder_average_all_simulation_logs(path_best, correction="crossterm_corrected", angles="GE_6", xyres=xy, zres=z) 

#f_u_GE_16_best, Dstar_u_GE_16_best, D_u_GE_16_best = powder_average_all_simulation_logs(path_best, correction="uncorrected", angles="GE_16", xyres=xy, zres=z) 
#f_i_GE_16_best, Dstar_i_GE_16_best, D_i_GE_16_best = powder_average_all_simulation_logs(path_best, correction="imaging_corrected", angles="GE_16", xyres=xy, zres=z) 
#f_c_GE_16_best, Dstar_c_GE_16_best, D_c_GE_16_best = powder_average_all_simulation_logs(path_best, correction="crossterm_corrected", angles="GE_16", xyres=xy, zres=z) 


## NEW
path_best = os.path.join(save_path, "20241101", folder_best_case)
path_worst = os.path.join(save_path, "20241101", folder_worst_case)

# Froeling 200dir sets
#f_u_worst, Dstar_u_worst, D_u_worst, bvals_nom_u_worst, minimum_bvalues_u_worst_z, maximum_bvalues_u_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="uncorrected", angles="froeling_200", xyres=xy, zres=z)
#f_i_worst, Dstar_i_worst, D_i_worst, bvals_nom_i_worst, minimum_bvalues_i_worst_z, maximum_bvalues_i_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="imaging_corrected", angles="froeling_200", xyres=xy, zres=z)
#f_c_worst, Dstar_c_worst, D_c_worst, bvals_nom_c_worst, minimum_bvalues_c_worst_z, maximum_bvalues_c_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="crossterm_corrected", angles="froeling_200", xyres=xy, zres=z)

#f_u_best, Dstar_u_best, D_u_best, bvals_nom_u_best, minimum_bvalues_u_best_z, maximum_bvalues_u_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="uncorrected", angles="froeling_200", xyres=xy, zres=z)
#f_i_best, Dstar_i_best, D_i_best, bvals_nom_i_best, minimum_bvalues_i_best_z, maximum_bvalues_i_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="imaging_corrected", angles="froeling_200", xyres=xy, zres=z)
#f_c_best, Dstar_c_best, D_c_best, bvals_nom_c_best, minimum_bvalues_c_best_z, maximum_bvalues_c_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="crossterm_corrected", angles="froeling_200", xyres=xy, zres=z)

# Worst
f_u_xyz_worst, Dstar_u_xyz_worst, D_u_xyz_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
f_i_xyz_worst, Dstar_i_xyz_worst, D_i_xyz_worst = powder_average_all_simulation_logs_new(path_worst, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
f_c_xyz_worst, Dstar_c_xyz_worst, D_c_xyz_worst = powder_average_all_simulation_logs_new(path_worst, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

f_u_xyz_antipodal_worst, Dstar_u_xyz_antipodal_worst, D_u_xyz_antipodal_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_i_xyz_antipodal_worst, Dstar_i_xyz_antipodal_worst, D_i_xyz_antipodal_worst = powder_average_all_simulation_logs_new(path_worst, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_c_xyz_antipodal_worst, Dstar_c_xyz_antipodal_worst, D_c_xyz_antipodal_worst = powder_average_all_simulation_logs_new(path_worst, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 

# Best
f_u_xyz_best, Dstar_u_xyz_best, D_u_xyz_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
f_i_xyz_best, Dstar_i_xyz_best, D_i_xyz_best = powder_average_all_simulation_logs_new(path_best, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
f_c_xyz_best, Dstar_c_xyz_best, D_c_xyz_best = powder_average_all_simulation_logs_new(path_best, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

f_u_xyz_antipodal_best, Dstar_u_xyz_antipodal_best, D_u_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_i_xyz_antipodal_best, Dstar_i_xyz_antipodal_best, D_i_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_best, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_c_xyz_antipodal_best, Dstar_c_xyz_antipodal_best, D_c_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_best, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 


def fig_3_inset(ax, x_axis, pwd_avg_lines, linestyles, colors, xlim, ylim, position=[0.45, 0.65, 0.5, 0.3], areas=None):
    axins = ax.inset_axes(position, xlim=xlim, ylim=ylim)



    for line_idx in range(len(pwd_avg_lines)):
        if linestyles[line_idx] == (0, (2,5)):
            axins.plot(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=2, ls=linestyles[line_idx])
        elif linestyles[line_idx] != "-":
            axins.plot(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=1.5, ls=linestyles[line_idx])
        else:
            axins.plot(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=1, ls=linestyles[line_idx])

    if areas:
        for area_idx in range(len(areas)):
            axins.fill_between(x=x_axis, y1=areas[area_idx][:,0], y2=areas[area_idx][:,2], facecolor=colors[area_idx+1], alpha=alpha_area, ls="")


    ax.indicate_inset_zoom(axins)

colors = ["tab:blue", "tab:orange"]*2
colors = ["black", *colors]

ls_nominal = (0, (2,5))
#ls_blue = (0, (5,5))
#ls_orange = (5, (5,5))
#ls_green = (0, (5,5))
ls_blue = (0, (3,6))
ls_orange = (4.5, (3,6))
ls_green = (6, (3,6))

f_reference = np.array([10*1e-2 for i in range(len(z))])
Dstar_reference = np.array([20 for i in range(len(z))])
D_reference = np.array([1 for i in range(len(z))])

linestyles = [ls_nominal, ls_blue, ls_orange, "-", "-"]
# Plot results
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6), sharex=True, tight_layout=True)
lw = 1.5
lw_straight = 1
alpha_area = 0.3
alpha_lines = 1
###### Worst-case
### f
# Nominal
#axs[0,0].plot(z*1e3, f_u_worst[:,1]*100, color="black", label="Reference", ls=(0, (2, 5)), lw=2)
axs[0,0].plot(z*1e3, f_reference*100, color="black", label="Reference", ls=(0, (2, 5)), lw=2)
# Uncorrected
#axs[0,0].fill_between(x=z*1e3, y1=f_u_worst[:,0]*100, y2=f_u_worst[:,2]*100, facecolor="tab:blue", alpha=alpha_area, ls="", label="Uncorrected")
#axs[0,0].fill_between(x=z*1e3, y1=f_u_worst[:,0]*100, y2=f_u_worst[:,2]*100, color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[0,0].fill_between(x=z*1e3, y1=f_i_worst[:,0]*100, y2=f_i_worst[:,2]*100, facecolor="tab:orange", alpha=alpha_area, ls="", label="Accounting for imaging")
#axs[0,0].fill_between(x=z*1e3, y1=f_i_worst[:,0]*100, y2=f_i_worst[:,2]*100, color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[0,0].fill_between(x=z*1e3, y1=f_c_worst[:,0]*100, y2=f_c_worst[:,2]*100, facecolor="tab:green", alpha=alpha_area, ls="", label="Accounting for cross-terms")
#axs[0,0].fill_between(x=z*1e3, y1=f_c_worst[:,0]*100, y2=f_c_worst[:,2]*100, color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[0,0].plot(z*1e3, f_u_xyz_worst*100, color="tab:blue", ls=ls_blue, label="[x,y,z] nominal", lw=lw)
axs[0,0].plot(z*1e3, f_i_xyz_worst*100, color="tab:orange", ls=ls_orange, label="[x,y,z] + im", lw=lw)
#axs[0,0].plot(z*1e3, f_c_xyz_worst*100, color="tab:green", ls=ls_green, label="[x,y,z] + im + ct", lw=lw)

#axs[0,0].plot(z*1e3, f_u_GE_6_worst*100, color="tab:blue", ls=(0, (5,5)), label="6 dir", lw=lw)
#axs[0,0].plot(z*1e3, f_i_GE_6_worst*100, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,0].plot(z*1e3, f_c_GE_6_worst*100, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[0,0].plot(z*1e3, f_u_GE_16_worst*100, color="tab:blue", ls="-", label="16 dir", lw=lw)
#axs[0,0].plot(z*1e3, f_i_GE_16_worst*100, color="tab:orange", ls="-", label="", lw=lw)
#axs[0,0].plot(z*1e3, f_c_GE_16_worst*100, color="tab:green", ls="-", label="", lw=lw)

axs[0,0].plot(z*1e3, f_u_xyz_antipodal_worst*100, color="tab:blue", ls="-", label="[x,y,z,-x,-y,-z] nominal", lw=lw_straight)
axs[0,0].plot(z*1e3, f_i_xyz_antipodal_worst*100, color="tab:orange", ls="-", label="[x,y,z,-x,-y,-z] + im", lw=lw_straight)
#axs[0,0].plot(z*1e3, f_c_xyz_antipodal_worst*100, color="tab:green", ls="-", label="[x,y,z,-x,-y,-z] + im + ct", lw=lw_straight)

# Settings
axs[0,0].set_ylabel("$f$ [\%]")
axs[0,0].set_ylim([9.25, 10.75])

### D*
# Nominal
#axs[0,1].plot(z*1e3, Dstar_u_worst[:,1], color="black", ls=(0, (2, 5)), lw=2)
axs[0,1].plot(z*1e3, Dstar_reference, color="black", ls=(0, (2, 5)), lw=2)
# Uncorrected
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_u_worst[:,0], y2=Dstar_u_worst[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_u_worst[:,0], y2=Dstar_u_worst[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_i_worst[:,0], y2=Dstar_i_worst[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_i_worst[:,0], y2=Dstar_i_worst[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_c_worst[:,0], y2=Dstar_c_worst[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_c_worst[:,0], y2=Dstar_c_worst[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[0,1].plot(z*1e3, Dstar_u_xyz_worst, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[0,1].plot(z*1e3, Dstar_i_xyz_worst, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_xyz_worst, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,1].plot(z*1e3, Dstar_u_GE_6_worst, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_i_GE_6_worst, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_GE_6_worst, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[0,1].plot(z*1e3, Dstar_u_GE_16_worst, color="tab:blue", ls="-", label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_i_GE_16_worst, color="tab:orange", ls="-", label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_GE_16_worst, color="tab:green", ls="-", label="", lw=lw)

axs[0,1].plot(z*1e3, Dstar_u_xyz_antipodal_worst, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[0,1].plot(z*1e3, Dstar_i_xyz_antipodal_worst, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[0,1].plot(z*1e3, Dstar_c_xyz_antipodal_worst, color="tab:green", ls="-", label="", lw=lw_straight)
# Settings
axs[0,1].set_ylabel("$D$* [µm$^2$/ms]")
axs[0,1].set_ylim([15, 25])

### D
# Nominal
axs[0,2].plot(z*1e3, D_reference, color="black", ls=(0, (2, 5)), lw=2)
# Uncorrected
#axs[0,2].fill_between(x=z*1e3, y1=D_u_worst[:,0], y2=D_u_worst[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[0,2].fill_between(x=z*1e3, y1=D_u_worst[:,0], y2=D_u_worst[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[0,2].fill_between(x=z*1e3, y1=D_i_worst[:,0], y2=D_i_worst[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[0,2].fill_between(x=z*1e3, y1=D_i_worst[:,0], y2=D_i_worst[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[0,2].fill_between(x=z*1e3, y1=D_c_worst[:,0], y2=D_c_worst[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[0,2].fill_between(x=z*1e3, y1=D_c_worst[:,0], y2=D_c_worst[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[0,2].plot(z*1e3, D_u_xyz_worst, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[0,2].plot(z*1e3, D_i_xyz_worst, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_xyz_worst, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,2].plot(z*1e3, D_u_GE_6_worst, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_i_GE_6_worst, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_GE_6_worst, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[0,2].plot(z*1e3, D_u_GE_16_worst, color="tab:blue", ls="-", label="", lw=lw)
#axs[0,2].plot(z*1e3, D_i_GE_16_worst, color="tab:orange", ls="-", label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_GE_16_worst, color="tab:green", ls="-", label="", lw=lw)

axs[0,2].plot(z*1e3, D_u_xyz_antipodal_worst, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[0,2].plot(z*1e3, D_i_xyz_antipodal_worst, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[0,2].plot(z*1e3, D_c_xyz_antipodal_worst, color="tab:green", ls="-", label="", lw=lw_straight)
# Settings
axs[0,2].set_ylabel("$D$ [µm$^2$/ms]")
axs[0,2].set_ylim([0.95,1.05])
axs[0,2].set_xlim(2, 10)

###### Best-case
### f
# Nominal
axs[1,0].plot(z*1e3, f_reference*1e2, color="black", ls=(0, (2,5)), lw=2)
# Uncorrected
#axs[1,0].fill_between(x=z*1e3, y1=f_u_best[:,0]*100, y2=f_u_best[:,2]*100, facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[1,0].fill_between(x=z*1e3, y1=f_u_best[:,0]*100, y2=f_u_best[:,2]*100, color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[1,0].fill_between(x=z*1e3, y1=f_i_best[:,0]*100, y2=f_i_best[:,2]*100, facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[1,0].fill_between(x=z*1e3, y1=f_i_best[:,0]*100, y2=f_i_best[:,2]*100, color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[1,0].fill_between(x=z*1e3, y1=f_c_best[:,0]*100, y2=f_c_best[:,2]*100, facecolor="tab:green", alpha=alpha_area, ls="")
#axs[1,0].fill_between(x=z*1e3, y1=f_c_best[:,0]*100, y2=f_c_best[:,2]*100, color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[1,0].plot(z*1e3, f_u_xyz_best*100, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[1,0].plot(z*1e3, f_i_xyz_best*100, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[1,0].plot(z*1e3, f_c_xyz_best*100, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[1,0].plot(z*1e3, f_u_GE_6_worst*100, color="tab:blue", ls=(0, (5,5)), label="6 dir", lw=lw)
#axs[1,0].plot(z*1e3, f_i_GE_6_worst*100, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[1,0].plot(z*1e3, f_c_GE_6_worst*100, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[1,0].plot(z*1e3, f_u_GE_16_best*100, color="tab:blue", ls="-", label="", lw=lw)
#axs[1,0].plot(z*1e3, f_i_GE_16_best*100, color="tab:orange", ls="-", label="", lw=lw)
#axs[1,0].plot(z*1e3, f_c_GE_16_best*100, color="tab:green", ls="-", label="", lw=lw)

axs[1,0].plot(z*1e3, f_u_xyz_antipodal_best*100, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[1,0].plot(z*1e3, f_i_xyz_antipodal_best*100, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[1,0].plot(z*1e3, f_c_xyz_antipodal_best*100, color="tab:green", ls="-", label="", lw=lw_straight)

# TEST
#axs[1,0].plot(z*1e3, f_u_xyz_antipodal_best_test*100, color="tab:red", ls="-", label="", lw=lw+1)
# Settings
axs[1,0].set_ylabel("$f$ [\%]")
axs[1,0].set_xlabel("Slice thickness [mm]")
axs[1,0].set_ylim([9.25, 10.75])
# Inset
#f_pwd_avg_lines = [f_reference*100, f_u_xyz_best*100, f_i_xyz_best*100, f_c_xyz_best*100, f_u_xyz_antipodal_best*100, f_i_xyz_antipodal_best*100, f_c_xyz_antipodal_best*100]
f_pwd_avg_lines = [f_reference*100, f_u_xyz_best*100, f_i_xyz_best*100, f_u_xyz_antipodal_best*100, f_i_xyz_antipodal_best*100]
fig_3_inset(axs[1,0], z*1e3, pwd_avg_lines=f_pwd_avg_lines, linestyles=linestyles, colors=colors, xlim=(2, 6), ylim=(9.95, 10.05))

### D*
# Nominal
axs[1,1].plot(z*1e3, Dstar_reference, color="black", ls=(0, (2,5)), lw=2)
# Uncorrected
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_u_best[:,0], y2=Dstar_u_best[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_u_best[:,0], y2=Dstar_u_best[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_i_best[:,0], y2=Dstar_i_best[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_i_best[:,0], y2=Dstar_i_best[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_c_best[:,0], y2=Dstar_c_best[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_c_best[:,0], y2=Dstar_c_best[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[1,1].plot(z*1e3, Dstar_u_xyz_best, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[1,1].plot(z*1e3, Dstar_i_xyz_best, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[1,1].plot(z*1e3, Dstar_c_xyz_best, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,1].plot(z*1e3, Dstar_u_GE_6_best, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_i_GE_6_best, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_GE_6_best, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[1,1].plot(z*1e3, Dstar_u_GE_16_best, color="tab:blue", ls="-", label="", lw=lw)
#axs[1,1].plot(z*1e3, Dstar_i_GE_16_best, color="tab:orange", ls="-", label="", lw=lw)
#axs[1,1].plot(z*1e3, Dstar_c_GE_16_best, color="tab:green", ls="-", label="", lw=lw)

axs[1,1].plot(z*1e3, Dstar_u_xyz_antipodal_best, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[1,1].plot(z*1e3, Dstar_i_xyz_antipodal_best, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[1,1].plot(z*1e3, Dstar_c_xyz_antipodal_best, color="tab:green", ls="-", label="", lw=lw_straight)

# TEST
#axs[1,1].plot(z*1e3, Dstar_u_xyz_antipodal_best_test, color="tab:red", ls="-", label="", lw=lw+1)
# Settings
axs[1,1].set_ylabel("$D$* [µm$^2$/ms]")
axs[1,1].set_xlabel("Slice thickness [mm]")
axs[1,1].set_ylim([15, 25])
# Inset
Dstar_pwd_avg_lines = [Dstar_reference, Dstar_u_xyz_best, Dstar_i_xyz_best, Dstar_u_xyz_antipodal_best, Dstar_i_xyz_antipodal_best]
fig_3_inset(axs[1,1], z*1e3, pwd_avg_lines=Dstar_pwd_avg_lines, linestyles=linestyles, colors=colors, xlim=(2, 6), ylim=(19.5, 20.5))

### D
# Nominal
axs[1,2].plot(z*1e3, D_reference, color="black", ls=(0, (2,5)), lw=2)
# Uncorrected
#axs[1,2].fill_between(x=z*1e3, y1=D_u_best[:,0], y2=D_u_best[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[1,2].fill_between(x=z*1e3, y1=D_u_best[:,0], y2=D_u_best[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[1,2].fill_between(x=z*1e3, y1=D_i_best[:,0], y2=D_i_best[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[1,2].fill_between(x=z*1e3, y1=D_i_best[:,0], y2=D_i_best[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[1,2].fill_between(x=z*1e3, y1=D_c_best[:,0], y2=D_c_best[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[1,2].fill_between(x=z*1e3, y1=D_c_best[:,0], y2=D_c_best[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[1,2].plot(z*1e3, D_u_xyz_best, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[1,2].plot(z*1e3, D_i_xyz_best, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[1,2].plot(z*1e3, D_c_xyz_best, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,2].plot(z*1e3, D_u_GE_6_best, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_i_GE_6_best, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_GE_6_best, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[1,2].plot(z*1e3, D_u_GE_16_best, color="tab:blue", ls="-", label="", lw=lw)
#axs[1,2].plot(z*1e3, D_i_GE_16_best, color="tab:orange", ls="-", label="", lw=lw)
#axs[1,2].plot(z*1e3, D_c_GE_16_best, color="tab:green", ls="-", label="", lw=lw)

axs[1,2].plot(z*1e3, D_u_xyz_antipodal_best, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[1,2].plot(z*1e3, D_i_xyz_antipodal_best, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[1,2].plot(z*1e3, D_c_xyz_antipodal_best, color="tab:green", ls="-", label="", lw=lw_straight)

# TEST
#axs[1,2].plot(z*1e3, D_u_xyz_antipodal_best_test, color="tab:red", ls="-", label="", lw=lw+1)
# Settings
axs[1,2].set_ylabel("$D$ [µm$^2$/ms]")
axs[1,2].set_xlabel("Slice thickness [mm]")
axs[1,2].set_ylim([0.95,1.05])
# Inset
D_pwd_avg_lines = [D_reference, D_u_xyz_best, D_i_xyz_best, D_u_xyz_antipodal_best, D_i_xyz_antipodal_best]
fig_3_inset(axs[1,2], z*1e3, pwd_avg_lines=D_pwd_avg_lines, linestyles=linestyles, colors=colors, xlim=(2, 6), ylim=(0.995, 1.005))

#fig.suptitle("Well-designed sequence")
#fig.legend(loc=(.09, -.005), ncols=4, frameon=False)
#fig.legend(loc=(0, -.005), ncols=7, frameon=False)
fig.legend(loc=(0.02, -.005), ncols=5, frameon=False)
handles, labels = axs[0,0].get_legend_handles_labels()
#handles_reordered = [handles[1], handles[4], handles[2], handles[5], handles[3], handles[6], handles[0]]
#labels_reordered = [labels[1], labels[4], labels[2], labels[5], labels[3], labels[6], labels[0]]
#fig.legend(handles_reordered, labels_reordered, loc=(0.045, -.025), ncols=4, frameon=False)
#fig.legend(handles_reordered, labels_reordered, ncols=4, frameon=False, bbox_to_anchor=[0.99,0.035])
#fig.tight_layout()
#fig.text(1, .75, "Large cross-terms")
fig.text(.99, .7, "Large cross-terms")
fig.text(.99, .3, "Minimal cross-terms")
#fig.suptitle("IVIM parameter ranges vs. slice thickness", x=.53, y=1.02)
fig.suptitle("IVIM parameter ranges vs. slice thickness", x=.53, y=0.95)
plt.margins(y=2)

fig.savefig(os.path.join(fig_save_path, "fig3_20241213.pdf"), bbox_inches="tight")

# %% Figure 4 20241213

folder_best_case = "optimal_allCrushers_onlyCrushWhenNeeded_sequence"
folder_worst_case = "allCrushers_sequence"

xy = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 2.75e-3, 3e-3, 3.25e-3, 3.5e-3, 3.75e-3, 4e-3])
z = np.array([4e-3 for i in range(len(xy))])

#path_best = os.path.join(save_path, folder_best_case)
#path_worst = os.path.join(save_path, folder_worst_case)

path_test = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\test\optimal_allCrushers_onlyCrushWhenNeeded_sequence"

#nom_b0 = np.array([3.])
# 1000 dir sets
#f_u_worst, Dstar_u_worst, D_u_worst, bvals_nom_u_worst, minimum_bvalues_u_worst_z, maximum_bvalues_u_worst_z = read_all_generic_simulation_logs(path_worst, correction="uncorrected", angles="ndir1000", xyres=xy, zres=z)
#f_i_worst, Dstar_i_worst, D_i_worst, bvals_nom_i_worst, minimum_bvalues_i_worst_z, maximum_bvalues_i_worst_z = read_all_generic_simulation_logs(path_worst, correction="imaging_corrected", angles="ndir1000", xyres=xy, zres=z)
#f_c_worst, Dstar_c_worst, D_c_worst, bvals_nom_c_worst, minimum_bvalues_c_worst_z, maximum_bvalues_c_worst_z = read_all_generic_simulation_logs(path_worst, correction="crossterm_corrected", angles="ndir1000", xyres=xy, zres=z)

#f_u_best, Dstar_u_best, D_u_best, bvals_nom_u_best, minimum_bvalues_u_best_z, maximum_bvalues_u_best_z = read_all_generic_simulation_logs(path_best, correction="uncorrected", angles="ndir1000", xyres=xy, zres=z)
#f_i_best, Dstar_i_best, D_i_best, bvals_nom_i_best, minimum_bvalues_i_best_z, maximum_bvalues_i_best_z = read_all_generic_simulation_logs(path_best, correction="imaging_corrected", angles="ndir1000", xyres=xy, zres=z)
#f_c_best, Dstar_c_best, D_c_best, bvals_nom_c_best, minimum_bvalues_c_best_z, maximum_bvalues_c_best_z = read_all_generic_simulation_logs(path_best, correction="crossterm_corrected", angles="ndir1000", xyres=xy, zres=z)


### Powder averages
# Worst
#f_u_xyz_worst, Dstar_u_xyz_worst, D_u_xyz_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
#f_i_xyz_worst, Dstar_i_xyz_worst, D_i_xyz_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
#f_c_xyz_worst, Dstar_c_xyz_worst, D_c_xyz_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

#f_u_xyz_antipodal_worst, Dstar_u_xyz_antipodal_worst, D_u_xyz_antipodal_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_i_xyz_antipodal_worst, Dstar_i_xyz_antipodal_worst, D_i_xyz_antipodal_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_c_xyz_antipodal_worst, Dstar_c_xyz_antipodal_worst, D_c_xyz_antipodal_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 

#f_u_GE_6_worst, Dstar_u_GE_6_worst, D_u_GE_6_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="GE_6", xyres=xy, zres=z) 
#f_i_GE_6_worst, Dstar_i_GE_6_worst, D_i_GE_6_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="GE_6", xyres=xy, zres=z) 
#f_c_GE_6_worst, Dstar_c_GE_6_worst, D_c_GE_6_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="GE_6", xyres=xy, zres=z) 

#f_u_GE_16_worst, Dstar_u_GE_16_worst, D_u_GE_16_worst = powder_average_all_simulation_logs(path_worst, correction="uncorrected", angles="GE_16", xyres=xy, zres=z) 
#f_i_GE_16_worst, Dstar_i_GE_16_worst, D_i_GE_16_worst = powder_average_all_simulation_logs(path_worst, correction="imaging_corrected", angles="GE_16", xyres=xy, zres=z) 
#f_c_GE_16_worst, Dstar_c_GE_16_worst, D_c_GE_16_worst = powder_average_all_simulation_logs(path_worst, correction="crossterm_corrected", angles="GE_16", xyres=xy, zres=z) 
# Best
#f_u_xyz_best, Dstar_u_xyz_best, D_u_xyz_best = powder_average_all_simulation_logs(path_best, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
#f_i_xyz_best, Dstar_i_xyz_best, D_i_xyz_best = powder_average_all_simulation_logs(path_best, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
#f_c_xyz_best, Dstar_c_xyz_best, D_c_xyz_best = powder_average_all_simulation_logs(path_best, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

#f_u_xyz_antipodal_best, Dstar_u_xyz_antipodal_best, D_u_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_test, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_i_xyz_antipodal_best, Dstar_i_xyz_antipodal_best, D_i_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_test, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
#f_c_xyz_antipodal_best, Dstar_c_xyz_antipodal_best, D_c_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_test, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 

#f_u_xyz_antipodal_best_test, Dstar_u_xyz_antipodal_best_test, D_u_xyz_antipodal_best_test = powder_average_all_simulation_logs(path_test, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 

#f_u_GE_6_best, Dstar_u_GE_6_best, D_u_GE_6_best = powder_average_all_simulation_logs(path_best, correction="uncorrected", angles="GE_6", xyres=xy, zres=z) 
#f_i_GE_6_best, Dstar_i_GE_6_best, D_i_GE_6_best = powder_average_all_simulation_logs(path_best, correction="imaging_corrected", angles="GE_6", xyres=xy, zres=z) 
#f_c_GE_6_best, Dstar_c_GE_6_best, D_c_GE_6_best = powder_average_all_simulation_logs(path_best, correction="crossterm_corrected", angles="GE_6", xyres=xy, zres=z) 

#f_u_GE_16_best, Dstar_u_GE_16_best, D_u_GE_16_best = powder_average_all_simulation_logs(path_best, correction="uncorrected", angles="GE_16", xyres=xy, zres=z) 
#f_i_GE_16_best, Dstar_i_GE_16_best, D_i_GE_16_best = powder_average_all_simulation_logs(path_best, correction="imaging_corrected", angles="GE_16", xyres=xy, zres=z) 
#f_c_GE_16_best, Dstar_c_GE_16_best, D_c_GE_16_best = powder_average_all_simulation_logs(path_best, correction="crossterm_corrected", angles="GE_16", xyres=xy, zres=z) 


## NEW
path_best = os.path.join(save_path, "20241101", folder_best_case)
path_worst = os.path.join(save_path, "20241101", folder_worst_case)

# Froeling 200dir sets
#f_u_worst, Dstar_u_worst, D_u_worst, bvals_nom_u_worst, minimum_bvalues_u_worst_z, maximum_bvalues_u_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="uncorrected", angles="froeling_200", xyres=xy, zres=z)
#f_i_worst, Dstar_i_worst, D_i_worst, bvals_nom_i_worst, minimum_bvalues_i_worst_z, maximum_bvalues_i_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="imaging_corrected", angles="froeling_200", xyres=xy, zres=z)
#f_c_worst, Dstar_c_worst, D_c_worst, bvals_nom_c_worst, minimum_bvalues_c_worst_z, maximum_bvalues_c_worst_z = read_all_generic_simulation_logs_uvecs(path_worst, correction="crossterm_corrected", angles="froeling_200", xyres=xy, zres=z)

#f_u_best, Dstar_u_best, D_u_best, bvals_nom_u_best, minimum_bvalues_u_best_z, maximum_bvalues_u_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="uncorrected", angles="froeling_200", xyres=xy, zres=z)
#f_i_best, Dstar_i_best, D_i_best, bvals_nom_i_best, minimum_bvalues_i_best_z, maximum_bvalues_i_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="imaging_corrected", angles="froeling_200", xyres=xy, zres=z)
#f_c_best, Dstar_c_best, D_c_best, bvals_nom_c_best, minimum_bvalues_c_best_z, maximum_bvalues_c_best_z = read_all_generic_simulation_logs_uvecs(path_best, correction="crossterm_corrected", angles="froeling_200", xyres=xy, zres=z)

# Worst
f_u_xyz_worst, Dstar_u_xyz_worst, D_u_xyz_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
f_i_xyz_worst, Dstar_i_xyz_worst, D_i_xyz_worst = powder_average_all_simulation_logs_new(path_worst, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
f_c_xyz_worst, Dstar_c_xyz_worst, D_c_xyz_worst = powder_average_all_simulation_logs_new(path_worst, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

f_u_xyz_antipodal_worst, Dstar_u_xyz_antipodal_worst, D_u_xyz_antipodal_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_i_xyz_antipodal_worst, Dstar_i_xyz_antipodal_worst, D_i_xyz_antipodal_worst = powder_average_all_simulation_logs_new(path_worst, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_c_xyz_antipodal_worst, Dstar_c_xyz_antipodal_worst, D_c_xyz_antipodal_worst = powder_average_all_simulation_logs_new(path_worst, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 

# Best
f_u_xyz_best, Dstar_u_xyz_best, D_u_xyz_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz", xyres=xy, zres=z) 
f_i_xyz_best, Dstar_i_xyz_best, D_i_xyz_best = powder_average_all_simulation_logs_new(path_best, correction="imaging_corrected", angles="xyz", xyres=xy, zres=z) 
f_c_xyz_best, Dstar_c_xyz_best, D_c_xyz_best = powder_average_all_simulation_logs_new(path_best, correction="crossterm_corrected", angles="xyz", xyres=xy, zres=z) 

f_u_xyz_antipodal_best, Dstar_u_xyz_antipodal_best, D_u_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_i_xyz_antipodal_best, Dstar_i_xyz_antipodal_best, D_i_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_best, correction="imaging_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 
f_c_xyz_antipodal_best, Dstar_c_xyz_antipodal_best, D_c_xyz_antipodal_best = powder_average_all_simulation_logs_new(path_best, correction="crossterm_corrected", angles="xyz_antipodal", xyres=xy, zres=z) 


def fig_3_inset(ax, x_axis, pwd_avg_lines, linestyles, colors, xlim, ylim, position=[0.45, 0.65, 0.5, 0.3], areas=None):
    axins = ax.inset_axes(position, xlim=xlim, ylim=ylim)



    for line_idx in range(len(pwd_avg_lines)):
        if linestyles[line_idx] == (0, (2,5)):
            axins.plot(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=2, ls=linestyles[line_idx])
        elif linestyles[line_idx] != "-":
            axins.plot(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=1.5, ls=linestyles[line_idx])
        else:
            axins.plot(x_axis, pwd_avg_lines[line_idx], color=colors[line_idx], linewidth=1, ls=linestyles[line_idx])

    if areas:
        for area_idx in range(len(areas)):
            axins.fill_between(x=x_axis, y1=areas[area_idx][:,0], y2=areas[area_idx][:,2], facecolor=colors[area_idx+1], alpha=alpha_area, ls="")


    ax.indicate_inset_zoom(axins)

colors = ["tab:blue", "tab:orange"]*2
colors = ["black", *colors]

ls_nominal = (0, (2,5))
#ls_blue = (0, (5,5))
#ls_orange = (5, (5,5))
#ls_green = (0, (5,5))
ls_blue = (0, (3,6))
ls_orange = (4.5, (3,6))
ls_green = (6, (3,6))

f_reference = np.array([10*1e-2 for i in range(len(z))])
Dstar_reference = np.array([20 for i in range(len(z))])
D_reference = np.array([1 for i in range(len(z))])

linestyles = [ls_nominal, ls_blue, ls_orange, "-", "-"]
# Plot results
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10,6), sharex=True, tight_layout=True)
lw = 1.5
lw_straight = 1
alpha_area = 0.3
alpha_lines = 1
###### Worst-case
### f
# Nominal
#axs[0,0].plot(z*1e3, f_u_worst[:,1]*100, color="black", label="Reference", ls=(0, (2, 5)), lw=2)
axs[0,0].plot(xy*1e3, f_reference*100, color="black", label="Reference", ls=(0, (2, 5)), lw=2)
# Uncorrected
#axs[0,0].fill_between(x=z*1e3, y1=f_u_worst[:,0]*100, y2=f_u_worst[:,2]*100, facecolor="tab:blue", alpha=alpha_area, ls="", label="Uncorrected")
#axs[0,0].fill_between(x=z*1e3, y1=f_u_worst[:,0]*100, y2=f_u_worst[:,2]*100, color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[0,0].fill_between(x=z*1e3, y1=f_i_worst[:,0]*100, y2=f_i_worst[:,2]*100, facecolor="tab:orange", alpha=alpha_area, ls="", label="Accounting for imaging")
#axs[0,0].fill_between(x=z*1e3, y1=f_i_worst[:,0]*100, y2=f_i_worst[:,2]*100, color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[0,0].fill_between(x=z*1e3, y1=f_c_worst[:,0]*100, y2=f_c_worst[:,2]*100, facecolor="tab:green", alpha=alpha_area, ls="", label="Accounting for cross-terms")
#axs[0,0].fill_between(x=z*1e3, y1=f_c_worst[:,0]*100, y2=f_c_worst[:,2]*100, color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[0,0].plot(xy*1e3, f_u_xyz_worst*100, color="tab:blue", ls=ls_blue, label="[x,y,z] nominal", lw=lw)
axs[0,0].plot(xy*1e3, f_i_xyz_worst*100, color="tab:orange", ls=ls_orange, label="[x,y,z] + im", lw=lw)
#axs[0,0].plot(xy*1e3, f_c_xyz_worst*100, color="tab:green", ls=ls_green, label="[x,y,z] + im + ct", lw=lw)

#axs[0,0].plot(z*1e3, f_u_GE_6_worst*100, color="tab:blue", ls=(0, (5,5)), label="6 dir", lw=lw)
#axs[0,0].plot(z*1e3, f_i_GE_6_worst*100, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,0].plot(z*1e3, f_c_GE_6_worst*100, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[0,0].plot(z*1e3, f_u_GE_16_worst*100, color="tab:blue", ls="-", label="16 dir", lw=lw)
#axs[0,0].plot(z*1e3, f_i_GE_16_worst*100, color="tab:orange", ls="-", label="", lw=lw)
#axs[0,0].plot(z*1e3, f_c_GE_16_worst*100, color="tab:green", ls="-", label="", lw=lw)

axs[0,0].plot(xy*1e3, f_u_xyz_antipodal_worst*100, color="tab:blue", ls="-", label="[x,y,z,-x,-y,-z] nominal", lw=lw_straight)
axs[0,0].plot(xy*1e3, f_i_xyz_antipodal_worst*100, color="tab:orange", ls="-", label="[x,y,z,-x,-y,-z] + im", lw=lw_straight)
#axs[0,0].plot(xy*1e3, f_c_xyz_antipodal_worst*100, color="tab:green", ls="-", label="[x,y,z,-x,-y,-z] + im + ct", lw=lw_straight)

# Settings
axs[0,0].set_ylabel("$f$ [\%]")
axs[0,0].set_ylim([8.5, 11.5])

### D*
# Nominal
#axs[0,1].plot(z*1e3, Dstar_u_worst[:,1], color="black", ls=(0, (2, 5)), lw=2)
axs[0,1].plot(xy*1e3, Dstar_reference, color="black", ls=(0, (2, 5)), lw=2)
# Uncorrected
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_u_worst[:,0], y2=Dstar_u_worst[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_u_worst[:,0], y2=Dstar_u_worst[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_i_worst[:,0], y2=Dstar_i_worst[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_i_worst[:,0], y2=Dstar_i_worst[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_c_worst[:,0], y2=Dstar_c_worst[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[0,1].fill_between(x=z*1e3, y1=Dstar_c_worst[:,0], y2=Dstar_c_worst[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[0,1].plot(xy*1e3, Dstar_u_xyz_worst, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[0,1].plot(xy*1e3, Dstar_i_xyz_worst, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[0,1].plot(xy*1e3, Dstar_c_xyz_worst, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,1].plot(z*1e3, Dstar_u_GE_6_worst, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_i_GE_6_worst, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_GE_6_worst, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[0,1].plot(z*1e3, Dstar_u_GE_16_worst, color="tab:blue", ls="-", label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_i_GE_16_worst, color="tab:orange", ls="-", label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_GE_16_worst, color="tab:green", ls="-", label="", lw=lw)

axs[0,1].plot(xy*1e3, Dstar_u_xyz_antipodal_worst, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[0,1].plot(xy*1e3, Dstar_i_xyz_antipodal_worst, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[0,1].plot(xy*1e3, Dstar_c_xyz_antipodal_worst, color="tab:green", ls="-", label="", lw=lw_straight)
# Settings
axs[0,1].set_ylabel("$D$* [µm$^2$/ms]")
axs[0,1].set_ylim([14, 26])

### D
# Nominal
axs[0,2].plot(xy*1e3, D_reference, color="black", ls=(0, (2, 5)), lw=2)
# Uncorrected
#axs[0,2].fill_between(x=z*1e3, y1=D_u_worst[:,0], y2=D_u_worst[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[0,2].fill_between(x=z*1e3, y1=D_u_worst[:,0], y2=D_u_worst[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[0,2].fill_between(x=z*1e3, y1=D_i_worst[:,0], y2=D_i_worst[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[0,2].fill_between(x=z*1e3, y1=D_i_worst[:,0], y2=D_i_worst[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[0,2].fill_between(x=z*1e3, y1=D_c_worst[:,0], y2=D_c_worst[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[0,2].fill_between(x=z*1e3, y1=D_c_worst[:,0], y2=D_c_worst[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[0,2].plot(xy*1e3, D_u_xyz_worst, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[0,2].plot(xy*1e3, D_i_xyz_worst, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[0,2].plot(xy*1e3, D_c_xyz_worst, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,2].plot(z*1e3, D_u_GE_6_worst, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_i_GE_6_worst, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_GE_6_worst, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[0,2].plot(z*1e3, D_u_GE_16_worst, color="tab:blue", ls="-", label="", lw=lw)
#axs[0,2].plot(z*1e3, D_i_GE_16_worst, color="tab:orange", ls="-", label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_GE_16_worst, color="tab:green", ls="-", label="", lw=lw)

axs[0,2].plot(xy*1e3, D_u_xyz_antipodal_worst, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[0,2].plot(xy*1e3, D_i_xyz_antipodal_worst, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[0,2].plot(xy*1e3, D_c_xyz_antipodal_worst, color="tab:green", ls="-", label="", lw=lw_straight)
# Settings
axs[0,2].set_ylabel("$D$ [µm$^2$/ms]")
axs[0,2].set_ylim([0.94,1.06])
axs[0,2].set_xlim(1, 4)

###### Best-case
### f
# Nominal
axs[1,0].plot(xy*1e3, f_reference*1e2, color="black", ls=(0, (2,5)), lw=2)
# Uncorrected
#axs[1,0].fill_between(x=z*1e3, y1=f_u_best[:,0]*100, y2=f_u_best[:,2]*100, facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[1,0].fill_between(x=z*1e3, y1=f_u_best[:,0]*100, y2=f_u_best[:,2]*100, color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[1,0].fill_between(x=z*1e3, y1=f_i_best[:,0]*100, y2=f_i_best[:,2]*100, facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[1,0].fill_between(x=z*1e3, y1=f_i_best[:,0]*100, y2=f_i_best[:,2]*100, color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[1,0].fill_between(x=z*1e3, y1=f_c_best[:,0]*100, y2=f_c_best[:,2]*100, facecolor="tab:green", alpha=alpha_area, ls="")
#axs[1,0].fill_between(x=z*1e3, y1=f_c_best[:,0]*100, y2=f_c_best[:,2]*100, color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[1,0].plot(xy*1e3, f_u_xyz_best*100, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[1,0].plot(xy*1e3, f_i_xyz_best*100, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[1,0].plot(xy*1e3, f_c_xyz_best*100, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[1,0].plot(z*1e3, f_u_GE_6_worst*100, color="tab:blue", ls=(0, (5,5)), label="6 dir", lw=lw)
#axs[1,0].plot(z*1e3, f_i_GE_6_worst*100, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[1,0].plot(z*1e3, f_c_GE_6_worst*100, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[1,0].plot(z*1e3, f_u_GE_16_best*100, color="tab:blue", ls="-", label="", lw=lw)
#axs[1,0].plot(z*1e3, f_i_GE_16_best*100, color="tab:orange", ls="-", label="", lw=lw)
#axs[1,0].plot(z*1e3, f_c_GE_16_best*100, color="tab:green", ls="-", label="", lw=lw)

axs[1,0].plot(xy*1e3, f_u_xyz_antipodal_best*100, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[1,0].plot(xy*1e3, f_i_xyz_antipodal_best*100, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[1,0].plot(xy*1e3, f_c_xyz_antipodal_best*100, color="tab:green", ls="-", label="", lw=lw_straight)

# TEST
#axs[1,0].plot(z*1e3, f_u_xyz_antipodal_best_test*100, color="tab:red", ls="-", label="", lw=lw+1)
# Settings
axs[1,0].set_ylabel("$f$ [\%]")
axs[1,0].set_xlabel("In-plane resolution [mm]")
axs[1,0].set_ylim([8.5, 11.5])
# Inset
f_pwd_avg_lines = [f_reference*100, f_u_xyz_best*100, f_i_xyz_best*100, f_u_xyz_antipodal_best*100, f_i_xyz_antipodal_best*100]
fig_3_inset(axs[1,0], xy*1e3, pwd_avg_lines=f_pwd_avg_lines, linestyles=linestyles, colors=colors, xlim=(1, 2), ylim=(9.95, 10.05))

### D*
# Nominal
axs[1,1].plot(xy*1e3, Dstar_reference, color="black", ls=(0, (2,5)), lw=2)
# Uncorrected
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_u_best[:,0], y2=Dstar_u_best[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_u_best[:,0], y2=Dstar_u_best[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_i_best[:,0], y2=Dstar_i_best[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_i_best[:,0], y2=Dstar_i_best[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_c_best[:,0], y2=Dstar_c_best[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[1,1].fill_between(x=z*1e3, y1=Dstar_c_best[:,0], y2=Dstar_c_best[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[1,1].plot(xy*1e3, Dstar_u_xyz_best, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[1,1].plot(xy*1e3, Dstar_i_xyz_best, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[1,1].plot(xy*1e3, Dstar_c_xyz_best, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,1].plot(z*1e3, Dstar_u_GE_6_best, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_i_GE_6_best, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,1].plot(z*1e3, Dstar_c_GE_6_best, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[1,1].plot(z*1e3, Dstar_u_GE_16_best, color="tab:blue", ls="-", label="", lw=lw)
#axs[1,1].plot(z*1e3, Dstar_i_GE_16_best, color="tab:orange", ls="-", label="", lw=lw)
#axs[1,1].plot(z*1e3, Dstar_c_GE_16_best, color="tab:green", ls="-", label="", lw=lw)

axs[1,1].plot(xy*1e3, Dstar_u_xyz_antipodal_best, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[1,1].plot(xy*1e3, Dstar_i_xyz_antipodal_best, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[1,1].plot(xy*1e3, Dstar_c_xyz_antipodal_best, color="tab:green", ls="-", label="", lw=lw_straight)

# TEST
#axs[1,1].plot(z*1e3, Dstar_u_xyz_antipodal_best_test, color="tab:red", ls="-", label="", lw=lw+1)
# Settings
axs[1,1].set_ylabel("$D$* [µm$^2$/ms]")
axs[1,1].set_xlabel("In-plane resolution [mm]")
axs[1,1].set_ylim([14, 26])
# Inset
Dstar_pwd_avg_lines = [Dstar_reference, Dstar_u_xyz_best, Dstar_i_xyz_best, Dstar_u_xyz_antipodal_best, Dstar_i_xyz_antipodal_best]
fig_3_inset(axs[1,1], xy*1e3, pwd_avg_lines=Dstar_pwd_avg_lines, linestyles=linestyles, colors=colors, xlim=(1, 2), ylim=(19.5, 20.5))

### D
# Nominal
axs[1,2].plot(xy*1e3, D_reference, color="black", ls=(0, (2,5)), lw=2)
# Uncorrected
#axs[1,2].fill_between(x=z*1e3, y1=D_u_best[:,0], y2=D_u_best[:,2], facecolor="tab:blue", alpha=alpha_area, ls="")
#axs[1,2].fill_between(x=z*1e3, y1=D_u_best[:,0], y2=D_u_best[:,2], color="tab:blue", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Imaging corrected
#axs[1,2].fill_between(x=z*1e3, y1=D_i_best[:,0], y2=D_i_best[:,2], facecolor="tab:orange", alpha=alpha_area, ls="")
#axs[1,2].fill_between(x=z*1e3, y1=D_i_best[:,0], y2=D_i_best[:,2], color="tab:orange", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Cross-term corrected
#axs[1,2].fill_between(x=z*1e3, y1=D_c_best[:,0], y2=D_c_best[:,2], facecolor="tab:green", alpha=alpha_area, ls="")
#axs[1,2].fill_between(x=z*1e3, y1=D_c_best[:,0], y2=D_c_best[:,2], color="tab:green", facecolor="None", alpha=alpha_lines, ls="-", lw=lw)
# Powder avgs xyz
axs[1,2].plot(xy*1e3, D_u_xyz_best, color="tab:blue", ls=ls_blue, label="", lw=lw)
axs[1,2].plot(xy*1e3, D_i_xyz_best, color="tab:orange", ls=ls_orange, label="", lw=lw)
#axs[1,2].plot(xy*1e3, D_c_xyz_best, color="tab:green", ls=ls_green, label="", lw=lw)

#axs[0,2].plot(z*1e3, D_u_GE_6_best, color="tab:blue", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_i_GE_6_best, color="tab:orange", ls=(0, (5,5)), label="", lw=lw)
#axs[0,2].plot(z*1e3, D_c_GE_6_best, color="tab:green", ls=(0, (5,5)), label="", lw=lw)

#axs[1,2].plot(z*1e3, D_u_GE_16_best, color="tab:blue", ls="-", label="", lw=lw)
#axs[1,2].plot(z*1e3, D_i_GE_16_best, color="tab:orange", ls="-", label="", lw=lw)
#axs[1,2].plot(z*1e3, D_c_GE_16_best, color="tab:green", ls="-", label="", lw=lw)

axs[1,2].plot(xy*1e3, D_u_xyz_antipodal_best, color="tab:blue", ls="-", label="", lw=lw_straight)
axs[1,2].plot(xy*1e3, D_i_xyz_antipodal_best, color="tab:orange", ls="-", label="", lw=lw_straight)
#axs[1,2].plot(xy*1e3, D_c_xyz_antipodal_best, color="tab:green", ls="-", label="", lw=lw_straight)

# TEST
#axs[1,2].plot(z*1e3, D_u_xyz_antipodal_best_test, color="tab:red", ls="-", label="", lw=lw+1)
# Settings
axs[1,2].set_ylabel("$D$ [µm$^2$/ms]")
axs[1,2].set_xlabel("In-plane resolution [mm]")
axs[1,2].set_ylim([0.94,1.06])
# Inset
D_pwd_avg_lines = [D_reference, D_u_xyz_best, D_i_xyz_best, D_u_xyz_antipodal_best, D_i_xyz_antipodal_best]
fig_3_inset(axs[1,2], xy*1e3, pwd_avg_lines=D_pwd_avg_lines, linestyles=linestyles, colors=colors, xlim=(1, 2), ylim=(0.995, 1.005))

#fig.suptitle("Well-designed sequence")
#fig.legend(loc=(.09, -.005), ncols=4, frameon=False)
#fig.legend(loc=(0, -.005), ncols=7, frameon=False)
fig.legend(loc=(0.02, -.005), ncols=5, frameon=False)
#handles, labels = axs[0,0].get_legend_handles_labels()
#handles_reordered = [handles[1], handles[4], handles[2], handles[5], handles[3], handles[6], handles[0]]
#labels_reordered = [labels[1], labels[4], labels[2], labels[5], labels[3], labels[6], labels[0]]
#fig.legend(handles_reordered, labels_reordered, loc=(0.045, -.025), ncols=4, frameon=False)
#fig.legend(handles_reordered, labels_reordered, ncols=4, frameon=False, bbox_to_anchor=[0.99,0.035])
#fig.tight_layout()
#fig.text(1, .75, "Large cross-terms")
fig.text(.99, .7, "Large cross-terms")
fig.text(.99, .3, "Minimal cross-terms")
#fig.suptitle("IVIM parameter ranges vs. slice thickness", x=.53, y=1.02)
fig.suptitle("IVIM parameter ranges vs. in-plane resolution", x=.53, y=0.95)
plt.margins(y=2)

fig.savefig(os.path.join(fig_save_path, "fig4_20241213.pdf"), bbox_inches="tight")
