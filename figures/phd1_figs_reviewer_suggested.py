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
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid
import matplotlib.colors as colors

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

fig_save_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\MRM_submission\revision 1\figure drafts"

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

def powder_average_all_simulation_logs_new(path, correction="uncorrected", angles="xyz", xyres=[1e-3], zres=[1e-3], f=.1, Dstar=20e-3, D=1e-3):
    f_array = np.zeros(len(xyres))
    Dstar_array = np.zeros(len(xyres))
    D_array = np.zeros(len(xyres))

    for idx in range(len(xyres)):
        # Build the filename string
        reported_fname = f"{correction}_{angles}_xy{xyres[idx]}_z{zres[idx]}_bvalues_actual.npy"
        actual_fname = f"crossterm_corrected_{angles}_xy{xyres[idx]}_z{zres[idx]}"

        # Get true powder averaged signals using actual bvals
        #powder_averaged_signals = bval_calc_tools.powder_average_signals_from_file(os.path.join(path, actual_fname))
        signals = bval_calc_tools.signals_from_file(os.path.join(path, actual_fname), f=f, Dstar=Dstar, D=D)
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

        f_array[idx] = estimates.perfusion_fraction
        Dstar_array[idx] = estimates.D_star
        D_array[idx] = estimates.D

    return f_array, Dstar_array, D_array

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

# %% Curved text from https://stackoverflow.com/questions/19353576/curved-text-rendering-in-matplotlib

#from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
#import numpy as np
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used

# %% b0 vs isotropic resolution

resolutions = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3])
resolutions = np.round(np.linspace(1,4,13)*1e-3, decimals=7)

bvalues_nominal_worst, bvalues_actual_worst = read_simulation_files_and_average(path_worst, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)
bvalues_nominal_best, bvalues_actual_best = read_simulation_files_and_average(path_best, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)

# Get the b0's for each isotropic resolution
# b0 is independent of diffusion direction so we only show for one of them
#b0_actual_xy_worst = bvalues_actual_worst[0, -1, :]
#b0_actual_xy_best = bvalues_actual_best[0, -1, :]

b0_actual_z_worst = bvalues_actual_worst[-1, -1, :]
b0_actual_z_best = bvalues_actual_best[-1, -1, :]

fig, ax = plt.subplots(figsize=(4,4))
ax.plot(resolutions*1e3, b0_actual_z_worst, label="Large cross-terms", color="tab:blue", ls="-", lw=lw+.5)
ax.plot(resolutions*1e3, b0_actual_z_best, label="Minimal cross-terms", color="tab:blue", ls="--", lw=lw+.5)

ax.set_xlim(1.0, 4.0)
ax.set_ylim(0)

ax.set_xlabel("Isotropic resolution [mm]")
ax.set_ylabel("Actual b0 [s/mm$^2$]")

ax.legend(frameon=False)

#fig.savefig(os.path.join(fig_save_path, "Figure_reviewer_1.tiff"), dpi=550, bbox_inches="tight")


# %% Relative b-value error vs b-value

resolutions = np.array([1e-3, 4e-3]) # High-res and low-res

bvalues_nominal_worst, bvalues_actual_worst = read_simulation_files_and_average(path_worst, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)
bvalues_nominal_best, bvalues_actual_best = read_simulation_files_and_average(path_best, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)

bvalues_actual_best_xy_highres = bvalues_actual_best[-1, :-1, 0]
bvalues_actual_best_xy_lowres = bvalues_actual_best[-1, :-1, 1]

bvalues_actual_worst_xy_highres = bvalues_actual_worst[-1, :-1, 0]
bvalues_actual_worst_xy_lowres = bvalues_actual_worst[-1, :-1, 1]

bvalues_actual_best_xy_highres_relative_error = (bvalues_actual_best_xy_highres-bvalues_nominal_best[:-1])/bvalues_nominal_best[:-1]
bvalues_actual_best_xy_lowres_relative_error = (bvalues_actual_best_xy_lowres-bvalues_nominal_best[:-1])/bvalues_nominal_best[:-1]

bvalues_actual_worst_xy_highres_relative_error = (bvalues_actual_worst_xy_highres-bvalues_nominal_worst[:-1])/bvalues_nominal_worst[:-1]
bvalues_actual_worst_xy_lowres_relative_error = (bvalues_actual_worst_xy_lowres-bvalues_nominal_worst[:-1])/bvalues_nominal_worst[:-1]

fig, axs = plt.subplots(figsize=(4,6.5), nrows=2, sharex=True)
#axs[0].plot(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_highres_relative_error*100)
#axs[0].plot(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_lowres_relative_error*100)
axs[0].fill_between(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_lowres_relative_error*100, bvalues_actual_worst_xy_highres_relative_error*100, alpha=alpha_area)

#axs[0].plot(bvalues_nominal_best[:-1], bvalues_actual_best_xy_highres_relative_error*100)
#axs[0].plot(bvalues_nominal_best[:-1], bvalues_actual_best_xy_lowres_relative_error*100)
axs[0].fill_between(bvalues_nominal_best[:-1], bvalues_actual_best_xy_lowres_relative_error*100, bvalues_actual_best_xy_highres_relative_error*100, alpha=alpha_area)

# xy
#high_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[13:], y=np.flip(bvalues_actual_best_xy_highres_relative_error*140)[13:], text="High resolution", axes=axs[0])
#low_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[11:], y=np.flip(bvalues_actual_best_xy_lowres_relative_error*140)[11:], text="Low resolution", axes=axs[0])
#crushers_off_text = axs[0].annotate("Crushers off", (200,100), xytext=(10,5), rotation=-77, fontsize=10)

# z
high_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[12:], y=np.flip(bvalues_actual_best_xy_highres_relative_error*115)[12:], text="High resolution", axes=axs[0])
low_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[11:], y=np.flip(bvalues_actual_best_xy_lowres_relative_error*115)[11:], text="Low resolution", axes=axs[0])
crushers_off_text = axs[0].annotate("Crushers off", (200,100), xytext=(10,19), rotation=-77, fontsize=10)

axs[0].set_xscale("log")
axs[0].set_yscale("log")
import matplotlib.ticker as ticker
#axs[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
#axs[0].get_yaxis().set_major_formatter(ticker.ScalarFormatter())

axs[0].set_ylim(0)
axs[0].set_xlim(1,800)

#axs[0].set_xlabel("Nominal b-value [s/mm$^2$]")
axs[0].set_ylabel("Relative b-value deviation [\%]")


#fig, ax = plt.subplots(figsize=(4,4))
#axs[1].plot(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_highres - bvalues_nominal_worst[:-1])
#axs[1].plot(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_lowres - bvalues_nominal_worst[:-1])
axs[1].fill_between(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_lowres - bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_highres - bvalues_nominal_worst[:-1], alpha=alpha_area, label="Large cross-terms")

#axs[1].plot(bvalues_nominal_best[:-1], bvalues_actual_best_xy_highres - bvalues_nominal_best[:-1])
#axs[1].plot(bvalues_nominal_best[:-1], bvalues_actual_best_xy_lowres - bvalues_nominal_best[:-1])
axs[1].fill_between(bvalues_nominal_best[:-1], bvalues_actual_best_xy_lowres - bvalues_nominal_best[:-1], bvalues_actual_best_xy_highres - bvalues_nominal_best[:-1], alpha=alpha_area, label="Minimal cross-terms")

# xy
#high_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+18, y=np.flip(bvalues_actual_worst_xy_highres - bvalues_nominal_worst[:-1])[11:]+3.5, text="High resolution", axes=axs[1])
#low_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+20, y=np.flip(bvalues_actual_worst_xy_lowres - bvalues_nominal_worst[:-1])[11:]+2, text="Low resolution", axes=axs[1])

# z
high_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+18, y=np.flip(bvalues_actual_worst_xy_highres - bvalues_nominal_worst[:-1])[11:]+6, text="High resolution", axes=axs[1])
low_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+20, y=np.flip(bvalues_actual_worst_xy_lowres - bvalues_nominal_worst[:-1])[11:]+3, text="Low resolution", axes=axs[1])

axs[1].set_xscale("log")
#axs[1].set_yscale("log")
axs[1].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
axs[1].set_ylim(0)
axs[1].set_xlim(1,800)
axs[1].set_xticks([1, 10, 100, 800])

axs[1].set_xlabel("Nominal b-value [s/mm$^2$]")
axs[1].set_ylabel("Absolute b-value deviation [s/mm$^2$]")

axs[1].legend(frameon=False, loc="upper left")

fig.tight_layout()

# %% Relative b-value error vs b-value, 2x2

resolutions = np.array([1e-3, 4e-3]) # High-res and low-res

bvalues_nominal_worst, bvalues_actual_worst = read_simulation_files_and_average(path_worst, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)
bvalues_nominal_best, bvalues_actual_best = read_simulation_files_and_average(path_best, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)

bvalues_actual_best_highres = bvalues_actual_best[:, :-1, 0]
bvalues_actual_best_lowres = bvalues_actual_best[:, :-1, 1]

bvalues_actual_worst_highres = bvalues_actual_worst[:, :-1, 0]
bvalues_actual_worst_lowres = bvalues_actual_worst[:, :-1, 1]

bvalues_actual_best_xy_highres_relative_error = (bvalues_actual_best_highres[0,:]-bvalues_nominal_best[:-1])/bvalues_nominal_best[:-1]
bvalues_actual_best_xy_lowres_relative_error = (bvalues_actual_best_lowres[0,:]-bvalues_nominal_best[:-1])/bvalues_nominal_best[:-1]

bvalues_actual_worst_xy_highres_relative_error = (bvalues_actual_worst_highres[0,:]-bvalues_nominal_worst[:-1])/bvalues_nominal_worst[:-1]
bvalues_actual_worst_xy_lowres_relative_error = (bvalues_actual_worst_lowres[0,:]-bvalues_nominal_worst[:-1])/bvalues_nominal_worst[:-1]

### Create the figure ###
fig, axs = plt.subplots(figsize=(7.5,6.5), nrows=2, ncols=2, sharex=True, sharey="row")
## First column, xy gradient errors
axs[0,0].fill_between(bvalues_nominal_worst[:-1], bvalues_actual_worst_xy_lowres_relative_error*100, bvalues_actual_worst_xy_highres_relative_error*100, alpha=alpha_area)
axs[0,0].fill_between(bvalues_nominal_best[:-1], bvalues_actual_best_xy_lowres_relative_error*100, bvalues_actual_best_xy_highres_relative_error*100, alpha=alpha_area)

# xy text
high_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[13:], y=np.flip(bvalues_actual_best_xy_highres_relative_error*140)[13:], text="High resolution", axes=axs[0,0])
low_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[11:], y=np.flip(bvalues_actual_best_xy_lowres_relative_error*140)[11:], text="Low resolution", axes=axs[0,0])
crushers_off_text = axs[0,0].annotate("Crushers off", (200,100), xytext=(10,5), rotation=-76, fontsize=10)

# z text
#high_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[12:], y=np.flip(bvalues_actual_best_xy_highres_relative_error*115)[12:], text="High resolution", axes=axs[0,1])
#low_res_text = CurvedText(x=np.flip(bvalues_nominal_best[:-1])[11:], y=np.flip(bvalues_actual_best_xy_lowres_relative_error*115)[11:], text="Low resolution", axes=axs[0,1])
#crushers_off_text = axs[0,1].annotate("Crushers off", (200,100), xytext=(10,19), rotation=-77, fontsize=10)

axs[0,0].set_xscale("log")
axs[0,0].set_yscale("log")
import matplotlib.ticker as ticker
#axs[0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
#axs[0].get_yaxis().set_major_formatter(ticker.ScalarFormatter())

axs[0,0].set_ylim(0)
axs[0,0].set_xlim(1,800)

#axs[0].set_xlabel("Nominal b-value [s/mm$^2$]")
axs[0,0].set_ylabel("Relative b-value deviation [\%]")

# Aboslute error
axs[1,0].fill_between(bvalues_nominal_worst[:-1], bvalues_actual_worst_lowres[0,:] - bvalues_nominal_worst[:-1], bvalues_actual_worst_highres[0,:] - bvalues_nominal_worst[:-1], alpha=alpha_area, label="Large cross-terms")
axs[1,0].fill_between(bvalues_nominal_best[:-1], bvalues_actual_best_lowres[0,:] - bvalues_nominal_best[:-1], bvalues_actual_best_highres[0,:] - bvalues_nominal_best[:-1], alpha=alpha_area, label="Minimal cross-terms")

# xy
high_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+18, y=np.flip(bvalues_actual_worst_highres[0,:] - bvalues_nominal_worst[:-1])[11:]+4, text="High resolution", axes=axs[1,0])
low_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+20, y=np.flip(bvalues_actual_worst_lowres[0,:] - bvalues_nominal_worst[:-1])[11:]+2.5, text="Low resolution", axes=axs[1,0])

# z
#high_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+18, y=np.flip(bvalues_actual_worst_xy_highres - bvalues_nominal_worst[:-1])[11:]+6, text="High resolution", axes=axs[1,0])
#low_res_text = CurvedText(x=np.flip(bvalues_nominal_worst[:-1])[11:]+20, y=np.flip(bvalues_actual_worst_xy_lowres - bvalues_nominal_worst[:-1])[11:]+3, text="Low resolution", axes=axs[1,0])

axs[1,0].set_xscale("log")
#axs[1].set_yscale("log")
axs[1,0].get_xaxis().set_major_formatter(ticker.ScalarFormatter())
axs[1,0].set_xlim(1,800)
axs[1,0].set_xticks([1, 10, 100, 800])

axs[1,0].set_xlabel("Nominal b-value [s/mm$^2$]")
axs[1,0].set_ylabel("Absolute b-value deviation [s/mm$^2$]")

axs[1,0].legend(frameon=False, loc="upper left")


## Second column, z gradient errors
bvalues_actual_best_z_highres_relative_error = (bvalues_actual_best_highres[2,:]-bvalues_nominal_best[:-1])/bvalues_nominal_best[:-1]
bvalues_actual_best_z_lowres_relative_error = (bvalues_actual_best_lowres[2,:]-bvalues_nominal_best[:-1])/bvalues_nominal_best[:-1]

bvalues_actual_worst_z_highres_relative_error = (bvalues_actual_worst_highres[2,:]-bvalues_nominal_worst[:-1])/bvalues_nominal_worst[:-1]
bvalues_actual_worst_z_lowres_relative_error = (bvalues_actual_worst_lowres[2,:]-bvalues_nominal_worst[:-1])/bvalues_nominal_worst[:-1]
axs[0,1].fill_between(bvalues_nominal_worst[:-1], bvalues_actual_worst_z_lowres_relative_error*100, bvalues_actual_worst_z_highres_relative_error*100, alpha=alpha_area)
axs[0,1].fill_between(bvalues_nominal_best[:-1], bvalues_actual_best_z_lowres_relative_error*100, bvalues_actual_best_z_highres_relative_error*100, alpha=alpha_area)

#axs[0,1].set_yscale("log")
# Aboslute error
axs[1,1].fill_between(bvalues_nominal_worst[:-1], bvalues_actual_worst_lowres[2,:] - bvalues_nominal_worst[:-1], bvalues_actual_worst_highres[2,:] - bvalues_nominal_worst[:-1], alpha=alpha_area, label="Large cross-terms")
axs[1,1].fill_between(bvalues_nominal_best[:-1], bvalues_actual_best_lowres[2,:] - bvalues_nominal_best[:-1], bvalues_actual_best_highres[2,:] - bvalues_nominal_best[:-1], alpha=alpha_area, label="Minimal cross-terms")

axs[1,1].set_ylim(0)
axs[1,1].set_xlabel("Nominal b-value [s/mm$^2$]")


fig.text(0, 1, "(a)", fontsize=18)
fig.text(0.515, 1, "(b)", fontsize=18)
fig.tight_layout()

#fig.savefig(os.path.join(fig_save_path, "Figure_reviewer_2.tiff"), dpi=300, bbox_inches="tight")

# %% Error heatmaps for nominal xyz analysis 

# Stepsizes
#resolutions = np.array([1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 2.75e-3, 3e-3, 3.25e-3, 3.5e-3, 3.75e-3, 4e-3])
resolutions = np.round(np.linspace(1,4,13)*1e-3, decimals=7)


f_range = np.linspace(1e-2, 30e-2, 50)
Dstar_range = np.linspace(5e-3, 50e-3, 50)
D_range = np.linspace(0.5e-3, 3e-3, 50)

f_constant = 0.1
Dstar_constant = 20e-3
D_constant = 1e-3 

#bvalues_nominal_worst, bvalues_actual_worst = read_simulation_files_and_average(path_worst, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions)
#bvalues_nominal_best, bvalues_actual_best = read_simulation_files_and_average(path_best, correction="crossterm_corrected", angles="xyz", xyres=resolutions, zres=resolutions)

### Varying f
f_f_heatmap_estimates_worst = np.zeros((f_range.shape[0], resolutions.shape[0]))
f_f_heatmap_estimates_best = np.zeros((f_range.shape[0], resolutions.shape[0]))
f_f_heatmap_truth = np.zeros((f_range.shape[0], resolutions.shape[0]))

f_Dstar_heatmap_estimates_worst = np.zeros((f_range.shape[0], resolutions.shape[0]))
f_Dstar_heatmap_estimates_best = np.zeros((f_range.shape[0], resolutions.shape[0]))
f_Dstar_heatmap_truth = np.zeros((f_range.shape[0], resolutions.shape[0]))

f_D_heatmap_estimates_worst = np.zeros((f_range.shape[0], resolutions.shape[0]))
f_D_heatmap_estimates_best = np.zeros((f_range.shape[0], resolutions.shape[0]))
f_D_heatmap_truth = np.zeros((f_range.shape[0], resolutions.shape[0]))
for i in range(f_range.shape[0]):
    f_estimates_worst, Dstar_estimates_worst, D_estimates_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions, f=f_range[i], Dstar=Dstar_constant, D=D_constant)
    f_estimates_best, Dstar_estimates_best, D_estimates_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions, f=f_range[i], Dstar=Dstar_constant, D=D_constant)

    f_f_heatmap_estimates_worst[i, :] = f_estimates_worst
    f_f_heatmap_estimates_best[i, :] = f_estimates_best
    f_f_heatmap_truth[i, :] = f_range[i]

    f_Dstar_heatmap_estimates_worst[i, :] = Dstar_estimates_worst
    f_Dstar_heatmap_estimates_best[i, :] = Dstar_estimates_best
    f_Dstar_heatmap_truth[:, :] = Dstar_constant*1000

    f_D_heatmap_estimates_worst[i, :] = D_estimates_worst
    f_D_heatmap_estimates_best[i, :] = D_estimates_best
    f_D_heatmap_truth[:, :] = D_constant*1000

### Varying Dstar
Dstar_f_heatmap_estimates_worst = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
Dstar_f_heatmap_estimates_best = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
Dstar_f_heatmap_truth = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))

Dstar_Dstar_heatmap_estimates_worst = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
Dstar_Dstar_heatmap_estimates_best = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
Dstar_Dstar_heatmap_truth = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))

Dstar_D_heatmap_estimates_worst = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
Dstar_D_heatmap_estimates_best = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
Dstar_D_heatmap_truth = np.zeros((Dstar_range.shape[0], resolutions.shape[0]))
for i in range(Dstar_range.shape[0]):
    f_estimates_worst, Dstar_estimates_worst, D_estimates_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions, f=f_constant, Dstar=Dstar_range[i], D=D_constant)
    f_estimates_best, Dstar_estimates_best, D_estimates_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions, f=f_constant, Dstar=Dstar_range[i], D=D_constant)

    Dstar_f_heatmap_estimates_worst[i, :] = f_estimates_worst
    Dstar_f_heatmap_estimates_best[i, :] = f_estimates_best
    Dstar_f_heatmap_truth[:, :] = f_constant

    Dstar_Dstar_heatmap_estimates_worst[i, :] = Dstar_estimates_worst
    Dstar_Dstar_heatmap_estimates_best[i, :] = Dstar_estimates_best
    Dstar_Dstar_heatmap_truth[i, :] = Dstar_range[i]*1000

    Dstar_D_heatmap_estimates_worst[i, :] = D_estimates_worst
    Dstar_D_heatmap_estimates_best[i, :] = D_estimates_best
    Dstar_D_heatmap_truth[:, :] = D_constant*1000


### Varying D
D_f_heatmap_estimates_worst = np.zeros((D_range.shape[0], resolutions.shape[0]))
D_f_heatmap_estimates_best = np.zeros((D_range.shape[0], resolutions.shape[0]))
D_f_heatmap_truth = np.zeros((D_range.shape[0], resolutions.shape[0]))

D_Dstar_heatmap_estimates_worst = np.zeros((D_range.shape[0], resolutions.shape[0]))
D_Dstar_heatmap_estimates_best = np.zeros((D_range.shape[0], resolutions.shape[0]))
D_Dstar_heatmap_truth = np.zeros((D_range.shape[0], resolutions.shape[0]))

D_D_heatmap_estimates_worst = np.zeros((D_range.shape[0], resolutions.shape[0]))
D_D_heatmap_estimates_best = np.zeros((D_range.shape[0], resolutions.shape[0]))
D_D_heatmap_truth = np.zeros((D_range.shape[0], resolutions.shape[0]))
for i in range(D_range.shape[0]):
    f_estimates_worst, Dstar_estimates_worst, D_estimates_worst = powder_average_all_simulation_logs_new(path_worst, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions, f=f_constant, Dstar=Dstar_constant, D=D_range[i])
    f_estimates_best, Dstar_estimates_best, D_estimates_best = powder_average_all_simulation_logs_new(path_best, correction="uncorrected", angles="xyz", xyres=resolutions, zres=resolutions, f=f_constant, Dstar=Dstar_constant, D=D_range[i])

    D_f_heatmap_estimates_worst[i, :] = f_estimates_worst
    D_f_heatmap_estimates_best[i, :] = f_estimates_best
    D_f_heatmap_truth[:, :] = f_constant

    D_Dstar_heatmap_estimates_worst[i, :] = Dstar_estimates_worst
    D_Dstar_heatmap_estimates_best[i, :] = Dstar_estimates_best
    D_Dstar_heatmap_truth[:, :] = Dstar_constant*1000

    D_D_heatmap_estimates_worst[i, :] = D_estimates_worst
    D_D_heatmap_estimates_best[i, :] = D_estimates_best
    D_D_heatmap_truth[i, :] = D_range[i]*1000


# %% Error heatmaps as imagegrid

colormap = "BrBG"
aspect = 0.2
fig = plt.figure(figsize=(8,4))
grid = ImageGrid(fig, 111,
                nrows_ncols=(2,3),
                axes_pad=0.4,
                label_mode="L",
                cbar_location="top",
                cbar_mode="edge",
                cbar_pad=0.3,
                cbar_size="10%",
                share_all=False)

#f = grid[0].imshow(np.flip(f_f_heatmap_truth, axis=0))
#f = grid[0].imshow(np.flip((f_f_heatmap_estimates-f_f_heatmap_truth)/f_f_heatmap_truth, axis=0), cmap=colormap, vmin=-0.25, vmax=0.25, aspect=aspect, interpolation="bilinear")
#f = grid[0].imshow((f_f_heatmap_estimates-f_f_heatmap_truth)/f_f_heatmap_truth, origin="lower", cmap=colormap, clim=(-0.2, 1.0), norm=MidpointNormalize(midpoint=0.0, vmin=-0.2, vmax=1.0), aspect=aspect, interpolation="bilinear")
#norm=colors.TwoSlopeNorm(vmin=vmin-1e-5, vcenter=0., vmax=vmax+1e-5)
norm=colors.TwoSlopeNorm(vmin=-0.2, vcenter=0., vmax=1.0)
#f = grid[0].imshow((f_f_heatmap_estimates-f_f_heatmap_truth)/f_f_heatmap_truth, origin="lower", cmap=colormap, clim=(-0.2, 1.0), norm=MidpointNormalize(midpoint=0.0, vmin=-0.2, vmax=0.2), aspect=aspect, interpolation="bilinear")
f = grid[0].imshow((f_f_heatmap_estimates-f_f_heatmap_truth)/f_f_heatmap_truth, origin="lower", cmap=colormap, norm=norm, aspect=aspect, interpolation="none")
cax = grid.cbar_axes[0].colorbar(f)
cax.set_ticks([-0.2, 0, 1])
grid[0].set_ylabel("Ground truth $f$ [\%]")
grid[0].set_yticks([0, 15, 32, 49])
grid[0].set_yticklabels([0, 10, 20, 30])

#Dstar = grid[1].imshow(np.flip((Dstar_Dstar_heatmap_estimates-Dstar_Dstar_heatmap_truth)/Dstar_Dstar_heatmap_truth, axis=0), vmin=0.1, vmax=0.7, aspect=aspect)
Dstar = grid[1].imshow((Dstar_Dstar_heatmap_estimates-Dstar_Dstar_heatmap_truth)/Dstar_Dstar_heatmap_truth, origin="lower", vmin=0.1, vmax=1.0, aspect=aspect, interpolation="none")
cax = grid.cbar_axes[1].colorbar(Dstar)
cax.set_ticks([0.1, 1])
grid[1].set_ylabel("Ground truth $D^*$ [µm$^2$/ms]")
grid[1].set_yticks([0, 5, 27, 49])
grid[1].set_yticklabels([5, 10, 30, 50])

#D = grid[2].imshow(np.flip((D_D_heatmap_estimates-D_D_heatmap_truth)/D_D_heatmap_truth, axis=0), aspect=aspect)
D = grid[2].imshow((D_D_heatmap_estimates-D_D_heatmap_truth)/D_D_heatmap_truth, origin="lower", aspect=aspect, interpolation="bilinear")
grid.cbar_axes[2].colorbar(D)


# %% Error heatmaps as subplots

cbar_aspect = 10
cbar_shrink = 0.72
aspect = 0.26
factor = 100 # To show relative errors as percentages

colormap = "BrBG"
fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(9,6), sharex=True)

norm = colors.TwoSlopeNorm(vmin=-0.2*factor, vcenter=0.*factor, vmax=1.0*factor)
f = axs[0,0].imshow((f_f_heatmap_estimates_worst-f_f_heatmap_truth)/f_f_heatmap_truth*factor, origin="lower", cmap=colormap, norm=norm, aspect=aspect, interpolation="none")
cbar = fig.colorbar(f, ax=axs[0,0], location="top", shrink=cbar_shrink, aspect=cbar_aspect)
cbar.set_ticks([-0.2*factor, 0, 1*factor])
cbar.set_label("Relative $f$ error [\%]")
axs[0,0].set_ylabel("$f$ [\%]")
axs[0,0].set_yticks([0, 15, 32, 49])
axs[0,0].set_yticklabels(np.array([0, 10, 20, 30]))


norm = colors.TwoSlopeNorm(vmin=-1.0*factor, vcenter=0.*factor, vmax=1.0*factor)
Dstar = axs[0,1].imshow((Dstar_Dstar_heatmap_estimates_worst-Dstar_Dstar_heatmap_truth)/Dstar_Dstar_heatmap_truth*factor, origin="lower", norm=norm, cmap=colormap, aspect=aspect, interpolation="none")
cbar = fig.colorbar(Dstar, ax=axs[0,1], location="top", shrink=cbar_shrink, aspect=cbar_aspect)
cbar.set_ticks(np.array([-1, -0.5, 0, 0.5, 1])*factor)
cbar.set_label("Relative $D^*$ error [\%]")
axs[0,1].set_ylabel("$D^*$ [µm$^2$/ms]")
axs[0,1].set_yticks([0, 5, 27, 49])
axs[0,1].set_yticklabels([5, 10, 30, 50])

norm = colors.TwoSlopeNorm(vmin=-0.1*factor, vcenter=0.*factor, vmax=0.1*factor)
D = axs[0,2].imshow((D_D_heatmap_estimates_worst-D_D_heatmap_truth)/D_D_heatmap_truth*factor, origin="lower", norm=norm, cmap=colormap, aspect=aspect, interpolation="none")
cbar = fig.colorbar(D, ax=axs[0,2], location="top", shrink=cbar_shrink, aspect=cbar_aspect)
cbar.set_ticks(np.array([-0.1, -0.05, 0, 0.05, 0.1])*factor)
cbar.set_label("Relative $D$ error [\%]")
axs[0,2].set_ylabel("$D$ [µm$^2$/ms]")
axs[0,2].set_yticks([0, 10, 30, 49])
axs[0,2].set_yticklabels([0.5, 1, 2, 3])

### Optimal sequence
norm = colors.TwoSlopeNorm(vmin=-0.2*factor, vcenter=0.*factor, vmax=1.0*factor)
f = axs[1,0].imshow((f_f_heatmap_estimates_best-f_f_heatmap_truth)/f_f_heatmap_truth*factor, origin="lower", cmap=colormap, norm=norm, aspect=aspect, interpolation="none")
#cbar = fig.colorbar(f, ax=axs[1,0], location="top", shrink=cbar_shrink, aspect=cbar_aspect)
#cbar.set_ticks([-0.2*factor, 0, 1*factor])
#cbar.set_label("Relative $f$ error [\%]")
axs[1,0].set_ylabel("$f$ [\%]")
axs[1,0].set_yticks([0, 15, 32, 49])
axs[1,0].set_yticklabels(np.array([0, 10, 20, 30]))

norm = colors.TwoSlopeNorm(vmin=-1.0*factor, vcenter=0.*factor, vmax=1.0*factor)
Dstar = axs[1,1].imshow((Dstar_Dstar_heatmap_estimates_best-Dstar_Dstar_heatmap_truth)/Dstar_Dstar_heatmap_truth*factor, origin="lower", norm=norm, cmap=colormap, aspect=aspect, interpolation="none")
#cbar = fig.colorbar(Dstar, ax=axs[1,1], location="top", shrink=cbar_shrink, aspect=cbar_aspect)
#cbar.set_ticks(np.array([-1, -0.5, 0, 0.5, 1])*factor)
#cbar.set_label("Relative $D^*$ error [\%]")
axs[1,1].set_ylabel("$D^*$ [µm$^2$/ms]")
axs[1,1].set_yticks([0, 5, 27, 49])
axs[1,1].set_yticklabels([5, 10, 30, 50])

norm = colors.TwoSlopeNorm(vmin=-0.1*factor, vcenter=0.*factor, vmax=0.1*factor)
D = axs[1,2].imshow((D_D_heatmap_estimates_best-D_D_heatmap_truth)/D_D_heatmap_truth*factor, origin="lower", norm=norm, cmap=colormap, aspect=aspect, interpolation="none")
#cbar = fig.colorbar(D, ax=axs[1,2], location="top", shrink=cbar_shrink, aspect=cbar_aspect)
#cbar.set_ticks(np.array([-0.1, -0.05, 0., 0.05, 0.1])*factor)
#cbar.set_label("Relative $D$ error [\%]")
axs[1,2].set_ylabel("$D$ [µm$^2$/ms]")
axs[1,2].set_yticks([0, 10, 30, 49])
axs[1,2].set_yticklabels([0.5, 1, 2, 3])

axs[1,0].set_xlabel("Isotropic resolution [mm]")
axs[1,0].set_xticks([0, 4, 8, 12])
axs[1,0].set_xticklabels([1, 2, 3, 4])

axs[1,1].set_xlabel("Isotropic resolution [mm]")
axs[1,1].set_xticks([0, 4, 8, 12])
axs[1,1].set_xticklabels([1, 2, 3, 4])

axs[1,2].set_xlabel("Isotropic resolution [mm]")
axs[1,2].set_xticks([0, 4, 8, 12])
axs[1,2].set_xticklabels([1, 2, 3, 4])


fig.text(0.01, 0.95, "(a)", fontsize=18)
fig.text(0.01, 0.47, "(b)", fontsize=18)
fig.tight_layout()

#fig.savefig(os.path.join(fig_save_path, "Figure_reviewer_3.tiff"), dpi=300, bbox_inches="tight")

# %%
