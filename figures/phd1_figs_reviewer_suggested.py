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
# %%
