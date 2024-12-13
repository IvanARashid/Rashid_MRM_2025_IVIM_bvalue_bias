import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl
import seaborn as sns

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
#mpl.rcParams["text.color"] = "#4D4C44"
#mpl.rcParams["axes.labelcolor"] = "#4D4C44"
#mpl.rcParams["xtick.color"] = "#4D4C44"
#mpl.rcParams["ytick.color"] = "#4D4C44"

class parametric_map:
    def __init__(self, data, windowing):
        self.data = data
        self.windowing = windowing

base_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\figures\fig5 data"

files = ["nom", "corr"]
patient = "pat1"
parameters = ["f", "Dstar", "D"]
no_of_image_squares_to_skip = 3
#folders_MIX = [f"{algorithms[1]} {folders[i]}" for i in range(len(folders))]

#path = os.path.join(base_path, folders_MIX[0])

#dicom_data = utilities_dicom.read_dicom_folder(path)

slice = 16
#slice = 13
x1 = 64
x2 = 102
y1 = 70
y2 = 96


windowing = [[0,.3], [0, 0.06], [0, 0.0025]]
#fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(8,8))
#axs[0, 0].imshow(GTV_contour, cmap="gray")
data_expanded = []
for row_idx in range(len(files)):
    for col_idx in range(len(parameters)):
        filename = f"{patient}_{parameters[col_idx]}_{files[row_idx]}.npy"
        path = os.path.join(base_path, filename)
        img = np.load(path)
        img = img[x1:x2, y1:y2, slice].T
        img_filtered = ndimage.gaussian_filter(img, sigma=(1,1))
        
        #axs[row_idx, col_idx].imshow(img_filtered, cmap="inferno", vmin=windowing[col_idx][0], vmax=windowing[col_idx][1])
        data_expanded.append(parametric_map(img, windowing[col_idx]))
        

fig5a = plt.figure(figsize=(10.8,6))
grid = ImageGrid(fig5a, 111,
                nrows_ncols=(len(files),len(parameters)),
                axes_pad=0.3, # 0.25
                share_all=False,
                cbar_location="top",
                cbar_mode="edge",
                cbar_pad=0.1,
                cbar_size="15%",
                direction="row")

for ax_idx in range(len(grid)):
    #masked_array = np.ma.masked_where(data_expanded[ax_idx].data <= np.max(data_expanded[ax_idx].data)*1e-3, data_expanded[ax_idx].data)
    #cmap = mpl.cm.get_cmap("inferno").copy()
    #cmap.set_bad(color="white")
    cmap = "viridis"
    fontsize = 13
    im = grid[ax_idx].imshow(data_expanded[ax_idx].data, vmin=data_expanded[ax_idx].windowing[0], vmax=data_expanded[ax_idx].windowing[1], cmap=cmap)
    #im = grid[ax_idx].imshow(masked_array, vmin=data_expanded[ax_idx].windowing[0], vmax=data_expanded[ax_idx].windowing[1], cmap=cmap)
    
    grid[ax_idx].set_xticks([])
    grid[ax_idx].set_yticks([])
    grid[0].set_ylabel("Nominal", fontsize=fontsize+3)
    grid[3].set_ylabel("Actual", fontsize=fontsize+3)
    #grid[6].set_ylabel("TopoPro", fontsize=fontsize+3)
            
   
    cblabel = ["$f$ [\%]", "$D^*$ [µm$^2$/ms]", "$D$ [µm$^2$/ms]"] 
    if ax_idx in [0, 1, 2]:
        fontsize = 14
        
        cbar = grid.cbar_axes[ax_idx].colorbar(im)
        cbar.set_label(cblabel[ax_idx], fontsize=fontsize+2, labelpad=10)
        cbar.ax.xaxis.set_label_position("top")
        cbar.ax.xaxis.set_ticks_position("top")
        
        if ax_idx == 0:
            ticks = [0, .1, .2, .3]
            cbar.set_ticks(ticks, labels=ticks, fontsize=fontsize)
            cbar.set_ticklabels([int(tick*1e2) for tick in ticks], fontsize=fontsize)
            
            #grid[ax_idx].set_title("Perfusion fraction", fontsize=fontsize+3)
        if ax_idx == 1:
            ticks = [0, 15e-3, 30e-3, 45e-3, 60e-3]
            cbar.set_ticks(ticks, labels=ticks, fontsize=fontsize)
            cbar.set_ticklabels([int(tick*1e3) for tick in ticks], fontsize=fontsize)
            
            #grid[ax_idx].set_title("Pseudo-diffusion\ncoefficient", fontsize=fontsize+3)
        if ax_idx == 2:
            ticks = [0, .5e-3, 1e-3, 1.5e-3, 2e-3, 2.5e-3]
            cbar.set_ticks(ticks, labels=ticks, fontsize=fontsize)
            cbar.set_ticklabels([tick*1e3 for tick in ticks], fontsize=fontsize)
            
            #grid[ax_idx].set_title("Diffusion coefficient", fontsize=fontsize+3)

## Arrows
## Appearing voxels
#grid[0].arrow(37,18, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[0+3].arrow(37,18, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")




## Outlier change
#grid[1].arrow(37,18, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(37,18, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")

## Outlier change
#grid[1].arrow(33,23, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(33,23, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")

## Outlier change
#grid[1].arrow(19, 11, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(19, 11, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")

## Middle
#grid[1].arrow(25, 12, -1, 1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(25, 12, -1, 1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")


### Left side of the prostate
## Outlier change
##grid[1].arrow(15, 9, -1, 1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
##grid[1+3].arrow(15, 9, -1, 1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
## Outlier change
#grid[1].arrow(15, 15, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(15, 15, -1, -1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
## Outlier change
#grid[1].arrow(4, 11, 1, 0, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(4, 11, 1, 0, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
## Outlier change
#grid[1].arrow(11, 5.5, 0, 1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")
#grid[1+3].arrow(11, 5.5, 0, 1, width=0, head_width=1.5, head_starts_at_zero=True, length_includes_head=True, color="white")





fig5a.text(0.1, 0.95, "(a)", fontsize=fontsize+3)


# Generate histogram data
# Load the mask
mask = np.load(os.path.join(base_path, f"{patient}_mask.npy"))

histogram_data = []
for parameter_idx in range(len(parameters)):
    nominal_data = np.load(os.path.join(base_path, f"{patient}_{parameters[parameter_idx]}_nom.npy"))
    corrected_data = np.load(os.path.join(base_path, f"{patient}_{parameters[parameter_idx]}_corr.npy"))

    # Calculate element-wise difference
    difference = nominal_data - corrected_data

    # Apply mask
    difference_masked = difference[mask]

    histogram_data.append(difference_masked)


fig5b, axs = plt.subplots(ncols=3, figsize=(9,3), sharey=True)

density = False
axs[0].hist(histogram_data[0]*100, density=density, bins=50, range=(-.5, .5), alpha=.5)
#sns.histplot(ax=axs[0], data=histogram_data[0]*100, bins=60)
#axs[0].set_xlim(-6, 6)
axs[0].set_xticks([-.5, 0, .5])
axs[0].set_yscale("linear")
axs[0].set_xlabel("$\Delta f$ [\%]")
axs[0].set_ylabel("Difference histogram\nNominal -- Actual", fontsize=fontsize+3)

axs[1].hist(histogram_data[1]*1000, density=density, bins=50, range=(-3,3), alpha=.5)
axs[1].set_xlim(-3, 3)
#axs[1].set_xticks([-30, 0, 30])
axs[1].set_yscale("linear")
axs[1].set_xlabel("$\Delta D$* [µm$^2$/ms]")

axs[2].hist(histogram_data[2]*1000, density=density, bins=50, range=(-.12, .12), alpha=.5)
#axs[2].set_xlim(-0.12, 0.12)
axs[2].set_xticks([-0.12, -0.06, 0, 0.06, 0.12])
axs[2].set_yscale("linear")
axs[2].set_xlabel("$\Delta D$ [µm$^2$/ms]")
fig5b.text(0.02, 1, "(b)", fontsize=fontsize+3, clip_on=True)
fig5b.tight_layout()
#axs[0].text(-9, 1e4, "(b)", fontsize=fontsize+3, clip_on=False)

            
fig_path = r"C:\Users\ivan5\Box\PhD\Resor\Diffusion workshop 2025\b_value_correction"
#fig5a.savefig(os.path.join(fig_path, "fig5a.pdf"), bbox_inches="tight") 
#fig5b.savefig(os.path.join(fig_path, "clinical_histograms.pdf"), bbox_inches="tight") 
        

    
    

