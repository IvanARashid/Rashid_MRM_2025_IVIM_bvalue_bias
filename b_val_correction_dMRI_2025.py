import numpy as np
import os
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13
fontsize = 13

base_path = r"C:\Users\ivan5\Box\PhD\Resor\Diffusion workshop 2025\b_value_correction"

parameters = ["f", "vd", "D"]


# Generate histogram data
# Load the mask
mask = np.load(os.path.join(base_path, f"mask.npy"))

histogram_data = []
hisotgram_data_no_diff = []
for parameter_idx in range(len(parameters)):
    nominal_data = np.load(os.path.join(base_path, f"{parameters[parameter_idx]}_nominal.npy"))
    corrected_data = np.load(os.path.join(base_path, f"{parameters[parameter_idx]}_actual.npy"))

    # Calculate element-wise difference
    difference = nominal_data - corrected_data

    # Apply mask
    difference_masked = difference[mask]

    histogram_data.append(difference_masked)
    hisotgram_data_no_diff.append(corrected_data[mask])


fig5b, axs = plt.subplots(ncols=3, figsize=(9,3), sharey=True)

density = False
axs[0].hist(histogram_data[0]*100, density=density, bins=50, range=(-1, 1), alpha=.5)
#sns.histplot(ax=axs[0], data=histogram_data[0]*100, bins=60)
axs[0].set_xlim(-1, 1)
axs[0].set_xticks([-1, 0, 1])
axs[0].set_yscale("linear")
axs[0].set_xlabel("$\Delta f$ [\%]")
axs[0].set_ylabel("Difference histogram\nNominal -- Actual", fontsize=fontsize+3)

axs[1].hist(histogram_data[1], density=density, bins=50, range=(-.2,.2), alpha=.5)
axs[1].set_xlim(-.2, .2)
#axs[1].set_xticks([-30, 0, 30])
axs[1].set_yscale("linear")
axs[1].set_xlabel("$\Delta v_d$ [µm/ms]")

axs[2].hist(histogram_data[2]*1000, density=density, bins=50, range=(-.025, .025), alpha=.5)
axs[2].set_xlim(-0.025, 0.025)
axs[2].set_xticks([-0.025, 0, 0.025])
axs[2].set_yscale("linear")
axs[2].set_xlabel("$\Delta D$ [µm$^2$/ms]")
fig5b.text(0.02, 1, "(b)", fontsize=fontsize+3, clip_on=True)
fig5b.tight_layout()
#axs[0].text(-9, 1e4, "(b)", fontsize=fontsize+3, clip_on=False)

            
fig_path = r"C:\Users\ivan5\Box\PhD\Resor\Diffusion workshop 2025\b_value_correction"
#fig5a.savefig(os.path.join(fig_path, "fig5a.pdf"), bbox_inches="tight") 
fig5b.savefig(os.path.join(fig_path, "FCNC_histograms.pdf"), bbox_inches="tight") 
        

    