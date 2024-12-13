import numpy as np
import simulate_pulse_sequences
import sys
import os

# Commandline arguments
try:
	save_path = sys.argv[1]
	print(f"save_path = {save_path}")
except:
	# Default on IAR's laptop
	save_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\new_sims"

full_id = False
sequence_config = {"xres" : 2e-3,
                   "yres" : 2e-3,
                   "zres" : 4e-3,
                   "crushers" : True,
                   "optimal" : False,
                   "all_crushers" : True,
                   "only_crush_when_needed" : False}

uvec_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\bvecs"
uvec_fname = "xyz.bvec"
uvec_file = os.path.join(uvec_path, uvec_fname)

# Nominal b-values
nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
#nominal_bvalues = [800, 200, 50, 0]
# Number of diffusion directions
n = 1000
# Include B_im
include_imaging = False
# Include B_ct
include_cross_terms = False

# Resolution lists
resolution_lst = np.round(np.linspace(1,4,13)*1e-3, decimals=7)
print(resolution_lst)

# Uncorrected
for resolution in resolution_lst:
    sequence_config["xres"] = resolution
    sequence_config["yres"] = resolution
    #sequence_config["zres"] = resolution
    # Run simulation
    bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)

## Accounting for imaging
include_imaging = True	
for resolution in resolution_lst:
    sequence_config["xres"] = resolution
    sequence_config["yres"] = resolution
    #sequence_config["zres"] = resolution
    # Run simulation
    bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)

# Accounting for cross-terms
include_cross_terms = True
for resolution in resolution_lst:
    sequence_config["xres"] = resolution
    sequence_config["yres"] = resolution
    #sequence_config["zres"] = resolution
    # Run simulation
    bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)