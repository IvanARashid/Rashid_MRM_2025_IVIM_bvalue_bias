import numpy as np
import dMRItools.simulate_pulse_sequences as simulate_pulse_sequences
import sys
import os
import multiprocessing as mp

# Commandline arguments
try:
	save_path = sys.argv[1]
	print(f"save_path = {save_path}")
except:
	# Default on IAR's laptop
	save_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\test"

full_id = False
sequence_config = {"xres" : 2e-3,
                   "yres" : 2e-3,
                   "zres" : 2e-3,
                   "crushers" : True,
                   "optimal" : True,
                   "all_crushers" : True,
                   "only_crush_when_needed" : True}

uvec_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\bvecs"
uvec_fname = "xyz_antipodal.bvec"
uvec_file = os.path.join(uvec_path, uvec_fname)

chunks = 6

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
resolution_lst = np.round(np.linspace(2,10,9)*1e-3, decimals=7)
print(resolution_lst)

# Uncorrected
for resolution in resolution_lst:
    #sequence_config["xres"] = resolution
    #sequence_config["yres"] = resolution
    sequence_config["zres"] = resolution


    sequence_config_chunks = [sequence_config for i in range(chunks)]
    nominal_bvalues_chunks = np.array_split(np.asarray(nominal_bvalues), indices_or_sections=chunks)
    uvec_file_chunks = [uvec_file for i in range(chunks)]
    return_angles_chunks = [True for i in range(chunks)]
    plot_chunks = [False for i in range(chunks)]
    save_path_chunks = [save_path for i in range(chunks)]
    include_imaging_chunks = [include_imaging for i in range(chunks)]
    include_cross_terms_chunks = [include_cross_terms for i in range(chunks)]
    full_id_chunks = [full_id for i in range(chunks)]

    input_list = []
    for i in range(chunks):
        input_list.append([sequence_config_chunks[i], nominal_bvalues_chunks[i], uvec_file_chunks[i], return_angles_chunks[i], plot_chunks[i], save_path_chunks[i], include_imaging_chunks[i], include_cross_terms_chunks[i], full_id_chunks[i]])

    # Run simulation
    #bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)
    pool = mp.Pool(chunks)
    res = pool.starmap_async(simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation, input_list)
    pool.close()

## Accounting for imaging
include_imaging = True	
for resolution in resolution_lst:
    #sequence_config["xres"] = resolution
    #sequence_config["yres"] = resolution
    sequence_config["zres"] = resolution
    # Run simulation
    bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)

# Accounting for cross-terms
include_cross_terms = True
for resolution in resolution_lst:
    #sequence_config["xres"] = resolution
    #sequence_config["yres"] = resolution
    sequence_config["zres"] = resolution
    # Run simulation
    bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)