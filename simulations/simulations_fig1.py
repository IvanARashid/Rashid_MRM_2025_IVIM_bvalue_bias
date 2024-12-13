import numpy as np
import dMRItools.simulate_pulse_sequences as simulate_pulse_sequences
import sys
import os


# Commandline arguments
try:
	save_path = sys.argv[1]
	print(f"save_path = {save_path}")
except:
	# Default on IAR's laptop
	save_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\20241101"


full_id = False
sequence_config = {"xres" : 2e-3,
                   "yres" : 2e-3,
                   "zres" : 2e-3,
                   "crushers" : True,
                   "optimal" : True,
                   "all_crushers" : True,
                   "only_crush_when_needed" : True}

uvec_path = r"C:\Users\ivan5\Box\PhD\Articles\PhD1 - IVIM incl imaging gradients\bvalue simulations\bvecs"
uvec_fname = "froeling_200.bvec"
uvec_file = os.path.join(uvec_path, uvec_fname)

# Nominal b-values
nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
# Number of diffusion directions
n = 1000
# Include B_im
include_imaging = False
# Include B_ct
include_cross_terms = False


sequence_config["xres"] = 1e-3
sequence_config["yres"] = 1e-3
sequence_config["zres"] = 1e-3

# Run simulation
bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)

include_imaging = True
bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)


include_cross_terms = True
bvalues_many, angles = simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation(sequence_config=sequence_config, nominal_bvalues=nominal_bvalues, uvec_file=uvec_file, return_angles=True, plot=False, save_path=save_path, include_imaging=include_imaging, include_cross_terms=include_cross_terms, full_id=full_id)