
import sys
sys.path.append(".")
import numpy as np
import dMRItools.simulate_pulse_sequences as simulate_pulse_sequences
import os
import multiprocessing as mp
import copy

save_path = r".\simulation_data\figs_reviewer_suggested"

uvec_path = r".\simulations\bvecs"
uvec_fname = "xyz.bvec"
uvec_file = os.path.join(uvec_path, uvec_fname)

resolution_lst = np.round(np.linspace(1,4,13)*1e-3, decimals=7)

sequence_config = {"xres" : 2e-3,
                "yres" : 2e-3,
                "zres" : 2e-3,
                "crushers" : True,
                "optimal" : True,
                "all_crushers" : True,
                "only_crush_when_needed" : True,
				"qc" : False}

def main1(sequence_config=sequence_config, save_path=save_path, uvec_file=uvec_file, resolution_list=resolution_lst):
    full_id = False

    # Nominal b-values
    nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # Include B_im
    include_imaging = False
    # Include B_ct
    include_cross_terms = False


    chunks = resolution_lst.shape[0]

    sequence_configs = []
    for resolution in resolution_lst:
        sequence_config["xres"] = resolution
        sequence_config["yres"] = resolution
        sequence_config["zres"] = resolution
        sequence_configs.append(copy.deepcopy(sequence_config))


    nominal_bvalues_chunks = [nominal_bvalues for i in range(chunks)]
    uvec_file_chunks = [uvec_file for i in range(chunks)]
    return_angles_chunks = [True for i in range(chunks)]
    plot_chunks = [False for i in range(chunks)]
    save_path_chunks = [save_path for i in range(chunks)]
    include_imaging_chunks = [include_imaging for i in range(chunks)]
    include_cross_terms_chunks = [include_cross_terms for i in range(chunks)]
    full_id_chunks = [full_id for i in range(chunks)]

    input_list = []
    for i in range(chunks):
        input_list.append([sequence_configs[i], nominal_bvalues_chunks[i], uvec_file_chunks[i], return_angles_chunks[i], save_path_chunks[i], plot_chunks[i], include_imaging_chunks[i], include_cross_terms_chunks[i], full_id_chunks[i]])


    pool = mp.Pool(chunks)
    res = pool.starmap_async(simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation, input_list)
    pool.close()

    results = res.get()

def main2(sequence_config=sequence_config, save_path=save_path, uvec_file=uvec_file):
    full_id = False

    # Nominal b-values
    nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # Include B_im
    include_imaging = True
    # Include B_ct
    include_cross_terms = False


    #resolution_lst = np.round(np.linspace(1,4,13)*1e-3, decimals=7)

    chunks = resolution_lst.shape[0]

    sequence_configs = []
    for resolution in resolution_lst:
        sequence_config["xres"] = resolution
        sequence_config["yres"] = resolution
        sequence_config["zres"] = resolution
        sequence_configs.append(copy.deepcopy(sequence_config))


    nominal_bvalues_chunks = [nominal_bvalues for i in range(chunks)]
    uvec_file_chunks = [uvec_file for i in range(chunks)]
    return_angles_chunks = [True for i in range(chunks)]
    plot_chunks = [False for i in range(chunks)]
    save_path_chunks = [save_path for i in range(chunks)]
    include_imaging_chunks = [include_imaging for i in range(chunks)]
    include_cross_terms_chunks = [include_cross_terms for i in range(chunks)]
    full_id_chunks = [full_id for i in range(chunks)]

    input_list = []
    for i in range(chunks):
        input_list.append([sequence_configs[i], nominal_bvalues_chunks[i], uvec_file_chunks[i], return_angles_chunks[i], save_path_chunks[i], plot_chunks[i], include_imaging_chunks[i], include_cross_terms_chunks[i], full_id_chunks[i]])

    pool = mp.Pool(chunks)
    res = pool.starmap_async(simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation, input_list)
    pool.close()

    results = res.get()

def main3(sequence_config=sequence_config, save_path=save_path, uvec_file=uvec_file):
    full_id = False


    # Nominal b-values
    nominal_bvalues = [800, 700, 600, 500, 400, 300, 200, 175, 150, 125, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    # Include B_im
    include_imaging = True
    # Include B_ct
    include_cross_terms = True


    #resolution_lst = np.round(np.linspace(2,10,9)*1e-3, decimals=7)

    chunks = resolution_lst.shape[0]

    sequence_configs = []
    for resolution in resolution_lst:
        sequence_config["xres"] = resolution
        sequence_config["yres"] = resolution
        sequence_config["zres"] = resolution
        sequence_configs.append(copy.deepcopy(sequence_config))


    nominal_bvalues_chunks = [nominal_bvalues for i in range(chunks)]
    uvec_file_chunks = [uvec_file for i in range(chunks)]
    return_angles_chunks = [True for i in range(chunks)]
    plot_chunks = [False for i in range(chunks)]
    save_path_chunks = [save_path for i in range(chunks)]
    include_imaging_chunks = [include_imaging for i in range(chunks)]
    include_cross_terms_chunks = [include_cross_terms for i in range(chunks)]
    full_id_chunks = [full_id for i in range(chunks)]

    input_list = []
    for i in range(chunks):
        input_list.append([sequence_configs[i], nominal_bvalues_chunks[i], uvec_file_chunks[i], return_angles_chunks[i], save_path_chunks[i], plot_chunks[i], include_imaging_chunks[i], include_cross_terms_chunks[i], full_id_chunks[i]])

    pool = mp.Pool(chunks)
    res = pool.starmap_async(simulate_pulse_sequences.simulate_n_generic_sequences_with_uvec_rotation, input_list)
    pool.close()

    results = res.get()

if __name__ == "__main__":
    main1()
    main2()
    main3()