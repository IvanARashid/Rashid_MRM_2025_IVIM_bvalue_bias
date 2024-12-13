import numpy as np
from tqdm import tqdm
import time
import os
import dMRItools.waveforms as waveforms
import dMRItools.dMRI_io as dMRI_io

def get_b_using_custom_nominal_bvalues_on_generic_sequence_with_randomized_encoding_direction(sequence_config, nominal_bvalues, return_angles=False, include_imaging=False, include_cross_terms=False, plot=False):
    """
    1. Samples a random diffusion encoding direction
    2. Creates the diffusion encoding
    3. Creates a generic sequence with the specified diffusion encoding and resolution and sequence design parameters from the sequence_config dictionary
    4. Loops over the nominal bvalues
    5. The diffusion encoding is scaled to the bvalue in question, with relevant b-tensor calculations, and the bvalues and angles are recorded
    """
    bvalues = []

    dt = 12e-6

    ### Initialize sequence
    slice_select_trap = waveforms.trapezoid(delta=5e-3, slew_rate=1e6, amplitude=10e-3, gradient_update_rate=dt, return_time_axis=False)
    slice_select_zeros = np.zeros(slice_select_trap.shape)
    gwf_excitation = np.vstack((slice_select_zeros, slice_select_zeros, slice_select_trap))
    gwf_excitation = np.transpose(gwf_excitation)
    sequence = waveforms.Sequence_base(gwf_excitation, dt)

    ### Generate diffusion gradients
    # Create a diffusion waveform object. delta 20ms, DELTA 40 ms
    trapezoid = waveforms.trapezoid(delta=20e-3, slew_rate=1e6, amplitude=30e-3, gradient_update_rate=dt, return_time_axis=False)
    trapezoid_pre = np.concatenate((trapezoid, np.zeros(int(10e-3/dt))))
    trapezoid_post = np.concatenate((np.zeros(int(3e-3/dt)), trapezoid))
    gwf = waveforms.Gwf(dt=dt, duration180=7e-3, pre180=trapezoid_pre, post180=trapezoid_post)

    # Randomize angles and rotate the encoding
    angles_xyz = np.random.uniform(0, 2*np.pi, 3)
    gwf.rotate_encoding(angles_xyz[0], angles_xyz[1], angles_xyz[2])
    gwf.get_pre180_and_post180()

    ### Create the generic sequence
    
    sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"])
    sequence.get_rf(optimize=True, start_time=sequence.t_180)
    sequence.get_optimal_TE(start_time=3e-3)
    #sequence.set_b_by_scaling_amplitude(20e6)
    #print(f"Sequence b-value: {sequence.get_b(start_time=3e-3)*1e-6:.1f} s/mm2")

    for nominal_bvalue in nominal_bvalues:
        #sequence, start_time = initiate_sequence(path_to_unindexed_xml_file, volume, extwd=extwd)

        # Get the optimal RF and TE
        #sequence.get_rf(optimize=True, start_time=start_time, print_result=False)
        #TE = sequence.get_optimal_TE(start_time)

        # Set the encoding bvalue
        sequence.set_b_by_scaling_amplitude(nominal_bvalue*1e6, print_results=False, include_imaging=include_imaging, include_cross_terms=include_cross_terms)


        # Get the bvalue for the full sequence
        bvalues.append(sequence.get_b(start_time=3e-3)*1e-6)

        if plot:
            sequence.plot_gwf(start_time=3e-3)
            sequence.plot_q(start_time=3e-3)

    if return_angles:
        return np.array(bvalues), angles_xyz
    else:
        return np.array(bvalues)

def simulate_n_generic_sequences_with_random_encoding_directions(sequence_config, nominal_bvalues, n, return_angles=False, save_path=False, plot=False, include_imaging=False, include_cross_terms=False, full_id=True):
    bvalues = np.zeros((len(nominal_bvalues), n))
    rotation_angles = np.zeros((3, n))

    for i in tqdm(range(n), desc=f"xyres: {sequence_config['xres']}, zres: {sequence_config['zres']:}"):
        single_bval_set, angles_xyz = get_b_using_custom_nominal_bvalues_on_generic_sequence_with_randomized_encoding_direction(sequence_config, nominal_bvalues, return_angles=return_angles, include_imaging=include_imaging, include_cross_terms=include_cross_terms)

        bvalues[:, i] = single_bval_set

        if return_angles:
            rotation_angles[:, i] = angles_xyz

    if save_path != False:
        if include_imaging:
            id_string = "imaging_corrected"
        if include_cross_terms:
            id_string = "crossterm_corrected"
        if include_imaging == False and include_cross_terms == False:
            id_string = "uncorrected"

        resolution_string = f"xy{sequence_config['xres']}_z{sequence_config['zres']}"

        folder = ""
        if sequence_config["optimal"]:
            folder += "optimal_"
        if sequence_config["all_crushers"]:
            folder += "allCrushers_"
        if sequence_config["only_crush_when_needed"]:
            folder += "onlyCrushWhenNeeded_"
        
        folder += "sequence"

        ndir = f"ndir{n}"

        time_string = time.strftime("%Y%m%d_%H%M%S")

        if full_id:
            full_id = f"PhD1_{time_string}_"
        else:
            full_id = ""

        np.save(os.path.join(save_path, folder, f"{full_id}{id_string}_{ndir}_{resolution_string}_bvalues_actual"), bvalues) 
        np.save(os.path.join(save_path, folder, f"{full_id}{id_string}_{ndir}_{resolution_string}_bvalues_nominal"), np.asarray(nominal_bvalues)) 
        np.save(os.path.join(save_path, folder, f"{full_id}{id_string}_{ndir}_{resolution_string}_rotation_angles"), rotation_angles) 
    
    if return_angles:
        return bvalues, rotation_angles
    else:
        return bvalues

def get_b_using_custom_nominal_bvalues_on_generic_sequence_with_uvec_rotation(sequence_config, nominal_bvalues, uvec, return_angles=False, include_imaging=False, include_cross_terms=False, plot=False):
    """
    1. Samples a random diffusion encoding direction
    2. Creates the diffusion encoding
    3. Creates a generic sequence with the specified diffusion encoding and resolution and sequence design parameters from the sequence_config dictionary
    4. Loops over the nominal bvalues
    5. The diffusion encoding is scaled to the bvalue in question, with relevant b-tensor calculations, and the bvalues and angles are recorded
    """
    bvalues = []

    ### Initialize sequence
    slice_select_trap = waveforms.trapezoid(delta=5e-3, slew_rate=1e6, amplitude=10e-3, gradient_update_rate=4e-6, return_time_axis=False)
    slice_select_zeros = np.zeros(slice_select_trap.shape)
    gwf_excitation = np.vstack((slice_select_zeros, slice_select_zeros, slice_select_trap))
    gwf_excitation = np.transpose(gwf_excitation)
    sequence = waveforms.Sequence_base(gwf_excitation, 4e-6)

    ### Generate diffusion gradients
    # Create a diffusion waveform object. delta 20ms, DELTA 40 ms
    trapezoid = waveforms.trapezoid(delta=20e-3, slew_rate=1e6, amplitude=30e-3, gradient_update_rate=4e-6, return_time_axis=False)
    trapezoid = np.stack((trapezoid, trapezoid, trapezoid)).T
    trapezoid_pre = np.concatenate((trapezoid, np.zeros((int(10e-3/4e-6), 3))))
    trapezoid_post = np.concatenate((np.zeros((int(3e-3/4e-6),3)), trapezoid))
    gwf = waveforms.Gwf(dt=4e-6, duration180=7e-3, pre180=trapezoid_pre, post180=trapezoid_post)

    # Randomize angles and rotate the encoding
    #angles_xyz = np.random.uniform(0, 2*np.pi, 3)
    #gwf.rotate_encoding(angles_xyz[0], angles_xyz[1], angles_xyz[2])
    gwf.rotate_encoding_with_uvec(uvec)
    gwf.get_pre180_and_post180()

    ### Create the generic sequence
    try:
        sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"], qc=sequence_config["qc"])
    except:
        sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"])

    sequence.get_rf(optimize=True, start_time=sequence.t_180)
    sequence.get_optimal_TE(start_time=3e-3)
    #sequence.set_b_by_scaling_amplitude(20e6)
    #print(f"Sequence b-value: {sequence.get_b(start_time=3e-3)*1e-6:.1f} s/mm2")

    for nominal_bvalue in nominal_bvalues:
        #sequence, start_time = initiate_sequence(path_to_unindexed_xml_file, volume, extwd=extwd)

        # Get the optimal RF and TE
        #sequence.get_rf(optimize=True, start_time=start_time, print_result=False)
        #TE = sequence.get_optimal_TE(start_time)

        # Set the encoding bvalue
        sequence.set_b_by_scaling_amplitude(nominal_bvalue*1e6, print_results=False, include_imaging=include_imaging, include_cross_terms=include_cross_terms)

        # Revisit crushers
        if sequence_config["only_crush_when_needed"]:
            sequence.generic_sequence_crusher_check(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"])
            sequence.get_rf(optimize=True, start_time=sequence.t_180)
            sequence.get_optimal_TE(start_time=3e-3)


        # Get the bvalue for the full sequence
        bvalues.append(sequence.get_b(start_time=3e-3)*1e-6)

        if plot:
            sequence.plot_gwf(start_time=3e-3)
            sequence.plot_q(start_time=3e-3)

    if return_angles:
        return np.array(bvalues), uvec
    else:
        return np.array(bvalues)

def get_b_using_custom_nominal_bvalues_on_generic_sequence_with_uvec_rotation_no_opt(sequence_config, nominal_bvalues, uvec, return_angles=False, include_imaging=False, include_cross_terms=False, plot=False):
    """
    1. Samples a random diffusion encoding direction
    2. Creates the diffusion encoding
    3. Creates a generic sequence with the specified diffusion encoding and resolution and sequence design parameters from the sequence_config dictionary
    4. Loops over the nominal bvalues
    5. The diffusion encoding is scaled to the bvalue in question, with relevant b-tensor calculations, and the bvalues and angles are recorded
    """
    bvalues = []
    dt = 12e-6

    ### Initialize sequence
    slice_select_trap = waveforms.trapezoid(delta=5e-3, slew_rate=1e6, amplitude=10e-3, gradient_update_rate=dt, return_time_axis=False)
    slice_select_zeros = np.zeros(slice_select_trap.shape)
    gwf_excitation = np.vstack((slice_select_zeros, slice_select_zeros, slice_select_trap))
    gwf_excitation = np.transpose(gwf_excitation)
    sequence = waveforms.Sequence_base(gwf_excitation, dt)

    ### Generate diffusion gradients
    # Create a diffusion waveform object. delta 20ms, DELTA 40 ms
    trapezoid = waveforms.trapezoid(delta=20e-3, slew_rate=1e6, amplitude=30e-3, gradient_update_rate=dt, return_time_axis=False)
    trapezoid = np.stack((trapezoid, trapezoid, trapezoid)).T
    trapezoid_pre = np.concatenate((trapezoid, np.zeros((int(10e-3/dt), 3))))
    trapezoid_post = np.concatenate((np.zeros((int(3e-3/dt),3)), trapezoid))
    gwf = waveforms.Gwf(dt=dt, duration180=7e-3, pre180=trapezoid_pre, post180=trapezoid_post)

    # Randomize angles and rotate the encoding
    #angles_xyz = np.random.uniform(0, 2*np.pi, 3)
    #gwf.rotate_encoding(angles_xyz[0], angles_xyz[1], angles_xyz[2])
    gwf.rotate_encoding_with_uvec(uvec)
    gwf.get_pre180_and_post180()



    for nominal_bvalue in nominal_bvalues:
        ### Create the generic sequence
        try:
            sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], nominal_bvalue=nominal_bvalue*1e6, gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"], qc=sequence_config["qc"])
            #sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], nominal_bvalue=nominal_bvalue*1e6, gwf_diffusion_pre180=None, gwf_diffusion_post180=None, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"], qc=sequence_config["qc"])
        except:
            sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], nominal_bvalue=nominal_bvalue*1e6, gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"])
            #sequence.generic_sequence(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], nominal_bvalue=nominal_bvalue*1e6, gwf_diffusion_pre180=gwf.gwf_pre180, gwf_diffusion_post180=gwf.gwf_post180, optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"])

        sequence.get_rf(optimize=True, start_time=sequence.t_180)
        sequence.get_optimal_TE(start_time=3e-3)

        # Set the encoding bvalue
        sequence.set_b_by_scaling_amplitude(nominal_bvalue*1e6, print_results=False, include_imaging=False, include_cross_terms=False)

        # Calculate the b-value
        b = sequence.get_b_separate(include_imaging=include_imaging, include_cross_terms=include_cross_terms)

        # Revisit crushers
        #if sequence_config["only_crush_when_needed"]:
            #sequence.generic_sequence_crusher_check(sequence_config["xres"], sequence_config["yres"], sequence_config["zres"], optimal=sequence_config["optimal"], all_crushers=sequence_config["all_crushers"], crushers=sequence_config["crushers"], only_crush_when_needed=sequence_config["only_crush_when_needed"])
            #sequence.get_rf(optimize=True, start_time=sequence.t_180)
            #sequence.get_optimal_TE(start_time=3e-3)


        # Get the bvalue for the full sequence
        #bvalues.append(sequence.get_b(start_time=3e-3)*1e-6)
        bvalues.append(b*1e-6)

        if plot:
            sequence.plot_gwf(start_time=3e-3)
            sequence.plot_q(start_time=3e-3)

    if return_angles:
        return np.array(bvalues), uvec
    else:
        return np.array(bvalues)

def simulate_n_generic_sequences_with_uvec_rotation(sequence_config, nominal_bvalues, uvec_file, return_angles=False, save_path=False, plot=False, include_imaging=False, include_cross_terms=False, full_id=True):
    bvecs = dMRI_io.read_bvec_file(uvec_file)
    #uvec_array = dMRI_io.bvec_to_uvec(bvecs)

    no_of_uvecs = bvecs.shape[1]
    bvalues = np.zeros((len(nominal_bvalues), no_of_uvecs))
    rotation_angles = np.zeros((3, no_of_uvecs))

    for i in tqdm(range(no_of_uvecs), desc=f"xyres: {sequence_config['xres']}, zres: {sequence_config['zres']:}"):
        single_bval_set, angles_xyz = get_b_using_custom_nominal_bvalues_on_generic_sequence_with_uvec_rotation_no_opt(sequence_config, nominal_bvalues, bvecs[:,i], return_angles=return_angles, plot=plot, include_imaging=include_imaging, include_cross_terms=include_cross_terms)

        bvalues[:, i] = single_bval_set

        if return_angles:
            rotation_angles[:, i] = bvecs[:,i]

    if save_path != False:
        if include_imaging:
            id_string = "imaging_corrected"
        if include_cross_terms:
            id_string = "crossterm_corrected"
        if include_imaging == False and include_cross_terms == False:
            id_string = "uncorrected"

        resolution_string = f"xy{sequence_config['xres']}_z{sequence_config['zres']}"

        folder = ""
        if sequence_config["optimal"]:
            folder += "optimal_"
        if sequence_config["all_crushers"]:
            folder += "allCrushers_"
        if sequence_config["only_crush_when_needed"]:
            folder += "onlyCrushWhenNeeded_"
        
        folder += "sequence"

        uvec_fname = os.path.basename(uvec_file)
        uvec_fname = os.path.splitext(uvec_fname)[0]

        time_string = time.strftime("%Y%m%d_%H%M%S")

        if full_id:
            full_id = f"PhD1_{time_string}_"
        else:
            full_id = ""

        np.save(os.path.join(save_path, folder, f"{full_id}{id_string}_{uvec_fname}_{resolution_string}_bvalues_actual"), bvalues) 
        np.save(os.path.join(save_path, folder, f"{full_id}{id_string}_{uvec_fname}_{resolution_string}_bvalues_nominal"), np.asarray(nominal_bvalues)) 
        np.save(os.path.join(save_path, folder, f"{full_id}{id_string}_{uvec_fname}_{resolution_string}_uvecs"), rotation_angles) 
    
    if return_angles:
        return bvalues, rotation_angles
    else:
        return bvalues
    
def find_worst_case_waveforms(path_to_folder, filename_until_end_of_datetime, plot=False):
    """
    Function that finds the rotation angles of the worst case waveforms generated by the simulate_n_sequences_with_random_encoding_directions() function.
    The angles are applied to a pair of trapezoids and plotted to illustrate the two scenarios.
    The two worst-case scenarios are defined as the encodings that yield the lowest and highest b-value deviation from the normal. 

    path_to_folder : Path to the folder containing the files
    filename_until_end_of_datetime : str. The filename until the end of the time section, i.e. just before e.g. _bvalues_actual.npy
    """
    filename_bvalues_actual = filename_until_end_of_datetime + "_bvalues_actual.npy"
    filename_bvalues_nominal = filename_until_end_of_datetime + "_bvalues_nominal.npy"
    filename_rotation_angles = filename_until_end_of_datetime + "_rotation_angles.npy"

    # Load the arrays
    bvalues_actual = np.load(os.path.join(path_to_folder, filename_bvalues_actual))
    bvalues_nominal = np.load(os.path.join(path_to_folder, filename_bvalues_nominal))
    rotation_angles = np.load(os.path.join(path_to_folder, filename_rotation_angles))

    worst_angles = np.zeros((2, 3, len(bvalues_nominal)))
    worst_bvalues = np.zeros((2, len(bvalues_nominal)))
    # For each nominal b-value, find the index containing the minimum actual b-value
    for i in range(len(bvalues_nominal)):
        # Get the indices of min and max actual b-values for the nominal b-value
        worst_min_idx = np.argmin(bvalues_actual[i,:])
        worst_max_idx = np.argmax(bvalues_actual[i,:])

        #Idx 0 is worst-case minimum b, idx 1 is worst-case maximum b
        worst_angles[0, :, i] = rotation_angles[:,worst_min_idx]
        worst_angles[1, :, i] = rotation_angles[:,worst_max_idx]

        worst_bvalues[0, i] = bvalues_actual[i, worst_min_idx]
        worst_bvalues[1, i] = bvalues_actual[i, worst_max_idx]

        if plot:
            fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(6,3))

            # Generate some Stejskal-Tanner waveforms
            trap = waveforms.trapezoid(30e-3, 50, 10e-3, 4e-6)
            gwf = waveforms.Gwf(4e-6, 7e-3, trap, trap)

            # Set the b-value
            gwf.set_b_by_scaling_amplitude(bvalues_nominal[i]*1e6)

            # Apply rotation (min)
            gwf.rotate_encoding(worst_angles[0, 0, i], worst_angles[0, 1, i], worst_angles[0, 2, i])

            # Plot the encoding (min)
            #norm = np.max(np.abs(np.concatenate((gwf.gwf_x, gwf.gwf_y, gwf.gwf_z))))*1e3
            norm = 1
            axs[0].plot(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, label="x")
            axs[0].plot(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, label="y")
            axs[0].plot(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, label="z")
            axs[0].fill_between(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, alpha=0.2)
            axs[0].fill_between(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, alpha=0.2)
            axs[0].fill_between(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, alpha=0.2)
            axs[0].set_ylabel("Gradient amplitude [mT/m]")
            axs[0].set_xlabel("Time [ms]")
            axs[0].set_title("Worst-case minimum")

            # Reverse rotation (min)
            gwf.rotate_encoding(worst_angles[0, 0, i], worst_angles[0, 1, i], worst_angles[0, 2, i], inverse=True)

            # Apply rotation (max)
            gwf.rotate_encoding(worst_angles[1, 0, i], worst_angles[1, 1, i], worst_angles[1, 2, i])

            # Plot the encoding (max)
            #norm = np.max(np.abs(np.concatenate((gwf.gwf_x, gwf.gwf_y, gwf.gwf_z))))*1e3
            axs[1].plot(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, label="x")
            axs[1].plot(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, label="y")
            axs[1].plot(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, label="z")
            axs[1].fill_between(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, alpha=0.2)
            axs[1].fill_between(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, alpha=0.2)
            axs[1].fill_between(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, alpha=0.2)
            axs[1].set_xlabel("Time [ms]")
            axs[1].set_title("Worst-case maximum")

            # Reverse rotation (max)
            gwf.rotate_encoding(worst_angles[1, 0, i], worst_angles[1, 1, i], worst_angles[1, 2, i], inverse=True)

            axs[1].legend(loc=0)
            fig.suptitle(f"Nominal b-value {bvalues_nominal[i]:.0f} s/mm2", y=1.05)
            #fig.tight_layout()




    return worst_bvalues, worst_angles, bvalues_nominal

def find_worst_case_waveforms_uvecs(path_to_folder, filename_until_end_of_datetime, plot=False):
    """
    Function that finds the rotation angles of the worst case waveforms generated by the simulate_n_sequences_with_random_encoding_directions() function.
    The angles are applied to a pair of trapezoids and plotted to illustrate the two scenarios.
    The two worst-case scenarios are defined as the encodings that yield the lowest and highest b-value deviation from the normal. 

    path_to_folder : Path to the folder containing the files
    filename_until_end_of_datetime : str. The filename until the end of the time section, i.e. just before e.g. _bvalues_actual.npy
    """
    filename_bvalues_actual = filename_until_end_of_datetime + "_bvalues_actual.npy"
    filename_bvalues_nominal = filename_until_end_of_datetime + "_bvalues_nominal.npy"
    filename_rotation_angles = filename_until_end_of_datetime + "_uvecs.npy"

    # Load the arrays
    bvalues_actual = np.load(os.path.join(path_to_folder, filename_bvalues_actual))
    bvalues_nominal = np.load(os.path.join(path_to_folder, filename_bvalues_nominal))
    rotation_angles = np.load(os.path.join(path_to_folder, filename_rotation_angles))

    worst_angles = np.zeros((2, 3, len(bvalues_nominal)))
    worst_bvalues = np.zeros((2, len(bvalues_nominal)))
    # For each nominal b-value, find the index containing the minimum actual b-value
    for i in range(len(bvalues_nominal)):
        # Get the indices of min and max actual b-values for the nominal b-value
        worst_min_idx = np.argmin(bvalues_actual[i,:])
        worst_max_idx = np.argmax(bvalues_actual[i,:])

        #Idx 0 is worst-case minimum b, idx 1 is worst-case maximum b
        worst_angles[0, :, i] = rotation_angles[:,worst_min_idx]
        worst_angles[1, :, i] = rotation_angles[:,worst_max_idx]

        worst_bvalues[0, i] = bvalues_actual[i, worst_min_idx]
        worst_bvalues[1, i] = bvalues_actual[i, worst_max_idx]

        if plot:
            fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(6,3))

            # Generate some Stejskal-Tanner waveforms
            trap = waveforms.trapezoid(30e-3, 50, 10e-3, 4e-6)
            gwf = waveforms.Gwf(4e-6, 7e-3, trap, trap)

            # Set the b-value
            gwf.set_b_by_scaling_amplitude(bvalues_nominal[i]*1e6)

            # Apply rotation (min)
            gwf.rotate_encoding_with_uvec(worst_angles[0, 0, i], worst_angles[0, 1, i], worst_angles[0, 2, i])

            # Plot the encoding (min)
            #norm = np.max(np.abs(np.concatenate((gwf.gwf_x, gwf.gwf_y, gwf.gwf_z))))*1e3
            norm = 1
            axs[0].plot(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, label="x")
            axs[0].plot(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, label="y")
            axs[0].plot(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, label="z")
            axs[0].fill_between(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, alpha=0.2)
            axs[0].fill_between(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, alpha=0.2)
            axs[0].fill_between(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, alpha=0.2)
            axs[0].set_ylabel("Gradient amplitude [mT/m]")
            axs[0].set_xlabel("Time [ms]")
            axs[0].set_title("Worst-case minimum")

            # Reverse rotation (min)
            gwf.rotate_encoding_with_uvec(worst_angles[0, 0, i], worst_angles[0, 1, i], worst_angles[0, 2, i], inverse=True)

            # Apply rotation (max)
            gwf.rotate_encoding_with_uvec(worst_angles[1, 0, i], worst_angles[1, 1, i], worst_angles[1, 2, i])

            # Plot the encoding (max)
            #norm = np.max(np.abs(np.concatenate((gwf.gwf_x, gwf.gwf_y, gwf.gwf_z))))*1e3
            axs[1].plot(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, label="x")
            axs[1].plot(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, label="y")
            axs[1].plot(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, label="z")
            axs[1].fill_between(gwf.t*1e3, gwf.gwf_x*gwf.rf[:,0]*1e3/norm, alpha=0.2)
            axs[1].fill_between(gwf.t*1e3, gwf.gwf_y*gwf.rf[:,1]*1e3/norm, alpha=0.2)
            axs[1].fill_between(gwf.t*1e3, gwf.gwf_z*gwf.rf[:,2]*1e3/norm, alpha=0.2)
            axs[1].set_xlabel("Time [ms]")
            axs[1].set_title("Worst-case maximum")

            # Reverse rotation (max)
            gwf.rotate_encoding_with_uvec(worst_angles[1, 0, i], worst_angles[1, 1, i], worst_angles[1, 2, i], inverse=True)

            axs[1].legend(loc=0)
            fig.suptitle(f"Nominal b-value {bvalues_nominal[i]:.0f} s/mm2", y=1.05)
            #fig.tight_layout()




    return worst_bvalues, worst_angles, bvalues_nominal

def read_generic_simulation_log(filename, path, ax=False, alpha=0.2, label=None, color="red"):
    # Read the file
    worst_bvalues, worst_angles, bvals_nom = find_worst_case_waveforms(path, filename, plot=False)
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
    signals_lower = ivim_signal(lower_b, Dstar=Dstar)
    signals_upper = ivim_signal(upper_b, Dstar=Dstar)

    if ax:
        ax.plot(nominal_b, signals_lower, ls="")
        ax.plot(nominal_b, signals_upper, ls="")
        ax.fill_between(x=nominal_b, y1=signals_upper, y2=signals_lower, alpha=alpha, color=color, label=label)

    return signals_lower, signals_upper

def ivim_signal(bvalues, S0=1, f=0.1, Dstar=0.05, D=0.001):
    return S0*f*np.exp(-bvalues*Dstar) + S0*(1-f)*np.exp(-bvalues*D)

