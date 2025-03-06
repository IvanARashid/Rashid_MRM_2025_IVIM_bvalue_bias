import numpy as np
import scipy
import os




def ivim(b, S0=1, f=.1, Dstar=20e-3, D=1e-3):
    return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

def powder_average_signals_from_file(path_to_file):

    # Build the filename string
    filename_bvalues_actual = path_to_file + "_bvalues_actual.npy"

    # Load the array
    bvalues_actual = np.load(filename_bvalues_actual)
    bvalues_actual = np.flip(bvalues_actual)

    # Calculate signals
    signals = ivim(bvalues_actual)

    # Geometric mean
    powder_averaged_signals = scipy.stats.mstats.gmean(signals, axis=1)

    return powder_averaged_signals

def signals_from_file(path_to_file, f=.1, Dstar=20e-3, D=1e-3):
    # Build the filename string
    filename_bvalues_actual = path_to_file + "_bvalues_actual.npy"

    # Load the array
    bvalues_actual = np.load(filename_bvalues_actual)
    bvalues_actual = np.flip(bvalues_actual)

    # Calculate signals
    signals = ivim(bvalues_actual, f=f, Dstar=Dstar, D=D)

    # Geometric mean
    #powder_averaged_signals = scipy.stats.mstats.gmean(signals, axis=1)

    return signals