import os
import numpy as np
import pandas as pd

def write_array_to_bval(bvals, path, fname):
    """
    Creates a .bval-file from array.
    """

    file_path = os.path.join(path, f"{fname}.bval")

    # Create file
    file = open(file_path, "w")

    # Loop over the bvals in the array
    for i in range(len(bvals)):
        # Write to file
        file.write(f"{bvals[i]} ")
    
    file.close()

def write_array_to_btens(btens, path, fname):
    """Create

    Args:
        btens (_type_): _description_
        path (_type_): _description_
        fname (_type_): _description_
    """

    file_path = os.path.join(path, f"{fname}.btens")

    # Create file
    file = open(file_path, "w")

    for row in range(btens.shape[0]):
        file.write(f"{btens[row,0]} {btens[row,1]} {btens[row,2]} {btens[row,3]} {btens[row,4]} {btens[row,5]}\n")

    file.close()

def read_bvec_file(file):
    bvecs = pd.read_csv(file, header=None, delimiter="\s+").to_numpy(dtype="float")
    return bvecs

def bvec_to_uvec(bvec):
    """
    Converts a bvec array to array of uvecs. Essentially only the transpose.
    """
    return bvec.T