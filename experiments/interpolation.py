'''
THIS FILE SHOULD BE RUN WITH WHATEVER TESTING MODEL IS BEING USED
AS IT USES THE MODEL'S DECODER TO CONVERT A LATENT SPACE VECTOR
INTO ITS RECONSTRUCTED SHAPE.

THE CODE THAT IS SHOWN IS TAKEN FROM THE 3D-GAE LINKED IN THE README
'''
import re
import os
import sys
import time
import numpy as np
import h5py
import numpy as np
import json
import time
import re
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from GAE import *
from utils import npytar, encoder_predict

from scipy.spatial import distance_matrix


# ---------- INPUT FILES ----------

# The z vector file is the latent space generated from the training model:
SHAPE_TYPE = 'Chair'
z_vectors = 'z_vectors.csv'
parameter_vectors = '../datasets/parameter_vectors/parameter_vectors_' + SHAPE_TYPE.lower() + '.h5'
parameter_config = '../datasets/parameter_config/parameter_config_' + SHAPE_TYPE.lower() + '.json'

# ---------- OUTPUT FILES ----------
OUTPUT_FILE_DIR = 'interpolations/'

param_indices_file = OUTPUT_FILE_DIR+'param_indices.npy'
# These output the euclidean distance matricies for the following datasets:
z_distances_file = OUTPUT_FILE_DIR+'z_distances.npy'
param_distances_file = OUTPUT_FILE_DIR+'param_distances.npy'

OVERWRITE_FILE = False

# This outputs the specific set of interpolations at index:
OUTPUT_INTERPOLATION_INDEX = 0

def output_data(filename, data):
    print("Outputted ", filename)
    np.save(filename, data)

def read_h5(filename):
    with h5py.File(filename, "r") as f:
        # Get the data
        np_arr = np.array(f[SHAPE_TYPE])
        np_arr = np_arr[0, :, :]
        np_arr = np.swapaxes(np_arr,0,1)
        return np_arr

def read_json(filename):
    f = open(filename,)
    data = json.load(f)
    return data

def calculate_distances(x, y):
    distances = x - y
    distances = np.square(distances)
    sum_distances = np.sum(distances)
    euclidean = np.sqrt(sum_distances)

    return euclidean

def calculate_step(min_val, max_val, step):
    step_val = float(max_val - min_val)/ float(step-1)
    return step_val

def get_same_indices(arr, in_vectors, index, step_val):
    vectors = np.delete(in_vectors, index, axis=1)
    indices = np.array(range(vectors.shape[0]))

    vectors_unique = np.unique(vectors, axis=0)

    for i in range(vectors_unique.shape[0]):
        unique_row = vectors_unique[i]
        same_arr = []
        for j in range(vectors.shape[0]):
            vector = vectors[j]
            if(np.array_equal(unique_row,vector)):
                same_arr.append(j)

        if(len(same_arr) > 2):
            same_arr = np.array(same_arr)
            vectors_found = in_vectors[same_arr]
            reshaped_indices = np.expand_dims(same_arr, axis=1)
            vectors_found = np.append(arr = vectors_found, values = reshaped_indices, axis= 1)

            vectors_sorted = vectors_found[vectors_found[:, index].argsort()]

            for j in range(vectors_sorted.shape[0]-2):
                start = vectors_sorted[j][index]
                end = vectors_sorted[j+2][index]
                avg =vectors_sorted[j+1][index]

                diff_1 = round(end - avg, 2)
                diff_2 = round(avg - start, 2)
                step_val_rounded = round(step_val, 2)

                shape_indices = []
                if(diff_1 == step_val_rounded and diff_2 == step_val_rounded):

                    shape_indices.append(vectors_sorted[j][-1])
                    shape_indices.append(vectors_sorted[j+1][-1])
                    shape_indices.append(vectors_sorted[j+2][-1])

                if(len(shape_indices) > 2):
                    arr.append(shape_indices)

    return arr

def calculate_param_indices(param_vectors):
    parameter_data = read_json(parameter_config)[SHAPE_TYPE]['config']

    indices = []
    i = 0
    for param in parameter_data:
        if(param['type'] == 'scalar'):
            step_val = calculate_step(param['data'][0], param['data'][1], param['step'])
            indices = get_same_indices(indices, param_vectors, i, step_val)
        i += 1
    indices = np.array(indices).astype(int)
    return indices

#The decoder is your network model's decoder:
def interpolate(decoder, input_z_vectors, i, j):

    step = 0.5
    z_vectors = []
    z_vector_i = np.array([input_z_vectors[i]])
    z_vector_j = np.array([input_z_vectors[j]])

    z_vector_interpolate = 0.5*z_vector_i+ 0.5*z_vector_j
    generation = decoder.predict([z_vector_interpolate, np.array([0])])
    generation[generation >= 0.5] = 1
    generation[generation < 0.5] = 0

    return generation[0, 0, :]

if __name__ == '__main__':

    if not os.path.exists(OUTPUT_FILE_DIR):
        os.makedirs(OUTPUT_FILE_DIR)

    if not os.path.exists(OUTPUT_FILE_DIR+'real/'):
        os.makedirs(OUTPUT_FILE_DIR+'real/')

    if not os.path.exists(OUTPUT_FILE_DIR+'interp/'):
        os.makedirs(OUTPUT_FILE_DIR+'interp/')

    if not os.path.exists(OUTPUT_FILE_DIR+'neigh_start/'):
        os.makedirs(OUTPUT_FILE_DIR+'neigh_start/')

    if not os.path.exists(OUTPUT_FILE_DIR+'neigh_end/'):
        os.makedirs(OUTPUT_FILE_DIR+'neigh_end/')
    start_time = time.time()

    z_vectors = np.genfromtxt(z_vectors, delimiter=',')
    param_vectors = read_h5(parameter_vectors)

    print("Vectors: Z - {}. Parameters - {}".format(z_vectors.shape, param_vectors.shape))

    param_indices = []
    if not os.path.exists(param_indices_file) or OVERWRITE_FILE:
        param_indices = calculate_param_indices(param_vectors)
        output_data(param_indices_file, param_indices)
    else:
        param_indices = np.load(param_indices_file)
    param_indices = calculate_param_indices(param_vectors)
    print("Parameters indices- {}".format(param_indices.shape))

    #------------------
    # This section contains the specific model architecture from the 3D-GAE
    # for computing the interpolations. Replace as you see fit.

    model = get_model()

    inputs = model['inputs']
    indices = model['indices']
    outputs = model['outputs']
    z = model['z']

    encoder = model['encoder']
    decoder = model['decoder']
    gae = model['gae']

    z_vectors, data_test = encoder_predict.get_encoded_data(encoder, gae)

    shape_indices = np.array(range(data_test.shape[0]))

    reconstructions = gae.predict([data_test, shape_indices])
    reconstructions[reconstructions >= 0.5] = 1
    reconstructions[reconstructions < 0.5] = 0
    #------------------

    print("z_vectors shape: {}. Reconstructions shape: {}".format(z_vectors.shape, reconstructions.shape))

    distance_scalars = []
    distance_neighbours = []
    distance_to_interps = []
    distance_to_recons = []
    OUTPUT_SHAPE = False
    for i in range(param_indices.shape[0]):
        print("Calculating distances for shape: {}".format(i))
        recon_interpolate = interpolate(decoder, z_vectors, param_indices[i][0], param_indices[i][2])

        recon_start = reconstructions[param_indices[i][0]][0]
        recon_mid = reconstructions[param_indices[i][1]][0]
        recon_end = reconstructions[param_indices[i][2]][0]
        distance_scalar = calculate_distances(recon_interpolate, recon_mid)
        distance_neighbour = calculate_distances(recon_start, recon_end)

        distance_to_interp_start = calculate_distances(recon_start, recon_interpolate)
        distance_to_interp_end = calculate_distances(recon_end, recon_interpolate)
        distance_to_recon_start = calculate_distances(recon_start, recon_mid)
        distance_to_recon_end = calculate_distances(recon_end, recon_mid)

        distance_scalars.append(distance_scalar)
        distance_neighbours.append(distance_neighbour)

        distance_to_interps.append(distance_to_interp_start)
        distance_to_interps.append(distance_to_interp_end)
        distance_to_recons.append(distance_to_recon_start)
        distance_to_recons.append(distance_to_recon_end)

        if(i == OUTPUT_INTERPOLATION_INDEX):
            p1 = param_vectors[param_indices[i][0]]
            p2 = param_vectors[param_indices[i][1]]
            p3 = param_vectors[param_indices[i][2]]
            save_volume.save_output(recon_interpolate, 32, OUTPUT_FILE_DIR+'interp/', i)
            save_volume.save_output(recon_mid, 32, OUTPUT_FILE_DIR+'real/', i)
            save_volume.save_output(recon_start, 32, OUTPUT_FILE_DIR+'neigh_start/', i)
            save_volume.save_output(recon_end, 32, OUTPUT_FILE_DIR+'neigh_end/', i)

    distance_scalars = np.array(distance_scalars)
    distance_neighbours = np.array(distance_neighbours)
    distance_to_interps = np.array(distance_to_interps)
    distance_to_recons = np.array(distance_to_recons)

    avg_distance_scalar = np.mean(distance_scalars)
    avg_distance_neighbour = np.mean(distance_neighbours)
    avg_distance_to_interp = np.mean(distance_to_interps)
    avg_distance_to_recon = np.mean(distance_to_recons)

    print("Distances - Scalar: {}. Neighbour: {}".format(avg_distance_scalar, avg_distance_neighbour))
    print("Distances to - interp: {}. recon: {}".format(avg_distance_to_interp, avg_distance_to_recon))

    end_time = time.time()
    print("Took {} seconds to complete.".format(end_time - start_time))
