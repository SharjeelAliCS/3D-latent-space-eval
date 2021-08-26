import re
import os
import sys
import time
import numpy as np
import h5py
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from scipy.spatial import distance_matrix

# ---------- INPUT FILES ----------

# The z vector file is the latent space generated from the training model:
SHAPE_TYPE = 'Chair'
z_vectors = 'z_vectors.csv'
parameter_vectors = '../datasets/parameter_vectors/parameter_vectors_' + SHAPE_TYPE.lower() + '.h5'
parameter_config = '../datasets/parameter_config/parameter_config_' + SHAPE_TYPE.lower() + '.json'

# ---------- OUTPUT FILES ----------
OUTPUT_FILE_DIR = 'distances/'
# These output the euclidean distance matricies for the following datasets:
z_distances_file = OUTPUT_FILE_DIR+'z_distances.npy'
param_distances_file = OUTPUT_FILE_DIR+'param_distances.npy'

# This output an h5 file containing all the datasets.
correlation_file = OUTPUT_FILE_DIR+'correlations.npy'

def output_data(filename, data):
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
    sum_distances = np.sum(distances)
    euclidean = np.sqrt(sum_distances)

    #euclidean = distance_matrix(x,y)
    return euclidean

def calculate_step(min_val, max_val, step):
    step_val = float(max_val - min_val)/ float(step-1)
    return step_val

def get_sorted_indices(z_vectors):
    sorted_arr = []
    for i in range(z_vectors.shape[1]):
        arr = [z_vectors[:, i].argsort()]
        sorted_arr.append(arr[0])
    return np.array(sorted_arr)

def calculate_pairwise(sorted_param):
    pairwise_list = []
    for i in range(sorted_param.shape[0]):
        pairwise = np.diff(sorted_param[i], axis = 0)
        pairwise_list.append(pairwise)

    return np.array(pairwise_list)

def calculate_sums(pairwise_list):
    param_sums = []
    for i in range(pairwise_list.shape[0]):
        sums = np.count_nonzero(pairwise_list[i], axis=0)
        param_sums.append(sums)

    return np.array(param_sums)

def calculate_smallest_change(param_sums):
    min_values = []
    min_indices = []
    for i in range(param_sums.shape[0]):
        min_val = np.where(param_sums[i] == np.amin(param_sums[i]))

        min_values.append(param_sums[i][min_val[0][0]])
        min_indices.append(min_val[0][0])
    return np.array(min_indices), np.array(min_values)

def within_range(val, ranges):
    return val > ranges[0] and val < ranges[1]

def calculate_change_less_than_iqr(param_sums):
    values = []
    indices = []
    for i in range(param_sums.shape[0]):

        q25, q75 = outlier_treatment(param_sums[i])

        val_less = np.where(param_sums[i] < q25)[0]
        val_greater = np.where(param_sums[i] > q75)[0]
        if(val_greater.shape[0] <= val_less.shape[0] ):
            indices.append(val_less)
            values.append(param_sums[i][val_less])
        else:
            indices.append([])
            values.append([])

    return np.array(indices, dtype=object), np.array(values,dtype=object)

def outlier_treatment(x):

    x = np.sort(x)
    q1,q3 = np.percentile(x , [25,75])
    iq = q3 - q1
    lower_range = q1 - (1.5 * iq)
    upper_range = q3 + (1.5 * iq)
    return lower_range,upper_range

def num_attributes_change(indices):
    values = []
    for i in range(indices.shape[0]):
        values.append(len(indices[i]))
    return np.array(values)

def num_times_param_change(param_sums):
    values = []
    indices = []

def param_onezeros(config,param_type, val):
    for param in config:
        if(param_type == param['name'] and param['type'] == 'scalar'):
            param_arr = param['data']

            stepVal = float(param_arr[1] - param_arr[0])/ float(param['step']-1)
            scalarData = [param_arr[0] + float(x)*stepVal for x in range(param['step'])]
            arr = []
            for i in range(len(scalarData)):
                if(val == scalarData[i]):
                    arr.append(1)
                else:
                    arr.append(0)
            return arr
    return [val]

def convert_param_to_binary(param_vectors, param_config):
    binary_arr = []
    for i in range(param_vectors.shape[0]):
        v = []
        for j in range(param_vectors.shape[1]):
            param_type = param_config['vector'][j]
            param_arr = param_onezeros(param_config['config'],param_type, param_vectors[i][j])

            v += param_arr
        v = np.array(v)
        binary_arr.append(v)
    binary_arr = np.array(binary_arr)
    return binary_arr

def param_step_mean(config,param_type, val):
    for param in config:
        if(param_type == param['name'] and param['type'] == 'scalar'):
            param_arr = param['data']
            steps = param['step']

            return float(val) / float(steps)
    return float(val) / float(2.0)

def param_dis(param_sums, param_config):
    param_sum_dis = []
    for i in range(param_sums.shape[0]):
        v = []
        for j in range(param_sums.shape[1]):
            param_type = param_config['vector'][j]
            param_val = param_step_mean(param_config['config'],param_type, param_sums[i][j])
            v.append(param_val)
        v = np.array(v)
        param_sum_dis.append(v)
    param_sum_dis = np.array(param_sum_dis)
    return param_sum_dis

def main(param_vectors, z_vectors):

    sorted_indices = get_sorted_indices(z_vectors)
    sorted_param = []

    for i in range(sorted_indices.shape[0]):
        param_vec_list = []
        for j in range(sorted_indices.shape[1]):
            param_vec = param_vectors[sorted_indices[i][j]]
            param_vec_list.append(param_vec)

        sorted_param.append(param_vec_list)

    sorted_param = np.array(sorted_param)
    pairwise_list = calculate_pairwise(sorted_param)
    param_sums = calculate_sums(pairwise_list)
    min_indices, min_values = calculate_smallest_change(param_sums)
    min_values_avg = np.mean(min_values)

    mean = np.mean(param_sums)

    param_sums = param_dis(param_sums, read_json(parameter_config)[SHAPE_TYPE])
    less_than_iqr_indices, less_than_iqr_values = calculate_change_less_than_iqr(param_sums)

    num_changes_iqr = 0
    for i in range(less_than_iqr_indices.shape[0]):
        if(len(less_than_iqr_indices[i]) > 0):
            num_changes_iqr +=1
    print('Number of interquartle dimension changes are: ', num_changes_iqr)

if __name__ == '__main__':
    start_time = time.time()
    z_vectors = np.genfromtxt(z_vectors, delimiter=',')
    param_vectors = read_h5(parameter_vectors)

    main(param_vectors, z_vectors)
    end_time = time.time()
    print("Took {} seconds to complete.".format(end_time - start_time))
