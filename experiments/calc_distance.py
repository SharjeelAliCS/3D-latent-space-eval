import re
import os
import sys
import time
import numpy as np
import h5py
import numpy as np

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

# One of 'Chair', 'Bed', 'Table', 'Shelf'
SHAPE_TYPE = 'Chair'
OVERWRITE_FILE = True
FILTER = True
FILTER_NUM = 100

# ---------- INPUT FILES ----------

# The z vector file is the latent space generated from the training model:
z_vectors = 'z_vectors.csv'
parameter_vectors = '../datasets/parameter_vectors/parameter_vectors_' + SHAPE_TYPE.lower() + '.h5'

# ---------- OUTPUT FILES ----------
OUTPUT_FILE_DIR = 'distances/'
# These output the euclidean distance matricies for the following datasets:
z_distances_file = OUTPUT_FILE_DIR+'z_distances.npy'
param_distances_file = OUTPUT_FILE_DIR+'param_distances.npy'

# This output an h5 file containing all the datasets.
correlation_file = OUTPUT_FILE_DIR+'correlations.npy'

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

def calculate_distances(x):
    euclidean = distance_matrix(x,x)
    return euclidean

def calculate_smallest_pairs(x, n):
    dis_arr = []
    for (j, k), _ in np.ndenumerate(x):
        if(j != k):
            dis_arr.append([j, k, x[j, k]])

    dis_arr = np.array(dis_arr)
    dis_arr = dis_arr[dis_arr[:, 2].argsort()]
    dis_arr = dis_arr[:n]
    if(FILTER):
        temp_matrix = np.zeros(x.shape)
        j = dis_arr[:,0].astype(int)
        k = dis_arr[:,1].astype(int)
        temp_matrix[j, k] = 1

        return temp_matrix

    return np.ones(x.shape)
def element_mean(j, k, x, col_means, row_means, total_mean):
    #print(col_means[k])
    e = x[j][k]
    e -= row_means[j]
    e -= col_means[k]
    e += total_mean
    return e

def calculate_mean(x, axis, pair_indices):
    means = []
    if(axis == -1):
        sums = 0
        c = 0
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                if(pair_indices[j][k] == 1):
                    c += 1
                    sums += x[j][k]
        return float(sums) / float(c)

    shape1 = 1
    shape2 = 0
    if(axis == 1):
        shape1 = 0
        shape2 = 1
    for j in range(x.shape[shape1]):
        sums = 0
        c = 0
        for k in range(x.shape[shape2]):
            if(pair_indices[j][k] == 1):
                c += 1
                sums += x[j][k]
        if(c == 0):
            means.append(0)
        else:
            means.append(float(sums) / float(c))
    return np.array(means)

def calculate_covariance(x, y, x_means, y_means, pair_indices):
    products = []

    for (j, k), _ in np.ndenumerate(x):
        if(pair_indices[j][k] == 1):
            x_dd = element_mean(j, k, x, x_means['col'], x_means['row'], x_means['total'])
            y_dd = element_mean(j, k, y, y_means['col'], y_means['row'], y_means['total'])

            products.append(x_dd * y_dd)

    products = np.array(products)
    sums = np.sum(products)

    n2 = np.count_nonzero(pair_indices)
    average = sums / n2
    return np.sqrt(average)

def calculate_correlation(covariance, x, y, x_means, y_means, pair_indices):
    std_x = calculate_covariance(x,x, x_means, x_means, pair_indices)
    std_y = calculate_covariance(y,y, y_means, y_means, pair_indices)
    correlation = covariance / np.sqrt(std_x*std_y)
    return correlation

def calculate_all_means(x, pair_indices):
    means = {}
    means['total'] = np.mean(x)
    means['col'] = np.mean(x, axis=1)
    means['row'] =np.mean(x, axis=0)
    return means

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FILE_DIR):
        os.makedirs(OUTPUT_FILE_DIR)

    z_vectors = np.genfromtxt(z_vectors, delimiter=',')
    param_vectors = read_h5(parameter_vectors)

    print("Vectors: Z - {}. Parameters - {}".format(z_vectors.shape, param_vectors.shape))

    z_distances = 0
    param_distances = 0

    if not os.path.exists(z_distances_file) or OVERWRITE_FILE:
        z_distances = calculate_distances(z_vectors)
        output_data(z_distances_file, z_distances)
    else:
        z_distances = np.load(z_distances_file)

    if not os.path.exists(param_distances_file) or OVERWRITE_FILE:
        param_distances = calculate_distances(param_vectors)
        output_data(param_distances_file, param_distances)
    else:
        param_distances = np.load(param_distances_file)

    print("Distances: Z - {}. Parameters - {}".format(z_distances.shape, param_distances.shape))

    filters = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]

    filter_n = 1

    if (FILTER):
        filter_n = FILTER_NUM

    corel = []

    for i in filters:
        filter_n = i
        print("filter size: ", filter_n)
        pair_indices = calculate_smallest_pairs(param_distances, filter_n)

        print("pair_indices is {}".format(pair_indices.shape))

        param_means = calculate_all_means(param_distances, pair_indices)
        z_means = calculate_all_means(z_distances, pair_indices)
        print("calculated means")

        covariance = calculate_covariance(z_distances, param_distances, z_means, param_means, pair_indices)
        print("covariance is {}".format(covariance))

        correlation = calculate_correlation(covariance, z_distances, param_distances, z_means, param_means, pair_indices)
        print("correlation is {}".format(correlation))

        corel.append(correlation)

    # Output the data into an h5 file:
    output_data(correlation_file, np.array(corel))
