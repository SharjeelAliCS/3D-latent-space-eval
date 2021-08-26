# Code taken from the 3D-GAE found here: https://github.com/IsaacGuan/3D-GAE
import re
import os
import sys
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from GAE import *
from numpy import genfromtxt
from utils import npytar, save_volume, paths, encoder_predict

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def data_loader(fname):
    x_dic = {}
    reader = npytar.NpyTarReader(fname)
    for ix, (x, name) in enumerate(reader):
        x_dic[name] = x.astype(np.float32)
    reader.reopen()
    xc = np.zeros((reader.length(), ) + input_shape, dtype = np.float32)
    i = 0
    for ik in sorted(x_dic.keys(), key = natural_keys):
        xc[i] = x_dic[ik]
        i += 1
    return xc

def get_encoded_data(encoder, gae):

    gae.load_weights(paths.ENCODER_SHAPE_PATH+'results/gae.h5')
    print("loaded weights from {}".format(paths.ENCODER_SHAPE_PATH+'results/gae.h5'))
    gae.trainable = False

    data_test = data_loader(paths.TAR_PATH)

    if not os.path.exists(paths.Z_FILE_PATH) or True:
        z_vectors = encoder.predict(data_test)
        np.savetxt(paths.Z_FILE_PATH, z_vectors, delimiter = ',')
        return (z_vectors, data_test)
    else:
        z_vectors = genfromtxt(paths.Z_FILE_PATH, delimiter=',')
        return (z_vectors, data_test)

