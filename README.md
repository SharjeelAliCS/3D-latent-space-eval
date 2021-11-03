# 3D-Latent-Space-Evaluation

As part of the URSA (Undergraduate Student Research Award), funded by the NSERC, we design a 3D-procedural modeling library that can generate various types of furniture shapes, such as Chairs, Beds, Tables and Shelves.

For this paper (url ommitted), we develop a novel evaulation of the latent space of 3D voxelized data, generated from various network architecture models.



## Updates

November 3, 2021: Updated the `Chair` datasets for both kaggle and here. Note: The new ones are named `Chair_updated`. If using the kaggle dataset, **make sure to rename the `chair_updated` to `chair` in order to be compatiable with the autoencoder models. **
## Datasets

The entirety of the dataset (both checkpoints and inputs) are found on the kaggle page here: https://www.kaggle.com/sharjeelalics/3d-latent-space-evaluation

This repository contains all the datasets used for training. It is seperated into the following:

- Parameter vectors (.h5 format): Contains the parameterized vectors used to procedurally generate the shapes.
- Parameter configs (.json format): Contains the configuration settings used to procedurally generate the shapes, and run the experiments.
- Procedually generated meshes (.obj format)
- Train data (.tar format): The input voxelized data in ztar format (numpy) that is used for training the network models.

## Network models

We evaulate on two specific network architectures, which we then modify for our experiments and various datasets. Their implementations are listed below:
- [3D-GAE](https://github.com/IsaacGuan/3D-GAE)
- [3D-VAE](https://github.com/IsaacGuan/3D-VAE)
- 3D-AE: We modify the [3D-GAE](https://github.com/IsaacGuan/3D-GAE) and replace the loss function with tensorflow's default `mean_squared_error` loss.
- 3D-PGAE: We modify the [3D-GAE](https://github.com/IsaacGuan/3D-GAE) and use the Euclidean distances instead of the Chamfur distances for calculating the Nearest Neighbours. We also compute these Euclidean distances on the parameter vector data:

```
def read_h5(filename):
    with h5py.File(filename, "r") as f:
        np_arr = np.array(f['SHAPE_TYPE, ONE OF CHAIR/BED/TABLE/SHELF'])
        np_arr = np_arr[0, :, :]
        np_arr = np.swapaxes(np_arr,0,1)
        return np_arr

param_vectors = read_h5('param_vectors_file')
euclidean_distances = distance_matrix(param_vectors, param_vectors)
euclidean_distances = (euclidean_distances - euclidean_distances.min()) / (euclidean_distances.max() - euclidean_distances.min())
```

## Evalation methods

This page also includes three files, each of which handles our experiments that we list in our paper above.

