#!/usr/bin/env python

##### Data Format
#
# face_data
# |
# - ['FaceData'] (face collection)
#   |
#   - [0] (face image collection)
#   | |
#   | - [0] (face image)
#   | | |
#   | | - [0] (image)
#   | .
#   | |
#   | - [9] (face image)
#   |   |
#   |   - [0] (image)
#   .
#   |
#   - [39] (face image collection)
#     |
#     - [0] (face image)
#     | |
#     | - [0] (image)
#     .
#     |
#     - [9] (face image)
#       |
#       - [0] (image)
#
# Get an individual face: face_data['FaceData'][0-39][0-9][0]
# show a face:
# pyplot.imshow(face_data['FaceData'][0-39][0-9][0])
# pyplot.show()
#
# 40 face image collections
# 10 images per face image collection
# 56x46 pixels per face

from matplotlib import pyplot as plt
import numpy
from scipy.io import loadmat


def list_collection_images(collection):
    return [collection[i][0] for i in range(collection.shape[0])]


def list_data_images(data):
    data = data['FaceData']
    images = []

    for i in range(data.shape[0]):
        images += list_collection_images(data[i])

    return images


def select_components(eigen_values, eigen_vectors, m):
    if m < 1 or m > eigen_values.shape[0]:
        m = eigen_values.shape[0]

    ind = numpy.argpartition(eigen_values, -m)[-m:]
    ind = ind[numpy.argsort(eigen_values[ind])]
    return eigen_vectors[:, ind]


def train(X):
    mean = numpy.mean(X, axis=0)
    X_0 = X - mean
    cov = 1 / (X.shape[0] - 1) * numpy.transpose(X_0) @ X_0
    eig_val, eig_vec = numpy.linalg.eig(cov)
    return eig_val, eig_vec, mean


def vectorize(image):
    return image.flatten(order='F')


if __name__ == "__main__":
    face_data = loadmat('FaceData.mat')
    # plt.imshow(face_data['FaceData'][0-39][0-9][0])
    # plt.show()
    m = 3
    vectorized_images = numpy.array([vectorize(i) for i in list_data_images(face_data)])
    eigen_values, eigen_vectors, mean = train(vectorized_images)
    components = select_components(eigen_values, eigen_vectors, m)
