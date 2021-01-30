#!/usr/bin/env python

import numpy
from scipy.io import loadmat

if __name__ == "__main__":
    face_data = loadmat('FaceData.mat')
    print(face_data['FaceData'].shape)
    print(numpy.array([[1], [2], [3]]).shape)
