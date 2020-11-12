from scipy.io import loadmat
import numpy as np


face_data = loadmat('FaceData.mat')
print(face_data['FaceData'].shape)
print(np.array([[1], [2], [3]]).shape)
