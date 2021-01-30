import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import assignment_2.pca as pca


def imagesc(data):
    plt.imshow(data)
    plt.show()


def display_eigenface(eigenvalues, eigenvectors, mean, m):
    phi = pca.select_components(eigenvalues, eigenvectors, m)
    for i in range(10):
        eigenface = np.reshape(phi[:, i], (56, 46))
        imagesc(eigenface.real)
    eigenface = np.reshape(mean, (56, 46))
    imagesc(eigenface.real)
    pass


if __name__ == "__main__":
    # Load data
    face_data = loadmat('FaceData.mat')

    # Display image
    # imagesc(face_data['FaceData'][0-39][0-9][0])

    # Split database into training and testing images
    vectorized_images = np.array([pca.vectorize(i) for i in pca.list_data_images(face_data)])
    Xtr = vectorized_images[:len(vectorized_images) // 2]
    Xte = vectorized_images[len(vectorized_images) // 2:]

    # Create identity labels for the columns of Xte
    id = [i for i in range(len(Xte))]

    # Compute components and eigenvalues from training set
    eigenvalues, eigenvectors, mean = pca.train(Xtr)
    components = pca.select_components(eigenvalues, eigenvectors, eigenvalues.shape[0])

    # Plot v(m)
    # TODO: plot v(m)

    # Display mean face and first 10 eigenfaces
    display_eigenface(eigenvalues, eigenvectors, mean, 10)

    # Compute score matrix
    # TODO: step 4

    # TODO: step 5

