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
    v = np.empty(eigenvalues.shape)
    v_m = np.sum(eigenvalues)
    for m in range(eigenvalues.shape[0]):
        v[m] = np.sum(eigenvalues[:m]) / v_m

    plt.plot(v)
    plt.show()

    # Display mean face and first 10 eigenfaces
    display_eigenface(eigenvalues, eigenvectors, mean, 10)

    # Compute score matrix
    m = 10
    phi_m = pca.select_components(eigenvalues, eigenvectors, m)
    phi_m.transpose()
    a = np.empty(m)
    for i in range(m):
        a[i] = phi_m * (Xte[i] - mean)  # TODO: fix shape incompatibility
    print(a)

    # Step 5
    for i in range(10, 110, 10):
        print('m={0}'.format(i))
        display_eigenface(eigenvalues, eigenvectors, mean, i)

