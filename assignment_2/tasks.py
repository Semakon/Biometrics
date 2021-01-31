import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import assignment_2.pca as pca
import assignment_1.assignment_one as as1


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


def d_E(a, b):
    return np.sqrt((a - b) ** 2)


if __name__ == "__main__":
    # Load data
    face_data = loadmat('FaceData.mat')

    # Display image
    # imagesc(face_data['FaceData'][0-39][0-9][0])

    # Split database into training and testing images
    vectorized_images = np.array([pca.vectorize(i) for i in pca.list_data_images(face_data)])
    Xtr = vectorized_images[:len(vectorized_images) // 2].astype(np.float) / 255
    Xte = vectorized_images[len(vectorized_images) // 2:].astype(np.float) / 255

    # Create identity labels for the columns of Xte
    id = np.empty(len(Xte))
    for i in range(10):
        for j in range(20):
            id[i * 20 + j] = i

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
    for m in [10, 20, 30, 40, 50, 100]:
        phi_m = pca.select_components(eigenvalues, eigenvectors, m).transpose()
        a = np.zeros((len(id), m))
        for i in range(len(id)):
            a[i] = (phi_m.dot(Xte[i] - mean)).real

        # Construct dissimilarity score matrix from coefficients
        s = np.zeros((len(a), len(a)))
        for i in range(len(a)):
            for j in range(len(a)):
                s[i, j] = np.sum(d_E(a[i], a[j]))

        gen, imp, thresholds, fmr_t, fnmr_t = as1.calc_fmr_fnmr(id, s)

        # Calculate EER
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnmr_t - fmr_t)))]
        eer = fmr_t[np.nanargmin(np.absolute((fnmr_t - fmr_t)))]
        print('EER with m={0} is {1}'.format(m, eer))

        # Plot ROC
        plt.plot(fmr_t, 1 - fnmr_t)
        plt.xlabel('FMR(t)')
        plt.ylabel('TMR(t)')
        plt.title('ROC with m={0} and EER={1}'.format(m, eer))
        plt.show()
