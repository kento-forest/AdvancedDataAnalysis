import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.sparse.linalg import eigs

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

np.random.seed(1)


def data_generation(n=1000):
    a = 3 * np.pi * np.random.rand(n)
    x = np.stack(
        [a * np.cos(a), 30 * np.random.random(n), a * np.sin(a)], axis=1)
    return a, x


def LapEig(x, d=2):
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(x)
    _, indices = nn.kneighbors(x)
    size = x.shape[0]
    W = np.zeros((size, size))
    for i in range(size):
        W[i][indices[i]] = [1] * len(indices[0])
        W[indices[i], i] = [1] * len(indices[0])
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eig_val, eig_vec = eigs(L, k=3, M=D, sigma=0)
    index = np.argsort(eig_val)
    eig_vec = eig_vec[:, index]
    return eig_vec[:, 1:d+1]

def visualize(x, z, a):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=a, marker='o')
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(z[:, 0], z[:, 1], c=a, marker='o')
    plt.savefig('../../output/Lec12/result.png', dpi=200, bbox_inches='tight')

n = 1000
a, x = data_generation(n)
x = x - np.mean(x, axis=0)
z = LapEig(x)
visualize(x, z, a)
