import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

np.random.seed(0)

def generate_data(sample_size=100, pattern='two_cluster'):
    if pattern not in ['two_cluster', 'three_cluster']:
        raise ValueError('Dataset pattern must be one of [two_cluster, three_cluster].')
    x = np.random.normal(size=(sample_size, 2))
    if pattern == 'two_cluster':
        x[:sample_size // 2, 0] -= 4
        x[sample_size // 2:, 0] += 4
    else:
        x[:sample_size // 4, 0] -= 4
        x[sample_size // 4:sample_size // 2, 0] += 4
    y = np.ones(sample_size, dtype=np.int64)
    y[sample_size // 2:] = 2
    return x, y


def get_scatter_matrix(x, y):
    # cal C
    xxT = np.array([np.dot(x[i].reshape(len(x[i]), 1), x[i].reshape(1, len(x[i]))) for i in range(len(x))])
    C = np.sum(xxT, axis=0)

    # cal S_b, S_w
    dim = x.shape[1]
    S_b = np.zeros((dim, dim))
    S_w = np.zeros((dim, dim))
    for c in np.unique(y):
        ny = np.count_nonzero(y==c)
        mean = np.mean(x[np.where(y==c)], axis=0).reshape((dim, 1))
        S_b += ny * np.dot(mean, mean.T)

        for x_ in x[np.where(y==c)]:
            x_ = x_.reshape((dim, 1))
            S_w += np.dot((x_ - mean), (x_ - mean).T)
    return C, S_b, S_w


def fda(x, y):
    _, S_b, S_w = get_scatter_matrix(x, y)
    sqrt_S_w_inv = sqrtm(np.linalg.inv(S_w))
    w_sq_b_w_sq = np.dot(sqrt_S_w_inv, np.dot(S_b, sqrt_S_w_inv))
    _, phi = np.linalg.eig(w_sq_b_w_sq)
    xi = np.dot(sqrt_S_w_inv, phi)
    # print(xi[:, 0] @ S_w @ xi[:, 0])
    # print(xi[:, 0] @ S_w @ xi[:, 1])
    return xi


def visualize(x, y, T):
    plt.figure(1, (6, 6))
    plt.clf()
    plt.xlim(-7., 7.)
    plt.ylim(-7., 7.)
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo', label='class-1')
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'rx', label='class-2')
    plt.plot(np.array([-T[:, 0], T[:, 0]]) * 10000,
             np.array([-T[:, 1], T[:, 1]]) * 10000, 'k-')
    plt.legend()
    plt.savefig('../../output/Lec11/result_three.png', dpi=200, bbox_inches='tight')


sample_size = 100
x, y = generate_data(sample_size=sample_size, pattern='two_cluster')
x, y = generate_data(sample_size=sample_size, pattern='three_cluster')
print(x[0])
x = x - np.mean(x, axis=0)

C, S_w, S_b = get_scatter_matrix(x, y)
T = fda(x, y)
visualize(x, y, T[0].reshape(1, 2))
