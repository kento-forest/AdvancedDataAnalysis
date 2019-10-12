import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(1)


def generate_data(n=200):
    x = np.linspace(0, np.pi, n // 2)
    u = np.stack([np.cos(x) + .5, -np.sin(x)], axis=1) * 10.
    u += np.random.normal(size=u.shape)
    v = np.stack([np.cos(x) - .5, np.sin(x)], axis=1) * 10.
    v += np.random.normal(size=v.shape)
    x = np.concatenate([u, v], axis=0)
    y = np.zeros(n)
    y[0] = 1
    y[-1] = -1
    return x, y


def get_KMatrix(x, c, h):
    K = np.zeros((x.shape[0], c.shape[0]))
    for i in range(c.shape[0]):
        tmp = x - c[i]
        K[:, i] = np.exp(- np.sum(tmp**2, axis=1) / (2 * h ** 2))
    return K


def lrls(x, y, h=1., l=1., nu=1.):
    """
    :param x: data points
    :param y: labels of data points
    :param h: width parameter of the Gaussian kernel
    :param l: weight decay
    :param nu: Laplace regularization
    :return:
    """

    Phi = get_KMatrix(x, x, h)
    Phi_tilde = get_KMatrix(x[[0, -1]], x, h)
    W = Phi
    tmp = np.sum(W, axis=1)
    D = np.eye(tmp.shape[0]) * tmp
    L = D - W
    A = Phi_tilde.T.dot(Phi_tilde) + l * np.eye(Phi_tilde.shape[1]) + 2 * nu * Phi.T.dot(L.dot(Phi))
    b = Phi_tilde.T.dot(y[[0, -1]])
    theta_hat = np.linalg.solve(A, b)
    return theta_hat


def visualize(x, y, theta, h=1.):
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.xlim(-20., 20.)
    plt.ylim(-20., 20.)
    grid_size = 1000
    grid = np.linspace(-20., 20., grid_size)
    X, Y = np.meshgrid(grid, grid)
    mesh_grid = np.stack([np.ravel(X), np.ravel(Y)], axis=1)
    k = np.exp(-np.sum((x.astype(np.float32)[:, None] - mesh_grid.astype(
        np.float32)[None]) ** 2, axis=2).astype(np.float64) / (2 * h ** 2))
    plt.contourf(X, Y, np.reshape(np.sign(k.T.dot(theta)),
                                  (grid_size, grid_size)),
                 alpha=.4, cmap=plt.cm.coolwarm)
    plt.scatter(x[y == 0][:, 0], x[y == 0][:, 1], marker='$.$', c='black')
    plt.scatter(x[y == 1][:, 0], x[y == 1][:, 1], marker='$X$', c='red')
    plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='$O$', c='blue')
    plt.savefig('../../output/Lec9/result.png', dpi=200, bbox_inches='tight')


x, y = generate_data(n=200)
theta = lrls(x, y, h=1.)
visualize(x, y, theta)

