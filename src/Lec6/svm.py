import numpy as np
import matplotlib.pyplot as plt
np.random.seed(16)


def generate_data(sample_size):
    x = np.random.normal(size=(sample_size, 3))
    x[:, 2] = 1.
    x[:sample_size // 2, 0] -= 5.
    x[sample_size // 2:, 0] += 5.
    y = np.concatenate([np.ones(sample_size // 2, dtype=np.int64),
                        -np.ones(sample_size // 2, dtype=np.int64)])
    x[:3, 1] -= 5.
    y[:3] = -1
    x[-3:, 1] += 5.
    y[-3:] = 1
    return x, y


def svm(x, y, l, lr):
    w = np.zeros(3)
    prev_w = w.copy()
    grad = np.zeros(3)
    R = np.dot(x[:, :2].T, x[:, :2])
    for i in range(10000):
        f_x = np.dot(x, w)
        tmp = 1 - f_x * y
        for i in range(tmp.shape[0]):
            if tmp[i] <= 0:
                grad += np.zeros(3)
            else:
                grad -= y[i] * x[i]
        w -= lr * (grad + np.append(l * np.dot(R, w[:2]), 0))
        if np.linalg.norm(w - prev_w) < 1e-3:
            break
        prev_w = w.copy()
    return w


def visualize(x, y, w):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.scatter(x[y == 1, 0], x[y == 1, 1], c='b', marker='x')
    plt.scatter(x[y == -1, 0], x[y == -1, 1], facecolors='none',
                edgecolors='r', marker='o', s=30)
    plt.plot([-10, 10], -(w[2] + np.array([-10, 10]) * w[0]) / w[1], c='limegreen', lw=1)
    return fig


x, y = generate_data(200)
w = svm(x, y, l=0.03, lr=0.01)
fig = visualize(x, y, w)
plt.savefig('../../output/Lec6/result.png', bbox_inches='tight')
