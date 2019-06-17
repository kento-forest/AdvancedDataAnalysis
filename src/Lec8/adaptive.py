import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

np.random.seed(1)  # set the random seed for reproducibility
gamma = 0.01

def data_generate(n=100):
    x = np.random.randn(n, 3)
    x[:n // 2, 0] -= 15
    x[n // 2:, 0] -= 5
    x[1:3, 0] += 10
    x[:, 2] = 1
    y = np.concatenate((np.ones(n // 2), -np.ones(n // 2)))
    index = np.random.permutation(np.arange(n))
    return x[index], y[index]


def init_params(d):
    mu = np.random.randn(d)
    A = np.random.uniform(-1, 1, (d, d))
    sigma = np.matmul(A, A.T)
    return mu.reshape((d, 1)), sigma


def update_params(mu_pre, sigma_pre, X, y):
    global gamma

    mu_deltas = []
    sigma_deltas = []
    for i in range(X.shape[0]):
        x = X[i].reshape((len(X[i]), 1))
        beta = np.matmul(x.T, np.matmul(sigma_pre, x)) + gamma
        mu_bunsi = y[i] * max(0, 1 - np.matmul(mu_pre.T,  x) * y[i]) * np.matmul(sigma_pre, x)
        sigma_bunsi = np.matmul(np.matmul(sigma_pre, x), np.matmul(x.T, sigma_pre))
        mu_deltas.append(mu_bunsi / beta)
        sigma_deltas.append(sigma_bunsi / beta)
    mu = mu_pre + sum(mu_deltas) / len(mu_deltas)
    sigma = sigma_pre - sum(sigma_deltas) / len(sigma_deltas)
    return mu, sigma


def train(X, y, batch_size):
    train_size = X.shape[0]
    batch_num = train_size // batch_size
    mu, sigma = init_params(X.shape[1])

    while True:
        X, y = shuffle(X, y)
        mu_prev = mu
        for i in range(batch_num):
            X_batched = X[i*batch_size:(i+1)*batch_size]
            y_batched = y[i*batch_size:(i+1)*batch_size]
            mu, sigma = update_params(mu, sigma, X_batched, y_batched)
        
        if train_size % batch_size != 0:
            X_batched = X[batch_num*batch_size:]
            y_batched = y[batch_num*batch_size:]
            mu, sigma = update_params(mu, sigma, X_batched, y_batched)
        
        if np.sum((mu - mu_prev)**2) < 1e-7:
            break
    return mu

def visualize(X_train, y_train, theta):
    fig = plt.figure(figsize=(8, 6), dpi=200)
    plt.xlim(-20, 0)
    plt.ylim(-3, 3)
    plt.scatter(X_train[y_train == -1, 0], X_train[y_train == -1, 1], c='r', marker='x')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], facecolors='none', edgecolors='b', marker='o', s=30)
    plt.plot([-20, 0], -(theta[2] + np.array([-20, 0]) * theta[0]) / theta[1], c='limegreen', lw=1)
    return fig

X, y = data_generate()
theta = train(X, y, batch_size=10)
fig = visualize(X, y, theta)
plt.savefig('../../output/Lec8/result.png', bbox_inches='tight')
