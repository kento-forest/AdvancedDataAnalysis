import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)

def generate_data(sample_size=90, n_class=3):
    x = (np.random.normal(size=(sample_size // n_class, n_class))
         + np.linspace(-3., 3., n_class)).flatten()
    y = np.broadcast_to(np.arange(n_class),
                        (sample_size // n_class, n_class)).flatten()
    return x, y


def cal_K(x, c, h):
    return np.exp(-(x-c[:, None]) ** 2 / (2 * h ** 2))


def cal_pi(y):
    global n_class
    pi = np.zeros((y.shape[0], n_class))
    for i in range(y.shape[0]):
        pi[i, y[i]] = 1
    return pi


def train(x, y, l, h):
    global n_class

    K = cal_K(x, x, h)
    pi = cal_pi(y)
    theta = np.linalg.solve((K.T.dot(K) + l * np.identity(len(K))), K.T.dot(pi))
    return theta


def visualize(x, y, theta, h):
    X = np.linspace(-5, 5, num=10000)
    K = np.exp(-(x - X[:, None]) ** 2 / (2 * h ** 2))
    
    _ = plt.figure(figsize=(6, 4), dpi=200)
    plt.xlim(-5, 5)
    plt.ylim(-0.3, 1.5)
    unnormalized_prob = K.dot(theta)
    for i in range(len(unnormalized_prob)):
        unnormalized_prob[i] = np.array(list(map(lambda x: x if x >= 0 else 0, unnormalized_prob[i])))
    prob = unnormalized_prob / np.sum(unnormalized_prob, axis=1, keepdims=1)

    plt.plot(X, prob[:, 0], c='blue', lw=1)
    plt.plot(X, prob[:, 1], c='red', lw=1)
    plt.plot(X, prob[:, 2], c='green', lw=1)

    plt.scatter(x[y == 0], -0.1 * np.ones(len(x) // 3), c='blue', marker='o')
    plt.scatter(x[y == 1], -0.2 * np.ones(len(x) // 3), c='red', marker='x')
    plt.scatter(x[y == 2], -0.1 * np.ones(len(x) // 3), c='green', marker='v')
    plt.savefig('../../output/Lec7/result.png', bbox_inches='tight')


def train_predict(x, y, l, h):
    theta = train(x, y, l, h)
    visualize(x, y, theta, h)


n_class = 3
x, y = generate_data(sample_size=120, n_class=n_class)

train_predict(x, y, h=1, l=0.2)
