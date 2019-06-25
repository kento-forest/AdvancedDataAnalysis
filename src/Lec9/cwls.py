import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

np.random.seed(10)

def generate_data(n_total, n_positive):
    x = np.random.normal(size=(n_total, 2))
    x[:n_positive, 0] -= 2
    x[n_positive:, 0] += 2
    x[:, 1] *= 2
    y = np.empty(n_total, dtype=np.int64)
    y[:n_positive] = -1
    y[n_positive:] = +1
    return x, y


def cwls(train_x, train_y, test_x):
    num_positive = (train_y==-1).sum()

    App, Apm, Amm = 0, 0, 0
    for i in range(num_positive):
        App += np.sum(np.linalg.norm(train_x[i] - train_x[:num_positive], axis=1)) / (num_positive * num_positive)
        Apm += np.sum(np.linalg.norm(train_x[i] - train_x[num_positive:], axis=1)) / (num_positive * (len(train_y) - num_positive))
    for i in range(len(train_y) - num_positive):
        Amm += np.sum(np.linalg.norm(train_x[num_positive+i] - train_x[num_positive:], axis=1)) / ((len(train_y) - num_positive) * (len(train_y) - num_positive))    

    bp, bm = 0, 0
    for i in range(len(test_x)):
        bp += np.sum(np.linalg.norm(test_x[i] - train_x[:num_positive], axis=1)) / (len(test_x) * num_positive)
        bm += np.sum(np.linalg.norm(test_x[i] - train_x[num_positive:], axis=1)) / (len(test_x) * (len(train_y) - num_positive))

    
    pi_tilde = (Apm - Amm - bp + bm) / (2 * Apm - App - Amm)
    pi_hat = min(1, max(0, pi_tilde))

    print(pi_hat)

    W = np.zeros((train_x.shape[0], train_x.shape[0]))
    for i in range(len(W)):
        if train_y[i] == -1:
            W[i][i] = pi_hat
        else:
            W[i][i] = 1 - pi_hat
    W = np.eye(train_x.shape[0])
    Phi = np.ones((train_x.shape[0], 3))
    Phi[:, 1:] = train_x
    A = Phi.T.dot(W.dot(Phi))
    b = Phi.T.dot(W.dot(train_y))
    theta = np.linalg.solve(A, b)
    return theta

def visualize(train_x, train_y, test_x, test_y, theta):
    for x, y, name in [(train_x, train_y, 'train'), (test_x, test_y, 'test')]:
        plt.clf()
        plt.figure(figsize=(6, 6))
        plt.xlim(-5., 5.)
        plt.ylim(-7., 7.)
        lin = np.array([-5., 5.])
        plt.plot(lin, -(theta[0] + lin * theta[1]) / theta[2], c='limegreen')
        plt.scatter(x[y == -1][:, 0], x[y == -1][:, 1], marker='$O$', c='blue')
        plt.scatter(x[y == +1][:, 0], x[y == +1][:, 1], marker='$X$', c='red')
        plt.savefig('../../output/Lec9/result_no_weight_{}.png'.format(name), dpi=200, bbox_inches='tight')


train_x, train_y = generate_data(n_total=100, n_positive=90)
eval_x, eval_y = generate_data(n_total=100, n_positive=10)
theta = cwls(train_x, train_y, eval_x)
visualize(train_x, train_y, eval_x, eval_y, theta)
