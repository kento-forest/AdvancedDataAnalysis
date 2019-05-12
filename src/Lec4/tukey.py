import numpy as np
import matplotlib.pyplot as plt

np.random.seed(15)

def generate_sample(x_min=-3., x_max=3., sample_size=10):
    x = np.linspace(x_min, x_max, num=sample_size)
    y = x + np.random.normal(loc=0., scale=.2, size=sample_size)
    y[-1] = -4
    return x, y


def cal_phi(x_list):
    phi = np.array([[1, x_list[i]] for i in range(len(x_list))])
    return phi


def cal_W(err, eta=1):
    calw = lambda x: (1 - x**2 / eta**2)**2 if x <= eta else 0

    W = np.zeros((err.shape[0], err.shape[0]))
    w = list(map(calw, err))
    for i in range(err.shape[0]):
        W[i][i] = w[i]
    return W

def train(x_list, y_list, eta=1, max_iter=1000):
    phi = cal_phi(x_list)
    
    theta = theta_pre = theta_init = np.linalg.solve(
        phi.T.dot(phi) + 1e-4 * np.identity(phi.shape[1]), phi.T.dot(y_list))
    
    for _ in range(max_iter):
        err = np.abs(phi.dot(theta_pre) - y_list)
        W = cal_W(err, eta)
        phit_w_phi = phi.T.dot(W).dot(phi)
        phit_w_y = phi.T.dot(W).dot(y_list)
        theta = np.linalg.solve(phit_w_phi, phit_w_y)
        if np.linalg.norm(theta - theta_pre) < 1e-4:
            break
        theta_pre = theta
    return theta, theta_init


def predict(x_list, theta):
    phi = cal_phi(x_list)
    y_predicted = phi.dot(theta)
    return y_predicted


def plot_data(x_true, y_true, x_test, y_predicted, y_predicted2):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    plt.scatter(x_true, y_true, marker='o', s=10)
    plt.plot(x_test, y_predicted, c='r', lw=1, label='Robust')
    plt.plot(x_test, y_predicted2, c='green', lw=1, linestyle='--', label='l2norm')
    plt.legend(fontsize=15)
    return fig


x, y = generate_sample()
theta, theta_init = train(x, y, eta=1, max_iter=1000)

x_test = np.arange(-3, 3, 0.001)
# robust
y_predicted = predict(x_test, theta)

# not robust
y_predicted2 = predict(x_test, theta_init)

fig = plot_data(x, y, x_test, y_predicted, y_predicted2)
plt.savefig('../../output/Lec4/result.png', bbox_inches='tight')
