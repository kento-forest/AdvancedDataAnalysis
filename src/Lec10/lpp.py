import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
np.random.seed(0)

def data_generation1(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, np.random.randn(n, 1)],
                          axis=1)


def data_generation2(n=100):
    return np.concatenate([np.random.randn(n, 1) * 2, 2 * np.round(
        np.random.rand(n, 1)) - 1 + np.random.randn(n, 1) / 3.], axis=1)



class LPP:
    def __init__(self):
        self.L = None
        self.D = None
        self.xi = None
    

    def fit(self, x):
        x = x - np.mean(x, axis=0)
        self.L, self.D = self.get_L_D_Matrix(x)
        A = np.dot(x.T, np.dot(self.L, x))
        B = np.dot(x.T, np.dot(self.D, x))
        sqrt_B = linalg.sqrtm(B)
        sqrt_B_inv = np.linalg.inv(sqrt_B)
        BAB = np.dot(sqrt_B_inv, np.dot(A, sqrt_B_inv))
        _, phi = np.linalg.eig(BAB)
        self.xi = np.dot(sqrt_B_inv, phi)
    

    def get_L_D_Matrix(self, x):
        W = np.exp(-np.sum((x[:, None] - x[None]) ** 2, axis=-1))
        W_sum = np.sum(W, axis=0)
        D = np.diag(W_sum)
        L = D - W
        return L, D
    

    def get_tlpp(self, n_components):
        return self.xi[-n_components:][::-1]


def visualize(x, tlpp):
    fig = plt.figure(figsize=(6, 6))
    plt.xlim((-5, 5))
    plt.ylim((-6, 6))
    plt.scatter(x[:, 0], x[:, 1], c='r', marker='x', s=30)
    plt.plot([-5, 5], [-5 / tlpp[0] * tlpp[1], 5 / tlpp[0] * tlpp[1]], c='limegreen')
    plt.savefig('../../output/Lec10/result.png')



n = 100
n_components = 1
# x = data_generation1(n)
x = data_generation2(n)

lpp = LPP()
lpp.fit(x)
tlpp = lpp.get_tlpp(n_components)
visualize(x, tlpp.flatten())
