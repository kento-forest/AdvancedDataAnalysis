import numpy as np
import matplotlib.pyplot as plt

lambdalist = [0.001, 0.01, 0.1, 1]
hlist = [0.1, 1, 10]

# set seed
np.random.seed(16)

p = 1e-5

# sample data 
def generate_sample(xmin, xmax, sample_size):
    x = (xmax - xmin)*np.random.rand(sample_size) + xmin
    target = np.sin(np.pi * x) / (np.pi * x) + 0.1 * x
    noise = 0.1 * np.random.randn(sample_size)
    return x, target + noise

def cal_Kernel(x, c, h):
    return np.exp(-(x-c.reshape(c.shape[0], 1)) ** 2 / (2 * h ** 2))

def update_one_step(x, y, z, u, h, l):
    K = cal_Kernel(x, x, h)
    theta_next = np.linalg.solve(K.T.dot(K) + np.identity(len(K)), K.T.dot(y.reshape(y.shape[0], 1)) + z.reshape(z.shape[0], 1) - u.reshape(u.shape[0], 1)).flatten()
    z_next = np.array([max(0, theta_next[i] + u[i] - l) + min(0, theta_next[i] + u[i] + l) for i in range(len(z))])
    u_next = u + theta_next - z_next
    return theta_next, z_next, u_next

def train(x, y, h, l, p):
    z = np.random.rand(y.shape[0])
    u = np.random.rand(y.shape[0])
    theta = np.random.rand(y.shape[0])

    for i in range(100):
        theta, z, u = update_one_step(x, y, z, u, h, l)
        if np.linalg.norm(theta - z) < p:
            break
    return theta

def predict(x, X, h, l, theta):
    K = cal_Kernel(x, X, h)
    y_predicted = K.dot(theta)
    return y_predicted

def cross_validation(x_all, y_all, h, l, splited_size):
    global p
    num = int(len(x_all) / splited_size)
    errorlist = []
    # calc error with each data
    for k in range(splited_size):
        x_train, y_train = np.append(x_all[:num*k], x_all[num*(k+1):]), np.append(y_all[:num*k], y_all[num*(k+1):])
        x_test, y_test = x_all[num*k:num*(k+1)], y_all[num*k:num*(k+1)]
        theta = train(x_train, y_train, h, l, p)
        y_predicted = predict(x_train, x_test, h, l, theta)
        errorlist.append((np.linalg.norm(y_predicted - y_test))**2)
    return sum(errorlist)/len(errorlist)


def plot_func(x_all, y_all, xmin, xmax, h, l, error):
    global p
    # plot用のデータ作成
    x_true = np.arange(xmin, xmax, 0.01)
    y_true = np.sin(np.pi * x_true) / (np.pi * x_true) + 0.1 * x_true
    theta = train(x_all, y_all, h, l, p)
    y_pre = predict(x_all, x_true, h, l, theta)
    # plot
    plt.title('$\lambda$ = {}, h = {}, error = {:.3f}'.format(l, h, error))
    plt.plot(x_true, y_true, c='r', lw=1.5)
    plt.plot(x_true, y_pre, c='limegreen', lw=1.5)
    plt.scatter(x_all, y_all, s=5)
    
# create sample data
sample_size = 50
xmin, xmax = -3, 3
x_all, y_all = generate_sample(xmin=xmin, xmax=xmax, sample_size=sample_size)

error_list = [[] for _ in range(len(hlist))]

fig = plt.figure(figsize=(16, 10), dpi=200)
plt.rcParams["font.size"] = 10
for i in range(len(hlist)):
    for j in range(len(lambdalist)):
        err = cross_validation(x_all, y_all, hlist[i], lambdalist[j], 10)
        error_list[i].append(err)
        plt.subplot(len(hlist), len(lambdalist), int(i*len(lambdalist) + j + 1))
        plot_func(x_all, y_all, xmin, xmax, hlist[i], lambdalist[j], err)

plt.savefig('../../output/Lec3/result.png', bbox_inches='tight')

# serach best h, l
minlist = np.array([min(error_list[i]) for i in range(len(error_list))])
hidx = np.argmin(minlist)
lidx = np.argmin(np.array(error_list[hidx]))

print('h = {}, l = {}'.format(hlist[hidx], lambdalist[lidx]))
h, l = hlist[hidx], lambdalist[lidx]

# train with this h l
theta = train(x_all, y_all, h, l, p)

# visualize theta
param_idx = [i for i in range(len(theta))]
fig = plt.figure(figsize=(10, 8), dpi=100)
plt.scatter(param_idx, theta, marker='.', s=50)
plt.plot([param_idx[0], param_idx[-1]], [0, 0], c='r', lw=1)
plt.savefig('../../output/Lec3/params.png', bbox_inches='tight')
