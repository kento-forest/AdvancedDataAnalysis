import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = '../../input/Lec5/'
OUTPUT_DIR = '../../output/Lec5/'

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(10):
    label = i
    train_csv = INPUT_DIR + 'train/digit_train' + str(i) + '.csv'
    test_csv = INPUT_DIR + 'test/digit_test' + str(i) + '.csv'

    with open(train_csv, 'r') as f:
        for _ in range(500):
            line = f.readline()[:-1].split(',')
            tmp = list(map(float, line))
            x_train.append(tmp)
            y_train.append(label)

    with open(test_csv, 'r') as f:
        for _ in range(200):
            line = f.readline()[:-1].split(',')
            tmp = list(map(float, line))
            x_test.append(tmp)
            y_test.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)


def generate_train_data(data):
    x = np.concatenate([data['X'][:, :, 0], data['X'][:, :, 1]], axis=1)
    x = np.transpose(x, (1, 0))
    y = np.concatenate([np.ones(500), -np.ones(500)])
    return x, y


def generate_test_data(data):
    x = data['T'][:, :, 0:2]
    return x


def build_design_mat(x1, x2, bandwidth):
    return np.exp(
        -np.sum((x1[:, None] - x2[None]) ** 2, axis=-1) / (2 * bandwidth ** 2))


def optimize_param(design_mat, y, regularizer):
    print(y)
    return np.linalg.solve(
        design_mat.T.dot(design_mat) + regularizer * np.identity(len(y)),
        design_mat.T.dot(y))


def predict(train_data, test_data, theta):
    return build_design_mat(train_data, test_data, 10.).T.dot(theta)


def build_confusion_matrix(train_data, data, theta):
    confusion_matrix = np.zeros((2, 2), dtype=np.int64)
    for i in range(2):
        test_data = np.transpose(data[:, :, i], (1, 0))
        prediction = predict(train_data, test_data, theta)
        confusion_matrix[i][0] = np.sum(
            np.where(prediction > 0, 1, 0))
        confusion_matrix[i][1] = np.sum(
            np.where(prediction < 0, 1, 0))
    return confusion_matrix


data = loadmat('digit.mat')
x, y = generate_train_data(data)
design_mat = build_design_mat(x, x, 10.)
theta = optimize_param(design_mat, y, 1.)

confusion_matrix = build_confusion_matrix(x, data['T'], theta)
print('confusion matrix:')
print(confusion_matrix)
