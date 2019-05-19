import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

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


def get_KMatrix(x, c, h):
    K = np.zeros((x.shape[0], c.shape[0]))
    for i in range(c.shape[0]):
        tmp = x - c[i]
        K[:, i] = np.exp(- np.sum(tmp**2, axis=1) / (2 * h ** 2))
    #####################   this caused memory error   ########################
    # K = np.exp(-np.sum((x[:, None] - c[None]) ** 2, axis=-1) / (2 * h ** 2)) #
    ############################################################################
    return K


def train(x, y, l, K):
    theta = np.linalg.solve(
        K.T.dot(K) + l * np.identity(len(K)), K.T.dot(y.reshape(y.shape[0], 1)))
    return theta


def predict(x, X, theta, K):
    y_predicted = K.T.dot(theta)
    return y_predicted

def train_predict_for_class_c(c, K_train, K_test, x_train, y_train, x_test):
    check = lambda x: 1 if x==c else -1
    y_train_for_class_c = np.array(list(map(check, y_train)))
    theta = train(x_train, y_train_for_class_c, l=0.001, K=K_train)
    y_predicted = predict(x_train, x_test, theta=theta, K=K_test)
    return y_predicted.flatten()


# calc K Matrix
## for train
K_train = get_KMatrix(x_train, x_train, h=1)
K_test = get_KMatrix(x_train, x_test, h=1)

y_predicted = np.zeros((2000, 10))
for c in range(10):
    print('train for class:', c)
    y_predicted[:, c] = train_predict_for_class_c(c, K_train, K_test, x_train, y_train, x_test)

y_predicted = np.argmax(y_predicted, axis=1)
cMat = confusion_matrix(y_test, y_predicted)
print(cMat)

df = pd.DataFrame(cMat, columns=[i for i in range(10)])
df.to_csv(OUTPUT_DIR+'result.csv')
