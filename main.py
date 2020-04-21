import numpy as np
import time
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset


def sigmoid(z):
    res = 1/(1+np.exp(-z))
    return res


def J(hypo, y, m):
    costs = -(y * np.log(hypo) + (1-y) * np.log(1-hypo))
    res = np.sum(costs) / m
    return res


def gradient(theta, bias, x, y, m):
    z = np.dot(x, theta) + bias
    hypo = sigmoid(z)
    cost = J(hypo, y.T, m)
    diff = - y + hypo
    return diff, cost


def iterate(x, y, iterations, rate, n, m):
    theta = np.zeros(shape=(n, 1))
    bias = 0.0
    costs = []
    for i in range(0, iterations):
        diff, cost = gradient(theta, bias, x, y, m)
        costs.append(cost)
        theta = theta - np.dot(x.T, diff) * rate / m
        bias = bias - np.sum(diff) * rate / m
    return theta, bias, costs


def test(theta, bias, x):
    z = np.dot(x, theta) + bias
    res = sigmoid(z)
    for i in range(0, res.shape[0]):
        if res[i, 0] >= 0.5:
            res[i, 0] = 1
        else:
            res[i, 0] = 0
    return res


def rescmp(cal, data):
    len = cal.size
    cnt = 0
    for i in range(0, len):
        if cal[i] == data[i]:
            cnt += 1
    return cnt / len


# def lore(n, m, x, y):
#     theta = np.full(shape=(n, 1), fill_value=0.0)
#     alpha = 0.000001
#     p = 0
#     while True:
#         diff = sigmoid(x, theta) - y  # 计算出的 y 与数据 y 的差值
#         delta = np.dot(x, diff)
#         theta -= alpha * delta / m  # 更新 theta
#         p += 1
#         if p >= 10000:
#             break
#     return theta, p


def main():
    # theta, bias, X, Y, m = np.mat('1; 2'), 2, np.mat('1 3; 2 4'), np.mat('1; 0'), 2
    # diff, cost = gradient(theta, bias, X, Y, 2)
    # pram, b, costs = iterate(X, Y, 100, 0.009, 2, 2)
    # print(pram)
    # print(b)
    # print(costs)
    # print(cost)
    # print(np.dot(X.T, diff) / m)
    # print(np.sum(diff)/m)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
    train_set_count = train_set_x_orig.shape[0]
    test_set_count = test_set_x_orig.shape[0]
    train_set_x = train_set_x_orig.reshape(train_set_count, -1) / 255
    test_set_x = test_set_x_orig.reshape(test_set_count, -1) / 255
    feature_count = 64 * 64 * 3
    train_accs = []
    test_accs = []
    for i in range(1, 31):
        theta, bias, costs = iterate(train_set_x, train_set_y.T, 1000, 0.001 * i, feature_count, train_set_count)
        test_res = test(theta, bias, test_set_x)
        test_acc = rescmp(test_res, test_set_y.T)
        train_res = test(theta, bias, train_set_x)
        train_acc = rescmp(train_res, train_set_y.T)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    plt.plot(range(1, 31), train_accs)
    plt.plot(range(1, 31), test_accs)
    plt.show()
    # pram, ite = lore(feature_count, train_set_x_orig.shape[0], train_set_x, train_set_y.T)
    # results = sigmoid(test_set_x, pram)
    # for result in results:
    #     if result >= 0.5:
    #         result = 1
    #     else:
    #         result = 0
    # cnt = 0
    # for i in range(0, test_set_x_orig.shape[0]):
    #     if results[i] == np.squeeze(test_set_y[i]):
    #         cnt += 1
    # print(cnt / test_set_x_orig.shape[0])
    # plt.imshow(train_set_x_orig[i])
    #     # print(train_set_y_orig)
    #     # plt.show()


main()
