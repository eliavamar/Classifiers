import numpy as np


def print_price(test_x, test_y, W, b):
    x = np.dot(test_x, W) + b
    print("Mean Squared-Error: ", ((x - test_y) ** 2).mean(axis=None))


def linear():
    arr = np.loadtxt("prices.txt", dtype=float, encoding=None, delimiter=",")
    np.random.shuffle(arr)
    S = arr.shape[0]
    P = int((S * 85) / 100)
    R = S - P
    train_x = np.array([arr[i, 1:] for i in range(P)])
    train_y = np.array([arr[i, 1] for i in range(P)])
    test_x = np.array([arr[P + i, 1:] for i in range(R)])
    test_y = np.array([arr[P + i, 1] for i in range(R)])
    w = np.array([0.] * (arr.shape[1] - 1))
    b = 0
    alpha = 0.00001
    for iteration in range(350000):
        deriv_b = np.mean(1 * ((np.dot(train_x, w) + b) - train_y))
        gradient_w = (1.0 / len(train_y)) * np.dot(((np.dot(train_x, w) + b) - train_y), train_x)
        b -= alpha * deriv_b
        w -= alpha * gradient_w
    print_price(test_x, test_y, w, b)


if __name__ == '__main__':
    linear()
