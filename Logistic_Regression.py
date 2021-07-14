import numpy as np


def print_price(test_x, test_y, W, b):
    h_x = np.round(h(test_x, W, b))
    true_positive = 0.
    true_negative = 0.
    false_negative = 0.
    false_positive = 0.
    for i in range(len(h_x)):
        if h_x[i] == test_y[i] and h_x[i] == 1:
            true_positive += 1
        elif h_x[i] == test_y[i] and h_x[i] == 0:
            true_negative += 1
        elif h_x[i] == 1:
            false_positive += 1
        else:
            false_negative += 1
    Accuracy = ((true_positive + true_negative) / len(test_y))
    Recall = true_positive / (true_positive + false_negative)
    Precision = true_positive / (true_positive + false_positive)
    print("Accuracy: ", Accuracy)
    print("Recall:", Recall)
    print("Precision:", Precision)
    print("F-measure:", (2 * Precision * Recall) / (Precision + Recall))


def h(x, w, b):
    return 1 / (1 + np.exp(-(np.dot(x, w) + b)))


def logistic():
    arr = np.loadtxt("prices.txt", dtype=float, encoding=None, delimiter=",")
    np.random.shuffle(arr)
    S = arr.shape[0]
    P = int((S * 75) / 100)
    R = S - P
    train_x = np.array([arr[i, :] for i in range(P)])
    train_x = np.delete(train_x, [10], axis=1)
    train_y = np.array([arr[i, 10] for i in range(P)])
    test_x = np.array([arr[P + i, :] for i in range(R)])
    test_x = np.delete(test_x, [10], axis=1)
    test_y = np.array([arr[P + i, 10] for i in range(R)])
    w = np.array([0.] * (arr.shape[1] - 1))
    b = 0
    alpha = 0.0001
    # if not at least 1,000,000 iterations might cause divide by zero exception!!!!
    for iteration in range(500000):
        deriv_b = np.mean(1 * (h(train_x, w, b) - train_y))
        gradient_w = (1.0 / len(train_y)) * np.dot(((h(train_x, w, b)) - train_y), train_x)
        b -= alpha * deriv_b
        w -= alpha * gradient_w
    print_price(test_x, test_y, w, b)


if __name__ == '__main__':
    logistic()
