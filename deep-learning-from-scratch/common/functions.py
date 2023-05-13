import numpy as np


def numerical_gradient_1d(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        y1 = f(x)

        x[idx] = tmp_val - h
        y2 = f(x)

        grad[idx] = (y1 - y2) / (2 * h)
        x[idx] = tmp_val
    return grad


def numerical_gradient(f, x):
    if x.ndim == 1:
        return numerical_gradient_1d(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, xx in enumerate(x):
            grad[idx] = numerical_gradient_1d(f, xx)
    return grad


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# def cross_entropy_error(y, t):
#    delta = 1e-7
#    return -np.sum(t * np.log(y + delta))


# for one-hot coded t
def cross_entropy_error(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size

