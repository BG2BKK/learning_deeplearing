
import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

def show_img(x, y):
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()

def function_2(x):
    return np.sum(x ** 2)



def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        y1 = f(x)

        x[idx] = tmp_val - h
        y2 = f(x)

        grad[idx] = (y1 - y2)/(2*h)
        x[idx] = tmp_val
    return grad

print(numerical_gradient(function_2, np.array([0.1, 0.2])))
print(numerical_gradient(function_2, np.array([0.2, 0.1])))
print(numerical_gradient(function_2, np.array([2.0, 3.0])))

def gradient_descent(f, init_x, lr=0.1, step_num=100):
    x = init_x
    for idx in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

init_x = np.array([-3., 4.])
x = gradient_descent(function_2, init_x)
print(x)
