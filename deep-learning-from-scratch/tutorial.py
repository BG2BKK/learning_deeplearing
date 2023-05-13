import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x > 0, int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


x = np.arange(-5, 5, 0.1)
y = step_function(x)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.legend()
#plt.show()