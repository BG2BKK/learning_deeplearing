
import numpy as np

def mean_squared_error(y, t):
    return (y - t)**2 * 0.5

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))



x = np.array([0, 0, 1, 0, 0])
y = np.array([0.777, 0.63, 0.325, 0.333, 0.07])

print(y)
print(cross_entropy_error(y, x))

y = np.array([0.577, 0.03, 0.925, 0.333, 0.07])
print(y)
print(cross_entropy_error(y, x))
