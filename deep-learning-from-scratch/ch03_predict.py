import numpy as np
from common import softmax, cross_entropy_error, sigmoid
from dataset.mnist import load_mnist
from PIL import Image
import pickle


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        for k in network:
            print(k)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y


x, t = get_data()
network = init_network()

batch_size = 1
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    range_end = min(i+batch_size, len(x))
    y_batch = predict(network, x[i:range_end])
    p_batch = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p_batch == t[i:range_end])
print('Accuracy %s' % (accuracy_cnt * 1.0 / len(x)))