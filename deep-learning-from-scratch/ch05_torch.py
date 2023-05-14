# -*- coding: UTF-8 -*-

import sys, os, time
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt


InputSize = 784
HiddenSize = 100
OutputSize = 10

import torch
from torch import nn

net = nn.Sequential(nn.Linear(InputSize, HiddenSize), nn.ReLU(), nn.Linear(HiddenSize, OutputSize))
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print('train size x:t', x_train.shape, t_train.shape)
print('test size x:t', x_test.shape, t_test.shape)


train_loss_list = []
train_acc_list = []
test_acc_list = []

train_size = x_train.shape[0]
batch_size = 100

iter_per_epoch = max(1, train_size / batch_size)

iter_nums = 10000
learning_rate = .1


def accuracy(y_hat, y):  # @save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
        y = y.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


for i in range(iter_nums):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = torch.tensor(x_train[batch_mask])
    t_batch = torch.tensor(t_train[batch_mask])

    l = loss(net(x_batch), t_batch)

    # backward for gradient descent
    trainer.zero_grad()
    l.mean().backward()
    trainer.step()

    if i % iter_per_epoch == 0:
        with torch.no_grad():
            acc = accuracy(net(torch.tensor(x_train)), torch.tensor(t_train)) / t_train.shape[0]
            test = accuracy(net(torch.tensor(x_test)), torch.tensor(t_test)) / t_test.shape[0]
            train_acc_list.append(acc)
            test_acc_list.append(test)
            print('iter_nums %d acc %6f loss %6f' % (i, acc, l.sum() / t_batch.shape[0]))

for i in range(10):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = torch.tensor(x_train[batch_mask])
    t_batch = torch.tensor(t_train[batch_mask])
    with torch.no_grad():
        for i in range(len(batch_mask)):
            r = net(x_batch[i])
            x = r.argmax()
            t = t_batch[i].argmax()
            if x != t:
                print(batch_mask[i], x, t, x == t)

x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
