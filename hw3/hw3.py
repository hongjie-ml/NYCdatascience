import os
import random
import collections
from util import dotProduct
from util import increment
import numpy as np

neg_path = './data/neg'
pos_path = './data/pos'


### 4. The data ###
# input txt dataset

# output word_list with label

def read_data(file_path, file_name):
    with open(os.path.join(file_path, file_name)) as f:
        lines = f.read().split(" ")
        symbols = '${}()[].,:;+-*/&|<>=~" '
        words = map(lambda Element: Element.translate(str.maketrans("", "", symbols)).strip(), lines)
        words = list(filter(None, words))
        return words


def read_file(file_path, label):
    review_all = []
    file_list = os.listdir(file_path)
    for item in file_list:
        review = read_data(file_path, item)
        review.append(label)
        review_all.append(review)
    return review_all


neg_review = read_file(neg_path, -1)
pos_review = read_file(pos_path, 1)

all_data = neg_review + pos_review
random.shuffle(all_data)
training_data = all_data[:1500]
validation_data = all_data[1500:2000]


### Sparse Representations ###

def create_sparse_list(dataset):
    x = []
    y = []
    x_sparse = []
    for item in dataset:
        length = len(item)
        x.append(item[:length - 1])
        y.append(item[length - 1:])
    for text in x:
        temp = collections.Counter()
        for vocab in set(text):
            temp[vocab] = text.count(vocab)
        x_sparse.append(temp)
    return x_sparse, y


### Pegasos algorithm ###
def updating_grad_pegasos(x, y, weight, lambda_reg):
    """

    Args:
        x: list for all data, each one is a counter data type
        y: list for the label, each one is a list with a label,  y[i][0]
        weight: dict or Counter class word:frequency
        lambda_reg: float

    Returns:
        weight_updated

    """
    t = 0
    for i in range(len(x)):
        t += 1
        lr = 1 / (t * lambda_reg)
        if y[i][0] * dotProduct(weight, x[i]) < 1:
            increment(weight, -lr * lambda_reg, weight)
            increment(weight, lr * y[i][0], x[i])
        else:
            increment(weight, -lr * lambda_reg, weight)
    return weight


def compute_pegasos_loss(x, y, weight, lambda_reg):
    """

    Args:
        x: dict or Counter class {word：frequency}
        y: class -1 or 1
        weight: dict or Counter class {word:frequency}

    Returns:
        loss: float        yw.tx

    """
    reg_term = np.sum([weight[i] ** 2 for i in weight]) * lambda_reg / 2.0
    hinge_term = max(0, 1 - y * dotProduct(weight, x))
    return reg_term + hinge_term


def compute_pegasos_gradient(x, y, weight, lambda_reg):
    weight_using = weight.copy()
    for vac, fre in weight.items():
        weight_using[vac] *= lambda_reg
    if y * dotProduct(weight, x) < 1:
        increment(weight_using, -y, x)
    else:
        pass
    return weight_using


def pegasos_gradient_checker(x, y, weight, lambda_reg, epsilon=0.01, tolerance=1e-4):
    true_gradient = compute_pegasos_gradient(x, y, weight, lambda_reg)
    weight_checking = weight.copy()
    increment(weight_checking, 0, x)
    approx_grad = weight_checking.copy()

    for word in weight_checking:
        weight_plus = weight_checking.copy()
        weight_plus[word] += epsilon
        loss_plus = compute_pegasos_loss(x, y, weight_plus, lambda_reg)
        weight_minus = weight_checking.copy()
        weight_minus[word] -= epsilon
        loss_minus = compute_pegasos_loss(x, y, weight_minus, lambda_reg)
        approx_grad[word] = (loss_plus - loss_minus) / (2 * epsilon)
    distance = np.sum([(true_gradient[i] - approx_grad[i]) ** 2 for i in approx_grad])
    return distance < tolerance


def accuracy_rate(y, weight, x):
    accurate = 0
    total = len(y)
    for i in range(len(x)):
        if np.sign(y[i][0]) == np.sign(dotProduct(weight, x[i])):
            accurate += 1

    accuraterate = accurate / total
    return accuraterate


train_x_sparse, train_y = create_sparse_list(training_data)
val_x_sparse, val_y = create_sparse_list(validation_data)

max_epoch = 1
epoch = 0
weight_updated = collections.Counter()
"""avg_train_loss = []
avg_val_loss = []
weight = []
train_accuracy = []
val_accuracy = []"""
while epoch < max_epoch:
    epoch += 1
    weight_updated = updating_grad_pegasos(train_x_sparse, train_y, weight_updated, 0.1)

    for i in range(len(train_x_sparse)):
        train_loss.append(compute_pegasos_loss(train_x_sparse[i], train_y[i][0], weight_updated, lambda_reg))
    avg_train_loss.append(sum(train_loss) / len(train_loss))

    for i in range(len(val_x_sparse)):
        val_loss.append(compute_pegasos_loss(val_x_sparse[i], val_y[i][0], weight_updated, lambda_reg))
    avg_val_loss.append(sum(val_loss) / len(val_loss))

    train_accuracy.append(accuracy_rate(train_y, weight_updated, train_x_sparse))
    val_accuracy.append(accuracy_rate(val_y, weight_updated, val_x_sparse))

    print(
        f'Epoch {epoch} completed Train Loss:{avg_train_loss[epoch - 1]}  |Val Loss:{avg_val_loss[epoch - 1]}|Train_acc:{train_accuracy[epoch - 1]}|Val_acc:{val_accuracy[epoch - 1]}')

import matplotlib.pyplot as plt

train_list = []
val_list = []
max_epoch = 2
epoch = 0
for i in range(-5, 5):
    lambda_reg = 10 ** i
    print('==============')
    print('lambda= ', lambda_reg)
    epoch = 0
    weight_updated = collections.Counter()
    while epoch < max_epoch:
        epoch += 1
        weight_updated = updating_grad_pegasos(train_x_sparse, train_y, weight_updated, lambda_reg)
    train_acc = accuracy_rate(train_y, weight_updated, train_x_sparse)
    val_acc = accuracy_rate(val_y, weight_updated, val_x_sparse)
    train_list.append(train_acc)
    val_list.append(val_acc)

plt.plot(range(-5, 5), train_list, label='train accuracy')
plt.plot(range(-5, 5), val_list, label='validation accuracy')
plt.xlabel("log(lambda)")
plt.ylabel("accuracy")
plt.legend()
plt.show()


def updating_pegasos_fast(x, y, weight, lambda_reg, lr, s):
    for i in range(len(x)):
        s = (1 - lr * lambda_reg) * s
        if y * dotProduct(weight, x) < 1:
            increment(weight, lr * y / s, x)
    w = dict()
    increment(w, s, weight)
    return w


train_list = []
val_list = []
t = 1
max_epoch = 2
epoch = 0
s = 1
weight_updated = collections.Counter()
for i in range(-5, 5):
    lambda_reg = 10 ** (i)
    print('==============')
    print('lambda= ', lambda_reg)
    epoch = 0
    s = 1
    t = 1
    while epoch < max_epoch:
        t += 1
        epoch += 1
        lr = 1 / (t * lambda_reg)
        for i in range(len(train_x_sparse)):
            weight_updated = updating_pegasos_fast(train_x_sparse[i], train_y[i][0], weight_updated, lambda_reg, lr,
                                                   s=s)
    train_acc = accuracy_rate(train_y, weight_updated, train_x_sparse)
    val_acc = accuracy_rate(val_y, weight_updated, val_x_sparse)
    train_list.append(train_acc)
    val_list.append(val_acc)

accuracy_rate(val_y, weight_updated, val_x_sparse)


