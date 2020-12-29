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
def updating_grad_pegasos(x, y, weight, lambda_reg, lr):
    """

    Args:
        x: dict or Counter class  word: frequency
        y: class -1 or 1
        weight: dict or Counter class word:frequency
        lambda_reg: float

    Returns:
        weight_updated

    """
    weight_1 = weight.copy()
    for v, fre in weight_1.items():
        weight_1[v] = weight_1[v] * (1 - lr * lambda_reg)
    if y * dotProduct(weight, x) < 1:
        increment(weight_1, lr * y, x)
    else:
        pass
    return weight_1


def compute_pegasos_loss(x, y, weight, lambda_reg):
    """

    Args:
        x: dict or Counter class {word：frequency}
        y: class -1 or 1
        weight: dict or Counter class {word:frequency}

    Returns:
        loss: float        yw.tx

    """
    reg_term = np.sum([weight[i] ** 2 for i in weight]) * lambda_reg/2.0
    hinge_term = max(0, 1-y*dotProduct(weight, x))
    return reg_term+hinge_term


train_x_sparse, train_y = create_sparse_list(training_data)
val_x_sparse, val_y = create_sparse_list(validation_data)


def compute_pegasos_gradient(x, y, weight, lambda_reg):
    weight_using = weight.copy()
    for vac, fre in weight.item():
        weight_using[vac] *= lambda_reg
    if y * dotProduct(weight,x) < 1:
        increment(weight_using, -y, x)
    else:
        pass
    return weight_using





def pegasos_gradient_checker(x, y, weight, lambda_reg, epsilon=0.01, tolerance=1e-4):
    true_gradient = compute_pegasos_gradient(x, y, weight, lambda_reg)
    weight_checking = weight.copy
    increment(weight_checking, 0, x)
    approx_grad = weight_checking.copy()

    for word in weight_checking:
        weight_plus = weight_checking.copy()
        weight_plus[word] += epsilon
        loss_plus = compute_pegasos_loss(x, y, weight_plus, lambda_reg)
        weight_minus = weight_checking.copy()
        weight_minus[word] -= epsilon
        loss_minus = compute_pegasos_loss(x, y, weight_minus, lambda_reg)
        approx_grad[word] = (loss_plus - loss_minus) / (2*epsilon)
    distance = np.sum([(true_gradient[i] - approx_grad[i]) ** 2 for i in approx_grad])
    return distance < tolerance




max_epoch = 20
epoch = 0
t = 0
lambda_reg = 20
weight_updated = collections.Counter()
avg_train_loss = []
avg_val_loss = []
while epoch < max_epoch:
    t += 1
    epoch += 1
    lr = 1/(t * lambda_reg)
    train_loss = []
    val_loss = []

    for i in range(len(train_x_sparse)):
        weight_updated = updating_grad_pegasos(train_x_sparse[i], train_y[i][0], weight_updated, lambda_reg, lr)

    for i in range(len(train_x_sparse)):
        train_loss.append(compute_pegasos_loss(train_x_sparse[i], train_y[i][0], weight_updated, lambda_reg))
        avg_train_loss.append(sum(train_loss)/len(train_loss))

    for i in range(len(val_x_sparse)):
        val_loss.append(compute_pegasos_loss(val_x_sparse[i], val_y[i][0], weight_updated, lambda_reg))
        avg_val_loss.append(sum(val_loss)/len(val_loss))

    print(f'Epoch {epoch} completed          Train Loss:{avg_train_loss[epoch-1]}   Val Loss:{avg_val_loss[epoch-1]}')

weight_updated = updating_grad_pegasos(train_x_sparse[0], train_y[0][0], weight_updated, lambda_reg, 0.01)
compute_pegasos_loss(train_x_sparse[1], train_y[1][0], weight_updated, lambda_reg)


increment(weight_updated, 0.01 * train_y[0][0], train_x_sparse[0])

weight_updated = updating_grad_pegasos(train_x_sparse[5], train_y[5][0], weight_updated, lambda_reg, 0.001)


for i in range(len(train_x_sparse)):
    weight_updated = updating_grad_pegasos(train_x_sparse[i], train_y[i][0], weight_updated, lambda_reg, 0.001)



compute_pegasos_loss(train_x_sparse[1], train_y[1][0], weight_updated, 100)


val_predict = []
for item in range(len(val_x_sparse)):
    val_predict.append(dotProduct(weight_updated, val_x_sparse[item]))


compute_pegasos_loss(val_x_sparse[0], val_y[0][0], weight_updated, 100)

count = 0
for item in range(len(train_x_sparse)):
    if train_y[item][0] * dotProduct(weight_updated, train_x_sparse[item]) > 0:
        count += 1
    else:
        pass
