import numpy as np
import matplotlib.pyplot as plt
import sklearn
import scipy.spatial
import functools
import os
from sklearn.model_selection import GridSearchCV
### Kernel function generators
def linear_kernel(X1, X2):
    """
    Computes the linear kernel between two sets of vectors.
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
    Returns:
        matrix of size n1xn2, with x1_i^T x2_j in position i,j
    """
    return np.dot(X1, np.transpose(X2))


def RBF_kernel(X1, X2, sigma):
    """
    Computes the RBF kernel between two sets of vectors
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        sigma - the bandwidth (i.e. standard deviation) for the RBF/Gaussian kernel
    Returns:
        matrix of size n1xn2, with exp(-||x1_i-x2_j||^2/(2 sigma^2)) in position i,j
    """
    difference = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')
    return np.exp((-difference)/(2 * sigma**2))



def polynomial_kernel(X1, X2, offset, degree):
    """
    Computes the inhomogeneous polynomial kernel between two sets of vectors
    Args:
        X1 - an n1xd matrix with vectors x1_1,...,x1_n1 in the rows
        X2 - an n2xd matrix with vectors x2_1,...,x2_n2 in the rows
        offset, degree - two parameters for the kernel
    Returns:
        matrix of size n1xn2, with (offset + <x1_i,x2_j>)^degree in position i,j
    """
    return offset + np.dot(X1, np.transpose(X2)) ** degree


# PLot kernel machine functions

plot_step = .01
xpts = np.arange(-5.0, 6, plot_step).reshape(-1, 1)
prototypes = np.array([-4, -1, 0, 2]).reshape(-1, 1)

# Linear kernel
y = linear_kernel(prototypes, xpts)
for i in range(len(prototypes)):
    label = "Linear@" + str(prototypes[i, :])
    plt.plot(xpts, y[i, :], label=label)
plt.legend(loc='best')
plt.show()


class Kernel_Machine:
    def __init__(self, kernel, prototype_points, weights):
        """
        Args:
            kernel(X1,X2) - a function return the cross-kernel matrix between rows of X1 and rows of X2 for kernel k
            prototype_points - an Rxd matrix with rows mu_1,...,mu_R
            weights - a vector of length R with entries w_1,...,w_R
        """

        self.kernel = kernel
        self.prototype_points = prototype_points
        self.weights = weights

    def predict(self, X):
        """
        Evaluates the kernel machine on the points given by the rows of X
        Args:
            X - an nxd matrix with inputs x_1,...,x_n in the rows
        Returns:
            Vector of kernel machine evaluations on the n points in X.  Specifically, jth entry of return vector is
                Sum_{i=1}^R w_i k(x_j, mu_i)
        """
        kernel = self.kernel(X, self.prototype_points)
        return np.dot(kernel, self.weights)


prototypes = np.array([-1, 0, 1]).reshape(-1, 1)
weights = np.array([1, -1, 1]).reshape(-1, 1)
xpts = np.arange(-5.0, 6, plot_step).reshape(-1, 1)
rbf_model = Kernel_Machine('RBF_kernel', prototypes, weights)
plt.plot(xpts, np.transpose(rbf_model.predict(xpts)))
plt.show()
rbf_model.predict(xpts)



# 6.3 Kernel Ridge Regression
data_train, data_test = np.loadtxt("./hw4/krr-train.txt"),np.loadtxt("./hw4/krr-test.txt")
x_train, y_train = data_train[:, 0].reshape(-1, 1), data_train[:, 1].reshape(-1, 1)
x_test, y_test = data_test[:, 0].reshape(-1, 1), data_test[:, 1].reshape(-1, 1)

# 1) Plot the training data
plt.plot(x_train, y_train)
plt.show()


# 2) kernel ridge regression
def train_kernel_ridge_regression(X, y, kernel, l2reg):
    kernel_matrix = kernel(X,X)
    dim_k = kernel_matrix.shape[0]
    alpha = np.dot(np.linalg.inv((np.identity(dim_k)*l2reg + kernel_matrix)), y)
    return Kernel_Machine(kernel, X, alpha)


# 3) RBF kernel fixed regularization parameter(0.0001), different sigma(0.01, 0.1, 1)

plot_step = .001
xpts = np.arange(0 ,1, plot_step).reshape(-1,1)
plt.plot(x_train,y_train, 'o')
l2reg = 0.0001
for sigma in [.01,0.1,1]:
    k = functools.partial(RBF_kernel, sigma=sigma)
    f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=l2reg)
    label = "Sigma="+str(sigma)+",L2Reg="+str(l2reg)
    plt.plot(xpts, f.predict(xpts), label=label)
plt.legend(loc = 'best')
plt.ylim(-1,1.5)
plt.show()


# 4) RBF kernel fixed sigma  (0.02), different regularization parameters (0.0001, 0.001, 0.1)

plot_step = .001
xpts = np.arange(0 ,1, plot_step).reshape(-1,1)
plt.plot(x_train,y_train,'o')
for reg in [.0001,0.001,0.1]:
    k = functools.partial(RBF_kernel, sigma=0.02)
    f = train_kernel_ridge_regression(x_train, y_train, k, l2reg=reg)
    label = "Reg="+str(reg)+",Sigma="+str(0.02)
    plt.plot(xpts, f.predict(xpts), label=label)
plt.legend(loc = 'best')
plt.ylim(-1,1.5)
plt.show()

# 5) find the best hyper parameter settings


def compute_error(x_data, y_data, model, reg, sig):
    k = functools.partial(model, sigma=sig)
    predicted = train_kernel_ridge_regression(x_data, y_data, k, reg).predict(x_data)
    difference = np.dot(np.transpose(predicted - y_data), (predicted - y_data))
    return difference


sigma = [0.02,0.01,0.1,1]
reg = [0.0001,0.001,0.1]

print('RBF')
for s in sigma:
    for r in reg:
        print('Sigma is ', s)
        print('Reg is ', r)
        print('===============')
        print(compute_error(x_train, y_train, RBF_kernel, r, s))


print('linear')
print('===============')
for r in reg:
    predicted = train_kernel_ridge_regression(x_train, y_train, linear_kernel, r).predict(x_train)
    difference = np.dot(np.transpose(predicted - y_train), (predicted - y_train))
    print(difference)

print('polynomial_kernel')
print('===============')

# 6) Plot the best fitting prediction functions with Polynomial kernel and the RBF kernel  (-0.5, 1.5)



