import numpy as np
import random
# Dataset

def generate_data(num_instance, num_feature, theta, bias= False):

    X = np.random.rand(num_instance, num_feature)
    epsilon = 0.1 * np.random.rand(num_instance)

    if bias:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    y = X @ theta + epsilon
    X_train, X_validation, X_test = X[:80], X[80:100], X[100:150]
    y_train, y_validation, y_test = y[:80], y[80:100], y[100:150]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

# Ridge_regression

class ridge_regression():
    def __init__(self, theta):
        self.theta = ''

    def fit(self, X, y, lambda_reg):
        m = X.shape[0]
        def obj(theta):





# Shooting Algorithm (Coordinate Descent for Lasso)



def soft(a, delta):
    soft = np.sign(a) * max(abs(a) - delta, 0)
    return soft


def compute_loss(theta, X, y):
    loss = (np.linalg.norm(np.dot(X, theta) - y))**2
    return loss

def compute_objective_loss(theta, X, y, lambda_reg):
    loss = compute_loss(theta, X, y)
    loss += lambda_reg * np.linalg.norm(theta, ord=1)
    return loss

def shooting_algorithm(theta_init, X, y, lambda_reg, max_iter=10000, tol=10**-8):
    theta = theta_init
    diff = 1
    num_iter = 1
    while num_iter<max_iter and diff>tol:
        loss_previous = compute_objective_loss(theta, X, y, lambda_reg)

        for j in range(theta.shape[0]):
            a = X[:,j] @ X[:,j]
            c = X[:,j] @ (y - X @ theta + theta[j] * X[:,j])
            parameter_1 = c/a
            parameter_2 = lambda_reg/a
            theta[j] = soft(parameter_1, parameter_2)
        loss_current = compute_objective_loss(theta, X, y, lambda_reg)
        diff = abs(loss_previous - loss_current)
        num_iter+=1
    return theta


def homotopy(theta_init, X_train, y_train, X_val, y_val, lambda_max):
    theta = theta_init
    loss_hist = {}
    theta_hist = {}
    Lambda=lambda_max
    while Lambda>10**-5:
        theta = shooting_algorithm(theta, X_train, y_train, Lambda)
        loss_hist[Lambda] = compute_objective_loss(theta, X_val, y_val, Lambda)
        theta_hist[Lambda] = theta
        Lambda *= 0.8
    return theta_hist, loss_hist



# Projected SGD via Variable Splitting

def gradiant_descent_lasso(X, y,alpha=0.01,lambda_reg=0.001, max_iter=10000, tol=0):
    # X       i*j
    # theta   j*1
    #
    num_features = X.shape[1]

    def compute_gradient(X, y, theta_1, theta_2, lambda_reg):
        num_features = X.shape[1]
        predict = X @ (theta_1 - theta_2)
        error = y - predict                 #i*1

        grad_1 = - X.transpose() @ error + 2*lambda_reg * np.ones(num_features)
        grad_2 = X.transpose() @ error + 2*lambda_reg * np.ones(num_features)

        return grad_1, grad_2

    def floor(array):
        for i,e in enumerate(array):
            array[i] = max(e,0)
        return array

    num_instance, num_features = X.shape[0], X.shape[1]
    theta_1 = np.zeros(num_features)
    theta_2 = np.zeros(num_features)
    loss_hist = np.zeros(max_iter)
    theta = 0

    for i in np.arange(max_iter):
        theta_previous = theta_1 - theta_2
        grad_1, grad_2 = compute_gradient(X, y, theta_1, theta_2, lambda_reg)
        theta_1 -= alpha * grad_1
        theta_2 -= alpha * grad_2
        theta_1 = floor(theta_1)
        theta_2 = floor(theta_2)
        theta = theta_1 - theta_2

        loss_hist[i] = compute_loss(theta, X, y)

        diff = np.linalg.norm(theta_previous - theta, ord=1)

        if diff < tol:
            break
    return theta, loss_hist


def stochastic_gradient_descent_lasso(X, y, alpha='1/t', lambda_reg=0.001, max_iter=1000, tol=0):

    def compute_stochastic_gradient(X, y, theta_1, theta_2, lambda_reg):
        num_features = X.shape[1]
        predict = X @ (theta_1 - theta_2)
        error = y - predict

        grad_1 = - X @ error + 2*lambda_reg*np.ones(num_features)
        grad_2 =   X @ error + 2*lambda_reg*np.ones(num_features)
        return grad_1, grad_2

    def floor(array):
        for i,e in enumerate(array):
            array[i] = max(e,0)
        return array
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_1 = np.zeros(num_features)
    theta_2 = np.zeros(num_features)
    loss_hist = np.zeros(max_iter)
    theta = np.zeros(num_features)


    for i in np.arange(max_iter):
        theta_previous = theta_1 - theta_2
        index = np.random.permutation(num_instances)
        for i_index,j in enumerate(index):
            x,y= X[j],y[j]

            if isinstance(alpha, float):
                step_size = alpha
            elif alpha == '1/t':
                step_size = 1.0/(num_instances*i+i_index+1)
            else:
                step_size = 1.0/np.sqrt(num_instances*i + i_index + 1)

            grad_1, grad_2 = compute_stochastic_gradient(x,y,theta_1, theta_2, lambda_reg)
            theta_1 -= step_size*grad_1
            theta_2 -= step_size*grad_2
            theta_1 = floor(theta_1)
            theta_2 = floor(theta_2)

        theta = theta_1 - theta_2
        loss_hist[i] = compute_loss(theta, X, y)

        diff = np.linalg.norm(theta_previous - theta, ord=1)

        if diff < tol:
            print('Converged')
            break

    return theta, loss_hist



