import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt

### userful link to complete this https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote06.html


def f_objective(theta, X, y, l2_param=1):
    '''
    Args:
        theta: 1D numpy array of size num_features
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        l2_param: regularization parameter

    Returns:
        objective: scalar value of objective function
    '''
    num_instances, num_features = X.shape
    logistic_loss = 0
    for i in range(0, num_instances-1):
        logistic_loss = logistic_loss + np.logaddexp(0, -y[i] * np.dot(np.transpose(theta), X[i, :]))
    R_n = (logistic_loss + l2_param * np.dot(np.transpose(theta), theta))/num_instances
    return R_n



def fit_logistic_reg(X, y, objective_function, l2_param=1):
    '''
    Args:
        X: 2D numpy array of size (num_instances, num_features)
        y: 1D numpy array of size num_instances
        objective_function: function returning the value of the objective
        l2_param: regularization parameter
        
    Returns:
        optimal_theta: 1D numpy array of size num_features
    '''
    num_instances, num_features = X.shape
    theta_initial = np.random.rand(num_features, 1)
    optimal_theta = minimize(objective_function, theta_initial, args=(X, y, l2_param), method='SLSQP')
    return optimal_theta




def data_normalize(raw_data):
    num_instance, num_feature = raw_data.shape
    x_normalized = np.zeros(raw_data.shape)
    feature_min = np.amin(raw_data, 0)
    feature_max = np.amax(raw_data, 0)
    for i in range(num_feature):
        x_normalized[:, i] = (raw_data[:, i] - feature_min[i])/(feature_max[i] - feature_min[i])
    return x_normalized


X_train, y_train = np.loadtxt("./hw5/logistic-code/X_train.txt",delimiter=','), np.loadtxt("./hw5/logistic-code/y_train.txt",delimiter=',')
X_val, y_val = np.loadtxt("./hw5/logistic-code/X_val.txt",delimiter=','), np.loadtxt("./hw5/logistic-code/y_val.txt",delimiter=',')




normalized_x_train = data_normalize(X_train)
log_likelihood = []
reg = []
for i in range(-6, 0):
    lr2_reg = 10 ** i
    result = fit_logistic_reg(normalized_x_train, y_train, f_objective, 1)
    log_likelihood.append(f_objective(result.x, X_val, y_val, lr2_reg))
    reg.append(lr2_reg)

plt.plot(reg, log_likelihood)
plt.show()




def compute_correct_logaddexp(s):
    answer = 0
    if s <= 0:
        answer = np.log(1 + np.exp(-s))
    elif s > 0:
        answer = -s + np.log(np.exp(-s) + np.exp(-2*s))
    return answer