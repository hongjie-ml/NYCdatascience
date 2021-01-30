import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets.samples_generator import make_blobs
from numpy.random import shuffle
from random import choice
# Create the  training data
np.random.seed(2)
X, y = make_blobs(n_samples=300,cluster_std=.25, centers=np.array([(-3,1),(0,2),(3,1)]))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50)

from sklearn.base import BaseEstimator, ClassifierMixin, clone


class OneVsAllClassifier(BaseEstimator, ClassifierMixin):
    """
    One-vs-all classifier
    We assume that the classes will be the integers 0,..,(n_classes-1).
    We assume that the estimator provided to the class, after fitting, has a "decision_function" that
    returns the score for the positive class.
    """

    def __init__(self, estimator, n_classes):
        """
        Constructed with the number of classes and an estimator (e.g. an
        SVM estimator from sklearn)
        @param estimator : binary base classifier used
        @param n_classes : number of classes
        """
        self.n_classes = n_classes
        self.estimators = [clone(estimator) for _ in range(n_classes)]
        self.fitted = False

    def fit(self, X, y=None):
        """
        This should fit one classifier for each class.
        self.estimators[i] should be fit on class i vs rest
        @param X: array-like, shape = [n_samples,n_features], input data
        @param y: array-like, shape = [n_samples,] class labels
        @return returns self
        """
        # Your code goes here
        y_fit={}

        for i in range(self.n_classes):
            y_fit[i] = np.where(y == i, 1, 0)

        for i in range(self.n_classes):
            self.estimators[i].fit(X, y_fit[i])
        self.fitted = True
        return self

    def decision_function(self, X):
        """
        Returns the score of each input for each class. Assumes
        that the given estimator also implements the decision_function method (which sklearn SVMs do),
        and that fit has been called.
        @param X : array-like, shape = [n_samples, n_features] input data
        @return array-like, shape = [n_samples, n_classes]
        """
        if not self.fitted:
            raise RuntimeError("You must train classifer before predicting data.")

        if not hasattr(self.estimators[0], "decision_function"):
            raise AttributeError(
                "Base estimator doesn't have a decision_function attribute.")

        # Replace the following return statement with your code

        score = np.zeros([self.n_classes, X.shape[0]])
        for k in range(self.n_classes):
            score[k] = self.estimators[k].decision_function(X)
        score = score.T
        return score

    def predict(self, X):
        """
        Predict the class with the highest score.
        @param X: array-like, shape = [n_samples,n_features] input data
        @returns array-like, shape = [n_samples,] the predicted classes for each input
        """
        # Replace the following return statement with your code
        score = self.decision_function(X)
        predicted_y = np.zeros([score.shape[0]])
        for i in range(len(predicted_y)):
            predicted_y[i] = np.where(score[i] == max(score[i]))[0][0]
        return predicted_y


#Here we test the OneVsAllClassifier
from sklearn import svm
svm_estimator = svm.LinearSVC(loss='hinge', fit_intercept=False, C=200, max_iter=2000)
clf_onevsall = OneVsAllClassifier(svm_estimator, n_classes=3)
clf_onevsall.fit(X,y)

for i in range(3) :
    print("Coeffs %d"%i)
    print(clf_onevsall.estimators[i].coef_) #Will fail if you haven't implemented fit yet

# create a mesh to plot in
h = .02  # step size in the mesh
x_min, x_max = min(X[:,0])-3,max(X[:,0])+3
y_min, y_max = min(X[:,1])-3,max(X[:,1])+3
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
mesh_input = np.c_[xx.ravel(), yy.ravel()]

Z = clf_onevsall.predict(mesh_input)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)



from sklearn import metrics
metrics.confusion_matrix(y, clf_onevsall.predict(X))


def zeroOne(y, a):
    '''
    Computes the zero-one loss.
    @param y: output class
    @param a: predicted class
    @return 1 if different, 0 if same
    '''
    return int(y != a)


def featureMap(X, y, num_classes):
    '''
    Computes the class-sensitive features.
    @param X: array-like, shape = [n_samples,n_inFeatures] or [n_inFeatures,], input features for input data
    @param y: a target class (in range 0,..,num_classes-1)
    @return array-like, shape = [n_samples,n_outFeatures], the class sensitive features for class y
    '''
    # The following line handles X being a 1d-array or a 2d-array
    num_samples, num_inFeatures = (1, X.shape[0]) if len(X.shape) == 1 else (X.shape[0], X.shape[1])
    # your code goes here, and replaces following return
    n_outFeatures = num_classes * num_inFeatures
    output = np.zeros([num_samples, n_outFeatures])
    if num_samples == 1:
        output = np.zeros(n_outFeatures)
        output[y*num_inFeatures:(y+1)*num_inFeatures] = X
        return output
    for idx, value in enumerate(X):
        y_i = y[idx]
        output[idx][y_i*num_inFeatures:(y_i+1)*num_inFeatures] = value
    return output



def sgd(X, y, num_outFeatures, subgd, eta=0.1, T=10000):
    '''
    Runs subgradient descent, and outputs resulting parameter vector.
    @param X: array-like, shape = [n_samples,n_features], input training data
    @param y: array-like, shape = [n_samples,], class labels
    @param num_outFeatures: number of class-sensitive features
    @param subgd: function taking x,y and giving subgradient of objective
    @param eta: learning rate for SGD
    @param T: maximum number of iterations
    @return: vector of weights
    '''
    num_samples = X.shape[0]
    # your code goes here and replaces following return statement
    w = np.zeros(num_outFeatures)
    for t in range(T):
        index = np.random.randint(300)
        X_suffled = X[index]
        y_suffled = y[index]
        subgradient = subgd(X_suffled, y_suffled, w)
        w = w - eta * subgradient
    return w



class MulticlassSVM(BaseEstimator, ClassifierMixin):
    '''
    Implements a Multiclass SVM estimator.
    '''

    def __init__(self, num_outFeatures, lam=0.1, num_classes=3, Delta=zeroOne, Psi=featureMap):
        '''
        Creates a MulticlassSVM estimator.
        @param num_outFeatures: number of class-sensitive features produced by Psi
        @param lam: l2 regularization parameter
        @param num_classes: number of classes (assumed numbered 0,..,num_classes-1)
        @param Delta: class-sensitive loss function taking two arguments (i.e., target margin)
        @param Psi: class-sensitive feature map taking two arguments
        '''
        self.num_outFeatures = num_outFeatures
        self.lam = lam
        self.num_classes = num_classes
        self.Delta = Delta
        self.Psi = lambda X, y: Psi(X, y, num_classes)
        self.fitted = False

    def subgradient(self, x, y, w):
        '''
        Computes the subgradient at a given data point x,y
        @param x: sample input
        @param y: sample class
        @param w: parameter vector
        @return returns subgradient vector at given x,y,w
        '''
        y_max = 0
        local_max = self.Delta(y, y_max) + np.dot(w, (self.Psi(x, y_max) - self.Psi(x, y)).T)
        for y_i in range(self.num_classes):
            current = self.Delta(y, y_i) + np.dot(w, (self.Psi(x, y_i) - self.Psi(x, y)).T)
            if current > local_max:
                local_max = current
                y_max = y_i
        local_sgd = 2 * self.lam * w + self.Psi(x, y_max) - self.Psi(x, y)
        return local_sgd

    def fit(self, X, y, eta=0.1, T=10000):
        '''
        Fits multiclass SVM
        @param X: array-like, shape = [num_samples,num_inFeatures], input data
        @param y: array-like, shape = [num_samples,], input classes
        @param eta: learning rate for SGD
        @param T: maximum number of iterations
        @return returns self
        '''

        self.coef_ = sgd(X, y, self.num_outFeatures, self.subgradient, eta, T)
        self.fitted = True
        return self



    def decision_function(self, X):
        '''
        Returns the score on each input for each class. Assumes
        that fit has been called.
        @param X : array-like, shape = [n_samples, n_inFeatures]
        @return array-like, shape = [n_samples, n_classes] giving scores for each sample,class pairing
        '''
        if not self.fitted:
            raise RuntimeError("You must train classifer before predicting data.")

        num_samples = X.shape[0]
        score = np.zeros([num_samples, self.num_classes])
        for i in range(num_samples):
            for j in range(self.num_classes):
                output = self.Psi(X[i], j)
                score[i][j] = np.dot(self.coef_, output.T)
        return score



    def predict(self, X):
        '''
        Predict the class with the highest score.
        @param X: array-like, shape = [n_samples, n_inFeatures], input data to predict
        @return array-like, shape = [n_samples,], class labels predicted for each data point
        '''

        # Your code goes here and replaces following return statement
        prediction_score = self.decision_function(X)
        return np.argmax(prediction_score, axis=1)



#the following code tests the MulticlassSVM and sgd
#will fail if MulticlassSVM is not implemented yet
est = MulticlassSVM(6,lam=0.1)
est.fit(X,y)
print("w:")
print(est.coef_)
Z = est.predict(mesh_input)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)






