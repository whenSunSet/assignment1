import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *

class LinearClassifier(object):

  def __init__(self):
    self.W = None

  def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=256, verbose=False):
    num_train, dim = X.shape
    num_classes = np.max(y) + 1
    if self.W is None:
      self.W = 0.001 * np.random.randn(dim, num_classes)

    loss_history = []
    for it in range(num_iters):

      batch_idx = np.random.choice(num_train , batch_size , replace=True)
      X_batch = X[batch_idx]
      y_batch = y[batch_idx]

      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

      self.W += -1 * learning_rate * grad
      if verbose and it % 100 == 0:
        print( 'iteration %d / %d: loss %f' % (it, num_iters, loss))

    return loss_history

  def predict(self, X):
    y_pred = np.zeros(X.shape[1])
    scores = X.dot(self.W)
    y_pred = np.argmax(scores, axis = 1)

    return y_pred
  
  def loss(self, X_batch, y_batch, reg):
    """
    Compute the loss function and its derivative. 
    Subclasses will override this.

    Inputs:
    - X_batch: A numpy array of shape (N, D) containing a minibatch of N
      data points; each point has dimension D.
    - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
    - reg: (float) regularization strength.

    Returns: A tuple containing:
    - loss as a single float
    - gradient with respect to self.W; an array of the same shape as W
    """
    pass


class LinearSVM(LinearClassifier):
  """ A subclass that uses the Multiclass SVM loss function """

  def loss(self, X_batch, y_batch, reg):
    return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
  """ A subclass that uses the Softmax + Cross-entropy loss function """

  def loss(self, X_batch, y_batch, reg):
    return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

