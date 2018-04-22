import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):

  dW = np.zeros(W.shape)

  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j]+=X[i].T
        dW[:,y[i]] -= X[i].T

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  scores = X.dot(W)  # N by C
  num_train = X.shape[0]
  scores_correct = scores[np.arange(num_train), y]  # 1 by N
  scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1
  margins = scores - scores_correct + 1.0  # N by C
  margins[np.arange(num_train), y] = 0.0
  margins[margins <= 0] = 0.0
  loss += np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W * W)


  margins[margins > 0] = 1.0
  row_sum = np.sum(margins, axis=1)  # 1 by N
  margins[np.arange(num_train), y] = -row_sum
  dW += np.dot(X.T, margins) / num_train + reg * W  # D by C
  return loss, dW
