import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    num = y[i]
    correct = scores[num]
    scores = np.exp(scores)
    exp_sum = np.sum(scores)
    loss += np.log(exp_sum) - correct
    dW[:, y[i]] -= X[i]
    for j in range(W.shape[1]):
      dW[:,j] += (np.exp(scores[j]) / exp_sum) * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum( W*W )
  dW /= X.shape[0]
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W); #N*C
  scores -= np.max(scores, axis=1).reshape(scores.shape[0],-1)
  correct = scores[range(X.shape[0]), y]
  scores_sum = np.log(np.sum(np.exp(scores), axis=1))
  for i in xrange(num_train):
    dW += exp_scores[i] * X[i][:,np.newaxis]
    dW[:, y[i]] -= X[i]
  loss = np.sum(scores_sum) - np.sum(correct)
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum( W*W )

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

