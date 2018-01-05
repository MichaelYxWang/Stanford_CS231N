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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    # Normalization trick: http://cs231n.github.io/linear-classify/#softmax
    scores -= np.max(scores)
    exp_correct_score = np.exp(scores[y[i]])
    exp_sum = np.sum(np.exp(scores))
    loss += -np.log(float(exp_correct_score)/float(exp_sum))
    ####################################################################################################
    # Why we need gradient for both correct and incorrect rows?
    # Because the Loss function for each training example looks like:
    # Li = -syi + log(ej1 + ej2 + ej3 + ....)
    # sj and syi are all SCORES, which means they are RELATED to W
    # Gradient update for incorrect rows can be computed using chain rule:
    # which is: (1/sum)*exp(WX)*X
    # But gradient update for correct rows depends also on -syi, so we substract it for correct rows
    ###################################################################################################
    for k in range(num_classes):
      if(k != y[i]):
        dW[:,k] += (np.exp(scores[k])/exp_sum)*X[i]
      else:
        dW[:,k] += (np.exp(scores[k])/exp_sum)*X[i] - X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W*reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  # Normalization trick: http://cs231n.github.io/linear-classify/#softmax
  scores -= np.max(scores,axis=1,keepdims=True)
  exp_correct_score = np.exp(scores[range(num_train),y])
  exp_sum = np.sum(np.exp(scores),axis=1,keepdims=True)
  loss = np.sum(-np.log(np.divide(exp_correct_score[:,np.newaxis],exp_sum)))


  mask = np.zeros((np.exp(scores)/exp_sum).shape)
  mask[range(num_train),y] = 1
  dW = X.T.dot(np.exp(scores)/exp_sum - mask)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*W*reg
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

