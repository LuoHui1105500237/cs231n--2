import numpy as np
from random import shuffle


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
  class_size = W.shape[1]
  scores1 = np.dot(X,W)
  for i in range(num_train):
        scores = X[i].dot(W)
        scores = np.exp(scores)
        scores = normalized1(scores)
        for j in range(class_size):
            if j==y[i]:
                margin = -np.log(scores[y[i]])
                scores[j]=scores[j]-1
            if margin > 0:
                #a=(1-margin)margin
                loss+= margin
        scores1[i]=scores
  dW += np.dot(X.T,scores1)
  loss /= num_train
  dW /= num_train
  dW += reg*W
            
                
        
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = X.dot(W)
  scores = np.exp(scores)
  class_size = W.shape[1]
  scores = normalized(scores)
  margins = -np.log(scores)
  ve_sum = np.sum (margins,axis=1)/class_size
  y_true = np.zeros_like(margins)
  y_true[range(num_train),y] =1.0
  print('y_true',y_true)
  loss += np.sum(ve_sum)/num_train
  dW += np.dot(X.T,scores-y_true)/num_train    #i=j aj-1, i!=j,ai =>score-y_true <= >dZ
    
    
    
    
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
def normalized(a):
    sum_scores = np.sum(a,axis=1)
    sum_scores = 1/sum_scores
    result = a.T*sum_scores.T
    return (result.T)
def normalized1 (a):
    sum_scores = np.sum(a)
    sum_scores = 1/sum_scores
    result = a.T*sum_scores.T
    return (result.T)
    
