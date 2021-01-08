from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    train_nums = np.arange(X.shape[0])

    # X shape = (500, 3073) = (amount of training, the flat pixels)
    # W shape = (3073, 10) = (the flat pixels, the class nums)
    # X @ W = (500, 10) = (amount of training, the class nums)
    # Simple linear computation. Weights times the Training
    S =  X.dot(W)

    # We do this to so that the maximum number is zero
    # and therefore the numbers in case they are large
    # enough won't overflow from the expotentiation.
    # This is element-wise subtraction because we want
    # to subtract from every single element in this matrix
    S -= np.max(S)

    # expotentiation the scores. Pretty straight forward.
    # E.shape = (500, 10)
    E = np.exp(S)

    # sum up the expotentiation.
    # axis=1 will make it sum up the numbers in the rows
    # or the numbers cooresponding to each class in that matrix.
    # basically, sum up the 10 numbers 500 times (for each training example)
    # keepdims=True will make sure that G will retain the shape
    # but instead of 10 numbers, it will be one, the sum
    # G.shape = (500, 1)
    G = np.sum(E, axis=1, keepdims=True)

    # This is some meat of the softmax function.
    # Divide the expotentiation by the sum of the expotentiation.
    # Which is in other words, normalizing the values
    # P.shape = (500, 10)
    P = E / G

    # This is the final part of softmax where we have to sum
    # up the negative logs of all the normalized values to
    # get the loss. We want to basically select, from 0 to 499,
    # all the training examples in P. Since y is a list of the
    # actual labels, we want to select the item pertaining to
    # the actual label. So this is that.
    # We sum up all of that and do not keep dimensions because
    # the total loss is a scalar value.
    loss = np.sum(-np.log(P[train_nums, y]))

    # loss = (softmax(X, W, Y) or svm(X, W, Y)) / N + (l * L2(W))
    # l = some regularization scalar
    # L2 = just the sum of the squares => sum(W**2)

    # We divide by the total amount of training examples
    # to get the average loss which is part of the loss function.
    loss /= X.shape[0]

    # regularization scaler * the L2 distance function
    loss += reg * np.sum(np.square(W))

    # TODO WHYYY??
    # G.shape = (500, 1) <- sum of the expotention
    G[train_nums, y] -= 1

    # TODO WHYYY??
    # X shape = (500, 3073)
    dW = X.T.dot(G)

    # TODO WHYY???
    dW /= X.shape[0]

    # TODO WHY???
    # W shape = (3073, 10)
    dW += reg * 2 * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
