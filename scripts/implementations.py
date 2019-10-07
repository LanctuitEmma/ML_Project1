# -*- coding: utf-8 -*-
"""All function should return (w,loss)."""
import csv
import numpy as np

def compute_mse(y, tx, w):
    e=np.subtract(y,tx.dot(w.T))
    return (e**2).mean()

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y- tx.dot(w)
    coef = -1/tx.shape[0]
    return coef* (tx.T @ e)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.nanmean(x)
    x = x - mean_x
    std_x = np.nanstd(x)
    x = x / std_x
    return x, mean_x, std_x

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

            """****************************************************************************************"""

def least_squares_GD(y,tx, initial_w, max_iters, gamma):
    """calculate the least squares solution using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma*grad

    return (compute_mse(y,tx,w), w)

def least_squares_SGD(y,tx,initial_w, max_iters, gamma):
    """calculate the least squares solution using stochiastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,1):
            grad = compute_stoch_gradient(y,tx,w)
            w = w - gamma*grad
    return (compute_mse(y,tx,w), w)


def least_squares(y,tx):
    """calculate the least squares solution using normal equation."""
    opt_w = np.linalg.solve(tx.T@tx,tx.T@y)
    return (compute_mse(y,tx,opt_w), opt_w)
    
def ridge_regression(y,tx,lambda_):
    """calculate the ridge regression solution using normal equation."""
    l = (2*tx.shape[0]*lambda_)* np.identity(tx.shape[1])
    fact1 = tx.T@tx +l
    fact2 = tx.T@y
    opt_w = np.linalg.solve(fact1,fact2)
    "rmse = np.sqrt(2*compute_mse(y_train,x_train,opt_w))"
    return (compute_mse(y,tx,lambda_), opt_w)

def logistic_regression(y,tx,initial_w, max_iters, gamma):
    raise

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    """
   
    """
    raise