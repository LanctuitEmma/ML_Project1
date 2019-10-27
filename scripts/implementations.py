# -*- coding: utf-8 -*-
"""All function should return (w,loss)."""
import csv
import numpy as np
import matplotlib.pyplot as plt
import helpers as helpers

# ************************************************** least squares and variant **************************************************

def compute_mse(y, tx, w):
    """return the mean square error"""
    e=y-tx.dot(w)
    return (e**2).mean()

def compute_gradient(y, tx, w):
    """return the gradient of the MSE"""
    length = len(y)
    return - 1 / length * tx.T.dot(y - tx.dot(w))

def least_squares(y,tx):
    """calculate the least squares solution using normal equation."""
    opt_w = np.linalg.lstsq(tx.T@tx,tx.T@y)[0]
    return (opt_w, compute_mse(y,tx,opt_w))

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """calculate the least squares solution using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma * grad
    
    mse = compute_mse(y, tx, w)
    return (w, compute_mse(y, tx, w))

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """calculate the least squares solution using stochastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in helpers.batch_iter(y, tx, batch_size,1):
            grad = compute_gradient(y,tx,w)
            w = w - gamma*grad
    return (w, compute_mse(y,tx,w))

# ************************************************** ridge regression **************************************************

def ridge_regression(y, tx, lambda_):
    """Returns ridgre regression parameters and mse loss"""
    length = y.shape[0]
    lambda_p = 2 * length * lambda_
    w_rr = np.linalg.inv(tx.T.dot(tx) + lambda_p * np.eye(tx.shape[1])).dot(tx.T).dot(y)
    return (w_rr, compute_mse(y, tx, w_rr))

# ************************************************** logistic regression **************************************************

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))

def calculate_loss_lr(y, tx, w):
    """compute the logistic loss by negative log likelihood."""
    o = np.array([sigmoid(xi) for xi in tx @ w])
    #we added 1e-5 to our sigmoid so we don't get a log(0)
    log = y.T@np.log(o+1e-5)+(1-y.T)@np.log(1-o+1e-5)
    del o
    return -log.mean()

def compute_loss_lrr(y, tx, w):
    """compute the cost by negative log likelihood."""
    o = np.array([sigmoid(xi) for xi in tx @ w])
    #we added 1e-5 to our sigmoid so we don't get a log(0)
    log = y.T@np.log(o+1e-5)+(1-y.T)@np.log(1-o+1e-5)
    del o
    return -log.mean()

def compute_gradient_lr(y, tx, w):
    """compute the gradient for logistic regression"""
    o = np.array([sigmoid(xi) for xi in tx @ w])
    return tx.T @ (o - y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """returns loss and parameters for logistic regression"""
    threshold = 1e-8
    losses = []
    w = initial_w
    
    if np.any(np.isnan(tx)):
        print("logistic: training poisoned with nans")
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_loss_lr(y, tx ,w)
        gradient = compute_gradient_lr(y, tx, w)
        w = w - gamma * gradient
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """returns loss and parameters for regularized logistic regression"""
    losses = []
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        loss = compute_loss_lrr(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
        gradient = 2 * compute_gradient_lr(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    return w, loss
