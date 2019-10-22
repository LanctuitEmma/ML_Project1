# -*- coding: utf-8 -*-
"""All function should return (w,loss)."""
import csv
import numpy as np
import matplotlib.pyplot as plt

def compute_mse(y, tx, w):
    e=y-tx.dot(w)
    return (e**2).mean()

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    # w is (n_features)
    # e shape is n_rows
    coef = -1/tx.shape[0]
    return coef * (tx.T @ e)

def calculate_loss_logistic_reg(y, tx, w):
    """compute the cost by negative log likelihood."""
    o = sigmoid(tx@w)
    log = np.sum(y.T@np.log(o+1e-5)+(1-y.T)@np.log(1-o+1e-5))
    del o
    return -log

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1/(1+np.exp(-t))

def standardize(x):
    """Standardize the original data set."""
    num_valid_values = np.count_nonzero(~np.isnan(x), axis=0)
    valid_columns = num_valid_values > 0
    x_valid_cols = x[:, valid_columns]
    
    mean_x = np.nanmean(x_valid_cols, axis=0)
    std_x = np.nanstd(x_valid_cols, axis=0)
    x_norm = np.zeros(x.shape, dtype=x.dtype)
    x_norm[:,valid_columns] =  ( x_valid_cols - mean_x[None, :] ) / std_x[None, :]
    num_cols = x.shape[1]
    
    mean_x_ret = np.zeros(num_cols, dtype=x.dtype)
    mean_x_ret[valid_columns] = mean_x
    std_x_ret = np.zeros(num_cols, dtype=x.dtype)
    std_x_ret[valid_columns] = std_x    
    return x_norm, mean_x_ret, std_x_ret

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
def ridge_regression_s(y, tx, lambda_):
    """implement ridge regression."""
    length = y.shape[0]
    lambda_p = 2 * length * lambda_
    w_rr = np.linalg.inv(tx.T.dot(tx) + lambda_p * np.eye(tx.shape[1])).dot(tx.T).dot(y)
    return (w_rr, compute_mse(y, tx, w_rr))

def return_factors(x):
    """This function takes a number and returns its factors"""
    factors = []
    for i in range(2,10):
        if x % i == 0:
            factors.append(i)
    return factors

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_rr(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    x_tr_poly = build_poly(x_train, degree)
    x_te_poly = build_poly(x_test, degree)

    w, loss_tr = ridge_regression_s(y_train, x_tr_poly, lambda_)
    loss_te = compute_mse(y_test, x_te_poly, w)

    return loss_tr, loss_te

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation") 

def least_squares_GD(y,tx, initial_w, max_iters, gamma):
    """calculate the least squares solution using gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y,tx,w)
        w = w - gamma*grad
    mse = compute_mse(y,tx,w)
    return (compute_mse(y,tx,w), w)

def least_squares_SGD(y,tx,initial_w,batch_size, max_iters, gamma):
    """calculate the least squares solution using stochiastic gradient descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,1):
            grad = compute_gradient(y,tx,w)
            w = w - gamma*grad
    return (compute_mse(y,tx,w), w)

def least_squares(y,tx):
    """calculate the least squares solution using normal equation."""
    opt_w = np.linalg.lstsq(tx.T@tx,tx.T@y)[0]
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
    threshold = 1e-8
    losses = []
    w=initial_w
    
    if np.any(np.isnan(tx)):
        print("logistic: training poisoned with nans")
    #print(f"logistic: tx {np.mean(tx, axis=0)}{np.std(tx,axis=0)}")
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss = calculate_loss_logistic_reg(y,tx,w)
        gradient = compute_gradient(y,tx,w)
        #gradient /= np.linalg.norm(gradient)
        #print(f"logistic: g {np.linalg.norm(gradient)}")
        w = w-gamma*gradient
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss,w

def reg_logistic_regression(y,tx,lambda_,initial_w,max_iters,gamma):
    threshold = 1e-8
    losses = []
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iters):
        loss= calculate_loss_logistic_reg(y,tx,w)+lambda_*np.squeeze(w.T.dot(w))
        gradient= compute_gradient(y,tx,w)+ 2*lambda_*w
        w = w-gamma*gradient
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return loss,w