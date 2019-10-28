# -*- coding: utf-8 -*-
"""This file contains the functions that allowed us to find the best hyperparameters and test our models"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import implementations as imp
import helpers as helpers

# ************************************************** least squares **************************************************

def cross_validation_ls(y, x, k_indices, k):
    """train and test least square model using cross validation"""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    opt_w, mse_tr = imp.least_squares(y_train,x_train)
    mse_te = imp.compute_mse(y_test, x_test, opt_w)
    return mse_te, opt_w

def least_squares_demo(y, x, k):
    """return error for least square model"""
    seed = 1
    weights=[]
    mse_errors = []
    
    tx = helpers.build_poly(x, 1)

    # Initialization
    w_initial = np.zeros(tx.shape[1])

    # split data in k fold
    k_indices = helpers.build_k_indices(y, k, seed)
    
    for i in range(k):
            mse_te, opt_w = cross_validation_ls(y, tx, k_indices, i)
            mse_errors.append(mse_te)
            weights.append([opt_w])
    
    mse = np.min(mse_errors)
    opt_w = weights[np.argmin(mse_errors)]
    y_model = helpers.predict_labels(np.array(opt_w).T, tx)

    #Computing accuracy
    print("   mse={mse}".format(mse = mse))
    accuracy = (list(y_model.flatten() == y).count(True))/len(y_model)
    print("   accuracy={acc:.3f}".format(acc=accuracy))

# ************************************************** least squares GD **************************************************

def cross_validation_ls_GD(y, x, k_indices, k, gamma, max_iters, w_initial):
    """train and test least square GD model using cross validation"""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    opt_w, mse_tr = imp.least_squares_GD(y_train, x_train, w_initial, max_iters,gamma)
    mse_te = imp.compute_mse(y_test, x_test,opt_w)
    return mse_te, opt_w

def LS_GD_demo(y, x, k):
    """find best hyperparameters and return error for least square GD model"""
    seed=1
    max_iters = 50
    gammas = np.logspace(-3, 0, 10)
    
    tx = helpers.build_poly(x, 1)

    # Initialization
    w_initial = np.zeros(tx.shape[1])
    
    # split data in k fold
    k_indices = helpers.build_k_indices(y, k, seed)

    gen_opt_w = []
    gen_mse = []

    #gamma selection
    for gamma in gammas:
        weights=[]
        mse_errors = []
        for i in range(k):
            mse_te, opt_w = cross_validation_ls_GD(y, tx, k_indices, i, gamma,max_iters, w_initial)
            mse_errors.append(mse_te)
            weights.append([opt_w])
        
        gen_mse.append(np.mean(mse_errors))
        gen_opt_w.append(np.mean(weights, axis=0))
        
    del weights
    del mse_errors
    
    opt_gamma = gammas[np.nanargmin(gen_mse)]
    opt_w = gen_opt_w[np.nanargmin(gen_mse)]
    mse_LS_GD = np.nanmin(gen_mse)
    
    print("   gamma={l:.3f}, mse={mse:.3f}".format(mse = mse_LS_GD, l = opt_gamma))

    #Training Accuracy
    y_predicted = helpers.predict_labels(opt_w.T, tx)
    accuracy = (list(y == y_predicted.flatten()).count(True))/len(y)
    print("   accuracy={acc:.3f}".format(acc=accuracy))
    
# ************************************************** least squares SGD **************************************************

def cross_validation_ls_SGD(y, x, k_indices, k, gamma, max_iters, w_initial, batch_size):
    """train and test least square SGD model using cross validation"""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    opt_w, mse_tr = imp.least_squares_SGD(y_train, x_train, w_initial, batch_size, max_iters, gamma)
    mse_te = imp.compute_mse(y_test, x_test,opt_w)
    return mse_te, opt_w

def LS_SGD_demo(y, x, k):
    """find best hyperparameters and return error for least square SGD model"""

    #Adding constant term
    tx = helpers.build_poly(x, 1)

    seed = 1
    max_iters = 50
    gammas = np.logspace(-3, 0, 10)
    batch_sizes = np.array([1])
    
    # Initialization
    w_initial = np.zeros(tx.shape[1])
    
    # split data in k fold
    k_indices = helpers.build_k_indices(y, k, seed)

    temp_mse = []
    temp_opt_w = []
    
    hyperparams = [(batch_size,gamma) for batch_size in batch_sizes for gamma in gammas ]
    
    for batch_size, gamma in hyperparams:  
            mse_errors = []
            weights = []
            
            for i in range(k):
                mse_te, opt_w = cross_validation_ls_SGD(y, tx, k_indices, i, gamma, max_iters, w_initial, batch_size)
                mse_errors.append(mse_te)
                weights.append([opt_w])
    
            temp_mse.append(np.mean(mse_errors))
            temp_opt_w.append(np.mean(weights, axis=0))
    
    mse = np.min(temp_mse)
    hyper_opt= hyperparams[np.argmin(temp_mse)]
    print("   gamma={g:.3f}, batch={b:.2f}, mse={mse:.3f}".format(mse = mse, g = hyper_opt[1], b = hyper_opt[0]))

    opt_w = temp_opt_w[np.nanargmin(temp_mse)]

    #Training Accuracy
    y_predicted = helpers.predict_labels(opt_w.T, tx)
    accuracy = (list(y == y_predicted.flatten()).count(True))/len(y)
    print("   accuracy={acc:.3f}".format(acc = accuracy))
    
# ************************************************** ridge regression **************************************************
    
def cross_validation_rr(y, x, k_indices, k, lambda_, degree):
    """train and test ridge regression model using cross validation"""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    x_tr_poly = helpers.build_poly(x_train, degree)
    x_te_poly = helpers.build_poly(x_test, degree)

    w, loss_tr = imp.ridge_regression(y_train, x_tr_poly, lambda_)
    loss_te = imp.compute_mse(y_test, x_te_poly, w)
    
    return loss_tr, loss_te

def ridge_regression_demo(y, x, degree, k_fold):
    """find best hyperparameters and return error for ridge regression model"""
    seed = 1
    lambdas = np.logspace(-1.1, -0.8, 20)
    
    # split data in k fold
    k_indices = helpers.build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    
    # iterate over all the lambdas, compute model parameters, store the rmse
    for i in range(len(lambdas)):
        l = lambdas[i]
        avg_err_tr = 0
        avg_err_te = 0
        for k in range(k_fold):
            err = cross_validation_rr(y, x, k_indices, k, l, degree)
            avg_err_tr += err[0]
            avg_err_te += err[1]
        rmse_tr.append(np.sqrt(2 * avg_err_tr / k_fold))
        rmse_te.append(np.sqrt(2 * avg_err_te / k_fold))
    helpers.visualization(lambdas, rmse_tr, rmse_te)
    
    # find the best lambda
    min_err_index = 0
    for i in range(1, len(rmse_te)):
        if rmse_te[i] < rmse_te[min_err_index]:
            min_err_index = i
            
    lambda_opt = lambdas[min_err_index]
    
    x_poly = helpers.build_poly(x, degree)
    w_opt, mse = imp.ridge_regression(y, x_poly, lambda_opt)
    
    print("   lambda={l:.3f}, mse={mse:.3f}".format(mse = mse, l = lambda_opt))

    #Training Accuracy
    y_predicted = helpers.predict_labels(w_opt.T, x_poly)
    accuracy = (list(y == y_predicted.flatten()).count(True))/len(y)
    print("   accuracy={acc:.3f}".format(acc = accuracy))
    
    
    
    
    
    
    
# ************************************************** logistic regression **************************************************
    
    
def cross_validation_lr(y, x, k_indices, k, gamma, max_iters, w_initial):
    """train and test logistic regression model using cross validation"""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    opt_w, loss_tr = imp.logistic_regression(y_train, x_train, w_initial, max_iters, gamma)
    loss_te = imp.calculate_loss_lr(y_test, x_test, opt_w)
    return loss_te, opt_w
    
def lr_demo(y, x, k):
    """find best hyperparameters and return error for logistic regression model"""
    max_iters = 100
    gammas = np.logspace(-4, -3, 1)
    seed = 1
    
    # adding constant term
    tx = helpers.build_poly(x, 1)
    
    # Initialization
    w_initial = np.zeros(tx.shape[1])
    
    # split data in k fold
    k_indices = helpers.build_k_indices(y, k, seed)

    gen_opt_w = []
    gen_loss = []

    #gamma selection
    for gamma in gammas:
        weights=[]
        loss_errors = []
        
        for i in range(k):
            loss_te, opt_w = cross_validation_lr(y, tx, k_indices, i, gamma, max_iters, w_initial)
            loss_errors.append(loss_te)
            weights.append([opt_w])
    
        gen_loss.append(np.mean(loss_errors))
        gen_opt_w.append(np.mean(weights,axis=0))
    
    del weights
    del loss_errors
        
    opt_gamma = gammas[np.nanargmin(gen_loss)]
    opt_w = gen_opt_w[np.nanargmin(gen_loss)]
    print("   gamma={l:.3f},loss={loss:.3f}".format(loss = np.min(gen_loss), l = opt_gamma))

     #Training Accuracy
    y_predicted = helpers.predict_labels(opt_w.T, tx)
    accuracy = (list(y_predicted.flatten() == y).count(True))/len(y)
    print("   accuracy={acc:.3f}".format(acc = accuracy))
    
    del gen_opt_w
    del gen_loss
    
    
    
    
    
    
# ************************************************** regularized logistic regression **************************************************
    
    
def cross_validation_lrr(y, x, k_indices, k, lambda_, gamma, max_iters, w_initial):
    """train and test regularized logistic regression model using cross validation"""
    x_test = x[k_indices[k]]
    x_train = np.delete(x, [k_indices[k]], axis=0)
    y_test = y[k_indices[k]]
    y_train = np.delete(y, [k_indices[k]], axis=0)

    opt_w, loss = imp.reg_logistic_regression(y_train,x_train,lambda_,w_initial,max_iters,gamma)
    loss_te = imp.compute_loss_lrr(y_test, x_test,opt_w)
    return loss_te, opt_w


def lrr_demo(y, x, k):
    """find best hyperparameters and return error for regularized logistic regression model"""
    #Adding constant term
    tx = helpers.build_poly(x, 4)
    
    seed = 1
    max_iters = 50
    lambdas = np.logspace(-4, -3, 1)
    gammas = np.logspace(-4, -3, 1)
    hyperparams = [(gamma,lambda_) for gamma in gammas for lambda_ in lambdas]

    w_initial = np.zeros(tx.shape[1])
    
    # split data in k fold
    k_indices = helpers.build_k_indices(y, k, seed)

    result_loss =[]
    result_opt_w=[]
    for gamma,lambda_ in hyperparams:  
            loss_errors=[]
            weights=[]
            
            for i in range(k):
                loss_te, opt_w = cross_validation_lrr(y, tx, k_indices, i, lambda_, gamma, max_iters, w_initial)
                loss_errors.append(loss_te)
                weights.append([opt_w])
    
            result_loss.append(np.mean(loss_errors))
            result_opt_w.append(np.mean(weights,axis=0))

    
    del loss_errors
    del weights
    
    mse = np.min(result_loss)
    hyper_opt= hyperparams[np.argmin(result_loss)]
    print("   gamma={g:.3f}, mse={mse:.3f} lambda{l:.3f}".format(mse = mse, g=hyper_opt[0], l=hyper_opt[1]))

    opt_w = result_opt_w[np.argmin(result_loss)]
   
    #Training Accuracy
    y_predicted = helpers.predict_labels(opt_w.T, tx)
    accuracy = (list(y_predicted.flatten() == y).count(True))/len(y)
    print("   accuracy={acc:.3f}".format(acc=accuracy))
    
    del result_loss
    del result_opt_w
    
    
