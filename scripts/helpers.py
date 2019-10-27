# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
import matplotlib.pyplot as plt





# ************************************************** loading and output data **************************************************
            
def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
            

# ************************************************** cleaning of the data **************************************************

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


# All the Nan (corresponding to unknown values) were replaced by the mean value of the feature it is in.
def replace_mean(x):
    x_nan = x.copy()
    x_nan[x_nan == -999.0] = np.nan
    means_cols = np.nanmean(x_nan, axis=1)
    is_nan = np.isnan(x_nan)
    for col in range(x_nan.shape[1]):
        x_nan[is_nan[:, col], col] = means_cols[col]
    return x_nan

# Subgrouping
def subgrouping(x, ids, dict_):
    x_0=x[x[:,dict_['PRI_jet_num']]==0]
    x_1=x[x[:,dict_['PRI_jet_num']]==1]
    x_2=x[x[:,dict_['PRI_jet_num']]==2]
    x_3=x[x[:,dict_['PRI_jet_num']]==3]
    x_0 = np.delete(x_0,dict_['PRI_jet_num'],1)
    x_1 = np.delete(x_1,dict_['PRI_jet_num'],1)
    x_2 = np.delete(x_2,dict_['PRI_jet_num'],1)
    x_3 = np.delete(x_3,dict_['PRI_jet_num'],1)
    x_list = [x_0, x_1, x_2, x_3]

    ids_0=ids[x[:,dict_['PRI_jet_num']]==0]
    ids_1=ids[x[:,dict_['PRI_jet_num']]==1]
    ids_2=ids[x[:,dict_['PRI_jet_num']]==2]
    ids_3=ids[x[:,dict_['PRI_jet_num']]==3]
    ids_list = [ids_0]
    ids_list.append(ids_1)
    ids_list.append(ids_2)
    ids_list.append(ids_3)
    

    #Standardization of subgroups
    mean = []
    std = []
    x_nan_replaced = []
    for i in range(4):
        x_arr, m, s = standardize(x_list[i])
        x_nan_replaced.append(replace_mean(x_arr))
        mean.append(m)
        std.append(s)
    return x_nan_replaced, ids_list
    
# Grouping them back again
#Grouping them back again
def group(l,ids,dict):
    ls = l.copy()
    for i in range(4):
        ls[i] = np.insert(ls[i],dict['PRI_jet_num'],i+1,axis=1)
    data_ord = np.insert(ls[0],0,ids[0], axis=1)
    for i in range(1,4):
        a = np.insert(ls[i],0,ids[i], axis=1)
        data_ord = np.concatenate((data_ord, a))
    x_new = data_ord[data_ord[:,0].argsort()]
    x_new = x_new[:,1:]
    return x_new

# column indices to be kept
selected_columns0 = [1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29]
selected_columns1 = [1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29]
selected_columns_ideal = [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 29]

def select_non_nan_columns(x):
    return x[:, selected_columns0]



# ************************************************** model helpers **************************************************

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
            
def build_poly(x, degree):
    """
    returns augmented features vectors
    inputs:
    x -> samples vectors to be augmented
    degree -> degree of the augmentation
    comments:
    automatically add a 1 in front of each augmented vector 
    """
    nb_features = x.shape[1]
    nb_samples = x.shape[0]
    x_poly = np.ones((nb_samples, 1))
    for d in range(1, degree + 1):
        x_d = x**d
        x_poly = np.hstack((x_poly, x_d))
    return x_poly

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices) 

def visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")


