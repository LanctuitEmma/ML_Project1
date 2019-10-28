Here is the ReadMe which allows you to run our project and obtain the results we had in our rapport.

### In helpers

   Methods for loading/submitting the data:

	- load_csv_data(data_path, sub_sample=False): Loads data and returns y (class labels), tX (features) and ids (event ids)
	- predict_labels(weights, data): Generates class predictions given weights, and a test data matrix
	- create_csv_submission(ids, y_pred, name): Creates an output file in csv format for submission to kaggle

   Methods for pre-processing the dataset:

	- standardize(x): Standardize the original data set such that outliers don't have huge impacts, and such that the mean per feature i 0, and the standard deviation per feature is 1.
 	- replace_mean(x): The -999.0 values are replaced by the mean of the feature they are in.
	- subgrouping(x, ids, dict_): Generates a list of 4 groups, where each group corresponds to the value of the PRI_jet_num feature
	- group(l,ids,dict): Create a single dataset with the subgroups it is given.
	- select_non_nan_columns(x): Generates a dataset of specified features.

   Methods for constructing datasets:

	- batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): Generates a minibatch iterator for a dataset.
	- build_poly(x, degree): Returns augmented features vectors
	- build_k_indices(y, k_fold, seed): build k indices for k-fold.
	- visualization(lambds, mse_tr, mse_te): visualization the curves of mse_tr and mse_te.

### In implementations

   Methods for Least Squares and its variant:

	- compute_mse(y, tx, w): compute the MSE given the output y, the data tx and weights w.
	- compute_gradient(y, tx, w): compute a gradient normally given output y, the data tx and weights w.
	- least_squares(y,tx): compute the Least Square solution given data matrix tx and output vector y.
	- least_squares_GD(y, tx, initial_w, max_iters, gamma): compute the Least Square Gradient Descent solution given data matrix tx and output vector y, a specific value for the maximum iterations, and a lerning rate gamma value.
	- least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma): compute the Least Square Stochastic Gradient Descent solution given data matrix tx and output vector y, using a specific batch_size value, a number of maximum iterations and a learning rate gamma value.

   Methods for Ridge Regression:

	- ridge_regression(y, tx, lambda_): compute the Ridge Regression solution given data matrix tx and output vector y, with particular value lambda_.

   Methods for Logistic Regression and its variants:

	- sigmoid(t): An activation function which maps our values in the range [0,1].
	- calculate_loss_lr(y, tx, w): computes the cost by negative log likelihood.
	- compute_loss_lrr(y, tx, w): computes the cost by negative log likelihood.
	- compute_gradient_lr(y, tx, w): compute the gradient for the logistic regression
	- logistic_regression(y, tx, initial_w, max_iters, gamma):  computes the optimal weights w using logistic regression. We give matrix of training data tx, output labels y, an initial weight vector initial_w, a learning rate gamma and a number of maximum iterations.
	- reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): computes the optimal weights w using regularized logistic regression. We give matrix of training data tx, output labels y, a regularizer value lambda_ which will cancel the weights that overfit the data, an initial weight vector initial_w, a learning rate gamma and a number of maximum iterations.

### In cross_validation

   Methods for Least Squares and its variant:
	- cross_validation_ls(y, x, k_indices, k): returns the loss of least squares and the optimal weight, with k-1 folds as training set, and one as test set.
	- least_squares_demo(y, x, k): computes the optimal weights and least loss
	- cross_validation_ls_GD(y, x, k_indices, k, gamma, max_iters, w_initial):
	- LS_GD_demo(y, x, k): 
	- cross_validation_ls_SGD(y, x, k_indices, k, gamma, max_iters, w_initial, batch_size):
	- LS_SGD_demo(y, x, k):

   Methods for Ridge Regression:

	- cross_validation_rr(y, x, k_indices, k, lambda_, degree):
	- ridge_regression_demo(y, x, degree, k_fold):

   Methods for Logistic Regression and its variants:

	- cross_validation_lr(y, x, k_indices, k, gamma, max_iters, w_initial):
	- lr_demo(y, x, k):
	- cross_validation_lrr(y, x, k_indices, k, lambda_, gamma, max_iters, w_initial):
	- lrr_demo(y, x, k):

### The steps to obtain our best score
