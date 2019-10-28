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
	- cross_validation_ls(y, x, k_indices, k): returns the loss of least squares and the optimal weights vector, while doing the cross-validation using k-1 folds as training set and one as test set.
	- least_squares_demo(y, x, k): computes the optimal weights using cross_validation_ls and prints the minimal loss and its corresponding accuracy.
	- cross_validation_ls_GD(y, x, k_indices, k, gamma, max_iters, w_initial): returns the loss of least squares gradient descent and the optimal weights vector, while doing the cross-validation using k-1 folds as training set and one as test set.
	- LS_GD_demo(y, x, k): computes the optimal weights using cross_validation_ls_GD and prints the minimal loss and its corresponding accuracy. It also prints the optimal value for the learning rate gamma.
	- cross_validation_ls_SGD(y, x, k_indices, k, gamma, max_iters, w_initial, batch_size): returns the loss of least squares stochastic gradient descent and the optimal weights vector, while doing the cross-validation using k-1 folds as training set and one as test set.
	- LS_SGD_demo(y, x, k): computes the optimal weights using cross_validation_ls_SGD and prints the minimal loss and its corresponding accuracy. It also prints the optimal value for the learning rate gamma and the parameter batch_size.

   Methods for Ridge Regression:

	- cross_validation_rr(y, x, k_indices, k, lambda_, degree): returns the loss of ridge regression and the optimal weights vector, while doing the cross-validation using k-1 folds as training set and one as test set.
	- ridge_regression_demo(y, x, degree, k_fold): computes the optimal weights using cross_validation_rr and prints the minimal loss and its corresponding accuracy. It also prints the optimal value for the regularizer lambda.

   Methods for Logistic Regression and its variants:

	- cross_validation_lr(y, x, k_indices, k, gamma, max_iters, w_initial): returns the loss of logistic regression and the optimal weights vector, while doing the cross-validation using k-1 folds as training set and one as test set.
	- lr_demo(y, x, k): computes the optimal weights using cross_validation_lr and prints the minimal loss and its corresponding accuracy. It also prints the optimal value for the learning rate gamma.
	- cross_validation_lrr(y, x, k_indices, k, lambda_, gamma, max_iters, w_initial): returns the loss of regularized logistic regression and the optimal weights vector, while doing the cross-validation using k-1 folds as training set and one as test set.
	- lrr_demo(y, x, k): computes the optimal weights using cross_validation_lrr and prints the minimal loss and its corresponding accuracy. It also prints the optimal value for the regularizer lambda and the learning rate gamma.

### The steps to obtain our best score

	1) Loading training data samples and its corresponding labels and IDs.
	2) Create subgroups according to the PRI_jet_num discrete feature value.
	3) Standardize each subgroup individually.
	4) Create a single dataset by grouping the subgroups together.
	5) Build an augmented feature dataset, where each feature is augmented until degree 4.
	6) Compute Ridge Regression on the training dataset with augmented features to compute the optimal lambda.
	7) Loading testing data samples and its corresponding labels and IDs.
	8) Apply steps 2) to 5) to our test dataset.
	9) Compute Ridge Regression on the testing dataset with augmented features using the optimal lambda computed at step 6).
	10) Predict testing labels using the optimal weights computed at step 9).
	11) Submit predicted outpul labels to kaggle.

