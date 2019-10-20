How to use the functions in our project and obtain the result we have in the end.


1) Cost functions:

    
	- sigmoid(t): map our values in the range [0,1], just give it the value\values you want to be mapped.
    
	- compute_loss_MSE(y, tx, w): compute the MSE given the output y, the data tx and weights w.
    
	- compute_loss_MAE(y, tx, w): compute the MAE given the output y, the data tx and weights w.
    
	- compute_loss_log(y, tx, w): compute the log-loos given the ouput y, the data tx and the weights w
    


2) Gradients and subgradients:

    
	- compute_gradient(y,tx,w): compute a gradient normally given output y, the data tx and weights w.
    
	- compute_subgradient(y, tx, w): compute a sub-gradient given output y, the data tx and weights w. We use it when we worked with a non-continuous cost function as the MAE.
    
	- compute_gradient_log(y, tx, w): compute a gradient for the log-loss cost function given output y, the data tx and weights w.
    


3) Data processing and feature engineering:

    
	- standardize(x): standardize each column of our data matrix, return also the mean and the standard deviation fro each column so that we can use to standardize the test data.

	- standardize_test(groupped,mu,sigma): given our test groups and the mean and standard deviation for each of our train group, standardize the test data according to our input data
					       to stay consistent.
	- build_model_data(y,x): given the data x and the ouput y, it returns a linear model for our data in matrix form. We add the regularization constant in the matrix with a column
				 of one.
 
 	- batch_iter(y, tx, batch_size, num_batches=1, shuffle=True): Generate a minibatch iterator for a dataset (function from the labs). It allow us to use the Least Square Stochastic 
								      Gradient Descent.
   
	- build_poly(x, degree): given a data matrix x and a degree, computes a polynomial model for our data of the degree you desire. It works on each column of our input matrix
    
	- add_features_function(x): given an input matrix x, add new columns to our matrix using differents function as: cubic root, logarithm where we take the absolute value to avoid 
				    negative and null value in the log, square root where we take the absolute value to avoid negative value, inverse of abs(x) + 1 to avoid having 0 at 
			 	    the denominator, sinus, cosinus and exponetial.
 	- add_features(x):
 given a data matrix, add new features by multiplying columns with each other.   
	- replace_nan_values(x): given a data matrix, returns the same matrix where false value are replaced by the real mean of the column. to do this, for each column we remove the 
				 invalid values from it, then compute the mean of the shortened column and then replace in the original column, the false value by the mean.


	- subgroups(x,y,test=False): given a data matrix x and the output y, separate the data in subgroups according to the number of -999 in a line and the place where we can find them
				     ,ie, if we find a -999 in for data input x_1 in the column 3 and 15, it will place in an array containing only data points with the same properties.
	- expand_data(groupped,degree): given our groups and a degree, expand the data in the following order: 1) standardize the group, 2) build the poly of the given degree, 3) add the
					features to the group using add_features(x):, 4) expand the data with new functions using add_features_function(x), 5) add the column of one


4) Machine Learning algorithm:
    
	- least_squares(y, tx): compute the Least Square solution given data matrix tx and output vector y.
    
	- least_squares_GD(y, tx, initial_w,max_iters, gamma): compute the weights using gradient descent given datamatrix tx, output y, initial weights inititial_w, a learning rate gamma
							       and the number of iterations we want to do max_iters.
    
	- least_squares_SGD(y, tx, initial_w,max_iters, gamma): similar to least_squares_GD but this time we are working with minibatch of the full data tx.
    
	- ridge_regression(y, tx, lambda_): given data matrix tx, output y and a penalizer lamba_, compute the optimal weights by solving the normal equation.
    
	- logistic_regression(y, tx, initial_w,max_iters, gamma): compute the optimal weights w using logistic regressiong given data matrix tx, output y, an initial weights initial_w, 
								  a learning rate gamma and the maximum number of iterations we want to be done.
    
	- reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma): compute the optimal weights w using the regularized version of logistic regression where we add a 
										penalty for value to far away using the penalizer lambda_.
    


5) Main script : how we used our function to obtain the best score


	Our main work as follow:
		1) load the data x,y,ids
		2) replace the -1 in y by 0 so that we dont have problem of dividing by 0 or having negative value in some of our log
		3) create our groups of data using subgroups(x,t,test=False)
		4) standardize each of this groups and then expand them.
		5) afterwards we standardize one more time each group to avoid problem in the log loss function but because we are doing so, we need to re-add the column of one.
		6) Execute Regularized logistic regression for all the subsets
		7) load the test data x_test, y_test and ids_test
		8) create our groups of data using subgroups(x,t,test=True)
		9) standardize each of this groups using standardize_test(groupped,mu,sigma), with mu and sigma referring to the means and standard deviations of the training data,  
		and then expand them.
		10) Add the ones in front of each row in each subgroup.
		11) Predict each subgroup according to the corresponding model 
		12) Concatenate and place in the good order the predictions so that we can submit to Kaggle
		13) Create the CSV