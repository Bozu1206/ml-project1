import numpy as np
import itertools
import os
import helpers
import implementations as imp
import costs
import polynomial_exp 

class GradientFitter: 
    ## Here we supposed that data is already cleaned 
    def __init__(self, y, tx, max_iters, gamma, ratio): 
        self.y = y 
        self.tx = tx
        self.max_iters = max_iters
        self.gamma = gamma
        self.ratio = ratio
        
    def __train_and_validate(self): 
        # Split data
        y_tr, x_tr, y_test, x_test = helpers.split_data_rand(self.y, self.tx, self.ratio)
    
        # Train a GD Linear regression predictor
        # TODO: Ask 
        initial_w = [1] * x_tr.shape[1]
        w, loss = imp.mean_squared_error_gd(y_tr, x_tr, initial_w, self.max_iters, self.gamma)
        
        # Test error 
        test_loss = costs.compute_mse(y_test, x_test, w)
        
        # Prediction 
        y_pred_tr = helpers.predict_labels(w, x_tr)
        y_pred_te = helpers.predict_labels(w, x_test)
        
        # Training accuracy
        training_accuracy = helpers.compute_accuracy(y_tr, y_pred_tr)
        
        # Local Test accuracy
        local_test_acc = helpers.compute_accuracy(y_test, y_pred_te)
        
        # Todo: add f1-score
        
        print('Train-Validate: Training error={err}, Training accuracy={acc}'.format(err=loss, acc=training_accuracy))
        print('Train-Validate: Test error={err}, Test accuracy={acc}'.format(err=test_loss, acc=training_accuracy))
        return w, test_loss, (training_accuracy, local_test_acc) 
    
    def fit(self): 
        return self.__train_and_validate()


class StochasticGradientFitter: 
    ## Here we supposed that data is already cleaned 
    def __init__(self, y, tx, max_iters, gamma, ratio): 
        self.y = y 
        self.tx = tx
        self.max_iters = max_iters
        self.gamma = gamma
        self.ratio = ratio
        
    def __train_and_validate(self): 
        # Split data
        y_tr, x_tr, y_test, x_test = helpers.split_data_rand(self.y, self.tx, self.ratio)
    
        # Train a GD Linear regression predictor
        # TODO: Ask 
        initial_w = [1] * x_tr.shape[1]
        w, loss = imp.mean_squared_error_sgd(y_tr, x_tr, initial_w, self.max_iters, self.gamma)
        
        # Test error 
        test_loss = costs.compute_mse(y_test, x_test, w)
        
        # Prediction 
        y_pred_tr = helpers.predict_labels(w, x_tr)
        y_pred_te = helpers.predict_labels(w, x_test)
        
        # Training accuracy
        training_accuracy = helpers.compute_accuracy(y_tr, y_pred_tr)
        
        # Local Test accuracy
        local_test_acc = helpers.compute_accuracy(y_test, y_pred_te)
        
        # Todo: add f1-score
        
        print('Train-Validate: Training error={err}, Training accuracy={acc}'.format(err=loss, acc=training_accuracy))
        print('Train-Validate: Test error={err}, Test accuracy={acc}'.format(err=test_loss, acc=training_accuracy))
        return w, test_loss, (training_accuracy, local_test_acc) 
    
    def fit(self): 
        return self.__train_and_validate()

class LeastSquareFitter: 
    ## Here we supposed that data is already cleaned 
    def __init__(self, y, tx, ratio): 
        self.y = y 
        self.tx = tx
        self.ratio = ratio
        
    def __train_and_validate(self): 
        # Split data
        y_tr, x_tr, y_test, x_test = helpers.split_data_rand(self.y, self.tx, self.ratio)
    
        # Train a GD Linear regression predictor
        # TODO: Ask 
        w, loss = imp.least_squares(y_tr, x_tr)
        
        # Test error 
        test_loss = costs.compute_mse(y_test, x_test, w)
        
        # Prediction 
        y_pred_tr = helpers.predict_labels(w, x_tr)
        y_pred_te = helpers.predict_labels(w, x_test)
        
        # Training accuracy
        training_accuracy = helpers.compute_accuracy(y_tr, y_pred_tr)
        
        # Local Test accuracy
        local_test_acc = helpers.compute_accuracy(y_test, y_pred_te)
        
        # Todo: add f1-score
        
        print('Train-Validate: Training error={err}, Training accuracy={acc}'.format(err=loss, acc=training_accuracy))
        print('Train-Validate: Test error={err}, Test accuracy={acc}'.format(err=test_loss, acc=training_accuracy))
        return w, test_loss, (training_accuracy, local_test_acc) 
    
    def fit(self): 
        return self.__train_and_validate()
    
class RidgeRegressionFitter: 
    ## Here we supposed that data is already cleaned 
    def __init__(self, y, tx, lambda_, ratio): 
        self.y = y 
        self.tx = tx
        self.ratio = ratio
        self.lambda_ = lambda_
        
    def __train_and_validate(self): 
        # Split data
        y_tr, x_tr, y_test, x_test = helpers.split_data_rand(self.y, self.tx, self.ratio)
    
        # Train a GD Linear regression predictor
        # TODO: Ask 
        w, loss = imp.ridge_regression(y_tr, x_tr, self.lambda_)
        
        # Test error 
        test_loss = costs.compute_rmse(y_test, x_test, w)
        
        # Prediction 
        y_pred_tr = helpers.predict_labels(w, x_tr)
        y_pred_te = helpers.predict_labels(w, x_test)
        
        # Training accuracy
        training_accuracy = helpers.compute_accuracy(y_tr, y_pred_tr)
        
        # Local Test accuracy
        local_test_acc = helpers.compute_accuracy(y_test, y_pred_te)
        
        # Todo: add f1-score
        
        print('Train-Validate: Training error={err}, Training accuracy={acc}'.format(err=loss, acc=training_accuracy))
        print('Train-Validate: Test error={err}, Test accuracy={acc}'.format(err=test_loss, acc=training_accuracy))
        return w, test_loss, (training_accuracy, local_test_acc) 
    
    def fit(self): 
        return self.__train_and_validate()
    
## TODO: Add logistic regression and reg logistic regression fitters