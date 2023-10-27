from gradient_descent import *
from stochastic_gradient_descent import *
from costs import *
from helpers import batch_iter
import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    w = initial_w
    for n_iter in range(max_iters):
        for b_y, b_tx in batch_iter(y, tx, batch_size=1):
            gradient = compute_stoch_gradient(b_y, b_tx, w)
            w = w - gamma * gradient
    loss = compute_loss(b_y, b_tx, w)
    return w, loss


def least_squares(y, tx):
    """
    To do : Add docstring
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
 
    if np.linalg.cond(a) > 10e15: 
        w = np.linalg.pinv(a).dot(b)
        loss = compute_loss(y, tx, w)
        return w, loss
        
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    To do : Add docstring
    """
    N, D = tx.shape
    lambda_ = 2 * N * lambda_

    a = tx.T.dot(tx) + lambda_ * np.eye(D)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    w = initial_w
    for i in range(max_iters):
        grad = log_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_log_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    w = initial_w
    thres = 1e-8
    previous_loss = 0.0  # update previous loss
    loss = compute_log_loss(y, tx, w)
    for i in range(max_iters):
        grad = log_gradient(y, tx, w) + [2.0 * lambda_ * x for x in w]
        w = w - gamma * grad
        
        loss = compute_log_loss(y, tx, w)
        if np.abs(loss - previous_loss) < thres:
            break
        
        previous_loss = loss  # update previous loss    
    
    return w, loss
