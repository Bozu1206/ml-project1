from gradient_descent import *
from stochastic_gradient_descent import *
from costs import *
from helpers import batch_iter


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    w = initial_w
    loss = compute_loss(y, tx, w)
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        loss = compute_loss(y, tx, w)
        print(f"GD {n_iter}/{max_iters-1}: loss = {loss:.2f}")
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    w = initial_w
    loss = 0
    for n_iter in range(max_iters):
        for b_y, b_tx in batch_iter(y, tx, batch_size=1):
            gradient = compute_stoch_gradient(b_y, b_tx, w)
            w = w - gamma * gradient
            loss = compute_loss(b_y, b_tx, w)
            print(f"GD {n_iter}/{max_iters-1}: loss = {loss:.2f}")
    return w, loss


def least_squares(y, tx):
    """
    To do : Add docstring
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
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
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    To do : Add docstring
    """
    raise NotImplementedError
