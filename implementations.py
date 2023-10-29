from gradient_descent import *
from stochastic_gradient_descent import *
from costs import *
from helpers import batch_iter
import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using Gradient Descent (GD).

    Args:
    y (numpy.ndarray): The output values (labels).
    tx (numpy.ndarray): The input values (features).
    initial_w (numpy.ndarray): The initial weights for the model.
    max_iters (int): The maximum number of iterations for GD to perform.
    gamma (float): The learning rate.

    Returns:
    tuple: A tuple containing the final weights and the loss.
    """
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using Stochastic Gradient Descent (SGD).

    Args:
    y (numpy.ndarray): The output values (labels).
    tx (numpy.ndarray): The input values (features).
    initial_w (numpy.ndarray): The initial weights for the model.
    max_iters (int): The maximum number of iterations for SGD to perform.
    gamma (float): The learning rate.

    Returns:
    tuple: A tuple containing the final weights and the loss.
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
    Solve the least squares problem using normal equations.

    Args:
    y (numpy.ndarray): The output values (labels).
    tx (numpy.ndarray): The input values (features).

    Returns:
    tuple: A tuple containing the final weights and the loss.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    # Pseudo-inverse if the matrix is ill-conditionned
    if np.linalg.cond(a) > 10e15:
        w = np.linalg.pinv(a).dot(b)
        loss = compute_loss(y, tx, w)
        return w, loss

    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Perform ridge regression using normal equations.

    Args:
    y (numpy.ndarray): The output values (labels).
    tx (numpy.ndarray): The input values (features).
    lambda_ (float): The regularization parameter.

    Returns:
    tuple: A tuple containing the final weights and the loss.
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
    Perform logistic regression using Gradient Descent.

    Args:
    y (numpy.ndarray): The output values (labels).
    tx (numpy.ndarray): The input values (features).
    initial_w (numpy.ndarray): The initial weights for the model.
    max_iters (int): The maximum number of iterations.
    gamma (float): The learning rate.

    Returns:
    tuple: A tuple containing the final weights and the loss.
    """
    w = initial_w
    for i in range(max_iters):
        grad = log_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_log_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using Gradient Descent method.

    Args:
    y (numpy.ndarray): The output values (labels).
    tx (numpy.ndarray): The input values (features).
    lambda_ (float): The regularization parameter.
    initial_w (numpy.ndarray): The initial weights for the model.
    max_iters (int): The maximum number of iterations.
    gamma (float): The learning rate.

    Returns:
    tuple: A tuple containing the final weights and the loss.
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

        previous_loss = loss

    return w, loss
