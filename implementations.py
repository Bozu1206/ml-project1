from gradient_descent import *
from stochastic_gradient_descent import *
from costs import *
from helpers import batch_iter

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma): 
    w = initial_w
    for n_iter in range(max_iters): 
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        print(f"GD {n_iter}/{max_iters-1}: loss = {loss:.2f}")
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma): 
    w = initial_w
    for n_iter in range(max_iters): 
        for b_y, b_tx in batch_iter(y, tx, batch_size=1):
            gradient = compute_stoch_gradient(b_y, b_tx, w)
            loss = compute_loss(b_y, b_tx, w)
            w = w - gamma * gradient
            print(f"GD {n_iter}/{max_iters-1}: loss = {loss:.2f}")
    return w, loss