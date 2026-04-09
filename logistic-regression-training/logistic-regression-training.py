import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    W = np.zeros((X.shape[1], 1))
    b = 0.0

    for _ in range(steps):
        pi = _sigmoid(X @ W + b)
        W_grad = np.mean((pi - y) * X, axis=0).reshape(-1, 1)
        b_grad = np.mean(pi - y)
        W -= lr * W_grad
        b -= lr * b_grad
        
    # Flatten W to 1D and convert b to a standard Python float
    return W.flatten(), float(b)
