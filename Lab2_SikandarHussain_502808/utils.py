import numpy as np


def standardize_loop(X):
    n_samples, n_features = X.shape
    X_std = np.zeros_like(X)
    
    for j in range(n_features):
        mean_j = 0.0
        for i in range(n_samples):
            mean_j += X[i, j]
        mean_j /= n_samples
        
        std_j = 0.0
        for i in range(n_samples):
            std_j += (X[i, j] - mean_j) ** 2
        std_j = np.sqrt(std_j / n_samples)
        
        for i in range(n_samples):
            X_std[i, j] = (X[i, j] - mean_j) / (std_j + 1e-8)
    
    return X_std


def standardize_vectorized(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / (std + 1e-8)


def polynomial_features(X, degree=2):
    n_samples, n_features = X.shape
    
    if degree == 1:
        return X
    
    poly_features = [X]
    poly_features.append(X ** 2)
    
    if n_features > 1 and degree >= 2:
        interactions = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))
        if interactions:
            poly_features.append(np.hstack(interactions))
    
    return np.hstack(poly_features)


def add_bias(X):
    n_samples = X.shape[0]
    bias_column = np.ones((n_samples, 1))
    return np.hstack([bias_column, X])


def batch_linear_forward(X, W):
    return X @ W

