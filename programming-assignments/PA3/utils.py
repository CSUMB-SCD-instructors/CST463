#!/usr/bin/env python3

import numpy as np
from typing import Tuple, List, Dict

# Numerical stability constant (shared with student_code.py)
EPSILON = 1e-15


def initialize_network_weights(layer_sizes: List[int], method: str = 'xavier', seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Initialize weights and biases for multi-layer network

    Parameters
    ----------
    layer_sizes : List[int]
        List of layer sizes [input_size, hidden_size, ..., output_size]
    method : str
        Initialization method ('zeros', 'random', 'small_random', 'xavier', 'he')
    seed : int
        Random seed for reproducible initialization

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (W, b) tuples for each layer
    """
    # Set seed for reproducible results
    np.random.seed(seed)

    weights = []

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]
        n_out = layer_sizes[i + 1]

        if method == 'zeros':
            W = np.zeros((n_in, n_out))
        elif method == 'random':
            W = np.random.randn(n_in, n_out)
        elif method == 'small_random':
            # Small random initialization: weights ~ N(0, 0.01)
            W = 0.01 * np.random.randn(n_in, n_out)
        elif method == 'xavier':
            # Xavier initialization: weights ~ N(0, 1/n_in)
            W = np.random.randn(n_in, n_out) / np.sqrt(n_in)
        elif method == 'he':
            # He initialization: weights ~ N(0, 2/n_in)
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

        # Initialize biases to zero (common practice)
        b = np.zeros(n_out)

        weights.append((W, b))

    return weights


def generate_classification_data(n_samples: int, n_features: int = 2, n_classes: int = 2,
                               seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification dataset

    Parameters
    ----------
    n_samples : int
        Number of samples
    n_features : int
        Number of input features
    n_classes : int
        Number of classes
    seed : int
        Random seed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features and one-hot encoded labels
    """
    np.random.seed(seed)

    if n_classes == 2:
        # Binary classification: create two clusters
        samples_per_class = n_samples // 2

        # Class 0: cluster around (-1, -1)
        X0 = np.random.randn(samples_per_class, n_features) * 0.5 + np.array([-1, -1][:n_features])
        y0 = np.zeros((samples_per_class, 2))
        y0[:, 0] = 1  # One-hot: [1, 0]

        # Class 1: cluster around (1, 1)
        X1 = np.random.randn(n_samples - samples_per_class, n_features) * 0.5 + np.array([1, 1][:n_features])
        y1 = np.zeros((n_samples - samples_per_class, 2))
        y1[:, 1] = 1  # One-hot: [0, 1]

        X = np.vstack([X0, X1])
        y = np.vstack([y0, y1])
    else:
        # Multi-class: create multiple clusters
        samples_per_class = n_samples // n_classes
        X_list = []
        y_list = []

        for class_idx in range(n_classes):
            # Create cluster for this class
            n_samples_class = samples_per_class if class_idx < n_classes - 1 else n_samples - class_idx * samples_per_class

            # Place clusters in a circle
            angle = 2 * np.pi * class_idx / n_classes
            center = 2 * np.array([np.cos(angle), np.sin(angle)][:n_features])

            X_class = np.random.randn(n_samples_class, n_features) * 0.5 + center

            # One-hot encode labels
            y_class = np.zeros((n_samples_class, n_classes))
            y_class[:, class_idx] = 1

            X_list.append(X_class)
            y_list.append(y_class)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

    # Shuffle the data
    indices = np.random.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, y


def generate_nonlinear_data(n_samples: int = 200, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data requiring nonlinear decision boundary

    Parameters
    ----------
    n_samples : int
        Number of samples
    noise : float
        Noise level
    seed : int
        Random seed

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Features and binary labels
    """
    np.random.seed(seed)

    # Generate XOR-like dataset (requires nonlinear boundary)
    # Two features, binary classification
    X = np.random.uniform(-2, 2, size=(n_samples, 2))

    # XOR pattern: positive class when x1 and x2 have the same sign
    y_logic = (X[:, 0] > 0) == (X[:, 1] > 0)  # XOR pattern

    # Add noise to make it more realistic
    noise_mask = np.random.rand(n_samples) < noise
    y_logic = y_logic ^ noise_mask  # Flip labels for noise samples

    # Convert to integers (0, 1)
    y = y_logic.astype(int)

    return X, y


def predict_network(X: np.ndarray, weights: Dict, network_type: str = 'single') -> np.ndarray:
    """
    Make predictions with trained network

    Parameters
    ----------
    X : np.ndarray
        Input data
    weights : Dict
        Trained network weights
    network_type : str
        Type of network

    Returns
    -------
    np.ndarray
        Network predictions
    """
    # Import here to avoid circular imports
    from student_code import single_layer_forward

    if network_type == 'single':
        # Single layer prediction
        W = weights['W']
        b = weights['b']
        u, v = single_layer_forward(X, W, b, 'sigmoid')  # Default to sigmoid
        return v

    elif network_type == 'two_layer':
        # Two layer prediction - not implemented in simplified version
        raise NotImplementedError("Two-layer prediction moved to bonus/advanced section")

    else:
        raise ValueError(f"Unknown network type: {network_type}")


def full_gradient_check(X: np.ndarray, y: np.ndarray, weights: Dict, network_type: str = 'single',
                       epsilon: float = 1e-7) -> Dict:
    """
    Complete gradient checking for debugging (provided for students)

    This is the full implementation that students can use to verify their work,
    but they don't need to implement it themselves.
    """
    # Import here to avoid circular imports
    from student_code import (single_layer_forward, single_layer_backward,
                             two_layer_forward, two_layer_backward, mse_loss, mse_derivative)

    def compute_loss(w_dict):
        """Helper function to compute loss for given weights"""
        if network_type == 'single':
            u, v = single_layer_forward(X, w_dict['W'], w_dict['b'], 'sigmoid')
            return mse_loss(y, v)
        elif network_type == 'two_layer':
            cache = two_layer_forward(X, w_dict['W1'], w_dict['b1'], w_dict['W2'], w_dict['b2'])
            return mse_loss(y, cache['v2'])

    # Compute analytical gradients
    if network_type == 'single':
        u, v = single_layer_forward(X, weights['W'], weights['b'], 'sigmoid')
        dL_dv = mse_derivative(y, v)
        dL_dW, dL_db, _ = single_layer_backward(dL_dv, u, X, weights['W'], 'sigmoid')
        analytical_grads = {'W': dL_dW, 'b': dL_db}
    elif network_type == 'two_layer':
        cache = two_layer_forward(X, weights['W1'], weights['b1'], weights['W2'], weights['b2'])
        grads = two_layer_backward(cache, y, 'mse')
        analytical_grads = grads

    # Compute numerical gradients
    numerical_grads = {}
    for param_name in analytical_grads:
        param = weights[param_name]
        num_grad = np.zeros_like(param)

        # Use flat indexing for easier iteration
        flat_param = param.flatten()
        flat_num_grad = num_grad.flatten()

        for i in range(len(flat_param)):
            # Create perturbed weight copies
            weights_plus = {k: v.copy() for k, v in weights.items()}
            weights_minus = {k: v.copy() for k, v in weights.items()}

            # Perturb the parameter
            weights_plus[param_name].flat[i] += epsilon
            weights_minus[param_name].flat[i] -= epsilon

            # Compute losses
            loss_plus = compute_loss(weights_plus)
            loss_minus = compute_loss(weights_minus)

            # Numerical gradient
            flat_num_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical_grads[param_name] = num_grad

    # Compare gradients
    comparison = {}
    for param_name in analytical_grads:
        analytical = analytical_grads[param_name]
        numerical = numerical_grads[param_name]

        # Compute relative error
        diff = np.abs(analytical - numerical)
        rel_error = diff / (np.abs(numerical) + 1e-8)

        comparison[param_name] = {
            'analytical': analytical,
            'numerical': numerical,
            'max_diff': np.max(diff),
            'max_rel_error': np.max(rel_error),
            'passed': np.max(rel_error) < 1e-5
        }

    return comparison