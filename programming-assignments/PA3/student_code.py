#!/usr/bin/env python3

import numpy as np
from typing import Callable, Tuple, List, Dict, Union

# Numerical stability constant
EPSILON = 1e-15  # Small constant to prevent numerical instabilities (log(0), division by 0)


# ============================================================================
# FORWARD PASS FUNCTIONS
# ============================================================================

def linear_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute linear transformation: u = X @ W + b

    Following class notation: u represents pre-activation values

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix
    W : np.ndarray, shape (n_features, n_units)
        Weight matrix
    b : np.ndarray, shape (n_units,)
        Bias vector

    Returns
    -------
    np.ndarray, shape (n_samples, n_units)
        Pre-activation values (u in class notation)
    """
    return X @ W + b


def sigmoid_forward(u: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid activation: σ(u) = 1 / (1 + exp(-u))

    Following class notation: u → v (pre-activation → post-activation)

    Parameters
    ----------
    u : np.ndarray
        Pre-activation values

    Returns
    
    -------
    np.ndarray
        Sigmoid activation outputs (v in class notation)
    """
    # Clip input to prevent overflow/underflow
    u_clipped = np.clip(u, -500, 500)
    return 1 / (1 + np.exp(-u_clipped))


def relu_forward(u: np.ndarray) -> np.ndarray:
    """
    Compute ReLU activation: max(0, u)

    Following class notation: u → v (pre-activation → post-activation)

    Parameters
    ----------
    u : np.ndarray
        Pre-activation values

    Returns
    -------
    np.ndarray
        ReLU activation outputs (v in class notation)
    """
    return np.maximum(0, u)


def softmax_forward(u: np.ndarray) -> np.ndarray:
    """
    Compute softmax activation for multi-class classification

    Following class notation: u → v (pre-activation → post-activation)

    Parameters
    ----------
    u : np.ndarray, shape (n_samples, n_classes)
        Pre-activation values

    Returns
    -------
    np.ndarray, shape (n_samples, n_classes)
        Softmax probabilities (v in class notation, each row sums to 1)
    """
    # Subtract max for numerical stability
    u_stable = u - np.max(u, axis=1, keepdims=True)
    exp_u = np.exp(u_stable)
    return exp_u / np.sum(exp_u, axis=1, keepdims=True)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error loss

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    float
        MSE loss value
    """
    errors = y_true - y_pred
    return np.mean(errors ** 2)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = EPSILON) -> float:
    """
    Compute cross-entropy loss for classification

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True labels (one-hot encoded)
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities
    epsilon : float, optional
        Small value to prevent log(0). Default uses module constant EPSILON.

    Returns
    -------
    float
        Cross-entropy loss value
    """
    # Clip predictions to prevent log(0) which would give -inf
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)

    # Cross-entropy: -sum(y_true * log(y_pred)) averaged over samples
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))


# ============================================================================
# DERIVATIVE FUNCTIONS
# ============================================================================

def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute derivative of MSE loss with respect to predictions

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    np.ndarray
        Gradient of MSE w.r.t. predictions
    """
    # MSE = (1/n) * sum((y_pred - y_true)^2)
    # Derivative: (2/n) * (y_pred - y_true)
    n_samples = y_true.shape[0]
    return (2 / n_samples) * (y_pred - y_true)


def sigmoid_derivative(u: np.ndarray) -> np.ndarray:
    """
    Compute derivative of sigmoid function

    Following class notation: derivative w.r.t. u (pre-activation)

    Parameters
    ----------
    u : np.ndarray
        Pre-activation values (same u passed to sigmoid_forward)

    Returns
    -------
    np.ndarray
        Sigmoid derivative: σ(u) * (1 - σ(u)) = dv/du
    """
    # Compute sigmoid first
    sigmoid_u = sigmoid_forward(u)
    # Derivative: σ(u) * (1 - σ(u))
    return sigmoid_u * (1 - sigmoid_u)


def relu_derivative(u: np.ndarray) -> np.ndarray:
    """
    Compute derivative of ReLU function

    Following class notation: derivative w.r.t. u (pre-activation)

    Parameters
    ----------
    u : np.ndarray
        Pre-activation values (same u passed to relu_forward)

    Returns
    -------
    np.ndarray
        ReLU derivative: 1 if u > 0, else 0 = dv/du
    """
    # ReLU derivative: 1 where u > 0, 0 elsewhere
    return (u > 0).astype(np.float64)


# ============================================================================
# BACKPROPAGATION CORE FUNCTIONS
# ============================================================================

def linear_backward(dL_du: np.ndarray, X: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gradients through linear layer

    Following class notation: gradient flows backward from u to parameters

    Parameters
    ----------
    dL_du : np.ndarray, shape (n_samples, n_units)
        Gradient of loss w.r.t. pre-activation (u in class notation)
    X : np.ndarray, shape (n_samples, n_features)
        Input to this layer
    W : np.ndarray, shape (n_features, n_units)
        Weight matrix of this layer

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        dL_dW: Gradient w.r.t. weights, shape (n_features, n_units)
        dL_db: Gradient w.r.t. bias, shape (n_units,)
        dL_dX: Gradient w.r.t. input, shape (n_samples, n_features)
    """
    # Apply chain rule for linear transformation u = X @ W + b
    dL_dW = X.T @ dL_du
    dL_db = np.sum(dL_du, axis=0)  # Sum over samples
    dL_dX = dL_du @ W.T

    return dL_dW, dL_db, dL_dX


def single_layer_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forward pass through single layer with activation

    Following class notation: X → u → v (input → pre-activation → post-activation)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data
    W : np.ndarray, shape (n_features, n_units)
        Weight matrix
    b : np.ndarray, shape (n_units,)
        Bias vector
    activation : str
        Activation function ('linear', 'sigmoid', 'relu', 'softmax')

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        u: Pre-activation values, shape (n_samples, n_units)
        v: Post-activation values, shape (n_samples, n_units)
    """
    # 1. Compute linear transformation (get u)
    u = linear_forward(X, W, b)

    # 2. Apply activation function (get v)
    if activation == 'linear':
        v = u  # Linear activation is just identity
    elif activation == 'sigmoid':
        v = sigmoid_forward(u)
    elif activation == 'relu':
        v = relu_forward(u)
    elif activation == 'softmax':
        v = softmax_forward(u)
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    # 3. Return both u and v for backprop
    return u, v


def single_layer_backward(dL_dv: np.ndarray, u: np.ndarray, X: np.ndarray, W: np.ndarray,
                         activation: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass through single layer with activation

    Following class notation: gradients flow v → u → parameters

    Parameters
    ----------
    dL_dv : np.ndarray, shape (n_samples, n_units)
        Gradient of loss w.r.t. layer output (post-activation v)
    u : np.ndarray, shape (n_samples, n_units)
        Pre-activation values from forward pass
    X : np.ndarray, shape (n_samples, n_features)
        Input to this layer
    W : np.ndarray, shape (n_features, n_units)
        Weight matrix
    activation : str
        Activation function used in forward pass

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        dL_dW: Gradient w.r.t. weights
        dL_db: Gradient w.r.t. bias
        dL_dX: Gradient w.r.t. input
    """
    # 1. Compute dL_du = dL_dv * activation_derivative(u)
    if activation == 'linear':
        dL_du = dL_dv  # Linear derivative is 1
    elif activation == 'sigmoid':
        dL_du = dL_dv * sigmoid_derivative(u)
    elif activation == 'relu':
        dL_du = dL_dv * relu_derivative(u)
    elif activation == 'softmax':
        # For softmax, assume cross-entropy loss is used
        # Combined softmax + cross-entropy derivative simplifies to: y_pred - y_true
        # Here we assume dL_dv already incorporates this simplification
        dL_du = dL_dv
    else:
        raise ValueError(f"Unknown activation function: {activation}")

    # 2. Use linear_backward to get gradients w.r.t. W, b, X
    dL_dW, dL_db, dL_dX = linear_backward(dL_du, X, W)

    return dL_dW, dL_db, dL_dX


# ============================================================================
# TRAINING FUNCTIONS (SIMPLIFIED)
# ============================================================================

def train_single_layer(X: np.ndarray, y: np.ndarray, activation: str = 'sigmoid',
                      loss_type: str = 'mse', epochs: int = 100, learning_rate: float = 0.01) -> Tuple[Dict, List[float]]:
    """
    Train single layer network (logistic/linear regression)

    Uses utility functions for weight initialization

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training data
    y : np.ndarray
        Target values
    activation : str
        Activation function for output
    loss_type : str
        Loss function type
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate

    Returns
    -------
    Tuple[Dict, List[float]]
        Trained weights and loss history
    """
    # Import utility function
    from utils import initialize_network_weights

    # Initialize weights
    n_features = X.shape[1]
    n_outputs = y.shape[1] if len(y.shape) > 1 else 1

    layer_weights = initialize_network_weights([n_features, n_outputs], method='small_random', seed=42)
    W, b = layer_weights[0]

    # Flatten y if needed for consistency
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    loss_history = []

    for epoch in range(epochs):
        # Forward pass
        u, v = single_layer_forward(X, W, b, activation)

        # Compute loss
        if loss_type == 'mse':
            loss = mse_loss(y, v)
            # Compute loss derivative
            dL_dv = mse_derivative(y, v)
        elif loss_type == 'cross_entropy':
            loss = cross_entropy_loss(y, v)
            # For cross-entropy + softmax, derivative simplifies
            dL_dv = v - y
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        loss_history.append(loss)

        # Backward pass
        dL_dW, dL_db, dL_dX = single_layer_backward(dL_dv, u, X, W, activation)

        # Update weights
        W = W - learning_rate * dL_dW
        b = b - learning_rate * dL_db

    # Return weights in dictionary format
    weights = {'W': W, 'b': b}
    return weights, loss_history


# ============================================================================
# SIMPLIFIED GRADIENT CHECKING
# ============================================================================

def simple_gradient_check(param: np.ndarray, analytical_grad: np.ndarray,
                         loss_func: Callable, param_name: str,
                         epsilon: float = 1e-7) -> Dict:
    """
    Simple gradient checking for a single parameter

    This is the simplified version students implement to understand the concept.

    Parameters
    ----------
    param : np.ndarray
        Parameter to check (e.g., a weight matrix)
    analytical_grad : np.ndarray
        Analytical gradient computed by backprop
    loss_func : Callable
        Function that computes loss given perturbed parameter
        Should have signature: loss_func(perturbed_param) -> float
    param_name : str
        Name of parameter for reporting
    epsilon : float
        Small perturbation for numerical gradient

    Returns
    -------
    Dict
        Results of gradient check
    """
    # 1. Choose first element for simplicity (students can modify this)
    i, j = 0, 0

    # 2. Compute numerical gradient using finite differences
    # Create copy of parameter
    param_plus = param.copy()
    param_minus = param.copy()

    # Perturb the element
    param_plus[i, j] += epsilon
    param_minus[i, j] -= epsilon

    # Compute losses
    loss_plus = loss_func(param_plus)
    loss_minus = loss_func(param_minus)

    # Numerical gradient
    numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

    # 3. Compare with analytical gradient at the same position
    analytical_value = analytical_grad[i, j]

    # 4. Return results
    diff = abs(numerical_grad - analytical_value)
    rel_error = diff / (abs(numerical_grad) + 1e-8)

    return {
        'param_name': param_name,
        'position': (i, j),
        'numerical_grad': numerical_grad,
        'analytical_grad': analytical_value,
        'difference': diff,
        'relative_error': rel_error,
        'passed': rel_error < 1e-5
    }


if __name__ == "__main__":
    print("PA3: Backpropagation & Neural Networks")
    print("Run 'pytest tests.py -v' to test your implementations")

    # Quick demo when students have implemented functions
    print("\n=== Quick Demo ===")
    try:
        # Generate sample data
        from utils import generate_nonlinear_data
        X, y = generate_nonlinear_data(50, noise=0.1, seed=42)

        # Train single layer
        print("Training single layer network...")
        weights, costs = train_single_layer(
            X, y.reshape(-1, 1), activation='sigmoid', epochs=50, learning_rate=0.1
        )

        print(f"Final cost: {costs[-1]:.4f}")
        print("Demo completed successfully!")

    except Exception as e:
        print(f"Demo requires function implementations: {e}")
        import traceback
        traceback.print_exc()