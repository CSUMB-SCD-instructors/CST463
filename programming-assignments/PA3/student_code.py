#!/usr/bin/env python3
"""
PA3: Backpropagation & Neural Networks

This assignment implements backpropagation and neural network fundamentals from scratch.

Key functions to implement:
- Forward pass: linear_forward, sigmoid_forward, relu_forward, softmax_forward
- Loss functions: mse_loss, cross_entropy_loss
- Loss derivatives: mse_derivative
- Activation derivatives: sigmoid_derivative, relu_derivative
- Backpropagation: linear_backward (core chain rule), single_layer_backward
- Integration: single_layer_forward, train_single_layer
- Validation: simple_gradient_check (numerical gradient verification)

This builds the foundation for deep learning by implementing every component manually.
"""

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
    # TODO: Implement linear forward pass
    pass


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
    # TODO: Implement sigmoid activation function
    # Hint: For numerical stability, clip u to range [-500, 500] using np.clip()
    # Hint: This prevents overflow/underflow for extreme values
    pass


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
    # TODO: Implement ReLU activation
    pass


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
    # TODO: Implement softmax activation
    # Hint: For numerical stability, subtract max from u first: u_stable = u - max(u)
    # Hint: Result should have each row sum to 1 (probability distribution)
    pass


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
    # TODO: Implement Mean Squared Error loss
    pass


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
    # TODO: Implement cross-entropy loss
    # Hint: Clip y_pred to avoid log(0): np.clip(y_pred, epsilon, 1 - epsilon)
    pass


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
    # TODO: Compute the derivative of MSE with respect to predictions
    # Hint: MSE = (1/n) * sum((y_pred - y_true)^2)
    # Hint: Use calculus chain rule to find d(MSE)/d(y_pred)
    # Hint: The derivative of (y_pred - y_true)^2 w.r.t. y_pred is 2*(y_pred - y_true)
    # Hint: Don't forget to divide by the number of samples
    pass


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
    # TODO: Compute the derivative of the sigmoid function
    # Hint: Remember, the derivative of σ(u) = 1/(1+e^(-u)) is σ(u) * (1 - σ(u))
    pass


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
    # TODO: Compute the derivative of ReLU
    # Hint: ReLU(u) = max(0, u), so the derivative is 1 where u > 0, and 0 elsewhere
    # Hint: You can use boolean indexing or comparison operators
    # Hint: Convert boolean result to float64 for compatibility
    pass


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
    # TODO: Implement backpropagation through linear layer u = X @ W + b
    # Hint: Use the chain rule to compute gradients
    # Hint: Review the class notes on backprop through linear layers
    pass


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
    # TODO: Implement single layer forward pass
    pass


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
    # TODO: Implement backward pass through activation + linear layer
    pass


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
    # TODO: Implement training loop for single layer network
    # Step 1: Import and initialize weights

    # Step 2: Reshape y if needed

    # Step 3: Training loop

    # Step 4: Return results
    pass


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
    # TODO: Implement gradient checking using finite differences
    # Step 1: Choose an element to check (start with [0, 0] for simplicity)
    i, j = 0, 0

    # Step 2: Compute numerical gradient using finite differences

    # Step 3: Compare with analytical gradient

    # Step 4: Return dictionary with results

    pass


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