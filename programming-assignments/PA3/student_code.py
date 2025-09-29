#!/usr/bin/env python3

import numpy as np
from typing import Callable, Tuple, List, Dict, Union


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
    # TODO: Implement linear transformation
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
    # TODO: Implement sigmoid function
    # Hint: Use np.clip or other techniques to prevent overflow
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
    # TODO: Implement ReLU function
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
    # TODO: Implement softmax function
    # Hint: Subtract max for numerical stability
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
    # TODO: Implement MSE loss
    pass


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute cross-entropy loss for classification

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples, n_classes)
        True labels (one-hot encoded)
    y_pred : np.ndarray, shape (n_samples, n_classes)
        Predicted probabilities

    Returns
    -------
    float
        Cross-entropy loss value
    """
    # TODO: Implement cross-entropy loss
    # Hint: Add small epsilon to prevent log(0)
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
    # TODO: Implement MSE derivative
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
    # TODO: Implement sigmoid derivative
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
    # TODO: Implement ReLU derivative
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
    # TODO: Implement linear layer backpropagation
    # Use chain rule: dL_dW = X.T @ dL_du
    #                dL_db = sum over samples of dL_du
    #                dL_dX = dL_du @ W.T
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
    # 1. Compute linear transformation (get u)
    # 2. Apply activation function (get v)
    # 3. Return both u and v for backprop
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
    # TODO: Implement single layer backward pass
    # 1. Compute dL_du = dL_dv * activation_derivative(u)
    # 2. Use linear_backward to get gradients w.r.t. W, b, X
    pass


# ============================================================================
# TWO-LAYER NETWORK FUNCTIONS
# ============================================================================

def two_layer_forward(X: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray,
                     hidden_activation: str = 'relu', output_activation: str = 'linear') -> Dict:
    """
    Forward pass through two-layer network

    Following class notation: X → u1 → v1 → u2 → v2

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data
    W1, b1 : np.ndarray
        First layer weights and bias
    W2, b2 : np.ndarray
        Second layer weights and bias
    hidden_activation : str
        Activation for hidden layer
    output_activation : str
        Activation for output layer

    Returns
    -------
    Dict
        Forward pass cache containing all intermediate values needed for backprop
        Keys: X, u1, v1, u2, v2, W1, W2 (following class notation)
    """
    # TODO: Implement two-layer forward pass
    # 1. Forward through first layer: X → u1 → v1
    # 2. Forward through second layer: v1 → u2 → v2
    # 3. Return dict with all intermediate values: X, u1, v1, u2, v2, W1, W2
    pass


def two_layer_backward(forward_cache: Dict, y_true: np.ndarray, loss_type: str = 'mse') -> Dict:
    """
    Backward pass through two-layer network

    Parameters
    ----------
    forward_cache : Dict
        Cache from forward pass containing intermediate values
    y_true : np.ndarray
        True target values
    loss_type : str
        Type of loss function ('mse' or 'cross_entropy')

    Returns
    -------
    Dict
        Dictionary containing all gradients: dW1, db1, dW2, db2
    """
    # TODO: Implement two-layer backward pass
    # 1. Compute loss derivative w.r.t. output
    # 2. Backward through second layer
    # 3. Backward through first layer
    # 4. Return all gradients
    pass


# ============================================================================
# TRAINING & OPTIMIZATION
# ============================================================================

def initialize_network_weights(layer_sizes: List[int], method: str = 'xavier') -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Initialize weights and biases for multi-layer network

    Parameters
    ----------
    layer_sizes : List[int]
        List of layer sizes [input_size, hidden_size, ..., output_size]
    method : str
        Initialization method ('zeros', 'random', 'xavier', 'he')

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (W, b) tuples for each layer
    """
    # TODO: Implement weight initialization
    # Support different initialization strategies
    pass


def gradient_descent_step_network(gradients: Dict, weights: Dict, learning_rate: float) -> Dict:
    """
    Update network weights using computed gradients

    Parameters
    ----------
    gradients : Dict
        Dictionary containing gradients for all parameters
    weights : Dict
        Current network weights
    learning_rate : float
        Learning rate for gradient descent

    Returns
    -------
    Dict
        Updated network weights
    """
    # TODO: Implement gradient descent weight update
    # weights = weights - learning_rate * gradients
    pass


def train_single_layer(X: np.ndarray, y: np.ndarray, activation: str = 'sigmoid',
                      loss_type: str = 'mse', epochs: int = 100, learning_rate: float = 0.01) -> Tuple[Dict, List[float]]:
    """
    Train single layer network (logistic/linear regression)

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
    # TODO: Implement single layer training loop
    pass


def train_two_layer_network(X: np.ndarray, y: np.ndarray, hidden_size: int, epochs: int = 100,
                           learning_rate: float = 0.01) -> Tuple[Dict, List[float]]:
    """
    Train two-layer neural network

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Training data
    y : np.ndarray
        Target values
    hidden_size : int
        Number of hidden units
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate

    Returns
    -------
    Tuple[Dict, List[float]]
        Trained network weights and loss history
    """
    # TODO: Implement two-layer network training loop
    pass


# ============================================================================
# UTILITY & VERIFICATION FUNCTIONS
# ============================================================================

def numerical_gradient_check(X: np.ndarray, y: np.ndarray, weights: Dict, network_type: str = 'single',
                            epsilon: float = 1e-7) -> Dict:
    """
    Verify analytical gradients using numerical gradients

    Parameters
    ----------
    X : np.ndarray
        Input data
    y : np.ndarray
        Target values
    weights : Dict
        Current network weights
    network_type : str
        Type of network ('single' or 'two_layer')
    epsilon : float
        Small perturbation for numerical gradient

    Returns
    -------
    Dict
        Comparison between analytical and numerical gradients
    """
    # TODO: Implement gradient checking
    # Use finite differences to compute numerical gradients
    # Compare with analytical gradients from backprop
    pass


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
    # TODO: Implement prediction function
    pass


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
    # TODO: Generate synthetic classification data
    # Create data that requires nonlinear decision boundary for 2-layer vs 1-layer comparison
    pass


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
    # TODO: Generate nonlinear data (e.g., XOR-like, circles, spirals)
    pass


if __name__ == "__main__":
    print("PA3: Backpropagation & Neural Networks")
    print("Run 'pytest tests.py -v' to test your implementations")

    # Quick demo when students have implemented functions
    print("\n=== Quick Demo (implement functions first!) ===")
    try:
        # Generate sample data
        X, y = generate_nonlinear_data(100, noise=0.1, seed=42)

        # Train single layer (should struggle with nonlinear data)
        print("Training single layer network...")
        single_weights, single_costs = train_single_layer(X, y, epochs=100, learning_rate=0.1)

        # Train two layer (should handle nonlinear data better)
        print("Training two-layer network...")
        two_layer_weights, two_layer_costs = train_two_layer_network(X, y, hidden_size=10, epochs=100, learning_rate=0.1)

        print(f"Single layer final cost: {single_costs[-1]:.4f}")
        print(f"Two-layer final cost: {two_layer_costs[-1]:.4f}")

        # Gradient checking
        print("Running gradient check...")
        grad_check = numerical_gradient_check(X[:10], y[:10], two_layer_weights, 'two_layer')
        print("Gradient check completed!")

    except Exception as e:
        print(f"Demo requires function implementations: {e}")