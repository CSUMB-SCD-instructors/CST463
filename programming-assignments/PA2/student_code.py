#!/usr/bin/env python3

import numpy as np
from typing import Callable, Tuple, List


def numerical_derivative(f: Callable, x: np.ndarray, dimension: int, h: float = 1e-5) -> float:
    """
    Compute the numerical partial derivative of a function at a point.
    
    Uses the central difference formula: f'(x) ≈ (f(x+h) - f(x-h)) / (2*h)
    
    Parameters
    ----------
    f : Callable
        Function to compute derivative of. Should accept numpy array and return scalar.
    x : np.ndarray
        Point at which to evaluate the derivative.
    dimension : int
        Which dimension (variable) to take the partial derivative with respect to.
        For single-variable functions, use dimension=0.
    h : float, optional
        Small step size for numerical approximation. Default is 1e-5.
        
    Returns
    -------
    float
        The numerical partial derivative ∂f/∂x_dimension at point x.
        
    Examples
    --------
    >>> def f(x): return x[0]**2 + x[1]**2  # f(x,y) = x² + y²
    >>> x = np.array([2.0, 3.0])
    >>> numerical_derivative(f, x, dimension=0)  # ∂f/∂x = 2x = 4.0
    4.0
    """
    # TODO: Implement numerical derivative using central difference
    pass


def numerical_gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute the numerical gradient of a function at a point.
    
    The gradient is the vector of all partial derivatives: ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    
    Parameters
    ----------
    f : Callable
        Function to compute gradient of. Should accept numpy array and return scalar.
    x : np.ndarray, shape (n_features,)
        Point at which to evaluate the gradient.
    h : float, optional
        Small step size for numerical approximation. Default is 1e-5.
        
    Returns
    -------
    np.ndarray, shape (n_features,)
        The numerical gradient ∇f at point x.
        
    Examples
    --------
    >>> def f(x): return x[0]**2 + 2*x[1]**2  # f(x,y) = x² + 2y²
    >>> x = np.array([1.0, 2.0])
    >>> numerical_gradient(f, x)  # [∂f/∂x, ∂f/∂y] = [2x, 4y] = [2.0, 8.0]
    array([2.0, 8.0])
    """
    # TODO: Implement numerical gradient using numerical_derivative
    pass


def linear_predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Make predictions using linear model.
    
    Computes ŷ = X @ weights for linear regression predictions.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix where each row is a data point.
    weights : np.ndarray, shape (n_features,)
        Model parameters (coefficients).
        
    Returns
    -------
    np.ndarray, shape (n_samples,)
        Predicted values for each sample.
        
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])  # 2 samples, 2 features
    >>> weights = np.array([0.5, 1.0])
    >>> linear_predict(X, weights)
    array([2.5, 5.5])  # [1*0.5 + 2*1.0, 3*0.5 + 4*1.0]
    """
    # TODO: Implement linear prediction
    pass


def mse_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Squared Error between true and predicted values.
    
    MSE = (1/n) * Σ(yᵢ - ŷᵢ)²
    
    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        True target values.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    float
        Mean squared error cost.
        
    Examples
    --------
    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 1.9, 3.1])
    >>> mse_cost(y_true, y_pred)
    0.0067  # approximately
    """
    # TODO: Implement mean squared error
    pass


def mse_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute the analytical gradient of Mean Squared Error with respect to weights.
    
    For MSE = (1/n) * ||y - X@w||², the gradient is:
    ∇MSE = (2/n) * Xᵀ @ (X@w - y)
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target values.
    weights : np.ndarray, shape (n_features,)
        Current model parameters.
        
    Returns
    -------
    np.ndarray, shape (n_features,)
        Gradient of MSE with respect to weights.
        
    Examples
    --------
    >>> X = np.array([[1, 1], [1, 2]])
    >>> y = np.array([2, 3])
    >>> weights = np.array([0, 0])
    >>> mse_gradient(X, y, weights)
    array([-5.0, -8.0])  # gradient when predictions are all zeros
    """
    # TODO: Implement analytical MSE gradient
    pass


def initialize_weights(n_features: int, method: str = 'zeros') -> np.ndarray:
    """
    Initialize model weights using specified method.
    
    Parameters
    ----------
    n_features : int
        Number of features (size of weight vector).
    method : str, optional
        Initialization method. Options:
        - 'zeros': Initialize all weights to zero
        - 'random': Initialize with random values from standard normal
        - 'small_random': Initialize with small random values (scaled by 0.01)
        
    Returns
    -------
    np.ndarray, shape (n_features,)
        Initialized weight vector.
        
    Examples
    --------
    >>> initialize_weights(3, method='zeros')
    array([0., 0., 0.])
    >>> np.random.seed(42); initialize_weights(2, method='small_random')
    array([0.00496714, -0.00138264])  # small random values
    """
    # TODO: Implement weight initialization
    pass


def gradient_descent_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, 
                         learning_rate: float) -> Tuple[np.ndarray, float]:
    """
    Perform a single gradient descent update step.
    
    Updates weights using: w_new = w_old - learning_rate * gradient
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target values.
    weights : np.ndarray, shape (n_features,)
        Current model parameters.
    learning_rate : float
        Step size for gradient descent update.
        
    Returns
    -------
    Tuple[np.ndarray, float]
        Updated weights and current cost after the step.
        
    Examples
    --------
    >>> X = np.array([[1, 1], [1, 2]])
    >>> y = np.array([2, 3])
    >>> weights = np.array([0., 0.])
    >>> new_weights, cost = gradient_descent_step(X, y, weights, 0.1)
    >>> new_weights
    array([0.5, 0.8])  # weights moved in direction opposite to gradient
    """
    # TODO: Implement single gradient descent step
    pass


def gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float, 
                    epochs: int) -> Tuple[np.ndarray, List[float]]:
    """
    Perform full gradient descent optimization.
    
    Iteratively updates weights to minimize mean squared error using gradient descent.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Target values.
    learning_rate : float
        Step size for gradient descent updates.
    epochs : int
        Number of optimization iterations to perform.
        
    Returns
    -------
    Tuple[np.ndarray, List[float]]
        Final optimized weights and list of cost values at each epoch.
        
    Examples
    --------
    >>> X = np.array([[1, 1], [1, 2]])
    >>> y = np.array([2, 3])
    >>> weights, costs = gradient_descent(X, y, learning_rate=0.01, epochs=100)
    >>> len(costs)
    100
    >>> costs[-1] < costs[0]  # Cost should decrease
    True
    """
    # TODO: Implement full gradient descent training loop
    pass


def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Add intercept (bias) term to feature matrix.
    
    Prepends a column of ones to the feature matrix to account for bias term.
    This allows linear models to have a y-intercept: ŷ = w₀ + w₁x₁ + w₂x₂ + ...
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Original feature matrix.
        
    Returns
    -------
    np.ndarray, shape (n_samples, n_features + 1)
        Feature matrix with intercept column prepended.
        
    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> add_intercept(X)
    array([[1, 1, 2],
           [1, 3, 4]])  # First column is all ones for bias term
    """
    # TODO: Implement intercept addition
    pass


def generate_synthetic_data(n_samples: int, n_features: int, noise: float = 0.1, 
                          seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression dataset.
    
    Creates a dataset where y follows a linear relationship with X plus Gaussian noise.
    This provides clean, controlled data for testing gradient descent implementations.
    
    Parameters
    ----------
    n_samples : int
        Number of data points to generate.
    n_features : int
        Number of input features.
    noise : float, optional
        Standard deviation of Gaussian noise added to targets. Default is 0.1.
    seed : int, optional
        Random seed for reproducible results. Default is 42.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix X (n_samples, n_features) and target vector y (n_samples,).
        
    Examples
    --------
    >>> X, y = generate_synthetic_data(100, 2, noise=0.1, seed=42)
    >>> X.shape
    (100, 2)
    >>> y.shape
    (100,)
    >>> # Data follows linear relationship: y ≈ X @ true_weights + noise
    """
    # TODO: Implement synthetic data generation
    pass


def has_converged(cost_history: List[float], tolerance: float = 1e-6) -> bool:
    """
    Check if gradient descent has converged based on cost history.
    
    Determines convergence by examining if the change in cost over recent iterations
    is smaller than the tolerance threshold. This helps implement early stopping
    to avoid unnecessary computation.
    
    Parameters
    ----------
    cost_history : List[float]
        History of cost values from gradient descent iterations.
    tolerance : float, optional
        Threshold for determining convergence. Default is 1e-6.
        
    Returns
    -------
    bool
        True if converged (recent cost changes are below tolerance), False otherwise.
        
    Examples
    --------
    >>> cost_history = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.0001]
    >>> has_converged(cost_history, tolerance=1e-3)
    True  # Recent changes are smaller than 1e-3
    >>> has_converged([1.0, 0.8, 0.6, 0.4], tolerance=1e-6)
    False  # Still decreasing significantly
    """
    # TODO: Implement convergence checking
    pass


if __name__ == "__main__":
    print("PA2: Gradient Descent & Linear Regression")
    print("Run 'pytest test.py -v' to test your implementations")
    
    # Quick demo when students have implemented functions
    print("\n=== Quick Demo (implement functions first!) ===")
    try:
        # Generate sample data
        X, y = generate_synthetic_data(50, 2, noise=0.1, seed=42)
        X_with_intercept = add_intercept(X)
        
        # Train model
        final_weights, cost_history = gradient_descent(
            X_with_intercept, y, learning_rate=0.01, epochs=100
        )
        
        print(f"Training completed!")
        print(f"Final weights: {final_weights}")
        print(f"Initial cost: {cost_history[0]:.4f}")
        print(f"Final cost: {cost_history[-1]:.4f}")
        print(f"Converged: {has_converged(cost_history)}")
        
    except Exception as e:
        print(f"Demo requires function implementations: {e}")