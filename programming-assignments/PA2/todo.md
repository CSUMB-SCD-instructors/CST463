# PA2 Development Todo

## Assignment Overview
PA2: Gradient Descent & Linear Regression - focusing on derivatives, gradients, and optimization from scratch.

## Part 1: Technical Implementation (student_code.py)

### Core Mathematical Functions
- [ ] `numerical_derivative(f: Callable, x: np.ndarray, dimension: int, h: float = 1e-5) -> float`
  - Compute partial derivative w.r.t. specified dimension
  - Handle single-variable case when x is scalar
- [ ] `numerical_gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray`
  - Compute full gradient vector using numerical_derivative
  - Return array of partial derivatives

### Prediction & Cost Functions
- [ ] `linear_predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray`
  - Basic linear prediction: X @ weights
- [ ] `mse_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float`
  - Mean squared error calculation
- [ ] `mse_gradient(X: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray`
  - Analytical gradient of MSE w.r.t. weights

### Optimization Functions
- [ ] `initialize_weights(n_features: int, method: str = 'zeros') -> np.ndarray`
  - Support 'zeros', 'random', 'small_random' methods
- [ ] `gradient_descent_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float) -> Tuple[np.ndarray, float]`
  - Single gradient descent update step
  - Return updated weights and current cost
- [ ] `gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int) -> Tuple[np.ndarray, List[float]]`
  - Full gradient descent training loop
  - Return final weights and cost history

### Data Functions
- [ ] `add_intercept(X: np.ndarray) -> np.ndarray`
  - Add column of ones for bias term
- [ ] `generate_synthetic_data(n_samples: int, n_features: int, noise: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]`
  - Create synthetic linear regression dataset
  - Controlled noise and seeded randomness

### Utility Functions
- [ ] `has_converged(cost_history: List[float], tolerance: float = 1e-6) -> bool`
  - Check convergence based on cost history
  - Early stopping criteria

## Part 2: Visualization & Analysis (visualization_analysis.ipynb)

### Core Analysis Components
- [ ] Delta-h sensitivity exploration
  - Test different h values in numerical derivatives
  - Visualize accuracy vs. numerical stability trade-offs
- [ ] Learning rate effects analysis
  - Compare convergence with different learning rates
  - Identify oscillation/divergence patterns
- [ ] Convergence visualization
  - Plot cost function over iterations
  - Parameter trajectory visualization (if 2D)
- [ ] Gradient descent vs. analytical comparison
  - Compare custom implementation with closed-form solution

### Comparative Studies
- [ ] Initialization method comparison
  - Test zeros, random, small_random initialization
  - Impact on convergence speed and final result
- [ ] Feature scaling exploration
  - Effect of unscaled vs. scaled features
  - Gradient descent sensitivity to feature magnitude

### Standardized Stopping Condition Analysis
- [ ] Identical setup for all students (same data, same initial weights, same learning rate)
- [ ] Students analyze loss curves and choose optimal stopping point
- [ ] Justify stopping threshold choice with written reasoning
- [ ] Consider trade-offs: computational cost vs. diminishing returns vs. overfitting risk
- [ ] Peer review component: evaluate classmates' stopping criteria and reasoning
- [ ] Use moderately complex synthetic data where "best" stopping point is subjective

### Communication & Analysis
- [ ] Written interpretation of hyperparameter effects
- [ ] Executive summary for peer review
- [ ] Specific questions for peer reviewers

## Supporting Files
- [ ] `tests.py` - Comprehensive test suite
- [ ] `scoring.yaml` - Grading configuration
- [ ] `README.md` - Assignment instructions and learning objectives
- [ ] Sample synthetic datasets for testing

## Design Principles
- Remove hints compared to PA1 (progressive difficulty)
- Type annotations throughout for clarity
- Break functions into unit-testable components
- Focus on multidimensional case with single-variable test cases
- Synthetic data for clean, reproducible results