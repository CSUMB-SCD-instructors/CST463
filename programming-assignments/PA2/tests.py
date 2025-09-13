#!/usr/bin/env python3

import pytest
import numpy as np
from typing import Callable
from student_code import (
    numerical_derivative,
    numerical_gradient,
    linear_predict,
    mse_cost,
    mse_gradient,
    initialize_weights,
    gradient_descent_step,
    gradient_descent,
    add_intercept,
    generate_synthetic_data,
    has_converged
)


class TestNumericalDerivatives:
    """Test numerical derivative and gradient calculations."""
    
    def test_numerical_derivative_single_variable(self):
        """Test numerical derivative with single variable functions."""
        # Test f(x) = x^2, f'(x) = 2x
        def f_quadratic(x):
            if isinstance(x, np.ndarray):
                return x[0] ** 2
            return x ** 2
        
        x = np.array([3.0])
        derivative = numerical_derivative(f_quadratic, x, dimension=0)
        expected = 6.0  # 2 * 3
        assert abs(derivative - expected) < 1e-4
        
        # Test f(x) = 2x + 5, f'(x) = 2
        def f_linear(x):
            if isinstance(x, np.ndarray):
                return 2 * x[0] + 5
            return 2 * x + 5
        
        x = np.array([10.0])
        derivative = numerical_derivative(f_linear, x, dimension=0)
        expected = 2.0
        assert abs(derivative - expected) < 1e-6
    
    def test_numerical_derivative_multivariable(self):
        """Test numerical derivative with multivariable functions."""
        # Test f(x,y) = x^2 + y^2, df/dx = 2x, df/dy = 2y
        def f_quadratic_2d(x):
            return x[0] ** 2 + x[1] ** 2
        
        x = np.array([3.0, 4.0])
        
        # Partial derivative w.r.t. x (dimension 0)
        dx = numerical_derivative(f_quadratic_2d, x, dimension=0)
        expected_dx = 6.0  # 2 * 3
        assert abs(dx - expected_dx) < 1e-4
        
        # Partial derivative w.r.t. y (dimension 1)
        dy = numerical_derivative(f_quadratic_2d, x, dimension=1)
        expected_dy = 8.0  # 2 * 4
        assert abs(dy - expected_dy) < 1e-4
    
    def test_numerical_gradient(self):
        """Test full gradient computation."""
        # Test f(x,y) = x^2 + 2*x*y + y^2
        # df/dx = 2x + 2y, df/dy = 2x + 2y
        def f_mixed(x):
            return x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2
        
        x = np.array([2.0, 3.0])
        gradient = numerical_gradient(f_mixed, x)
        
        expected_gradient = np.array([10.0, 10.0])  # [2*2 + 2*3, 2*2 + 2*3]
        np.testing.assert_allclose(gradient, expected_gradient, atol=1e-4)
    
    def test_numerical_derivative_different_h_values(self):
        """Test that different h values give reasonable results."""
        def f_cubic(x):
            return x[0] ** 3
        
        x = np.array([2.0])
        expected = 12.0  # 3 * 2^2
        
        # Test different h values
        for h in [1e-3, 1e-5, 1e-7]:
            derivative = numerical_derivative(f_cubic, x, dimension=0, h=h)
            assert abs(derivative - expected) < 1e-2


class TestPredictionAndCost:
    """Test prediction and cost function calculations."""
    
    def test_linear_predict_simple(self):
        """Test basic linear prediction."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        weights = np.array([0.5, 1.0])
        
        predictions = linear_predict(X, weights)
        expected = np.array([2.5, 5.5, 8.5])  # [1*0.5 + 2*1, 3*0.5 + 4*1, 5*0.5 + 6*1]
        
        np.testing.assert_allclose(predictions, expected)
    
    def test_linear_predict_with_intercept(self):
        """Test linear prediction with intercept term."""
        X = np.array([[1, 2, 3], [1, 4, 5], [1, 6, 7]])  # First column is intercept
        weights = np.array([1.0, 0.5, 1.0])  # [bias, w1, w2]
        
        predictions = linear_predict(X, weights)
        expected = np.array([5.0, 8.0, 11.0])  # [1*1 + 2*0.5 + 3*1, etc.]
        
        np.testing.assert_allclose(predictions, expected)
    
    def test_mse_cost(self):
        """Test mean squared error calculation."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        
        cost = mse_cost(y_true, y_pred)
        
        # Manual calculation: ((0.1)^2 + (0.1)^2 + (0.2)^2 + (0.2)^2) / 4
        expected = (0.01 + 0.01 + 0.04 + 0.04) / 4
        assert abs(cost - expected) < 1e-10
    
    def test_mse_cost_perfect_prediction(self):
        """Test MSE when prediction is perfect."""
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1, 2, 3, 4])
        
        cost = mse_cost(y_true, y_pred)
        assert abs(cost) < 1e-10
    
    def test_mse_gradient(self):
        """Test analytical gradient of MSE."""
        # Simple case: X = [[1, 1], [1, 2]], y = [2, 3], weights = [0, 0]
        X = np.array([[1, 1], [1, 2]])
        y = np.array([2, 3])
        weights = np.array([0, 0])
        
        gradient = mse_gradient(X, y, weights)
        
        # Manual calculation: predictions = [0, 0], errors = [2, 3]
        # gradient = -2/n * X.T @ errors = -2/2 * [[1,1], [1,2]].T @ [2,3] = -[[1,1,2], [1,2]] @ [2,3]
        expected = np.array([-5.0, -8.0])  # -[2+3, 2+6]
        np.testing.assert_allclose(gradient, expected)


class TestOptimization:
    """Test optimization functions."""
    
    def test_initialize_weights_zeros(self):
        """Test zero initialization."""
        weights = initialize_weights(5, method='zeros')
        expected = np.zeros(5)
        np.testing.assert_array_equal(weights, expected)
    
    def test_initialize_weights_random(self):
        """Test random initialization."""
        weights = initialize_weights(5, method='random')
        
        assert weights.shape == (5,)
        assert not np.allclose(weights, 0)  # Should not be all zeros
    
    def test_initialize_weights_small_random(self):
        """Test small random initialization."""
        weights = initialize_weights(10, method='small_random')
        
        assert weights.shape == (10,)
        assert np.all(np.abs(weights) < 1.0)  # Should be small values
    
    def test_gradient_descent_step(self):
        """Test single gradient descent step."""
        # Simple linear problem
        X = np.array([[1, 1], [1, 2]])
        y = np.array([2, 3])
        weights = np.array([0, 0])
        learning_rate = 0.1
        
        new_weights, cost = gradient_descent_step(X, y, weights, learning_rate)
        
        # Should update weights based on gradient
        assert new_weights.shape == (2,)
        assert cost >= 0  # Cost should be non-negative
        assert not np.array_equal(new_weights, weights)  # Weights should change
    
    def test_gradient_descent_convergence(self):
        """Test that gradient descent converges on simple problem."""
        # Create simple linear problem: y = 2 + 3*x + noise
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 1)
        X_with_intercept = add_intercept(X)
        true_weights = np.array([2.0, 3.0])
        y = X_with_intercept @ true_weights + 0.01 * np.random.randn(n_samples)
        
        # Run gradient descent
        final_weights, cost_history = gradient_descent(
            X_with_intercept, y, learning_rate=0.01, epochs=1000
        )
        
        # Should converge close to true weights
        np.testing.assert_allclose(final_weights, true_weights, atol=0.1)
        
        # Cost should decrease
        assert cost_history[-1] < cost_history[0]
        assert len(cost_history) == 1000


class TestDataFunctions:
    """Test data generation and manipulation functions."""
    
    def test_add_intercept(self):
        """Test adding intercept column."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_with_intercept = add_intercept(X)
        
        expected = np.array([[1, 1, 2], [1, 3, 4], [1, 5, 6]])
        np.testing.assert_array_equal(X_with_intercept, expected)
    
    def test_add_intercept_single_feature(self):
        """Test adding intercept to single feature."""
        X = np.array([[1], [2], [3]])
        X_with_intercept = add_intercept(X)
        
        expected = np.array([[1, 1], [1, 2], [1, 3]])
        np.testing.assert_array_equal(X_with_intercept, expected)
    
    def test_generate_synthetic_data_shape(self):
        """Test synthetic data generation produces correct shapes."""
        n_samples, n_features = 100, 3
        X, y = generate_synthetic_data(n_samples, n_features, seed=42)
        
        assert X.shape == (n_samples, n_features)
        assert y.shape == (n_samples,)
    
    def test_generate_synthetic_data_reproducible(self):
        """Test that synthetic data is reproducible with same seed."""
        X1, y1 = generate_synthetic_data(50, 2, seed=123)
        X2, y2 = generate_synthetic_data(50, 2, seed=123)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_generate_synthetic_data_different_seeds(self):
        """Test that different seeds produce different data."""
        X1, y1 = generate_synthetic_data(50, 2, seed=123)
        X2, y2 = generate_synthetic_data(50, 2, seed=456)
        
        assert not np.array_equal(X1, X2)
        assert not np.array_equal(y1, y2)
    
    def test_generate_synthetic_data_noise_effect(self):
        """Test that noise parameter affects data generation."""
        X1, y1 = generate_synthetic_data(100, 2, noise=0.0, seed=42)
        X2, y2 = generate_synthetic_data(100, 2, noise=1.0, seed=42)
        
        # X should be the same (features), but y should be different due to noise
        np.testing.assert_array_equal(X1, X2)
        assert not np.array_equal(y1, y2)


class TestUtilities:
    """Test utility functions."""
    
    def test_has_converged_true(self):
        """Test convergence detection when converged."""
        # Cost history that has converged
        cost_history = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.0001, 0.0001]
        
        assert has_converged(cost_history, tolerance=1e-3)
    
    def test_has_converged_false(self):
        """Test convergence detection when not converged."""
        # Cost history that hasn't converged
        cost_history = [1.0, 0.8, 0.6, 0.4, 0.2]
        
        assert not has_converged(cost_history, tolerance=1e-6)
    
    def test_has_converged_insufficient_history(self):
        """Test convergence with insufficient history."""
        # Too few points to determine convergence
        cost_history = [1.0, 0.5]
        
        assert not has_converged(cost_history, tolerance=1e-6)
    
    def test_has_converged_empty_history(self):
        """Test convergence with empty history."""
        cost_history = []
        
        assert not has_converged(cost_history, tolerance=1e-6)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_complete_linear_regression_pipeline(self):
        """Test complete linear regression from data generation to training."""
        # Generate synthetic data
        X, y = generate_synthetic_data(100, 2, noise=0.1, seed=42)
        X_with_intercept = add_intercept(X)
        
        # Initialize weights
        weights = initialize_weights(X_with_intercept.shape[1], method='zeros')
        
        # Train model
        final_weights, cost_history = gradient_descent(
            X_with_intercept, y, learning_rate=0.01, epochs=500
        )
        
        # Make predictions
        predictions = linear_predict(X_with_intercept, final_weights)
        final_cost = mse_cost(y, predictions)
        
        # Verify reasonable results
        assert final_cost < 1.0  # Should achieve reasonable fit
        assert len(cost_history) == 500
        assert cost_history[-1] <= cost_history[0]  # Cost should decrease
    
    def test_numerical_vs_analytical_gradients(self):
        """Test that numerical and analytical gradients are close."""
        # Generate test data
        X, y = generate_synthetic_data(20, 2, noise=0.1, seed=42)
        X_with_intercept = add_intercept(X)
        weights = np.array([1.0, 0.5, -0.3])
        
        # Analytical gradient
        analytical_grad = mse_gradient(X_with_intercept, y, weights)
        
        # Numerical gradient
        def cost_function(w):
            predictions = linear_predict(X_with_intercept, w)
            return mse_cost(y, predictions)
        
        numerical_grad = numerical_gradient(cost_function, weights)
        
        # Should be very close
        np.testing.assert_allclose(analytical_grad, numerical_grad, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])