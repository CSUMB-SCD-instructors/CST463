#!/usr/bin/env python3

import numpy as np
import pytest
from student_code import (
    # Forward pass functions
    linear_forward, sigmoid_forward, relu_forward, softmax_forward,
    # Loss functions
    mse_loss, cross_entropy_loss,
    # Derivative functions
    mse_derivative, sigmoid_derivative, relu_derivative,
    # Backpropagation functions
    linear_backward, single_layer_forward, single_layer_backward,
    # Training functions
    train_single_layer,
    # Simplified gradient check
    simple_gradient_check
)

# Import utilities from separate file
from utils import (
    initialize_network_weights, predict_network,
    generate_classification_data, generate_nonlinear_data,
    full_gradient_check
)


# ============================================================================
# FORWARD PASS TESTS
# ============================================================================

class TestForwardPass:
    """Test forward pass functions with known inputs/outputs"""

    def test_linear_forward_simple(self):
        """Test linear_forward with simple known values"""
        # Simple case: 2 samples, 2 features, 1 output unit
        X = np.array([[1, 2], [3, 4]])
        W = np.array([[0.5], [0.3]])  # 2 features -> 1 unit
        b = np.array([0.1])

        u = linear_forward(X, W, b)

        # Expected: [1*0.5 + 2*0.3 + 0.1, 3*0.5 + 4*0.3 + 0.1] = [1.2, 2.8]
        expected = np.array([[1.2], [2.8]])

        assert u.shape == (2, 1)
        np.testing.assert_allclose(u, expected, rtol=1e-6)

    def test_linear_forward_batch(self):
        """Test linear_forward with larger batch and multiple outputs"""
        X = np.array([[1, 0], [0, 1], [-1, 1]])  # 3 samples, 2 features
        W = np.array([[1, -1], [2, 3]])  # 2 features -> 2 units
        b = np.array([0.5, -0.5])

        u = linear_forward(X, W, b)

        # Expected calculations for each sample
        expected = np.array([
            [1*1 + 0*2 + 0.5, 1*(-1) + 0*3 + (-0.5)],  # [1.5, -1.5]
            [0*1 + 1*2 + 0.5, 0*(-1) + 1*3 + (-0.5)],  # [2.5, 2.5]
            [-1*1 + 1*2 + 0.5, -1*(-1) + 1*3 + (-0.5)]  # [1.5, 3.5]
        ])

        assert u.shape == (3, 2)
        np.testing.assert_allclose(u, expected, rtol=1e-6)

    def test_sigmoid_forward_known_values(self):
        """Test sigmoid with known mathematical values"""
        # Test specific values where we know the exact output
        u = np.array([0, 1, -1, 100, -100])
        v = sigmoid_forward(u)

        # Known values: σ(0)=0.5, σ(1)≈0.731, σ(-1)≈0.269, σ(100)≈1, σ(-100)≈0
        expected = np.array([0.5, 1/(1+np.exp(-1)), 1/(1+np.exp(1)), 1.0, 0.0])

        np.testing.assert_allclose(v, expected, rtol=1e-6, atol=1e-15)

    def test_sigmoid_forward_stability(self):
        """Test sigmoid numerical stability with extreme values"""
        # Test that large positive/negative values don't cause overflow/underflow
        u = np.array([-1000, -100, 0, 100, 1000])
        v = sigmoid_forward(u)

        # Should be well-behaved: close to 0 for very negative, close to 1 for very positive
        assert v[0] < 1e-10  # Very small for large negative
        assert v[-1] > 1 - 1e-10  # Very close to 1 for large positive
        assert abs(v[2] - 0.5) < 1e-10  # Exactly 0.5 at 0
        assert np.all(v >= 0) and np.all(v <= 1)  # Valid probability range

    def test_relu_forward_basic(self):
        """Test ReLU activation function"""
        u = np.array([-2, -1, 0, 1, 2])
        v = relu_forward(u)

        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(v, expected)

    def test_relu_forward_batch(self):
        """Test ReLU with batch data"""
        u = np.array([[-1, 2], [3, -4], [0, 0]])
        v = relu_forward(u)

        expected = np.array([[0, 2], [3, 0], [0, 0]])
        np.testing.assert_array_equal(v, expected)

    def test_softmax_forward_simple(self):
        """Test softmax with simple values"""
        u = np.array([[0, 0, 0]])  # Single sample, 3 classes
        v = softmax_forward(u)

        # All zeros should give equal probabilities
        expected = np.array([[1/3, 1/3, 1/3]])
        np.testing.assert_allclose(v, expected, rtol=1e-6)

    def test_softmax_forward_batch(self):
        """Test softmax with batch data"""
        u = np.array([[1, 2, 3], [0, 0, 0]])  # 2 samples, 3 classes
        v = softmax_forward(u)

        # Check that each row sums to 1
        row_sums = np.sum(v, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], rtol=1e-10)

        # Check that probabilities are positive
        assert np.all(v > 0)


# ============================================================================
# LOSS FUNCTION TESTS
# ============================================================================

class TestLossFunctions:
    """Test loss functions with expected values"""

    def test_mse_loss_perfect_predictions(self):
        """Test MSE when predictions are perfect"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        loss = mse_loss(y_true, y_pred)
        assert abs(loss) < 1e-10  # Should be essentially zero

    def test_mse_loss_known_values(self):
        """Test MSE with known calculation"""
        y_true = np.array([1, 2])
        y_pred = np.array([1.5, 1.5])

        # MSE = ((1-1.5)^2 + (2-1.5)^2) / 2 = (0.25 + 0.25) / 2 = 0.25
        loss = mse_loss(y_true, y_pred)
        assert abs(loss - 0.25) < 1e-10

    def test_cross_entropy_loss_perfect_predictions(self):
        """Test cross-entropy with perfect predictions"""
        # One-hot encoded: class 0 for first sample, class 1 for second
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [0, 1]])  # Perfect predictions

        loss = cross_entropy_loss(y_true, y_pred)
        assert abs(loss) < 1e-10  # Should be essentially zero

    def test_cross_entropy_loss_known_values(self):
        """Test cross-entropy with calculated values"""
        y_true = np.array([[1, 0]])  # True class is 0
        y_pred = np.array([[0.8, 0.2]])  # 80% confidence in correct class

        # Cross-entropy = -log(0.8) for this case
        expected_loss = -np.log(0.8)
        loss = cross_entropy_loss(y_true, y_pred)

        assert abs(loss - expected_loss) < 1e-10


# ============================================================================
# DERIVATIVE TESTS
# ============================================================================

class TestDerivatives:
    """Test derivative functions against numerical derivatives"""

    def test_mse_derivative_known_values(self):
        """Test MSE derivative with known calculation"""
        y_true = np.array([1, 2])
        y_pred = np.array([1.5, 1.5])

        grad = mse_derivative(y_true, y_pred)

        # MSE derivative: (2/n) * (y_pred - y_true)
        # = (2/2) * ([1.5, 1.5] - [1, 2]) = [0.5, -0.5]
        expected = np.array([0.5, -0.5])
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_sigmoid_derivative_known_values(self):
        """Test sigmoid derivative with known values"""
        u = np.array([0])  # σ(0) = 0.5
        grad = sigmoid_derivative(u)

        # σ'(0) = σ(0) * (1 - σ(0)) = 0.5 * 0.5 = 0.25
        expected = np.array([0.25])
        np.testing.assert_allclose(grad, expected, rtol=1e-6)

    def test_sigmoid_derivative_numerical_verification(self):
        """Verify sigmoid derivative against numerical derivative"""
        u_test = np.array([-1.0, 0.0, 1.0])  # Use float values to avoid integer casting

        # Compute analytical derivative
        analytical_grad = sigmoid_derivative(u_test)

        # Compute numerical derivative with larger epsilon for stability
        eps = 1e-6
        numerical_grad = np.zeros_like(u_test, dtype=np.float64)  # Ensure float array

        for i, u_val in enumerate(u_test):
            u_plus = u_val + eps
            u_minus = u_val - eps

            # Use the actual sigmoid_forward function for consistency
            sig_plus = sigmoid_forward(np.array([u_plus]))[0]
            sig_minus = sigmoid_forward(np.array([u_minus]))[0]

            numerical_grad[i] = (sig_plus - sig_minus) / (2 * eps)

        np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-6)

    def test_relu_derivative_known_values(self):
        """Test ReLU derivative with known values"""
        u = np.array([-1, 0, 1, 2])
        grad = relu_derivative(u)

        expected = np.array([0, 0, 1, 1])  # 0 for u<=0, 1 for u>0
        np.testing.assert_array_equal(grad, expected)

    def test_derivatives_shape_consistency(self):
        """Test that derivatives have same shape as input"""
        shapes_to_test = [(5,), (3, 4), (2, 3, 4)]

        for shape in shapes_to_test:
            u = np.random.randn(*shape)

            # Test sigmoid derivative
            sig_grad = sigmoid_derivative(u)
            assert sig_grad.shape == shape

            # Test ReLU derivative
            relu_grad = relu_derivative(u)
            assert relu_grad.shape == shape


# ============================================================================
# BACKPROPAGATION TESTS
# ============================================================================

class TestBackpropagation:
    """Test backpropagation functions with known gradients"""

    def test_linear_backward_known_gradients(self):
        """Test linear_backward with manually calculated gradients"""
        # Simple setup
        X = np.array([[1, 2], [3, 4]])  # 2 samples, 2 features
        W = np.array([[0.5], [0.3]])    # 2 features -> 1 unit
        dL_du = np.array([[1], [2]])     # Gradient flowing backward

        dL_dW, dL_db, dL_dX = linear_backward(dL_du, X, W)

        # Manual calculations:
        # dL_dW = X.T @ dL_du = [[1, 3], [2, 4]] @ [[1], [2]] = [[7], [10]]
        # dL_db = sum(dL_du) = [3]
        # dL_dX = dL_du @ W.T = [[1], [2]] @ [[0.5, 0.3]] = [[0.5, 0.3], [1.0, 0.6]]

        expected_dW = np.array([[7], [10]])
        expected_db = np.array([3])
        expected_dX = np.array([[0.5, 0.3], [1.0, 0.6]])

        np.testing.assert_allclose(dL_dW, expected_dW, rtol=1e-6)
        np.testing.assert_allclose(dL_db, expected_db, rtol=1e-6)
        np.testing.assert_allclose(dL_dX, expected_dX, rtol=1e-6)

    def test_single_layer_forward_linear(self):
        """Test single layer forward pass with linear activation"""
        X = np.array([[1, 2]])
        W = np.array([[0.5], [0.3]])
        b = np.array([0.1])

        u, v = single_layer_forward(X, W, b, activation='linear')

        # Linear activation: v should equal u
        expected_u = np.array([[1.2]])  # 1*0.5 + 2*0.3 + 0.1

        np.testing.assert_allclose(u, expected_u, rtol=1e-6)
        np.testing.assert_allclose(v, u, rtol=1e-6)  # v == u for linear

    def test_single_layer_forward_sigmoid(self):
        """Test single layer forward pass with sigmoid activation"""
        X = np.array([[0, 0]])
        W = np.array([[1], [1]])
        b = np.array([0])

        u, v = single_layer_forward(X, W, b, activation='sigmoid')

        # u should be 0, v should be σ(0) = 0.5
        expected_u = np.array([[0]])
        expected_v = np.array([[0.5]])

        np.testing.assert_allclose(u, expected_u, rtol=1e-6)
        np.testing.assert_allclose(v, expected_v, rtol=1e-6)

    def test_gradient_shapes_consistency(self):
        """Test that all gradient shapes are consistent"""
        # Set up a simple network
        batch_size, n_features, n_units = 3, 4, 2

        X = np.random.randn(batch_size, n_features)
        W = np.random.randn(n_features, n_units)
        b = np.random.randn(n_units)
        dL_du = np.random.randn(batch_size, n_units)

        dL_dW, dL_db, dL_dX = linear_backward(dL_du, X, W)

        # Check shapes
        assert dL_dW.shape == W.shape
        assert dL_db.shape == b.shape
        assert dL_dX.shape == X.shape


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test complete workflows and integration between components"""

    def test_forward_backward_consistency(self):
        """Test that forward and backward pass shapes are consistent"""
        # Create small network
        X = np.random.randn(5, 3)  # 5 samples, 3 features
        W = np.random.randn(3, 2)  # 3 features -> 2 units
        b = np.random.randn(2)

        # Forward pass
        u, v = single_layer_forward(X, W, b, activation='sigmoid')

        # Create fake gradient flowing backward
        dL_dv = np.random.randn(*v.shape)

        # Backward pass
        dL_dW, dL_db, dL_dX = single_layer_backward(dL_dv, u, X, W, activation='sigmoid')

        # Check shapes are consistent
        assert dL_dW.shape == W.shape
        assert dL_db.shape == b.shape
        assert dL_dX.shape == X.shape

    def test_single_layer_integration(self):
        """Test single layer forward-backward integration"""
        X = np.random.randn(4, 3)   # 4 samples, 3 features
        W = np.random.randn(3, 2)   # 3 features -> 2 units
        b = np.random.randn(2)

        # Forward pass
        u, v = single_layer_forward(X, W, b, 'sigmoid')

        # Check shapes
        assert u.shape == (4, 2)
        assert v.shape == (4, 2)

        # Backward pass
        dL_dv = np.random.randn(4, 2)
        dL_dW, dL_db, dL_dX = single_layer_backward(dL_dv, u, X, W, 'sigmoid')

        # Check gradient shapes
        assert dL_dW.shape == W.shape
        assert dL_db.shape == b.shape
        assert dL_dX.shape == X.shape

    def test_training_reduces_loss(self):
        """Test that training actually reduces loss on simple problem"""
        # Create simple linearly separable data
        np.random.seed(42)
        X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
        y = np.array([[1], [1], [0], [0]])  # Binary classification

        # Train single layer (should be able to fit this perfectly)
        weights, loss_history = train_single_layer(
            X, y, activation='sigmoid', loss_type='mse',
            epochs=50, learning_rate=0.1
        )

        # Loss should decrease
        assert loss_history[-1] < loss_history[0]

        # Should converge to reasonable values
        assert loss_history[-1] < 0.5  # Arbitrary reasonable threshold


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_single_sample_batch(self):
        """Test that functions work with single sample"""
        X = np.array([[1, 2]])  # Single sample
        W = np.array([[0.5], [0.3]])
        b = np.array([0.1])

        u = linear_forward(X, W, b)
        assert u.shape == (1, 1)

        v = sigmoid_forward(u)
        assert v.shape == (1, 1)

    def test_zero_initialization(self):
        """Test weight initialization methods"""
        layer_sizes = [3, 5, 2]  # 3 -> 5 -> 2

        # Test different initialization methods
        for method in ['zeros', 'random', 'xavier', 'he']:
            weights = initialize_network_weights(layer_sizes, method=method)

            assert len(weights) == 2  # Should have 2 layers of weights
            assert weights[0][0].shape == (3, 5)  # W1 shape
            assert weights[0][1].shape == (5,)    # b1 shape
            assert weights[1][0].shape == (5, 2)  # W2 shape
            assert weights[1][1].shape == (2,)    # b2 shape

    def test_simple_gradient_checking(self):
        """Test simplified gradient checking function"""
        # Simple setup
        X = np.array([[1, 2], [3, 4]])
        y = np.array([[1], [0]])

        # Forward pass to get analytical gradient
        W = np.array([[0.1], [0.2]])
        b = np.array([0.1])

        u, v = single_layer_forward(X, W, b, 'sigmoid')
        dL_dv = mse_derivative(y, v)
        dL_dW, dL_db, _ = single_layer_backward(dL_dv, u, X, W, 'sigmoid')

        # Define loss function for gradient checking
        def loss_func(W_test):
            u_test, v_test = single_layer_forward(X, W_test, b, 'sigmoid')
            return mse_loss(y, v_test)

        # Test gradient checking
        result = simple_gradient_check(W, dL_dW, loss_func, 'W')

        # Should return expected structure
        assert isinstance(result, dict)
        assert 'param_name' in result
        assert 'numerical_grad' in result
        assert 'analytical_grad' in result
        assert 'passed' in result


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilities:
    """Test utility and data generation functions"""

    def test_generate_classification_data(self):
        """Test synthetic classification data generation"""
        X, y = generate_classification_data(100, n_features=2, n_classes=2, seed=42)

        assert X.shape == (100, 2)
        assert y.shape[0] == 100

        # Should be one-hot encoded for n_classes > 2
        if y.shape[1] == 2:  # Binary classification
            assert np.all(np.sum(y, axis=1) == 1)  # Each row sums to 1

    def test_generate_nonlinear_data(self):
        """Test nonlinear data generation"""
        X, y = generate_nonlinear_data(50, noise=0.1, seed=42)

        assert X.shape[0] == 50
        assert len(y) == 50

        # Should have both classes represented
        unique_classes = np.unique(y)
        assert len(unique_classes) <= 2  # Binary classification

    def test_predict_network_basic(self):
        """Basic test of network prediction function"""
        X = np.random.randn(5, 3)

        # Simple single layer weights
        weights = {
            'W': np.random.randn(3, 1),
            'b': np.random.randn(1)
        }

        predictions = predict_network(X, weights, network_type='single')
        assert predictions.shape[0] == 5  # Should predict for all samples


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])