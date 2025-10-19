#!/usr/bin/env python3

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from student_code import (
    # Model building functions
    build_sequential_cnn,
    build_functional_inception_cnn,
    # Custom callbacks
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
    # Training functions
    get_optimizer,
    train_model_with_config,
    run_grid_search
)

# Set random seeds for reproducible tests
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def sample_mnist_data():
    """Generate small sample MNIST-like data for testing."""
    n_train, n_val = 100, 20
    X_train = np.random.rand(n_train, 28, 28, 1).astype('float32')
    y_train = keras.utils.to_categorical(np.random.randint(0, 10, n_train), 10)
    X_val = np.random.rand(n_val, 28, 28, 1).astype('float32')
    y_val = keras.utils.to_categorical(np.random.randint(0, 10, n_val), 10)
    return (X_train, y_train), (X_val, y_val)


# ============================================================================
# MODEL BUILDING TESTS
# ============================================================================

class TestModelBuilding:
    """Test CNN model architecture functions"""

    def test_sequential_model_returns_keras_model(self):
        """Test that sequential CNN returns a Keras Model"""
        model = build_sequential_cnn()
        assert isinstance(model, keras.Model), "Should return keras.Model instance"

    def test_sequential_model_input_shape(self):
        """Test sequential model accepts correct input shape"""
        model = build_sequential_cnn(input_shape=(28, 28, 1))
        # Build model to initialize weights
        model.build(input_shape=(None, 28, 28, 1))

        # Test with sample data
        sample_input = np.random.rand(5, 28, 28, 1).astype('float32')
        try:
            output = model(sample_input, training=False)
            assert output.shape == (5, 10), f"Expected output shape (5, 10), got {output.shape}"
        except Exception as e:
            pytest.fail(f"Model failed to process input: {e}")

    def test_sequential_model_output_shape(self):
        """Test sequential model produces correct output shape"""
        model = build_sequential_cnn(input_shape=(28, 28, 1), num_classes=10)
        model.build(input_shape=(None, 28, 28, 1))

        batch_size = 3
        sample_input = np.random.rand(batch_size, 28, 28, 1).astype('float32')
        output = model(sample_input, training=False)

        assert output.shape == (batch_size, 10), \
            f"Expected output shape ({batch_size}, 10), got {output.shape}"

    def test_sequential_model_has_conv_layers(self):
        """Test that sequential model contains Conv2D layers"""
        model = build_sequential_cnn()

        # Check for Conv2D layers
        conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
        assert len(conv_layers) >= 2, "Model should have at least 2 Conv2D layers"

    def test_sequential_model_has_pooling(self):
        """Test that sequential model contains pooling layers"""
        model = build_sequential_cnn()

        # Check for MaxPooling2D layers
        pool_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.MaxPooling2D)]
        assert len(pool_layers) >= 1, "Model should have at least 1 pooling layer"

    def test_sequential_model_has_dense_layers(self):
        """Test that sequential model contains Dense layers"""
        model = build_sequential_cnn()

        # Check for Dense layers
        dense_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
        assert len(dense_layers) >= 2, "Model should have at least 2 Dense layers"

    def test_functional_model_returns_keras_model(self):
        """Test that functional CNN returns a Keras Model"""
        model = build_functional_inception_cnn()
        assert isinstance(model, keras.Model), "Should return keras.Model instance"

    def test_functional_model_input_shape(self):
        """Test functional model accepts correct input shape"""
        model = build_functional_inception_cnn(input_shape=(28, 28, 1))
        model.build(input_shape=(None, 28, 28, 1))

        sample_input = np.random.rand(5, 28, 28, 1).astype('float32')
        try:
            output = model(sample_input, training=False)
            assert output.shape == (5, 10), f"Expected output shape (5, 10), got {output.shape}"
        except Exception as e:
            pytest.fail(f"Model failed to process input: {e}")

    def test_functional_model_has_concatenate(self):
        """Test that functional model uses Concatenate layer (for inception module)"""
        model = build_functional_inception_cnn()

        # Check for Concatenate layer
        concat_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Concatenate)]
        assert len(concat_layers) >= 1, \
            "Model should have at least 1 Concatenate layer (for inception module)"

    def test_functional_model_parameter_count(self):
        """Test that functional model has reasonable parameter count"""
        model = build_functional_inception_cnn()
        model.build(input_shape=(None, 28, 28, 1))

        param_count = model.count_params()
        assert 10000 < param_count < 500000, \
            f"Model should have 10k-500k parameters for CPU efficiency, got {param_count}"

    def test_models_different_architectures(self):
        """Test that sequential and functional models have different structures"""
        seq_model = build_sequential_cnn()
        func_model = build_functional_inception_cnn()

        # They should have different numbers of layers due to inception module
        # Functional model should have more layers due to parallel branches
        assert len(seq_model.layers) != len(func_model.layers), \
            "Sequential and functional models should have different layer counts"


# ============================================================================
# CALLBACK TESTS
# ============================================================================

class TestCallbacks:
    """Test custom callback implementations"""

    def test_early_stopping_inherits_callback(self):
        """Test that EarlyStoppingCallback inherits from keras.callbacks.Callback"""
        callback = EarlyStoppingCallback(patience=3)
        assert isinstance(callback, keras.callbacks.Callback), \
            "EarlyStoppingCallback should inherit from keras.callbacks.Callback"

    def test_early_stopping_attributes(self):
        """Test that EarlyStoppingCallback has required attributes"""
        callback = EarlyStoppingCallback(monitor='val_loss', patience=5)

        assert hasattr(callback, 'monitor'), "Should have 'monitor' attribute"
        assert hasattr(callback, 'patience'), "Should have 'patience' attribute"
        assert hasattr(callback, 'best_value'), "Should have 'best_value' attribute"
        assert hasattr(callback, 'wait'), "Should have 'wait' attribute"

    def test_early_stopping_stops_training(self, sample_mnist_data):
        """Test that early stopping actually stops training"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        # Build simple model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train with early stopping (patience=2)
        callback = EarlyStoppingCallback(monitor='val_loss', patience=2)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,  # Set high number
            callbacks=[callback],
            verbose=0
        )

        # Should stop before 50 epochs
        actual_epochs = len(history.history['loss'])
        assert actual_epochs < 50, \
            f"Early stopping should stop before 50 epochs, but trained for {actual_epochs}"

    def test_lr_scheduler_inherits_callback(self):
        """Test that LearningRateSchedulerCallback inherits from keras.callbacks.Callback"""
        callback = LearningRateSchedulerCallback(initial_lr=0.01)
        assert isinstance(callback, keras.callbacks.Callback), \
            "LearningRateSchedulerCallback should inherit from keras.callbacks.Callback"

    def test_lr_scheduler_attributes(self):
        """Test that LearningRateSchedulerCallback has required attributes"""
        callback = LearningRateSchedulerCallback(initial_lr=0.01, decay_rate=0.5, decay_steps=10)

        assert hasattr(callback, 'initial_lr'), "Should have 'initial_lr' attribute"
        assert hasattr(callback, 'decay_rate'), "Should have 'decay_rate' attribute"
        assert hasattr(callback, 'decay_steps'), "Should have 'decay_steps' attribute"

    def test_lr_scheduler_changes_lr(self, sample_mnist_data):
        """Test that LR scheduler actually changes learning rate"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        # Build simple model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')

        # Track learning rates
        lr_values = []

        class LRTracker(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
                lr_values.append(lr)

        # Train with LR scheduler
        initial_lr = 0.01
        scheduler = LearningRateSchedulerCallback(
            initial_lr=initial_lr,
            decay_rate=0.5,
            decay_steps=5
        )

        model.fit(
            X_train, y_train,
            epochs=15,
            callbacks=[scheduler, LRTracker()],
            verbose=0
        )

        # Check that LR changed
        assert len(set(lr_values)) > 1, "Learning rate should change during training"

        # Check that LR decreased (for decay_rate < 1)
        assert lr_values[-1] < lr_values[0], \
            f"Learning rate should decrease: started at {lr_values[0]}, ended at {lr_values[-1]}"


# ============================================================================
# OPTIMIZER TESTS
# ============================================================================

class TestOptimizers:
    """Test optimizer creation function"""

    def test_get_optimizer_sgd(self):
        """Test creating SGD optimizer"""
        opt = get_optimizer('sgd', 0.01)
        assert isinstance(opt, keras.optimizers.SGD), "Should return SGD optimizer"
        assert abs(float(keras.backend.get_value(opt.learning_rate)) - 0.01) < 1e-6, \
            "Learning rate should be set correctly"

    def test_get_optimizer_adam(self):
        """Test creating Adam optimizer"""
        opt = get_optimizer('adam', 0.001)
        assert isinstance(opt, keras.optimizers.Adam), "Should return Adam optimizer"
        assert abs(float(keras.backend.get_value(opt.learning_rate)) - 0.001) < 1e-6, \
            "Learning rate should be set correctly"

    def test_get_optimizer_rmsprop(self):
        """Test creating RMSprop optimizer"""
        opt = get_optimizer('rmsprop', 0.002)
        assert isinstance(opt, keras.optimizers.RMSprop), "Should return RMSprop optimizer"
        assert abs(float(keras.backend.get_value(opt.learning_rate)) - 0.002) < 1e-6, \
            "Learning rate should be set correctly"

    def test_get_optimizer_invalid_name(self):
        """Test that invalid optimizer name raises error"""
        with pytest.raises(ValueError):
            get_optimizer('invalid_optimizer', 0.01)

    def test_get_optimizer_case_insensitive(self):
        """Test that optimizer names work case-insensitively"""
        opt1 = get_optimizer('adam', 0.001)
        opt2 = get_optimizer('Adam', 0.001)
        opt3 = get_optimizer('ADAM', 0.001)

        assert isinstance(opt1, keras.optimizers.Adam)
        assert isinstance(opt2, keras.optimizers.Adam)
        assert isinstance(opt3, keras.optimizers.Adam)


# ============================================================================
# TRAINING TESTS
# ============================================================================

class TestTraining:
    """Test training functions"""

    def test_train_model_returns_history(self, sample_mnist_data):
        """Test that train_model_with_config returns History object"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(10, activation='softmax')
        ])

        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            learning_rate=0.001,
            batch_size=32,
            epochs=3,
            verbose=0
        )

        assert isinstance(history, keras.callbacks.History), \
            "Should return keras.callbacks.History object"

    def test_train_model_reduces_loss(self, sample_mnist_data):
        """Test that training actually reduces loss"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(10, activation='softmax')
        ])

        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            learning_rate=0.01,
            batch_size=32,
            epochs=10,
            verbose=0
        )

        initial_loss = history.history['loss'][0]
        final_loss = history.history['loss'][-1]

        assert final_loss < initial_loss, \
            f"Training should reduce loss: started at {initial_loss:.4f}, ended at {final_loss:.4f}"

    def test_train_model_has_metrics(self, sample_mnist_data):
        """Test that training history contains expected metrics"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(10, activation='softmax')
        ])

        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            epochs=2,
            verbose=0
        )

        assert 'loss' in history.history, "History should contain 'loss'"
        assert 'accuracy' in history.history, "History should contain 'accuracy'"
        assert 'val_loss' in history.history, "History should contain 'val_loss'"
        assert 'val_accuracy' in history.history, "History should contain 'val_accuracy'"

    def test_train_model_with_callbacks(self, sample_mnist_data):
        """Test that training works with custom callbacks"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28, 1)),
            keras.layers.Dense(10, activation='softmax')
        ])

        callbacks = [EarlyStoppingCallback(patience=3)]

        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            epochs=20,
            callbacks=callbacks,
            verbose=0
        )

        # Should complete without errors
        assert len(history.history['loss']) > 0, "Training should produce history"


# ============================================================================
# GRID SEARCH TESTS
# ============================================================================

class TestGridSearch:
    """Test grid search functionality"""

    def test_grid_search_returns_list(self, sample_mnist_data):
        """Test that grid search returns a list of results"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        # Use small subset for fast test
        X_train_small = X_train[:50]
        y_train_small = y_train[:50]

        def model_builder():
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28, 1)),
                keras.layers.Dense(10, activation='softmax')
            ])
            return model

        param_grid = {
            'optimizer': ['adam'],
            'learning_rate': [0.001, 0.01],
            'batch_size': [16, 32]
        }

        results = run_grid_search(
            model_builder,
            X_train_small, y_train_small,
            X_val, y_val,
            param_grid,
            epochs=2,
            verbose=0
        )

        assert isinstance(results, list), "Should return a list"
        assert len(results) == 4, f"Should have 4 results (2 lrs × 2 batch sizes), got {len(results)}"

    def test_grid_search_result_structure(self, sample_mnist_data):
        """Test that each grid search result has reasonable structure"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        X_train_small = X_train[:50]
        y_train_small = y_train[:50]

        def model_builder():
            return keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28, 1)),
                keras.layers.Dense(10, activation='softmax')
            ])

        param_grid = {
            'optimizer': ['adam'],
            'learning_rate': [0.001],
            'batch_size': [32]
        }

        results = run_grid_search(
            model_builder,
            X_train_small, y_train_small,
            X_val, y_val,
            param_grid,
            epochs=2,
            verbose=0
        )

        assert len(results) > 0, "Should have at least one result"

        result = results[0]
        assert isinstance(result, dict), "Each result should be a dictionary"

        # Check that result includes hyperparameters
        assert 'optimizer' in result, "Result should include optimizer"
        assert 'learning_rate' in result, "Result should include learning_rate"
        assert 'batch_size' in result, "Result should include batch_size"

        # Check that result includes some performance metrics
        # (Don't specify exact keys - students can organize metrics reasonably)
        metric_keys = [k for k in result.keys() if 'loss' in k or 'acc' in k]
        assert len(metric_keys) > 0, "Result should include performance metrics (loss/accuracy)"

    def test_grid_search_multiple_optimizers(self, sample_mnist_data):
        """Test grid search with multiple optimizers"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        X_train_small = X_train[:30]
        y_train_small = y_train[:30]

        def model_builder():
            return keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28, 1)),
                keras.layers.Dense(10, activation='softmax')
            ])

        param_grid = {
            'optimizer': ['sgd', 'adam'],
            'learning_rate': [0.01],
            'batch_size': [32]
        }

        results = run_grid_search(
            model_builder,
            X_train_small, y_train_small,
            X_val, y_val,
            param_grid,
            epochs=2,
            verbose=0
        )

        assert len(results) == 2, f"Should have 2 results (2 optimizers), got {len(results)}"

        optimizers = [r['optimizer'] for r in results]
        assert 'sgd' in optimizers, "Results should include SGD"
        assert 'adam' in optimizers, "Results should include Adam"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test end-to-end workflows"""

    def test_full_training_pipeline_sequential(self, sample_mnist_data):
        """Test complete training pipeline with sequential model"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        # Build model
        model = build_sequential_cnn()

        # Train model
        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            learning_rate=0.001,
            batch_size=32,
            epochs=5,
            verbose=0
        )

        # Check that training worked
        assert len(history.history['loss']) == 5, "Should train for 5 epochs"
        assert history.history['val_accuracy'][-1] > 0, "Should have non-zero accuracy"

    def test_full_training_pipeline_functional(self, sample_mnist_data):
        """Test complete training pipeline with functional model"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        # Build model
        model = build_functional_inception_cnn()

        # Train model
        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            learning_rate=0.001,
            batch_size=32,
            epochs=5,
            verbose=0
        )

        # Check that training worked
        assert len(history.history['loss']) == 5, "Should train for 5 epochs"
        assert history.history['val_accuracy'][-1] > 0, "Should have non-zero accuracy"

    def test_training_with_all_callbacks(self, sample_mnist_data):
        """Test training with both custom callbacks"""
        (X_train, y_train), (X_val, y_val) = sample_mnist_data

        model = build_sequential_cnn()

        callbacks = [
            EarlyStoppingCallback(monitor='val_loss', patience=5),
            LearningRateSchedulerCallback(initial_lr=0.01, decay_rate=0.5, decay_steps=5)
        ]

        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='sgd',
            learning_rate=0.01,
            epochs=20,
            callbacks=callbacks,
            verbose=0
        )

        # Should complete without errors
        assert len(history.history['loss']) > 0, "Training should produce history"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, '-v'])
