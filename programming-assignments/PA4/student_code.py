#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Dict, List, Tuple, Optional, Any

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================================
# MODEL ARCHITECTURE FUNCTIONS
# ============================================================================

def build_sequential_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1),
                        num_classes: int = 10) -> keras.Model:
    """
    Build a sequential CNN for MNIST classification.

    Your architecture should include convolutional layers, pooling layers, and
    fully connected layers suitable for image classification. Aim for a
    CPU-friendly model with approximately 50k-150k parameters.

    Consider: How many conv/pool layers? What filter sizes? What activations?

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Shape of input images (height, width, channels)
    num_classes : int
        Number of output classes

    Returns
    -------
    keras.Model
        Uncompiled Keras model

    Notes
    -----
    DO NOT compile the model - tests will compile it separately.
    See: https://keras.io/guides/sequential_model/
    """
    # TODO: Implement sequential CNN architecture
    raise NotImplementedError("build_sequential_cnn not yet implemented")


def build_functional_inception_cnn(input_shape: Tuple[int, int, int] = (28, 28, 1),
                                   num_classes: int = 10) -> keras.Model:
    """
    Build a CNN with mini-inception module using Functional API.

    This model must include a mini-inception module: parallel convolutional
    branches that process the same input and are then merged. This demonstrates
    why the Functional API is necessary - Sequential API cannot express
    parallel branches.

    Consider: What filter sizes for the parallel branches? How to merge them?
    What comes before and after the inception module?

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        Shape of input images (height, width, channels)
    num_classes : int
        Number of output classes

    Returns
    -------
    keras.Model
        Keras model built with Functional API

    Notes
    -----
    DO NOT compile the model - tests will compile it separately.
    See: https://keras.io/guides/functional_api/
    """
    # TODO: Implement functional CNN with mini-inception module
    raise NotImplementedError("build_functional_inception_cnn not yet implemented")


# ============================================================================
# CUSTOM CALLBACK IMPLEMENTATIONS
# ============================================================================

class EarlyStoppingCallback(keras.callbacks.Callback):
    """
    Custom early stopping callback that monitors validation loss.

    Stops training when validation loss hasn't improved for `patience` epochs
    and restores the best weights found during training.

    Parameters
    ----------
    monitor : str
        Metric to monitor (e.g., 'val_loss', 'val_accuracy')
    patience : int
        Number of epochs with no improvement after which training will be stopped
    min_delta : float
        Minimum change in monitored value to qualify as an improvement
    restore_best_weights : bool
        Whether to restore model weights from the epoch with the best value

    Attributes
    ----------
    best_value : float
        Best value of monitored metric seen so far
    best_weights : list
        Model weights corresponding to best_value
    wait : int
        Number of epochs since last improvement
    stopped_epoch : int
        Epoch at which training was stopped (0 if not stopped)

    Examples
    --------
    >>> callback = EarlyStoppingCallback(monitor='val_loss', patience=5)
    >>> model.fit(X, y, validation_data=(X_val, y_val), callbacks=[callback])
    """

    def __init__(self, monitor: str = 'val_loss', patience: int = 5,
                 min_delta: float = 0.0, restore_best_weights: bool = True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        # Initialize tracking variables
        self.best_value = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        """
        Reset state at the beginning of training.

        Initialize tracking variables appropriately based on whether you're
        minimizing (loss) or maximizing (accuracy) the monitored metric.

        Hint: What value should best_value start at to ensure the first epoch
        always counts as an improvement? If you're minimizing loss, what's a value
        that any real loss will be better than? If you're maximizing accuracy,
        what's a value that any real accuracy will beat?
        """
        # TODO: Initialize tracking variables
        raise NotImplementedError("EarlyStoppingCallback.on_train_begin not yet implemented")

    def on_epoch_end(self, epoch, logs=None):
        """
        Check if we should stop training after each epoch.

        Implement the patience mechanism: track how many epochs have passed
        without improvement, and stop training if patience is exceeded.
        Remember to save/restore weights if restore_best_weights is True.

        Hint: You need to track two scenarios:
        1. Improvement: What should happen to 'wait' and 'best_weights'?
        2. No improvement: What should happen to 'wait'? What if wait >= patience?
        
        Think about what "improvement" means - should a tiny change count? That's
        what min_delta controls. When should you reset your patience counter?
        """

        # TODO: Implement early stopping logic
        raise NotImplementedError("EarlyStoppingCallback.on_epoch_end not yet implemented")


class LearningRateSchedulerCallback(keras.callbacks.Callback):
    """
    Custom learning rate scheduler that reduces learning rate over time.

    Implements step decay: the learning rate should be reduced by decay_rate
    every decay_steps epochs. For example, if decay_rate=0.5 and decay_steps=10,
    the learning rate should be halved every 10 epochs.

    Parameters
    ----------
    initial_lr : float
        Initial learning rate
    decay_rate : float
        Factor by which to reduce learning rate at each decay step
    decay_steps : int
        Number of epochs between each decay
    """

    def __init__(self, initial_lr: float = 0.01, decay_rate: float = 0.5,
                 decay_steps: int = 10):
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def on_epoch_begin(self, epoch, logs=None):
        """
        Update learning rate at the beginning of each epoch.

        Calculate the new learning rate based on the current epoch and the
        decay schedule, then update the optimizer's learning rate.

        Hint: The formula is lr = initial_lr * (decay_rate ^ num_decays).
        How many times have we decayed by epoch N if we decay every decay_steps epochs?
        Think about integer division.
        """
        # TODO: Implement learning rate scheduling
        raise NotImplementedError("LearningRateSchedulerCallback.on_epoch_begin not yet implemented")


# ============================================================================
# TRAINING AND HYPERPARAMETER SEARCH
# ============================================================================

def get_optimizer(optimizer_name: str, learning_rate: float) -> keras.optimizers.Optimizer:
    """
    Create and return a Keras optimizer instance.

    Support at least: 'sgd', 'adam', 'rmsprop' (case-insensitive).

    Parameters
    ----------
    optimizer_name : str
        Name of optimizer
    learning_rate : float
        Learning rate for the optimizer

    Returns
    -------
    keras.optimizers.Optimizer
        Configured optimizer instance

    Raises
    ------
    ValueError
        If optimizer_name is not recognized
    """
    # TODO: Implement optimizer creation
    raise NotImplementedError("get_optimizer not yet implemented")


def train_model_with_config(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimizer_name: str = 'adam',
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 20,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    verbose: int = 0
) -> keras.callbacks.History:
    """
    Train a model with specified configuration.

    Parameters
    ----------
    model : keras.Model
        Model to train
    X_train, y_train : np.ndarray
        Training data and labels
    X_val, y_val : np.ndarray
        Validation data and labels
    optimizer_name : str
        Name of optimizer to use
    learning_rate : float
        Learning rate for optimizer
    batch_size : int
        Batch size for training
    epochs : int
        Number of epochs to train
    callbacks : list of keras.callbacks.Callback, optional
        List of callbacks to use during training
    verbose : int
        Verbosity level (0=silent, 1=progress bar, 2=one line per epoch)

    Returns
    -------
    keras.callbacks.History
        Training history containing loss and metrics per epoch

    Notes
    -----
    Use categorical crossentropy loss and track accuracy.
    Return the History object from model.fit().
    """
    # TODO: Implement training with configuration
    raise NotImplementedError("train_model_with_config not yet implemented")


def run_grid_search(
    model_builder_func,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    param_grid: Dict[str, List[Any]],
    epochs: int = 15,
    verbose: int = 0
) -> List[Dict[str, Any]]:
    """
    Run grid search over hyperparameters.

    Parameters
    ----------
    model_builder_func : callable
        Function that builds and returns a fresh model (not compiled)
    X_train, y_train : np.ndarray
        Training data and labels
    X_val, y_val : np.ndarray
        Validation data and labels
    param_grid : dict
        Dictionary mapping parameter names to lists of values to try
        Example: {
            'optimizer': ['sgd', 'adam'],
            'learning_rate': [0.001, 0.01],
            'batch_size': [32, 64]
        }
    epochs : int
        Number of epochs to train each configuration
    verbose : int
        Verbosity level for training

    Returns
    -------
    list of dict
        List of results, one dict per configuration. Each result dict should
        include the hyperparameters used and the final training/validation
        metrics achieved.

    Notes
    -----
    Generate all combinations of parameters from param_grid, train a fresh
    model for each combination, and collect results.
    """
    # TODO: Implement grid search
    raise NotImplementedError("run_grid_search not yet implemented")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("PA4: Convolutional Neural Networks & Training Dynamics")
    print("Run 'pytest tests.py -v' to test your implementations")

    # Quick demo when students have implemented functions
    print("\n=== Quick Demo ===")
    try:
        from utils import load_mnist_data

        # Load small subset for quick test
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_data(
            train_samples=1000,
            val_samples=200,
            test_samples=200
        )

        print(f"Data shapes: X_train={X_train.shape}, y_train={y_train.shape}")

        # Build and train simple model
        print("\nBuilding sequential CNN...")
        model = build_sequential_cnn()
        print(f"Model has {model.count_params():,} parameters")

        print("\nTraining for 5 epochs...")
        history = train_model_with_config(
            model, X_train, y_train, X_val, y_val,
            optimizer_name='adam',
            learning_rate=0.001,
            batch_size=32,
            epochs=5,
            verbose=1
        )

        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print("Demo completed successfully!")

    except NotImplementedError as e:
        print(f"Demo requires function implementations: {e}")
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
