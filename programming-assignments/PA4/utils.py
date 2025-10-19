#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from typing import Tuple, List, Dict, Optional, Any

# Random seed for reproducibility
RANDOM_SEED = 42


def load_mnist_data(
    train_samples: int = 10000,
    val_samples: int = 2000,
    test_samples: int = 10000,
    seed: int = RANDOM_SEED
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Load and preprocess MNIST dataset.

    This function loads MNIST, normalizes pixel values to [0, 1],
    reshapes to add channel dimension, converts labels to one-hot encoding,
    and optionally subsamples for CPU-feasible training.

    Parameters
    ----------
    train_samples : int
        Number of training samples to use (max 60,000)
    val_samples : int
        Number of validation samples to use
    test_samples : int
        Number of test samples to use (max 10,000)
    seed : int
        Random seed for reproducible sampling

    Returns
    -------
    (X_train, y_train), (X_val, y_val), (X_test, y_test)
        Training, validation, and test sets
        X arrays have shape (n_samples, 28, 28, 1)
        y arrays have shape (n_samples, 10) - one-hot encoded

    Examples
    --------
    >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist_data()
    >>> print(X_train.shape)  # (10000, 28, 28, 1)
    >>> print(y_train.shape)  # (10000, 10)
    """
    # Load MNIST dataset from Keras
    (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test_full = X_test_full.astype('float32') / 255.0

    # Reshape to add channel dimension: (n, 28, 28) -> (n, 28, 28, 1)
    X_train_full = X_train_full.reshape(-1, 28, 28, 1)
    X_test_full = X_test_full.reshape(-1, 28, 28, 1)

    # Convert labels to one-hot encoding
    y_train_full = keras.utils.to_categorical(y_train_full, 10)
    y_test_full = keras.utils.to_categorical(y_test_full, 10)

    # Set random seed for reproducible sampling
    np.random.seed(seed)

    # Sample training data
    if train_samples < len(X_train_full):
        train_indices = np.random.choice(len(X_train_full), train_samples, replace=False)
        X_train = X_train_full[train_indices]
        y_train = y_train_full[train_indices]
    else:
        X_train = X_train_full
        y_train = y_train_full

    # Sample validation data (from remaining training data)
    # Get indices not used for training
    all_indices = set(range(len(X_train_full)))
    train_indices_set = set(train_indices) if train_samples < len(X_train_full) else set()
    remaining_indices = list(all_indices - train_indices_set)

    if val_samples <= len(remaining_indices):
        val_indices = np.random.choice(remaining_indices, val_samples, replace=False)
        X_val = X_train_full[val_indices]
        y_val = y_train_full[val_indices]
    else:
        # If not enough remaining, sample from full training set
        val_indices = np.random.choice(len(X_train_full), val_samples, replace=False)
        X_val = X_train_full[val_indices]
        y_val = y_train_full[val_indices]

    # Sample test data
    if test_samples < len(X_test_full):
        test_indices = np.random.choice(len(X_test_full), test_samples, replace=False)
        X_test = X_test_full[test_indices]
        y_test = y_test_full[test_indices]
    else:
        X_test = X_test_full
        y_test = y_test_full

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plot_training_history(
    history: keras.callbacks.History,
    metrics: List[str] = ['loss', 'accuracy'],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot training history using matplotlib only.

    Creates subplots for each metric showing both training and validation curves.

    Parameters
    ----------
    history : keras.callbacks.History
        History object returned from model.fit()
    metrics : list of str
        Metrics to plot (e.g., ['loss', 'accuracy'])
    title : str, optional
        Overall figure title
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plots

    Examples
    --------
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
    >>> fig = plot_training_history(history)
    >>> plt.show()
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    # Handle single metric case
    if n_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot training metric
        train_key = metric
        if train_key in history.history:
            epochs = range(1, len(history.history[train_key]) + 1)
            ax.plot(epochs, history.history[train_key], 'b-', label=f'Training {metric}', linewidth=2)

        # Plot validation metric
        val_key = f'val_{metric}'
        if val_key in history.history:
            epochs = range(1, len(history.history[val_key]) + 1)
            ax.plot(epochs, history.history[val_key], 'r--', label=f'Validation {metric}', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()
    return fig


def plot_sample_predictions(
    model: keras.Model,
    X_samples: np.ndarray,
    y_true: np.ndarray,
    n_samples: int = 10,
    figsize: Tuple[int, int] = (15, 3)
) -> plt.Figure:
    """
    Visualize model predictions on sample images using matplotlib.

    Parameters
    ----------
    model : keras.Model
        Trained model
    X_samples : np.ndarray
        Sample images, shape (n, 28, 28, 1)
    y_true : np.ndarray
        True labels, one-hot encoded shape (n, 10)
    n_samples : int
        Number of samples to display
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure showing images with predicted and true labels
    """
    # Get predictions
    y_pred = model.predict(X_samples[:n_samples], verbose=0)
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true[:n_samples], axis=1)

    # Create figure
    fig, axes = plt.subplots(1, n_samples, figsize=figsize)

    for idx in range(n_samples):
        ax = axes[idx] if n_samples > 1 else axes

        # Display image
        img = X_samples[idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')

        # Set title with prediction and true label
        pred = pred_classes[idx]
        true = true_classes[idx]
        color = 'green' if pred == true else 'red'
        ax.set_title(f'Pred: {pred}\nTrue: {true}', color=color, fontsize=10)

        ax.axis('off')

    plt.tight_layout()
    return fig


def plot_grid_search_results(
    results: List[Dict[str, Any]],
    x_param: str = 'learning_rate',
    color_param: str = 'optimizer',
    metric: str = 'final_val_acc',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Visualize grid search results using matplotlib bar plots.

    Parameters
    ----------
    results : list of dict
        Results from run_grid_search()
    x_param : str
        Parameter to use for x-axis grouping
    color_param : str
        Parameter to use for color grouping
    metric : str
        Metric to plot (e.g., 'final_val_acc', 'final_val_loss')
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with bar plot of results

    Examples
    --------
    >>> results = run_grid_search(...)
    >>> fig = plot_grid_search_results(results, x_param='learning_rate',
    ...                                color_param='optimizer', metric='final_val_acc')
    >>> plt.show()
    """
    # Extract unique values for grouping
    x_values = sorted(set(r[x_param] for r in results))
    color_values = sorted(set(r[color_param] for r in results))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up bar positions
    x_positions = np.arange(len(x_values))
    bar_width = 0.8 / len(color_values)

    # Define colors for different groups
    colors = plt.cm.Set3(np.linspace(0, 1, len(color_values)))

    # Plot bars for each color group
    for idx, color_val in enumerate(color_values):
        # Filter results for this color group
        group_results = [r for r in results if r[color_param] == color_val]

        # Get metric values for each x position
        metric_values = []
        for x_val in x_values:
            matching = [r[metric] for r in group_results if r[x_param] == x_val]
            metric_values.append(np.mean(matching) if matching else 0)

        # Plot bars
        offset = (idx - len(color_values)/2 + 0.5) * bar_width
        ax.bar(x_positions + offset, metric_values, bar_width,
               label=f'{color_param}={color_val}', color=colors[idx])

    # Customize plot
    ax.set_xlabel(x_param, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'Grid Search Results: {metric} by {x_param} and {color_param}', fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_values)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_overfitting_comparison(
    histories: Dict[str, keras.callbacks.History],
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Compare training dynamics across different scenarios using matplotlib.

    Useful for showing baseline, induced overfitting, and mitigation scenarios.

    Parameters
    ----------
    histories : dict
        Dictionary mapping scenario names to History objects
        Example: {
            'Baseline': history1,
            'Small Dataset (Overfitting)': history2,
            'With Early Stopping': history3
        }
    metric : str
        Metric to plot ('accuracy' or 'loss')
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with training and validation curves for each scenario
    """
    n_scenarios = len(histories)
    fig, axes = plt.subplots(1, n_scenarios, figsize=figsize)

    if n_scenarios == 1:
        axes = [axes]

    for idx, (name, history) in enumerate(histories.items()):
        ax = axes[idx]

        # Plot training and validation curves
        train_key = metric
        val_key = f'val_{metric}'

        if train_key in history.history:
            epochs = range(1, len(history.history[train_key]) + 1)
            ax.plot(epochs, history.history[train_key], 'b-',
                   label=f'Train {metric}', linewidth=2)

        if val_key in history.history:
            epochs = range(1, len(history.history[val_key]) + 1)
            ax.plot(epochs, history.history[val_key], 'r--',
                   label=f'Val {metric}', linewidth=2)

        # Calculate and display final gap
        if train_key in history.history and val_key in history.history:
            final_train = history.history[train_key][-1]
            final_val = history.history[val_key][-1]
            gap = abs(final_train - final_val)
            ax.text(0.95, 0.05, f'Gap: {gap:.4f}',
                   transform=ax.transAxes, ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Overfitting Analysis: {metric.capitalize()}', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def compare_optimizers(
    histories: Dict[str, keras.callbacks.History],
    metrics: List[str] = ['loss', 'accuracy'],
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Compare different optimizers on the same plot using matplotlib.

    Parameters
    ----------
    histories : dict
        Dictionary mapping optimizer names to History objects
    metrics : list of str
        Metrics to compare
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Figure with comparison plots
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    # Define colors and styles for different optimizers
    colors = plt.cm.Set2(np.linspace(0, 1, len(histories)))
    linestyles = ['-', '--', '-.', ':']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot each optimizer
        for opt_idx, (opt_name, history) in enumerate(histories.items()):
            val_key = f'val_{metric}'

            if val_key in history.history:
                epochs = range(1, len(history.history[val_key]) + 1)
                ax.plot(epochs, history.history[val_key],
                       color=colors[opt_idx],
                       linestyle=linestyles[opt_idx % len(linestyles)],
                       label=opt_name, linewidth=2, marker='o', markersize=3)

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(f'Validation {metric.capitalize()}', fontsize=12)
        ax.set_title(f'Optimizer Comparison: {metric.capitalize()}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
