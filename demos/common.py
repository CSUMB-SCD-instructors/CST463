#!env python

import typing
from typing import Callable, Optional, Sequence, Union, Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

ModelOrFactory = Union[keras.Model, Callable[[], keras.Model]]

def plot_multiple_histories(histories : typing.Dict, metric='loss', use_greyscale=True, show_validation=False, sort_agg_func=np.mean, sort_history_length=2, *args, **kwargs):
  """
  Plots the training history of multiple models on the same plot for comparison.

  Parameters:
    histories: List of History objects from multiple model's fit() calls.
    metric: The metric to plot ('loss', 'accuracy', etc.).
    labels: List of labels for the models. If None, defaults to 'Model 1', 'Model 2', etc.
  """
  # Check if labels are provided, otherwise generate default labels
  # if labels is None:
  #   labels = [f'Model {i+1}' for i in range(len(histories))]
  
  # Initialize the plot
  plt.figure(figsize=(10, 6))
  
  norm = plt.Normalize(vmin=0, vmax=len(histories))
  grayscale_cmap = plt.colormaps['gray']
  
  sorted_keys = sorted(histories.keys(), key=(lambda k: sort_agg_func(histories[k].history[metric][:-sort_history_length])))
  for i, label in enumerate(sorted_keys):
    history = histories[label]
    #for i, (label, history) in enumerate(histories.items()):
    epochs = range(1, len(history.history[metric]) + 1)
    
    # Plot training metric
    plt.plot(epochs, history.history[metric], label=f'{label} {metric.capitalize()}', color=grayscale_cmap(norm(i)))
    
    # Plot validation metric if available
    if show_validation and f'val_{metric}' in history.history:
      plt.plot(epochs, history.history[f'val_{metric}'], '--', label=f'{label} Validation {metric.capitalize()}', color=grayscale_cmap(norm(i)))
  
  # Add labels and title
  plt.title(f'Comparison of {metric.capitalize()} Between Models')
  plt.xlabel('Epochs')
  plt.ylabel(metric.capitalize())
  plt.legend()
  
  # Show plot
  plt.show()



def train_model(
    model: keras.Model,
    *,
    X,
    y,
    epochs: int = 100,
    batch_size: int = 128,
    validation_split: float = 0.0,
    validation_data: Optional[Union[Tuple, tf.data.Dataset]] = None,
    callbacks: Optional[List[keras.callbacks.Callback]] = None,
    # compile-if-needed options:
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    loss: Union[str, keras.losses.Loss] = "mse",
    metrics: Sequence[Union[str, keras.metrics.Metric]] = ("mae",),
    compile_kwargs: Optional[Dict] = None,
    fit_kwargs: Optional[Dict] = None,
) -> keras.callbacks.History:
  """Train a prebuilt Keras model on (X, y). Returns History."""
  
  # Compile only if the model isn't already compiled
  if not getattr(model, "_is_compiled", False):
    opt = optimizer or keras.optimizers.SGD(learning_rate=1e-3)
    model.compile(optimizer=opt, loss=loss, metrics=list(metrics), **(compile_kwargs or {}))
  
  history = model.fit(
    X,
    y,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=validation_split,
    validation_data=validation_data,
    callbacks=list(callbacks or []),
    **(fit_kwargs or {}),
    verbose=0
  )
  return history