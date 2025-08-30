#!env python

import pandas as pd
import numpy as np


def load_data(path_to_csv: str) -> np.ndarray:
  df = pd.read_csv(path_to_csv)
  return df.values


def center_data(X):
  """
  Center the data by subtracting the mean of each feature.
  
  Parameters:
  -----------
  X : numpy.ndarray, shape (n_samples, n_features)
      Input data matrix
  
  Returns:
  --------
  X_centered : numpy.ndarray, shape (n_samples, n_features)
      Centered data matrix
  feature_means : numpy.ndarray, shape (n_features,)
      Mean of each feature (needed for transforming new data)
  """
  # Calculate mean along axis 0 (across samples, for each feature)
  feature_means = np.mean(X, axis=0)
  
  # Subtract mean from each sample (broadcasting handles the subtraction)
  X_centered = X - feature_means
  
  return X_centered, feature_means

def compute_svd_decomposition(X_centered):
  """
  Perform Singular Value Decomposition on the centered data.
  
  Parameters:
  -----------
  X_centered : numpy.ndarray, shape (n_samples, n_features)
      Centered data matrix
  
  Returns:
  --------
  U : numpy.ndarray, shape (n_samples, min(n_samples, n_features))
      Left singular vectors
  sigma : numpy.ndarray, shape (min(n_samples, n_features),)
      Singular values
  Vt : numpy.ndarray, shape (min(n_samples, n_features), n_features)
      Right singular vectors (transposed)
  """
  # Use SVD with full_matrices=False for efficiency
  # This returns the "thin" or "reduced" SVD
  U, sigma, Vt = np.linalg.svd(X_centered, full_matrices=False)
  
  return U, sigma, Vt

def extract_principal_components(Vt, n_components=None):
  """
  Extract principal component directions from SVD output.
  
  Parameters:
  -----------
  Vt : numpy.ndarray, shape (min(n_samples, n_features), n_features)
      Right singular vectors from SVD (transposed)
  n_components : int or None
      Number of components to keep. If None, keep all.
  
  Returns:
  --------
  components : numpy.ndarray, shape (n_features, n_components)
      Principal component directions (columns are components)
  """
  # Transpose Vt to get V, where columns are the principal components
  V = Vt.T
  
  # Select the number of components to keep
  if n_components is None:
    n_components = V.shape[1]  # Keep all components
  
  # Return first n_components columns
  components = V[:, :n_components]
  
  return components

def calculate_explained_variance(sigma, n_samples, n_components=None):
  """
  Calculate explained variance and variance ratios from singular values.
  
  Parameters:
  -----------
  sigma : numpy.ndarray, shape (min(n_samples, n_features),)
      Singular values from SVD
  n_samples : int
      Number of samples in original data
  n_components : int or None
      Number of components to keep. If None, keep all.
  
  Returns:
  --------
  explained_variance : numpy.ndarray, shape (n_components,)
      Variance explained by each component
  explained_variance_ratio : numpy.ndarray, shape (n_components,)
      Proportion of total variance explained by each component
  """
  # Convert singular values to explained variance
  # Formula: variance = sigma^2 / (n_samples - 1)
  explained_variance_all = (sigma ** 2) / (n_samples - 1)
  
  # Select components to keep
  if n_components is None:
    n_components = len(explained_variance_all)
  
  explained_variance = explained_variance_all[:n_components]
  
  # Calculate variance ratios (proportion of total variance)
  total_variance = np.sum(explained_variance_all)
  explained_variance_ratio = explained_variance / total_variance
  
  return explained_variance, explained_variance_ratio

def transform_data(X_centered, components):
  """
  Project the centered data onto the principal components.
  
  Parameters:
  -----------
  X_centered : numpy.ndarray, shape (n_samples, n_features)
      Centered data matrix
  components : numpy.ndarray, shape (n_features, n_components)
      Principal component directions
  
  Returns:
  --------
  X_transformed : numpy.ndarray, shape (n_samples, n_components)
      Data projected onto principal components
  """
  # Matrix multiplication: X_centered @ components
  # Each column of components is a principal component direction
  # Result: each row is a data point in the new coordinate system
  X_transformed = X_centered @ components
  
  return X_transformed

def pca_fit_transform(X, n_components=None):
  """
  Perform complete PCA transformation using the functions above.
  
  Parameters:
  -----------
  X : numpy.ndarray, shape (n_samples, n_features)
      Input data matrix
  n_components : int or None
      Number of components to keep
  
  Returns:
  --------
  X_transformed : numpy.ndarray, shape (n_samples, n_components)
      Transformed data
  components : numpy.ndarray, shape (n_features, n_components)
      Principal components
  explained_variance_ratio : numpy.ndarray, shape (n_components,)
      Proportion of variance explained
  feature_means : numpy.ndarray, shape (n_features,)
      Feature means (for transforming new data)
  """
  # Step 1: Center the data
  X_centered, feature_means = center_data(X)
  
  # Step 2: Compute SVD
  U, sigma, Vt = compute_svd_decomposition(X_centered)
  
  # Step 3: Extract principal components
  components = extract_principal_components(Vt, n_components)
  
  # Step 4: Calculate explained variance
  explained_variance, explained_variance_ratio = calculate_explained_variance(
    sigma, X.shape[0], n_components
  )
  
  # Step 5: Transform the data
  X_transformed = transform_data(X_centered, components)
  
  return X_transformed, components, explained_variance_ratio, feature_means

def pca_transform_new_data(X_new, components, feature_means):
  """
  Transform new data using previously fitted PCA.
  
  Parameters:
  -----------
  X_new : numpy.ndarray, shape (n_samples_new, n_features)
      New data to transform
  components : numpy.ndarray, shape (n_features, n_components)
      Principal components from previous fit
  feature_means : numpy.ndarray, shape (n_features,)
      Feature means from previous fit
  
  Returns:
  --------
  X_new_transformed : numpy.ndarray, shape (n_samples_new, n_components)
      Transformed new data
  """
  # Center new data using the same means as training data
  X_new_centered = X_new - feature_means
  
  # Transform using the same components
  X_new_transformed = transform_data(X_new_centered, components)
  
  return X_new_transformed

def pca_inverse_transform(X_transformed, components, feature_means):
  """
  Transform data back to original space.
  
  Parameters:
  -----------
  X_transformed : numpy.ndarray, shape (n_samples, n_components)
      Data in PCA space
  components : numpy.ndarray, shape (n_features, n_components)
      Principal components
  feature_means : numpy.ndarray, shape (n_features,)
      Original feature means
  
  Returns:
  --------
  X_reconstructed : numpy.ndarray, shape (n_samples, n_features)
      Data reconstructed in original space
  """
  # Project back to original space
  X_centered_reconstructed = X_transformed @ components.T
  
  # Add back the original means
  X_reconstructed = X_centered_reconstructed + feature_means
  
  return X_reconstructed

if __name__ == "__main__":
  print("Golden solution for PCA implementation!")
  print("Run 'pytest pca_golden_solution.py -v' to run all tests")
  
  # Quick demo
  print("\n=== Quick Demo ===")
  np.random.seed(42)
  X = np.random.randn(20, 4)
  X_trans, comps, ratios, means = pca_fit_transform(X, n_components=2)
  
  print(f"Original data shape: {X.shape}")
  print(f"Transformed data shape: {X_trans.shape}")
  print(f"Explained variance ratios: {ratios}")
  print(f"Total variance explained: {np.sum(ratios):.3f}")

# def main():
#   arr = load_data()
#
# if __name__ == "__main__":
#   main()