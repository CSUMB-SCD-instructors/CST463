#!env python

import pandas as pd
import numpy as np


def matrix_multiply(A, B):
  """
  Perform matrix multiplication C = A @ B using nested loops.
  
  Students must implement this function without using numpy's @ operator
  or np.dot(). This helps internalize how matrix multiplication works.
  
  Parameters:
  -----------
  A : numpy.ndarray, shape (m, n)
      Left matrix
  B : numpy.ndarray, shape (n, p)  
      Right matrix
  
  Returns:
  --------
  C : numpy.ndarray, shape (m, p)
      Result matrix where C[i,j] = sum(A[i,k] * B[k,j] for k in range(n))
  
  Notes:
  ------
  - A.shape[1] must equal B.shape[0] for multiplication to be valid
  - Use nested loops to compute each element C[i,j]
  - Think about the mathematical definition: C[i,j] = Σ(A[i,k] * B[k,j])
  """
  # TODO
  # Hint: You'll need three nested loops (i, j, k)
  # Hint: Initialize result matrix with zeros first
  # Hint: Check that A.shape[1] == B.shape[0] for valid multiplication
  # Hint: Use A.shape to get dimensions (m, n) and B.shape to get (n, p)
  # Hint: Result matrix C should have shape (m, p)
  
  pass

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
  # TODO
  # Hint: Calculate mean along axis=0 (across samples, for each feature)
  # Hint: Use np.mean(X, axis=0) to get feature means
  # Hint: Subtract the mean from X using broadcasting: X - feature_means
  # Hint: Return both the centered data and the means (needed later)
  
  pass

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
  # TODO
  # Hint: Use np.linalg.svd() to perform Singular Value Decomposition
  # Hint: Set full_matrices=False for efficiency ("thin" SVD)
  # Hint: SVD returns U, sigma, Vt (note: V is transposed!)
  # Hint: U contains left singular vectors, sigma contains singular values, Vt contains right singular vectors (transposed)
  # Hint: This might be a great chance to dig into what SVD does!
  
  pass

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
  # TODO
  # Hint: Vt is transposed, so use Vt.T to get V where columns are principal components
  # Hint: If n_components is None, keep all components (V.shape[1])
  # Hint: Select first n_components columns: V[:, :n_components]
  # Hint: The result should have shape (n_features, n_components)
  
  pass

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
  # TODO
  # Hint: Convert singular values to variance using: sigma^2 / (n_samples - 1)
  # Hint: If n_components is None, use all components: len(explained_variance_all)
  # Hint: Select first n_components from explained_variance_all
  # Hint: Calculate ratios by dividing by total variance: explained_variance / total_variance
  # Hint: Total variance is the sum of ALL explained variances (not just selected ones)
  
  pass

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
  # TODO
  # Hint: Use your matrix_multiply function: matrix_multiply(X_centered, components)
  # Hint: This projects the centered data onto the principal component directions
  # Hint: Each column of components is a principal component direction
  # Hint: Result shape should be (n_samples, n_components)
  
  pass

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
  # TODO
  # This is the main PCA pipeline - use the functions you implemented above!
  # 
  # PCA Algorithm Steps:
  # 1. Center the data using center_data(X)
  # 2. Compute SVD using compute_svd_decomposition(X_centered) 
  # 3. Extract principal components using extract_principal_components(Vt, n_components)
  # 4. Calculate explained variance using calculate_explained_variance(sigma, n_samples, n_components)
  # 5. Transform the data using transform_data(X_centered, components)
  #
  # Hint: n_samples = X.shape[0]
  # Hint: Return X_transformed, components, explained_variance_ratio, feature_means
  
  pass

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
  # TODO
  # Hint: Center new data using the SAME means from training: X_new - feature_means
  # Hint: Transform using the same components from training: transform_data(X_new_centered, components)
  # Hint: Important: Don't recompute means - use the ones from training!
  
  pass

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
  # TODO
  # Hint: Project back to original space: matrix_multiply(X_transformed, components.T)
  # Hint: Note the .T (transpose) - we multiply by components transpose
  # Hint: Add back the original feature means to get final reconstruction
  # Hint: This reverses the centering step from the forward transform
  
  pass

if __name__ == "__main__":
  print("PCA implementation starter code")
  print("Implement the functions above and run tests with: pytest tests.py -v")
  print("\nOnce implemented, you can test with:")
  print("python student_code.py")

# def main():
#   arr = load_data()
#
# if __name__ == "__main__":
#   main()