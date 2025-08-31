#!env python

import numpy as np
import pytest
import yaml
import sys
from io import StringIO

import student_code


# ============================================================================
# UNIT TESTS - Comprehensive test suite
# ============================================================================

def test_matrix_multiply():
  """Test custom matrix multiplication implementation"""
  # Test basic 2x2 multiplication
  A = np.array([[1, 2], [3, 4]], dtype=float)
  B = np.array([[5, 6], [7, 8]], dtype=float)
  C = student_code.matrix_multiply(A, B)
  expected = np.array([[19, 22], [43, 50]], dtype=float)
  np.testing.assert_array_almost_equal(C, expected)
  
  # Test against numpy's @ operator
  np.testing.assert_array_almost_equal(C, A @ B)
  
  # Test different dimensions
  A_rect = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)  # 2x3
  B_rect = np.array([[7, 8], [9, 10], [11, 12]], dtype=float)  # 3x2
  C_rect = student_code.matrix_multiply(A_rect, B_rect)  # Should be 2x2
  expected_rect = A_rect @ B_rect
  np.testing.assert_array_almost_equal(C_rect, expected_rect)
  
  # Test matrix-vector multiplication
  A_vec = np.array([[1, 2], [3, 4]], dtype=float)
  b_vec = np.array([[5], [6]], dtype=float)  # Column vector
  c_vec = student_code.matrix_multiply(A_vec, b_vec)
  expected_vec = A_vec @ b_vec
  np.testing.assert_array_almost_equal(c_vec, expected_vec)
  
  # Test identity matrix
  I = np.eye(3)
  X = np.random.randn(3, 4)
  result = student_code.matrix_multiply(I, X)
  np.testing.assert_array_almost_equal(result, X)

def test_matrix_multiply_error_cases():
  """Test matrix multiplication error handling"""
  A = np.array([[1, 2], [3, 4]])  # 2x2
  B = np.array([[1], [2], [3]])   # 3x1 - incompatible
  
  # Should raise ValueError for incompatible dimensions
  with pytest.raises(ValueError):
    student_code.matrix_multiply(A, B)

def test_matrix_multiply_with_pca():
  """Test that matrix multiplication works correctly in PCA context"""
  # Generate test data
  np.random.seed(42)
  X = np.random.randn(10, 4)
  
  # Run PCA with our custom matrix multiplication
  X_transformed, components, exp_var_ratio, means = student_code.pca_fit_transform(X, n_components=2)
  
  # Verify results are reasonable
  assert X_transformed.shape == (10, 2)
  assert components.shape == (4, 2)
  
  # Test inverse transform using our matrix multiplication
  X_reconstructed = student_code.pca_inverse_transform(X_transformed, components, means)
  
  # Should have same shape as original
  assert X_reconstructed.shape == X.shape
  
  # With only 2 components, reconstruction won't be perfect but should be reasonable
  # (This is more of a sanity check that our matrix mult didn't break anything)

def test_center_data():
  """Test data centering"""
  X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
  X_centered, means = student_code.center_data(X)
  
  expected_means = np.array([4, 5, 6])
  expected_centered = np.array([[-3, -3, -3], [0, 0, 0], [3, 3, 3]])
  
  np.testing.assert_array_almost_equal(means, expected_means)
  np.testing.assert_array_almost_equal(X_centered, expected_centered)
  np.testing.assert_array_almost_equal(np.mean(X_centered, axis=0), [0, 0, 0])

def test_compute_svd_decomposition():
  """Test SVD computation"""
  X_centered = np.array([[-1, -1], [0, 0], [1, 1]], dtype=float)
  U, sigma, Vt = student_code.compute_svd_decomposition(X_centered)
  
  # Check shapes
  assert U.shape == (3, 2)
  assert sigma.shape == (2,)
  assert Vt.shape == (2, 2)
  
  # Check reconstruction
  reconstructed = U @ np.diag(sigma) @ Vt
  np.testing.assert_array_almost_equal(X_centered, reconstructed)
  
  # Check orthogonality of U and V
  np.testing.assert_array_almost_equal(U.T @ U, np.eye(2), decimal=10)
  np.testing.assert_array_almost_equal(Vt @ Vt.T, np.eye(2), decimal=10)

def test_extract_principal_components():
  """Test component extraction"""
  # Create a known Vt matrix
  Vt = np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                 [1/np.sqrt(2), -1/np.sqrt(2)]])
  
  # Test keeping 1 component
  components = student_code.extract_principal_components(Vt, n_components=1)
  assert components.shape == (2, 1)
  expected_first_comp = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]])
  np.testing.assert_array_almost_equal(components, expected_first_comp)
  
  # Test keeping all components
  components_all = student_code.extract_principal_components(Vt)
  assert components_all.shape == (2, 2)
  np.testing.assert_array_almost_equal(components_all, Vt.T)

def test_calculate_explained_variance():
  """Test variance calculations"""
  sigma = np.array([10.0, 5.0, 1.0])
  n_samples = 100
  
  # Test all components
  exp_var, exp_var_ratio = student_code.calculate_explained_variance(sigma, n_samples)
  
  expected_var = sigma**2 / (n_samples - 1)
  expected_ratio = expected_var / np.sum(expected_var)
  
  np.testing.assert_array_almost_equal(exp_var, expected_var)
  np.testing.assert_array_almost_equal(exp_var_ratio, expected_ratio)
  assert np.isclose(np.sum(exp_var_ratio), 1.0)
  
  # Test subset of components
  exp_var_2, exp_var_ratio_2 = student_code.calculate_explained_variance(sigma, n_samples, n_components=2)
  assert len(exp_var_2) == 2
  assert len(exp_var_ratio_2) == 2
  # Should sum to less than 1 since we're missing one component
  assert np.sum(exp_var_ratio_2) < 1.0

def test_transform_data():
  """Test data transformation"""
  X_centered = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
  
  # Test with identity transformation
  components = np.array([[1, 0], [0, 1]], dtype=float)
  X_transformed = student_code.transform_data(X_centered, components)
  np.testing.assert_array_almost_equal(X_transformed, X_centered)
  
  # Test with rotation (90 degrees)
  components_rot = np.array([[0, -1], [1, 0]], dtype=float)
  X_transformed_rot = student_code.transform_data(X_centered, components_rot)
  expected_rot = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float)
  np.testing.assert_array_almost_equal(X_transformed_rot, expected_rot)

def test_pca_complete():
  """Test complete PCA pipeline"""
  # Create test data with known structure
  np.random.seed(42)
  # Create data that varies more in first dimension
  X = np.random.randn(50, 3)
  X[:, 0] *= 3  # First feature has 3x more variance
  
  X_transformed, components, exp_var_ratio, means = student_code.pca_fit_transform(X, n_components=2)
  
  # Check shapes
  assert X_transformed.shape == (50, 2)
  assert components.shape == (3, 2)
  assert exp_var_ratio.shape == (2,)
  assert means.shape == (3,)
  
  # Check that variance ratios are in descending order
  assert exp_var_ratio[0] > exp_var_ratio[1]
  
  # Check that we explain substantial variance with 2 components
  assert np.sum(exp_var_ratio) > 0.5
  
  # Components should be orthogonal
  np.testing.assert_array_almost_equal(
    components.T @ components,
    np.eye(2),
    decimal=10
  )

def test_transform_new_data():
  """Test transforming new data"""
  # Fit PCA on training data
  np.random.seed(42)
  X_train = np.random.randn(100, 4)
  X_transformed, components, exp_var_ratio, means = student_code.pca_fit_transform(X_train, n_components=2)
  
  # Transform new data
  X_new = np.random.randn(20, 4)
  X_new_transformed = student_code.pca_transform_new_data(X_new, components, means)
  
  # Check shape
  assert X_new_transformed.shape == (20, 2)
  
  # Manual verification: should give same result as centering + transforming
  X_new_centered = X_new - means
  X_new_manual = X_new_centered @ components
  np.testing.assert_array_almost_equal(X_new_transformed, X_new_manual)

def test_inverse_transform():
  """Test inverse transformation"""
  # Create and fit PCA
  np.random.seed(42)
  X_original = np.random.randn(50, 4)
  X_transformed, components, exp_var_ratio, means = student_code.pca_fit_transform(X_original, n_components=4)
  
  # Inverse transform (should recover original data exactly with all components)
  X_reconstructed = student_code.pca_inverse_transform(X_transformed, components, means)
  
  # Should match original data (within numerical precision)
  np.testing.assert_array_almost_equal(X_original, X_reconstructed, decimal=10)
  
  # Test with fewer components (won't be exact reconstruction)
  X_transformed_2, components_2, _, _ = student_code.pca_fit_transform(X_original, n_components=2)
  X_reconstructed_2 = student_code.pca_inverse_transform(X_transformed_2, components_2, means)
  
  # Should be different from original (information loss)
  assert not np.allclose(X_original, X_reconstructed_2, rtol=1e-3)
  # But should have same shape
  assert X_reconstructed_2.shape == X_original.shape

def test_against_sklearn():
  """Test that our implementation matches sklearn"""
  try:
    from sklearn.decomposition import PCA
  except ImportError:
    pytest.skip("sklearn not available")
  
  # Generate test data
  np.random.seed(123)
  X = np.random.randn(100, 5)
  
  # Our implementation
  X_ours, components_ours, exp_var_ratio_ours, means_ours = student_code.pca_fit_transform(X, n_components=3)
  
  # Sklearn implementation
  pca_sklearn = PCA(n_components=3)
  X_sklearn = pca_sklearn.fit_transform(X)
  
  # Compare explained variance ratios (should match exactly)
  np.testing.assert_array_almost_equal(
    exp_var_ratio_ours,
    pca_sklearn.explained_variance_ratio_,
    decimal=10
  )
  
  # Compare means
  np.testing.assert_array_almost_equal(means_ours, pca_sklearn.mean_, decimal=10)
  
  # Components might have different signs, but should span same subspace
  # Check that absolute values of components are close
  for i in range(3):
    # Each component should match some sklearn component (possibly with sign flip)
    our_comp = components_ours[:, i]
    sklearn_comp = pca_sklearn.components_[i, :]
    
    # Check if they match (same direction or opposite direction)
    assert (np.allclose(our_comp, sklearn_comp, atol=1e-10) or
            np.allclose(our_comp, -sklearn_comp, atol=1e-10))

def test_edge_cases():
  """Test edge cases and error conditions"""
  # Test with more features than samples
  X_wide = np.random.randn(10, 20)
  X_trans, comps, ratios, means = student_code.pca_fit_transform(X_wide)
  
  # Should work and have sensible shapes
  assert X_trans.shape[0] == 10
  assert len(ratios) <= 10  # Can't have more components than samples
  
  # Test with single feature
  X_single = np.random.randn(50, 1)
  X_trans_single, comps_single, ratios_single, _ = student_code.pca_fit_transform(X_single)
  
  assert comps_single.shape == (1, 1)
  assert len(ratios_single) == 1
  assert np.isclose(ratios_single[0], 1.0)  # Should explain all variance
