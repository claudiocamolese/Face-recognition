"""Custom PCA implementation using eigenvalue decomposition."""

import numpy as np
from typing import Optional, Tuple
import warnings


class CustomPCA:
    """Custom implementation of Principal Component Analysis."""
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize CustomPCA.
        
        Args:
            tolerance: Numerical tolerance for computations
        """
        self.tolerance = tolerance
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
    def _center_data(self, X: np.ndarray) -> np.ndarray:
        """
        Center the data by subtracting the mean.
        
        Args:
            X: Input data matrix
            
        Returns:
            Centered data matrix
        """
        if self.mean_ is None:
            self.mean_ = np.mean(X, axis=0)
        return X - self.mean_
    
    def _compute_covariance_matrix(self, X_centered: np.ndarray) -> np.ndarray:
        """
        Compute the covariance matrix.
        
        Args:
            X_centered: Centered data matrix
            
        Returns:
            Covariance matrix
        """
        n_samples = X_centered.shape[0]
        if n_samples <= 1:
            return np.zeros((X_centered.shape[1], X_centered.shape[1]))
        
        return (X_centered.T @ X_centered) / (n_samples - 1)
    
    def _power_iteration_eigenvector(self, A: np.ndarray, max_iterations: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Find the dominant eigenvector using power iteration.
        
        Args:
            A: Symmetric matrix
            max_iterations: Maximum number of iterations
            
        Returns:
            Dominant eigenvector and eigenvalue
        """
        n = A.shape[0]
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for i in range(max_iterations):
            v_new = A @ v
            eigenvalue = np.dot(v, v_new)
            
            if np.linalg.norm(v_new) < self.tolerance:
                break
                
            v_new = v_new / np.linalg.norm(v_new)
            
            if np.linalg.norm(v_new - v) < self.tolerance:
                break
                
            v = v_new
            
        eigenvalue = np.dot(v, A @ v)
        return v, eigenvalue
    
    def _deflation(self, A: np.ndarray, eigenvalue: float, eigenvector: np.ndarray) -> np.ndarray:
        """
        Remove the influence of a found eigenvector from the matrix.
        
        Args:
            A: Original matrix
            eigenvalue: Found eigenvalue
            eigenvector: Found eigenvector
            
        Returns:
            Deflated matrix
        """
        return A - eigenvalue * np.outer(eigenvector, eigenvector)
    
    def _custom_eigh(self, A: np.ndarray, n_components: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Custom eigenvalue decomposition for symmetric matrices.
        
        Args:
            A: Symmetric matrix
            n_components: Number of components to find
            
        Returns:
            Eigenvalues and eigenvectors (sorted in descending order)
        """
        if n_components is None:
            n_components = A.shape[0]
        
        n_components = min(n_components, A.shape[0])
        eigenvalues = []
        eigenvectors = []
        A_deflated = A.copy()
        
        for i in range(n_components):
            try:
                eigenvector, eigenvalue = self._power_iteration_eigenvector(A_deflated)
                
                if eigenvalue < self.tolerance:
                    break
                    
                eigenvalues.append(eigenvalue)
                eigenvectors.append(eigenvector)
                
                # Deflate the matrix
                A_deflated = self._deflation(A_deflated, eigenvalue, eigenvector)
                
            except (np.linalg.LinAlgError, ZeroDivisionError):
                break
        
        if len(eigenvalues) == 0:
            # Fallback to numpy's implementation
            eigenvals, eigenvecs = np.linalg.eigh(A)
            idx = np.argsort(eigenvals)[::-1]
            return eigenvals[idx][:n_components], eigenvecs[:, idx][:, :n_components]
        
        eigenvalues = np.array(eigenvalues)
        eigenvectors = np.column_stack(eigenvectors)
        
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def fit(self, X: np.ndarray, n_components: Optional[int] = None) -> 'CustomPCA':
        """
        Fit the PCA model to the data.
        
        Args:
            X: Input data matrix of shape (n_samples, n_features)
            n_components: Number of components to keep
            
        Returns:
            Self
        """
        X = np.asarray(X)
        self.n_samples_, self.n_features_ = X.shape
        
        if n_components is None:
            n_components = min(self.n_samples_, self.n_features_)
        
        self.n_components_ = min(n_components, self.n_features_)
        
        # Center the data
        X_centered = self._center_data(X)
        
        # Compute covariance matrix
        cov_matrix = self._compute_covariance_matrix(X_centered)
        
        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = self._custom_eigh(cov_matrix, self.n_components_)
        
        # Store results
        self.explained_variance_ = eigenvalues
        self.components_ = eigenvectors.T  # Components are rows
        
        # Compute explained variance ratio
        total_variance = np.sum(eigenvalues) if len(eigenvalues) > 0 else 1.0
        self.explained_variance_ratio_ = eigenvalues / total_variance
        
        # Compute singular values (relationship: singular_value = sqrt(eigenvalue * (n_samples - 1)))
        self.singular_values_ = np.sqrt(eigenvalues * (self.n_samples_ - 1))
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to lower dimensional space.
        
        Args:
            X: Input data matrix
            
        Returns:
            Transformed data
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = np.asarray(X)
        X_centered = X - self.mean_
        
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray, n_components: Optional[int] = None) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Args:
            X: Input data matrix
            n_components: Number of components to keep
            
        Returns:
            Transformed data
        """
        return self.fit(X, n_components).transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X_transformed: Transformed data
            
        Returns:
            Data in original space
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X_transformed = np.asarray(X_transformed)
        
        # Reconstruct in original space
        X_reconstructed = X_transformed @ self.components_
        
        # Add back the mean
        return X_reconstructed + self.mean_
    
    def get_covariance(self) -> np.ndarray:
        """
        Compute data covariance with the fitted model.
        
        Returns:
            Estimated covariance matrix
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return (self.components_.T * self.explained_variance_) @ self.components_
    
    def get_precision(self) -> np.ndarray:
        """
        Compute data precision matrix with the fitted model.
        
        Returns:
            Estimated precision matrix
        """
        covariance = self.get_covariance()
        
        # Add small regularization for numerical stability
        reg = 1e-12 * np.eye(covariance.shape[0])
        
        try:
            precision = np.linalg.inv(covariance + reg)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            precision = np.linalg.pinv(covariance + reg)
        
        return precision
    
    def score(self, X: np.ndarray) -> float:
        """
        Return the average log-likelihood of the data.
        
        Args:
            X: Input data matrix
            
        Returns:
            Average log-likelihood
        """
        if self.components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = np.asarray(X)
        X_centered = X - self.mean_
        
        # Transform and inverse transform to get reconstruction
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        
        # Compute reconstruction error
        reconstruction_error = np.sum((X - X_reconstructed) ** 2, axis=1)
        
        # Return negative mean squared error as score
        return -np.mean(reconstruction_error)
    
    def explained_variance_ratio_cumsum(self) -> np.ndarray:
        """
        Get cumulative explained variance ratio.
        
        Returns:
            Cumulative explained variance ratio
        """
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return np.cumsum(self.explained_variance_ratio_)
    
    def get_feature_names_out(self, input_features: Optional[list] = None) -> list:
        """
        Get output feature names for transformation.
        
        Args:
            input_features: Input feature names
            
        Returns:
            Output feature names
        """
        if self.n_components_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        return [f"pca{i}" for i in range(self.n_components_)]