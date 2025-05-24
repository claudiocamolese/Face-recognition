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
        
        return eigen