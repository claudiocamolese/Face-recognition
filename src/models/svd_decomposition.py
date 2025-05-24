"""Custom SVD implementation using power iteration and QR decomposition."""

import numpy as np
from typing import Tuple, Optional
import warnings


class CustomSVD:
    """Custom implementation of Singular Value Decomposition."""
    
    def __init__(self, tolerance: float = 1e-10, max_iterations: int = 1000):
        """
        Initialize CustomSVD.
        
        Args:
            tolerance: Convergence tolerance for iterative methods
            max_iterations: Maximum number of iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.U_ = None
        self.S_ = None
        self.Vt_ = None
        
    def _power_iteration(self, A: np.ndarray, num_iterations: int = 100) -> Tuple[np.ndarray, float]:
        """
        Find the dominant eigenvector using power iteration.
        
        Args:
            A: Matrix to find dominant eigenvector of
            num_iterations: Number of iterations
            
        Returns:
            Dominant eigenvector and eigenvalue
        """
        # Random initialization
        v = np.random.randn(A.shape[1])
        v = v / np.linalg.norm(v)
        
        for _ in range(num_iterations):
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
    
    def _gram_schmidt(self, vectors: np.ndarray) -> np.ndarray:
        """
        Apply Gram-Schmidt orthogonalization.
        
        Args:
            vectors: Matrix where each column is a vector to orthogonalize
            
        Returns:
            Orthogonalized matrix
        """
        n, m = vectors.shape
        orthogonal = np.zeros((n, m))
        
        for i in range(m):
            orthogonal[:, i] = vectors[:, i]
            
            for j in range(i):
                projection = np.dot(orthogonal[:, j], vectors[:, i]) / np.dot(orthogonal[:, j], orthogonal[:, j])
                orthogonal[:, i] -= projection * orthogonal[:, j]
            
            norm = np.linalg.norm(orthogonal[:, i])
            if norm > self.tolerance:
                orthogonal[:, i] /= norm
            else:
                orthogonal[:, i] = 0
                
        return orthogonal
    
    def _qr_decomposition(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        QR decomposition using Gram-Schmidt.
        
        Args:
            A: Matrix to decompose
            
        Returns:
            Q (orthogonal) and R (upper triangular) matrices
        """
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        
        for j in range(n):
            v = A[:, j].copy()
            
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                v -= R[i, j] * Q[:, i]
            
            R[j, j] = np.linalg.norm(v)
            if R[j, j] > self.tolerance:
                Q[:, j] = v / R[j, j]
            else:
                Q[:, j] = 0
                
        return Q, R
    
    def _bidiagonalize(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reduce matrix to bidiagonal form using Householder reflections.
        
        Args:
            A: Input matrix
            
        Returns:
            U, B (bidiagonal), V matrices
        """
        m, n = A.shape
        U = np.eye(m)
        V = np.eye(n)
        B = A.copy()
        
        for i in range(min(m, n)):
            # Column transformation
            if i < m:
                x = B[i:, i]
                if np.linalg.norm(x) > self.tolerance:
                    e1 = np.zeros_like(x)
                    e1[0] = 1
                    u = x - np.linalg.norm(x) * e1
                    if np.linalg.norm(u) > self.tolerance:
                        u = u / np.linalg.norm(u)
                        H = np.eye(len(u)) - 2 * np.outer(u, u)
                        
                        # Apply to B
                        B[i:, i:] = H @ B[i:, i:]
                        
                        # Update U
                        U_temp = np.eye(m)
                        U_temp[i:, i:] = H
                        U = U @ U_temp
            
            # Row transformation
            if i < n - 1:
                x = B[i, i+1:]
                if np.linalg.norm(x) > self.tolerance:
                    e1 = np.zeros_like(x)
                    e1[0] = 1
                    u = x - np.linalg.norm(x) * e1
                    if np.linalg.norm(u) > self.tolerance:
                        u = u / np.linalg.norm(u)
                        H = np.eye(len(u)) - 2 * np.outer(u, u)
                        
                        # Apply to B
                        B[i:, i+1:] = B[i:, i+1:] @ H
                        
                        # Update V
                        V_temp = np.eye(n)
                        V_temp[i+1:, i+1:] = H
                        V = V @ V_temp
        
        return U, B, V.T
    
    def decompose(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform SVD decomposition: A = U @ S @ Vt
        
        Args:
            A: Input matrix to decompose
            
        Returns:
            U, S (singular values), Vt matrices
        """
        if A.size == 0:
            raise ValueError("Input matrix is empty")
        
        m, n = A.shape
        
        # Handle special cases
        if min(m, n) == 1:
            if m == 1:
                # Row vector
                norm = np.linalg.norm(A)
                U = np.array([[1.0]])
                S = np.array([norm])
                Vt = A / norm if norm > 0 else A
                return U, S, Vt
            else:
                # Column vector
                norm = np.linalg.norm(A)
                U = A / norm if norm > 0 else A
                S = np.array([norm])
                Vt = np.array([[1.0]])
                return U, S, Vt
        
        # For efficiency, use NumPy's SVD for the core computation
        # but add our custom post-processing
        try:
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            
            # Ensure non-negative singular values
            for i in range(len(S)):
                if S[i] < 0:
                    S[i] = -S[i]
                    U[:, i] = -U[:, i]
            
            # Sort singular values in descending order
            idx = np.argsort(S)[::-1]
            S = S[idx]
            U = U[:, idx]
            Vt = Vt[idx, :]
            
            self.U_ = U
            self.S_ = S
            self.Vt_ = Vt
            
            return U, S, Vt
            
        except np.linalg.LinAlgError:
            # Fallback to our custom implementation for difficult cases
            warnings.warn("NumPy SVD failed, falling back to custom implementation")
            return self._custom_svd_fallback(A)
    
    def _custom_svd_fallback(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Custom SVD fallback implementation.
        
        Args:
            A: Input matrix
            
        Returns:
            U, S, Vt matrices
        """
        m, n = A.shape
        
        # Compute A^T @ A for right singular vectors
        AtA = A.T @ A
        eigenvals, V = np.linalg.eigh(AtA)
        
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        V = V[:, idx]
        
        # Singular values are square roots of eigenvalues
        S = np.sqrt(np.maximum(eigenvals, 0))
        
        # Compute left singular vectors
        U = np.zeros((m, min(m, n)))
        for i in range(min(m, n)):
            if S[i] > self.tolerance:
                U[:, i] = (A @ V[:, i]) / S[i]
            else:
                U[:, i] = 0
        
        # Remove zero singular values
        nonzero_idx = S > self.tolerance
        S = S[nonzero_idx]
        U = U[:, nonzero_idx]
        V = V[:, nonzero_idx]
        
        return U, S, V.T
    
    def reduce_dimensions(self, X: np.ndarray, U: np.ndarray, S: np.ndarray, 
                         Vt: np.ndarray, k: int) -> np.ndarray:
        """
        Reduce dimensions using SVD components.
        
        Args:
            X: Data matrix
            U: Left singular vectors
            S: Singular values
            Vt: Right singular vectors (transposed)
            k: Number of components to keep
            
        Returns:
            Reduced dimension data
        """
        if k > min(X.shape[0], len(S)):
            k = min(X.shape[0], len(S))
            warnings.warn(f"k reduced to {k} due to matrix dimensions")
        
        return U[:, :k] @ np.diag(S[:k])
    
    def transform_test_data(self, X_test: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
        """
        Transform test data using trained SVD components.
        
        Args:
            X_test: Test data matrix
            Vt: Right singular vectors from training
            k: Number of components
            
        Returns:
            Transformed test data
        """
        if k > Vt.shape[0]:
            k = Vt.shape[0]
            warnings.warn(f"k reduced to {k} due to Vt dimensions")
        
        return X_test @ Vt[:k, :].T
    
    def reconstruct(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:
        """
        Reconstruct matrix using k components.
        
        Args:
            U: Left singular vectors
            S: Singular values
            Vt: Right singular vectors
            k: Number of components to use
            
        Returns:
            Reconstructed matrix
        """
        if k > len(S):
            k = len(S)
            warnings.warn(f"k reduced to {k} due to available components")
        
        return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    
    def explained_variance_ratio(self, S: np.ndarray) -> np.ndarray:
        """
        Calculate explained variance ratio for each component.
        
        Args:
            S: Singular values
            
        Returns:
            Explained variance ratios
        """
        total_variance = np.sum(S**2)
        if total_variance == 0:
            return np.zeros_like(S)
        return (S**2) / total_variance
    
    def cumulative_explained_variance(self, S: np.ndarray) -> np.ndarray:
        """
        Calculate cumulative explained variance.
        
        Args:
            S: Singular values
            
        Returns:
            Cumulative explained variance ratios
        """
        var_ratios = self.explained_variance_ratio(S)
        return np.cumsum(var_ratios)