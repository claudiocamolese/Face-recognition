"""Custom K-Nearest Neighbors classifier implementation."""

import numpy as np
from typing import Union, Optional, Literal
from collections import Counter
import warnings


class KNNClassifier:
    """Custom implementation of K-Nearest Neighbors classifier."""
    
    def __init__(
        self, 
        k: int = 5,
        weights: Literal['uniform', 'distance'] = 'uniform',
        metric: Literal['euclidean', 'manhattan', 'cosine'] = 'euclidean',
        p: int = 2
    ):
        """
        Initialize KNN classifier.
        
        Args:
            k: Number of neighbors to use
            weights: Weight function used in prediction ('uniform' or 'distance')
            metric: Distance metric to use ('euclidean', 'manhattan', 'cosine')
            p: Parameter for the Minkowski distance metric (when metric='minkowski')
        """
        self.k = k
        self.weights = weights
        self.metric = metric
        self.p = p
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
    def _validate_parameters(self):
        """Validate initialization parameters."""
        if self.k <= 0:
            raise ValueError("k must be a positive integer")
        
        if self.weights not in ['uniform', 'distance']:
            raise ValueError("weights must be 'uniform' or 'distance'")
        
        if self.metric not in ['euclidean', 'manhattan', 'cosine', 'minkowski']:
            raise ValueError("metric must be 'euclidean', 'manhattan', 'cosine', or 'minkowski'")
    
    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points."""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Manhattan distance between two points."""
        return np.sum(np.abs(x1 - x2))
    
    def _cosine_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Cosine distance between two points."""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0  # Maximum distance for zero vectors
        
        cosine_similarity = dot_product / (norm_x1 * norm_x2)
        # Clamp to [-1, 1] to avoid numerical errors
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        return 1 - cosine_similarity
    
    def _minkowski_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate Minkowski distance between two points."""
        return np.sum(np.abs(x1 - x2) ** self.p) ** (1/self.p)
    
    def _calculate_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculate distance between two points based on the specified metric."""
        if self.metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        elif self.metric == 'cosine':
            return self._cosine_distance(x1, x2)
        elif self.metric == 'minkowski':
            return self._minkowski_distance(x1, x2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _get_neighbors(self, x: np.ndarray) -> tuple:
        """
        Find k nearest neighbors for a given point.
        
        Args:
            x: Query point
            
        Returns:
            Tuple of (neighbor_labels, neighbor_distances)
        """
        distances = []
        
        # Calculate distances to all training points
        for i, train_point in enumerate(self.X_train):
            distance = self._calculate_distance(x, train_point)
            distances.append((distance, self.y_train[i], i))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        k_neighbors = distances[:self.k]
        
        neighbor_distances = [d[0] for d in k_neighbors]
        neighbor_labels = [d[1] for d in k_neighbors]
        neighbor_indices = [d[2] for d in k_neighbors]
        
        return neighbor_labels, neighbor_distances, neighbor_indices
    
    def _weighted_vote(self, neighbor_labels: list, neighbor_distances: list) -> any:
        """
        Perform weighted voting among neighbors.
        
        Args:
            neighbor_labels: Labels of the neighbors
            neighbor_distances: Distances to the neighbors
            
        Returns:
            Predicted class label
        """
        if self.weights == 'uniform':
            # Simple majority vote
            vote_counts = Counter(neighbor_labels)
            return vote_counts.most_common(1)[0][0]
        
        elif self.weights == 'distance':
            # Distance-weighted vote
            class_weights = {}
            
            for label, distance in zip(neighbor_labels, neighbor_distances):
                # Avoid division by zero
                weight = 1 / (distance + 1e-8)
                
                if label in class_weights:
                    class_weights[label] += weight
                else:
                    class_weights[label] = weight
            
            # Return class with highest weighted score
            return max(class_weights, key=class_weights.get)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNClassifier':
        """
        Fit the KNN classifier.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Returns:
            Self
        """
        self._validate_parameters()
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if X.shape[0] < self.k:
            warnings.warn(f"k={self.k} is greater than the number of training samples ({X.shape[0]}). "
                         f"Setting k to {X.shape[0]}")
            self.k = X.shape[0]
        
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = np.asarray(X)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"X has {X.shape[1]} features, but model was trained with {self.n_features_} features")
        
        predictions = []
        
        for x in X:
            neighbor_labels, neighbor_distances, _ = self._get_neighbors(x)
            prediction = self._weighted_vote(neighbor_labels, neighbor_distances)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Test data of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = np.asarray(X)
        probabilities = []
        
        for x in X:
            neighbor_labels, neighbor_distances, _ = self._get_neighbors(x)
            
            # Initialize probability array
            class_probs = np.zeros(self.n_classes_)
            
            if self.weights == 'uniform':
                # Count votes for each class
                for label in neighbor_labels:
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probs[class_idx] += 1
                
                # Normalize to get probabilities
                class_probs = class_probs / len(neighbor_labels)
                
            elif self.weights == 'distance':
                # Distance-weighted probabilities
                total_weight = 0
                
                for label, distance in zip(neighbor_labels, neighbor_distances):
                    weight = 1 / (distance + 1e-8)
                    class_idx = np.where(self.classes_ == label)[0][0]
                    class_probs[class_idx] += weight
                    total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    class_probs = class_probs / total_weight
            
            probabilities.append(class_probs)
        
        return np.array(probabilities)
    
    def kneighbors(self, X: Optional[np.ndarray] = None, n_neighbors: Optional[int] = None, 
                   return_distance: bool = True) -> Union[tuple, np.ndarray]:
        """
        Find the k-neighbors of a point.
        
        Args:
            X: Query points. If None, use training data
            n_neighbors: Number of neighbors to return. If None, use self.k
            return_distance: Whether to return distances
            
        Returns:
            If return_distance=True: (distances, indices)
            If return_distance=False: indices
        """
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X is None:
            X = self.X_train
        else:
            X = np.asarray(X)
        
        if n_neighbors is None:
            n_neighbors = self.k
        
        all_distances = []
        all_indices = []
        
        for x in X:
            distances = []
            
            for i, train_point in enumerate(self.X_train):
                distance = self._calculate_distance(x, train_point)
                distances.append((distance, i))
            
            # Sort by distance and get n_neighbors nearest
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:n_neighbors]
            
            neighbor_distances = [d[0] for d in neighbors]
            neighbor_indices = [d[1] for d in neighbors]
            
            all_distances.append(neighbor_distances)
            all_indices.append(neighbor_indices)
        
        distances_array = np.array(all_distances)
        indices_array = np.array(all_indices)
        
        if return_distance:
            return distances_array, indices_array
        else:
            return indices_array
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Mean accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self) -> dict:
        """Get parameters for this estimator."""
        return {
            'k': self.k,
            'weights': self.weights,
            'metric': self.metric,
            'p': self.p
        }
    
    def set_params(self, **params) -> 'KNNClassifier':
        """Set the parameters of this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        
        return self