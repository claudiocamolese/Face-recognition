
"""
Face Recognition System with Custom SVD Implementation
====================================================

Project Structure:
├── main.py
├── config/
│   └── settings.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── image_processor.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── svd_decomposition.py
│   │   ├── pca_decomposition.py
│   │   └── knn_classifier.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plotter.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
└── tests/
    ├── __init__.py
    └── test_svd.py
"""


import os
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import Config
from src.data.loader import DatasetLoader
from src.preprocessing.image_processor import ImageProcessor
from src.models.svd_decomposition import CustomSVD
from src.models.pca_decomposition import CustomPCA
from src.models.knn_classifier import KNNClassifier
from src.visualization.plotter import ResultPlotter
from src.utils.helpers import Logger


class FaceRecognitionSystem:
    """Main class for the face recognition system."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = Logger(__name__)
        
        # Initialize components
        self.data_loader = DatasetLoader(config)
        self.image_processor = ImageProcessor(config)
        self.svd = CustomSVD()
        self.pca = CustomPCA()
        self.knn_svd = KNNClassifier(config.KNN_NEIGHBORS, config.KNN_WEIGHTS, config.KNN_METRIC)
        self.knn_pca = KNNClassifier(config.KNN_NEIGHBORS, config.KNN_WEIGHTS, config.KNN_METRIC)
        self.plotter = ResultPlotter(config)
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        self.logger.info("Loading dataset...")
        images, labels = self.data_loader.load_dataset()
        
        self.logger.info("Preprocessing images...")
        processed_images = self.image_processor.preprocess_batch(images)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            processed_images, labels,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.RANDOM_STATE,
            stratify=labels
        )
        
        self.logger.info(f"Training set: {self.X_train.shape}")
        self.logger.info(f"Test set: {self.X_test.shape}")
        
    def train_svd_model(self):
        """Train the SVD-based model."""
        self.logger.info("Training SVD model...")
        
        # Apply SVD to training data
        U, S, Vt = self.svd.decompose(self.X_train)
        
        # Reduce dimensions
        X_train_reduced = self.svd.reduce_dimensions(
            self.X_train, U, S, Vt, self.config.SVD_COMPONENTS
        )
        X_test_reduced = self.svd.transform_test_data(
            self.X_test, Vt, self.config.SVD_COMPONENTS
        )
        
        # Train KNN
        self.knn_svd.fit(X_train_reduced, self.y_train)
        
        # Predict and evaluate
        y_pred = self.knn_svd.predict(X_test_reduced)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.logger.info(f"SVD Model Accuracy: {accuracy * 100:.2f}%")
        print(f"SVD Accuracy: {accuracy * 100:.2f}%")
        print("\nSVD Classification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=1))
        
        return U, S, Vt, accuracy
        
    def train_pca_model(self):
        """Train the PCA-based model."""
        self.logger.info("Training PCA model...")
        
        # Fit PCA and transform data
        self.pca.fit(self.X_train)
        X_train_reduced = self.pca.transform(self.X_train, self.config.PCA_COMPONENTS)
        X_test_reduced = self.pca.transform(self.X_test, self.config.PCA_COMPONENTS)
        
        # Train KNN
        self.knn_pca.fit(X_train_reduced, self.y_train)
        
        # Predict and evaluate
        y_pred = self.knn_pca.predict(X_test_reduced)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        self.logger.info(f"PCA Model Accuracy: {accuracy * 100:.2f}%")
        print(f"PCA Accuracy: {accuracy * 100:.2f}%")
        print("\nPCA Classification Report:")
        print(classification_report(self.y_test, y_pred, zero_division=1))
        
        return accuracy
        
    def visualize_results(self, U, S, Vt):
        """Visualize the results."""
        self.logger.info("Generating visualizations...")
        
        # Plot singular values analysis
        self.plotter.plot_singular_values_analysis(S, self.config.ENERGY_THRESHOLD)
        
        # Plot reconstructed images comparison
        self.plotter.plot_reconstruction_comparison(
            self.X_train[:7], U, S, Vt, self.pca,
            self.config.SVD_COMPONENTS, self.config.PCA_COMPONENTS,
            self.config.IMAGE_SHAPE
        )
        
    def run(self):
        """Run the complete face recognition pipeline."""
        self.logger.info("Starting Face Recognition System...")
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train SVD model
        U, S, Vt, svd_accuracy = self.train_svd_model()
        
        # Train PCA model
        pca_accuracy = self.train_pca_model()
        
        # Compare results
        print(f"\n{'='*50}")
        print("FINAL RESULTS COMPARISON")
        print(f"{'='*50}")
        print(f"SVD Accuracy: {svd_accuracy * 100:.2f}%")
        print(f"PCA Accuracy: {pca_accuracy * 100:.2f}%")
        print(f"Best Method: {'SVD' if svd_accuracy > pca_accuracy else 'PCA'}")
        
        # Visualize results
        if self.config.ENABLE_VISUALIZATION:
            self.visualize_results(U, S, Vt)
        
        self.logger.info("Face Recognition System completed successfully!")


def main():
    """Main entry point."""
    # Load configuration
    config = Config()
    
    # Initialize and run the system
    system = FaceRecognitionSystem(config)
    system.run()


if __name__ == "__main__":
    main()