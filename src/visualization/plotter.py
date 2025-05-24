"""Visualization utilities for the face recognition system."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import warnings
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultPlotter:
    """Class for creating visualizations of face recognition results."""
    
    def __init__(self, config):
        """
        Initialize the plotter.
        
        Args:
            config: Configuration object containing plot settings
        """
        self.config = config
        self.fig_size = getattr(config, 'FIGURE_SIZE', (12, 8))
        self.dpi = getattr(config, 'DPI', 100)
        self.save_plots = getattr(config, 'SAVE_PLOTS', True)
        self.output_dir = getattr(config, 'OUTPUT_DIR', 'outputs')
        
        # Create output directory if it doesn't exist
        if self.save_plots:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _save_figure(self, filename: str, tight_layout: bool = True):
        """
        Save figure to file if saving is enabled.
        
        Args:
            filename: Name of the file to save
            tight_layout: Whether to apply tight layout
        """
        if tight_layout:
            plt.tight_layout()
        
        if self.save_plots:
            filepath = Path(self.output_dir) / filename
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
    
    def plot_singular_values_analysis(self, singular_values: np.ndarray, 
                                    energy_threshold: float = 0.95):
        """
        Plot singular values analysis including energy retention.
        
        Args:
            singular_values: Array of singular values
            energy_threshold: Threshold for energy retention analysis
        """
        # Calculate energy (squared singular values)
        energy = singular_values ** 2
        total_energy = np.sum(energy)
        cumulative_energy = np.cumsum(energy) / total_energy
        
        # Find number of components for energy threshold
        n_components_threshold = np.argmax(cumulative_energy >= energy_threshold) + 1
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Singular values
        ax1.plot(singular_values, 'bo-', linewidth=2, markersize=4)
        ax1.set_title('Singular Values', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Component Index')
        ax1.set_ylabel('Singular Value')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Energy distribution
        ax2.bar(range(len(energy[:50])), energy[:50], alpha=0.7, color='skyblue')
        ax2.set_title('Energy Distribution (First 50 Components)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Component Index')
        ax2.set_ylabel('Energy')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cumulative energy
        ax3.plot(cumulative_energy, 'r-', linewidth=2)
        ax3.axhline(y=energy_threshold, color='g', linestyle='--', linewidth=2, 
                   label=f'{energy_threshold*100}% threshold')
        ax3.axvline(x=n_components_threshold, color='g', linestyle='--', linewidth=2,
                   label=f'{n_components_threshold} components')
        ax3.set_title('Cumulative Energy Retention', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Energy Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # Plot 4: Energy vs Components (zoomed)
        ax4.plot(cumulative_energy[:100], 'purple', linewidth=2)
        ax4.axhline(y=energy_threshold, color='g', linestyle='--', linewidth=2)
        ax4.axvline(x=n_components_threshold, color='g', linestyle='--', linewidth=2)
        ax4.set_title('Energy Retention (First 100 Components)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Number of Components')
        ax4.set_ylabel('Cumulative Energy Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Add text annotation
        ax4.text(n_components_threshold + 5, energy_threshold - 0.1,
                f'{n_components_threshold} components\nfor {energy_threshold*100}% energy',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.suptitle('Singular Value Decomposition Analysis', fontsize=16, fontweight='bold')
        self._save_figure('singular_values_analysis.png')
        plt.show()
        
        print(f"Components needed for {energy_threshold*100}% energy retention: {n_components_threshold}")
    
    def plot_reconstruction_comparison(self, original_images: np.ndarray,
                                     U: np.ndarray, S: np.ndarray, Vt: np.ndarray,
                                     pca_model, svd_components: int, pca_components: int,
                                     image_shape: Tuple[int, int]):
        """
        Plot comparison between original and reconstructed images using SVD and PCA.
        
        Args:
            original_images: Original images array
            U, S, Vt: SVD decomposition matrices
            pca_model: Fitted PCA model
            svd_components: Number of SVD components to use
            pca_components: Number of PCA components to use
            image_shape: Shape to reshape images for display
        """
        n_images = min(len(original_images), 7)
        
        # Reconstruct using SVD
        svd_reconstructed = self._reconstruct_svd_images(
            original_images[:n_images], U, S, Vt, svd_components
        )
        
        # Reconstruct using PCA
        pca_reconstructed = self._reconstruct_pca_images(
            original_images[:n_images], pca_model, pca_components
        )
        
        fig, axes = plt.subplots(4, n_images, figsize=(2*n_images, 8))
        
        for i in range(n_images):
            # Original image
            axes[0, i].imshow(original_images[i].reshape(image_shape), cmap='gray')
            axes[0, i].set_title(f'Original {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # SVD reconstruction
            axes[1, i].imshow(svd_reconstructed[i].reshape(image_shape), cmap='gray')
            axes[1, i].set_title(f'SVD ({svd_components})', fontsize=10)
            axes[1, i].axis('off')
            
            # PCA reconstruction
            axes[2, i].imshow(pca_reconstructed[i].reshape(image_shape), cmap='gray')
            axes[2, i].set_title(f'PCA ({pca_components})', fontsize=10)
            axes[2, i].axis('off')
            
            # Difference visualization (SVD vs Original)
            diff = np.abs(original_images[i] - svd_reconstructed[i])
            axes[3, i].imshow(diff.reshape(image_shape), cmap='hot')
            axes[3, i].set_title(f'SVD Diff', fontsize=10)
            axes[3, i].axis('off')
        
        # Add row labels
        row_labels = ['Original', 'SVD Reconstruction', 'PCA Reconstruction', 'SVD Difference']
        for i, label in enumerate(row_labels):
            axes[i, 0].set_ylabel(label, fontsize=12, fontweight='bold')
        
        plt.suptitle('Image Reconstruction Comparison', fontsize=16, fontweight='bold')
        self._save_figure('reconstruction_comparison.png')
        plt.show()
    
    def _reconstruct_svd_images(self, images: np.ndarray, U: np.ndarray, 
                               S: np.ndarray, Vt: np.ndarray, 
                               n_components: int) -> np.ndarray:
        """Reconstruct images using SVD with specified number of components."""
        # Project to reduced space
        reduced = images @ Vt[:n_components].T
        
        # Reconstruct
        reconstructed = reduced @ Vt[:n_components]
        
        return reconstructed
    
    def _reconstruct_pca_images(self, images: np.ndarray, pca_model, 
                               n_components: int) -> np.ndarray:
        """Reconstruct images using PCA with specified number of components."""
        # Transform to PCA space
        transformed = pca_model.transform(images)[:, :n_components]
        
        # Reconstruct (partial reconstruction with n_components)
        components_subset = pca_model.components_[:n_components]
        reconstructed = transformed @ components_subset + pca_model.mean_
        
        return reconstructed
    
    def plot_eigenfaces(self, pca_model, image_shape: Tuple[int, int], 
                       n_eigenfaces: int = 16):
        """
        Plot the first n eigenfaces (principal components).
        
        Args:
            pca_model: Fitted PCA model
            image_shape: Shape to reshape eigenfaces for display
            n_eigenfaces: Number of eigenfaces to display
        """
        n_eigenfaces = min(n_eigenfaces, len(pca_model.components_))
        n_cols = 4
        n_rows = (n_eigenfaces + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i in range(n_eigenfaces):
            eigenface = pca_model.components_[i].reshape(image_shape)
            
            # Normalize for better visualization
            eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
            
            axes[i].imshow(eigenface, cmap='gray')
            axes[i].set_title(f'Eigenface {i+1}\nVariance: {pca_model.explained_variance_[i]:.1f}',
                             fontsize=10)
            axes[i].axis('off')
        
        # Hide empty subplots
        for i in range(n_eigenfaces, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle('Principal Components (Eigenfaces)', fontsize=16, fontweight='bold')
        self._save_figure('eigenfaces.png')
        plt.show()
    
    def plot_explained_variance(self, pca_model, max_components: int = 50):
        """
        Plot explained variance analysis for PCA.
        
        Args:
            pca_model: Fitted PCA model
            max_components: Maximum number of components to show
        """
        n_components = min(max_components, len(pca_model.explained_variance_ratio_))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Individual explained variance
        ax1.bar(range(n_components), pca_model.explained_variance_ratio_[:n_components],
                alpha=0.7, color='lightblue')
        ax1.set_title('Individual Explained Variance Ratio', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum = np.cumsum(pca_model.explained_variance_ratio_[:n_components])
        ax2.plot(range(n_components), cumsum, 'ro-', linewidth=2, markersize=4)
        
        # Add threshold lines
        for threshold in [0.8, 0.9, 0.95]:
            idx = np.argmax(cumsum >= threshold)
            if cumsum[idx] >= threshold:
                ax2.axhline(y=threshold, color='g', linestyle='--', alpha=0.7,
                           label=f'{threshold*100}% ({idx+1} components)')
                ax2.axvline(x=idx, color='g', linestyle='--', alpha=0.7)
        
        ax2.set_title('Cumulative Explained Variance Ratio', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_ylim(0, 1)
        
        plt.suptitle('PCA Explained Variance Analysis', fontsize=16, fontweight='bold')
        self._save_figure('pca_explained_variance.png')
        plt.show()
    
    def plot_classification_results(self, y_true: np.ndarray, y_pred_svd: np.ndarray,
                                  y_pred_pca: np.ndarray, class_names: Optional[List[str]] = None):
        """
        Plot classification results comparison.
        
        Args:
            y_true: True labels
            y_pred_svd: SVD predictions
            y_pred_pca: PCA predictions
            class_names: Names of the classes
        """
        from sklearn.metrics import confusion_matrix, accuracy_score
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # SVD Confusion Matrix
        cm_svd = confusion_matrix(y_true, y_pred_svd)
        sns.heatmap(cm_svd, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'SVD Confusion Matrix\nAccuracy: {accuracy_score(y_true, y_pred_svd):.3f}',
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # PCA Confusion Matrix
        cm_pca = confusion_matrix(y_true, y_pred_pca)
        sns.heatmap(cm_pca, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title(f'PCA Confusion Matrix\nAccuracy: {accuracy_score(y_true, y_pred_pca):.3f}',
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        # Comparison
        methods = ['SVD', 'PCA']
        accuracies = [accuracy_score(y_true, y_pred_svd), accuracy_score(y_true, y_pred_pca)]
        
        bars = axes[2].bar(methods, accuracies, color=['skyblue', 'lightgreen'], alpha=0.8)
        axes[2].set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Classification Results Comparison', fontsize=16, fontweight='bold')
        self._save_figure('classification_results.png')
        plt.show()
    
    def plot_dimensionality_reduction_2d(self, X_original: np.ndarray, y: np.ndarray,
                                        X_svd: np.ndarray, X_pca: np.ndarray,
                                        class_names: Optional[List[str]] = None):
        """
        Plot 2D visualization of dimensionality reduction results.
        
        Args:
            X_original: Original high-dimensional data
            y: Labels
            X_svd: SVD-reduced data (first 2 components)
            X_pca: PCA-reduced data (first 2 components)
            class_names: Names of the classes
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Create color map
        unique_labels = np.unique(y)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        # SVD 2D plot
        for i, label in enumerate(unique_labels):
            mask = y == label
            label_name = class_names[i] if class_names else f'Class {label}'
            ax1.scatter(X_svd[mask, 0], X_svd[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.7, s=50)
        
        ax1.set_title('SVD - First 2 Components', fontsize=14, fontweight='bold')
        ax1.set_xlabel('First SVD Component')
        ax1.set_ylabel('Second SVD Component')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PCA 2D plot
        for i, label in enumerate(unique_labels):
            mask = y == label
            label_name = class_names[i] if class_names else f'Class {label}'
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                       c=[colors[i]], label=label_name, alpha=0.7, s=50)
        
        ax2.set_title('PCA - First 2 Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('First Principal Component')
        ax2.set_ylabel('Second Principal Component')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('2D Visualization of Dimensionality Reduction', fontsize=16, fontweight='bold')
        self._save_figure('dimensionality_reduction_2d.png')
        plt.show()
    
    def plot_reconstruction_error_analysis(self, original_images: np.ndarray,
                                         reconstructed_svd: np.ndarray,
                                         reconstructed_pca: np.ndarray):
        """
        Plot reconstruction error analysis.
        
        Args:
            original_images: Original images
            reconstructed_svd: SVD reconstructed images
            reconstructed_pca: PCA reconstructed images
        """
        # Calculate reconstruction errors
        svd_errors = np.mean((original_images - reconstructed_svd) ** 2, axis=1)
        pca_errors = np.mean((original_images - reconstructed_pca) ** 2, axis=1)
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Error distribution
        ax1.hist(svd_errors, bins=30, alpha=0.7, label='SVD', color='skyblue')
        ax1.hist(pca_errors, bins=30, alpha=0.7, label='PCA', color='lightgreen')
        ax1.set_title('Reconstruction Error Distribution', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Mean Squared Error')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error comparison scatter
        ax2.scatter(svd_errors, pca_errors, alpha=0.6, s=50)
        ax2.plot([min(min(svd_errors), min(pca_errors)), 
                 max(max(svd_errors), max(pca_errors))],
                [min(min(svd_errors), min(pca_errors)), 
                 max(max(svd_errors), max(pca_errors))], 
                'r--', alpha=0.8)
        ax2.set_title('SVD vs PCA Reconstruction Error', fontsize=12, fontweight='bold')
        ax2.set_xlabel('SVD Reconstruction Error')
        ax2.set_ylabel('PCA Reconstruction Error')
        ax2.grid(True, alpha=0.3)
        
        # Summary statistics
        methods = ['SVD', 'PCA']
        mean_errors = [np.mean(svd_errors), np.mean(pca_errors)]
        std_errors = [np.std(svd_errors), np.std(pca_errors)]
        
        x_pos = np.arange(len(methods))
        bars = ax3.bar(x_pos, mean_errors, yerr=std_errors, capsize=5,
                      color=['skyblue', 'lightgreen'], alpha=0.8)
        ax3.set_title('Mean Reconstruction Error', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Mean Squared Error')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(methods)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean_err in zip(bars, mean_errors):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(std_errors)*0.1,
                    f'{mean_err:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Reconstruction Error Analysis', fontsize=16, fontweight='bold')
        self._save_figure('reconstruction_error_analysis.png')
        plt.show()