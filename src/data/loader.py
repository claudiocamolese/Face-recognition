"""Data loading utilities for the face recognition system."""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm
import warnings

from ..utils.helpers import Logger


class DatasetLoader:
    """Handles loading and validation of image datasets."""
    
    def __init__(self, config):
        """
        Initialize DatasetLoader.
        
        Args:
            config: Configuration object containing dataset parameters
        """
        self.config = config
        self.logger = Logger(__name__)
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Statistics
        self.dataset_stats = {
            'total_images': 0,
            'classes': {},
            'failed_loads': 0,
            'corrupted_files': []
        }
    
    def _is_valid_image_file(self, filepath: str) -> bool:
        """
        Check if file is a valid image file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            True if file is a valid image file
        """
        return Path(filepath).suffix.lower() in self.supported_extensions
    
    def _load_single_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load a single image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image array or None if failed
        """
        try:
            # Load image in grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                self.logger.warning(f"Failed to load image: {image_path}")
                self.dataset_stats['failed_loads'] += 1
                self.dataset_stats['corrupted_files'].append(image_path)
                return None
            
            # Validate image dimensions
            if img.shape != self.config.IMAGE_SHAPE:
                # Resize image to expected dimensions
                img = cv2.resize(img, (self.config.IMAGE_SHAPE[1], self.config.IMAGE_SHAPE[0]))
                self.logger.debug(f"Resized image {image_path} to {self.config.IMAGE_SHAPE}")
            
            # Validate image content
            if np.all(img == 0) or np.all(img == 255):
                self.logger.warning(f"Image appears to be empty or corrupted: {image_path}")
                self.dataset_stats['corrupted_files'].append(image_path)
                return None
            
            return img
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            self.dataset_stats['failed_loads'] += 1
            self.dataset_stats['corrupted_files'].append(image_path)
            return None
    
    def _validate_dataset_structure(self, dataset_path: str) -> bool:
        """
        Validate the dataset directory structure.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            True if structure is valid
        """
        if not os.path.exists(dataset_path):
            self.logger.error(f"Dataset path does not exist: {dataset_path}")
            return False
        
        if not os.path.isdir(dataset_path):
            self.logger.error(f"Dataset path is not a directory: {dataset_path}")
            return False
        
        # Check for subdirectories (classes)
        subdirs = [d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))]
        
        if len(subdirs) == 0:
            self.logger.error("No class subdirectories found in dataset")
            return False
        
        # Check each class directory has images
        valid_classes = 0
        for subdir in subdirs:
            subdir_path = os.path.join(dataset_path, subdir)
            image_files = [f for f in os.listdir(subdir_path) 
                          if self._is_valid_image_file(os.path.join(subdir_path, f))]
            
            if len(image_files) > 0:
                valid_classes += 1
                self.logger.debug(f"Class '{subdir}': {len(image_files)} images")
            else:
                self.logger.warning(f"Class '{subdir}' has no valid images")
        
        if valid_classes < 2:
            self.logger.error("Need at least 2 classes with valid images")
            return False
        
        self.logger.info(f"Dataset validation passed: {valid_classes} valid classes found")
        return True
    
    def load_dataset(self, dataset_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load the complete dataset.
        
        Args:
            dataset_path: Path to dataset directory (uses config if None)
            
        Returns:
            Tuple of (images, labels) arrays
        """
        if dataset_path is None:
            dataset_path = self.config.DATASET_PATH
        
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        # Validate dataset structure
        if not self._validate_dataset_structure(dataset_path):
            raise ValueError(f"Invalid dataset structure at {dataset_path}")
        
        images = []
        labels = []
        
        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(dataset_path) 
                           if os.path.isdir(os.path.join(dataset_path, d))])
        
        self.logger.info(f"Found {len(class_dirs)} classes: {class_dirs}")
        
        # Process each class
        for class_name in tqdm(class_dirs, desc="Loading classes"):
            class_path = os.path.join(dataset_path, class_name)
            class_images = 0
            
            # Initialize class stats
            self.dataset_stats['classes'][class_name] = {
                'total_files': 0,
                'loaded_images': 0,
                'failed_loads': 0
            }
            
            # Get all image files in class directory
            image_files = [f for f in os.listdir(class_path) 
                          if self._is_valid_image_file(os.path.join(class_path, f))]
            
            self.dataset_stats['classes'][class_name]['total_files'] = len(image_files)
            
            # Load each image
            for image_file in tqdm(image_files, desc=f"Loading {class_name}", leave=False):
                image_path = os.path.join(class_path, image_file)
                
                img = self._load_single_image(image_path)
                if img is not None:
                    images.append(img)
                    labels.append(class_name)
                    class_images += 1
                    self.dataset_stats['classes'][class_name]['loaded_images'] += 1
                else:
                    self.dataset_stats['classes'][class_name]['failed_loads'] += 1
            
            self.logger.info(f"Class '{class_name}': {class_images} images loaded")
        
        if len(images) == 0:
            raise ValueError("No images could be loaded from the dataset")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Update total statistics
        self.dataset_stats['total_images'] = len(images)
        
        self.logger.info(f"Dataset loaded successfully:")
        self.logger.info(f"  Total images: {len(images)}")
        self.logger.info(f"  Image shape: {images[0].shape}")
        self.logger.info(f"  Number of classes: {len(np.unique(labels))}")
        self.logger.info(f"  Failed loads: {self.dataset_stats['failed_loads']}")
        
        return images, labels
    
    def get_dataset_stats(self) -> dict:
        """
        Get detailed dataset statistics.
        
        Returns:
            Dictionary containing dataset statistics
        """
        return self.dataset_stats.copy()
    
    def print_dataset_summary(self):
        """Print a summary of the loaded dataset."""
        if self.dataset_stats['total_images'] == 0:
            print("No dataset has been loaded yet.")
            return
        
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Total Images: {self.dataset_stats['total_images']}")
        print(f"Failed Loads: {self.dataset_stats['failed_loads']}")
        print(f"Number of Classes: {len(self.dataset_stats['classes'])}")
        
        print("\nClass Distribution:")
        print("-" * 30)
        for class_name, stats in self.dataset_stats['classes'].items():
            success_rate = (stats['loaded_images'] / stats['total_files'] * 100 
                          if stats['total_files'] > 0 else 0)
            print(f"{class_name:15}: {stats['loaded_images']:4} images "
                  f"({success_rate:5.1f}% success rate)")
        
        if self.dataset_stats['corrupted_files']:
            print(f"\nCorrupted Files ({len(self.dataset_stats['corrupted_files'])}):")
            print("-" * 30)
            for file_path in self.dataset_stats['corrupted_files'][:10]:  # Show first 10
                print(f"  {file_path}")
            if len(self.dataset_stats['corrupted_files']) > 10:
                print(f"  ... and {len(self.dataset_stats['corrupted_files']) - 10} more")
        
        print("="*50)
    
    def validate_loaded_data(self, images: np.ndarray, labels: np.ndarray) -> bool:
        """
        Validate loaded data integrity.
        
        Args:
            images: Loaded images array
            labels: Loaded labels array
            
        Returns:
            True if data is valid
        """
        try:
            # Check array shapes
            if len(images) != len(labels):
                self.logger.error("Images and labels arrays have different lengths")
                return False
            
            if len(images) == 0:
                self.logger.error("No data loaded")
                return False
            
            # Check image dimensions
            expected_shape = self.config.IMAGE_SHAPE
            if images[0].shape != expected_shape:
                self.logger.error(f"Image shape mismatch: expected {expected_shape}, "
                                f"got {images[0].shape}")
                return False
            
            # Check for consistent shapes
            for i, img in enumerate(images[:100]):  # Check first 100 images
                if img.shape != expected_shape:
                    self.logger.error(f"Inconsistent image shape at index {i}: {img.shape}")
                    return False
            
            # Check data types
            if not np.issubdtype(images.dtype, np.number):
                self.logger.error(f"Images array has non-numeric dtype: {images.dtype}")
                return False
            
            # Check for NaN or infinite values
            if np.any(np.isnan(images)) or np.any(np.isinf(images)):
                self.logger.error("Images contain NaN or infinite values")
                return False
            
            # Check label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            min_samples = np.min(counts)
            if min_samples < 2:
                self.logger.warning(f"Some classes have fewer than 2 samples: min={min_samples}")
            
            self.logger.info("Data validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def save_dataset_info(self, output_path: str):
        """
        Save dataset information to file.
        
        Args:
            output_path: Path to save the dataset info
        """
        import json
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.dataset_stats, f, indent=2)
            self.logger.info(f"Dataset info saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save dataset info: {str(e)}")