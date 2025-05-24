"""
Image preprocessing module for face recognition system.
Handles image loading, resizing, normalization, and enhancement.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from pathlib import Path
from ..utils.helpers import Logger, validate_image_array, normalize_data, ProgressBar


class ImageProcessor:
    """
    Image preprocessing class for face recognition.
    Handles various image preprocessing operations including resizing,
    normalization, histogram equalization, and noise reduction.
    """
    
    def __init__(self, config):
        """
        Initialize the image processor.
        
        Args:
            config: Configuration object containing preprocessing parameters
        """
        self.config = config
        self.logger = Logger(__name__)
        
        # Image processing parameters
        self.target_size = config.IMAGE_SHAPE[:2]  # (height, width)
        self.normalization_method = getattr(config, 'NORMALIZATION_METHOD', 'standard')
        self.enable_histogram_equalization = getattr(config, 'ENABLE_HIST_EQ', True)
        self.enable_gaussian_blur = getattr(config, 'ENABLE_GAUSSIAN_BLUR', False)
        self.gaussian_blur_kernel = getattr(config, 'GAUSSIAN_BLUR_KERNEL', (3, 3))
        self.enable_edge_enhancement = getattr(config, 'ENABLE_EDGE_ENHANCEMENT', False)
        
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load a single image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded image as numpy array or None if failed
        """
        try:
            image_path = Path(image_path)
            
            # Check if file exists
            if not image_path.exists():
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load image using PIL
            with Image.open(image_path) as img:
                # Convert to grayscale if needed
                if img.mode != 'L':
                    img = img.convert('L')
                
                # Convert to numpy array
                image_array = np.array(img, dtype=np.float32)
                
            return image_array
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def resize_image(self, image: np.ndarray, 
                    target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image array
            target_size: Target size as (height, width)
            
        Returns:
            Resized image array
        """
        if target_size is None:
            target_size = self.target_size
        
        # Use PIL for high-quality resizing
        pil_image = Image.fromarray(image.astype(np.uint8))
        resized_pil = pil_image.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
        resized_array = np.array(resized_pil, dtype=np.float32)
        
        return resized_array
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization to enhance contrast.
        
        Args:
            image: Input image array
            
        Returns:
            Histogram equalized image
        """
        # Convert to uint8 for OpenCV
        image_uint8 = image.astype(np.uint8)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(image_uint8)
        
        return equalized.astype(np.float32)
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for noise reduction.
        
        Args:
            image: Input image array
            
        Returns:
            Blurred image
        """
        # Apply Gaussian blur using OpenCV
        blurred = cv2.GaussianBlur(image, self.gaussian_blur_kernel, 0)
        return blurred
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance edges in the image.
        
        Args:
            image: Input image array
            
        Returns:
            Edge-enhanced image
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Apply edge enhancement
        enhancer = ImageEnhance.Sharpness(pil_image)
        enhanced = enhancer.enhance(1.5)  # Enhance by factor of 1.5
        
        return np.array(enhanced, dtype=np.float32)
    
    def remove_illumination_variations(self, image: np.ndarray) -> np.ndarray:
        """
        Remove illumination variations using homomorphic filtering.
        
        Args:
            image: Input image array
            
        Returns:
            Image with reduced illumination variations
        """
        # Add small constant to avoid log(0)
        image_log = np.log(image + 1)
        
        # Apply Gaussian filter to get low-frequency component
        low_freq = cv2.GaussianBlur(image_log, (21, 21), 0)
        
        # Subtract low-frequency component
        high_freq = image_log - low_freq
        
        # Reconstruct image
        result = np.exp(high_freq)
        
        # Normalize to [0, 255] range
        result = ((result - result.min()) / (result.max() - result.min()) * 255)
        
        return result.astype(np.float32)
    
    def preprocess_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to a single image.
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image array
        """
        # Validate input
        if not validate_image_array(image):
            raise ValueError("Invalid image array")
        
        processed_image = image.copy()
        
        # 1. Resize image
        if processed_image.shape[:2] != self.target_size:
            processed_image = self.resize_image(processed_image)
        
        # 2. Remove illumination variations
        processed_image = self.remove_illumination_variations(processed_image)
        
        # 3. Apply histogram equalization
        if self.enable_histogram_equalization:
            processed_image = self.apply_histogram_equalization(processed_image)
        
        # 4. Apply Gaussian blur for noise reduction
        if self.enable_gaussian_blur:
            processed_image = self.apply_gaussian_blur(processed_image)
        
        # 5. Enhance edges
        if self.enable_edge_enhancement:
            processed_image = self.enhance_edges(processed_image)
        
        # 6. Normalize pixel values
        processed_image = self.normalize_pixel_values(processed_image)
        
        return processed_image
    
    def normalize_pixel_values(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image: Input image array
            
        Returns:
            Normalized image array
        """
        # Ensure values are in valid range
        image = np.clip(image, 0, 255)
        
        # Normalize to [0, 1]
        normalized = image / 255.0
        
        return normalized
    
    def preprocess_batch(self, images: List[np.ndarray], 
                        show_progress: bool = True) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of image arrays
            show_progress: Whether to show progress bar
            
        Returns:
            Preprocessed images as 2D array (n_samples, n_features)
        """
        self.logger.info(f"Preprocessing {len(images)} images...")
        
        processed_images = []
        
        # Initialize progress bar
        if show_progress:
            progress_bar = ProgressBar(len(images), "Processing images")
        
        for i, image in enumerate(images):
            try:
                # Preprocess single image
                processed_image = self.preprocess_single_image(image)
                
                # Flatten image to 1D vector
                flattened_image = processed_image.flatten()
                processed_images.append(flattened_image)
                
                if show_progress:
                    progress_bar.update()
                    
            except Exception as e:
                self.logger.error(f"Error processing image {i}: {str(e)}")
                continue
        
        # Convert to numpy array
        processed_array = np.array(processed_images, dtype=np.float32)
        
        # Apply additional normalization if specified
        if self.normalization_method != 'none':
            processed_array = normalize_data(processed_array, self.normalization_method)
        
        self.logger.info(f"Preprocessing completed. Output shape: {processed_array.shape}")
        
        return processed_array
    
    def augment_image(self, image: np.ndarray, 
                     rotation_range: float = 10.0,
                     brightness_range: float = 0.2,
                     contrast_range: float = 0.2) -> List[np.ndarray]:
        """
        Generate augmented versions of an image.
        
        Args:
            image: Input image array
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Brightness variation range
            contrast_range: Contrast variation range
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        # Original image
        augmented_images.append(image)
        
        # Convert to PIL for augmentation
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # Rotation augmentation
        for angle in [-rotation_range, rotation_range]:
            rotated = pil_image.rotate(angle, fillcolor=0)
            augmented_images.append(np.array(rotated, dtype=np.float32))
        
        # Brightness augmentation
        enhancer = ImageEnhance.Brightness(pil_image)
        for factor in [1 - brightness_range, 1 + brightness_range]:
            brightened = enhancer.enhance(factor)
            augmented_images.append(np.array(brightened, dtype=np.float32))
        
        # Contrast augmentation
        enhancer = ImageEnhance.Contrast(pil_image)
        for factor in [1 - contrast_range, 1 + contrast_range]:
            contrasted = enhancer.enhance(factor)
            augmented_images.append(np.array(contrasted, dtype=np.float32))
        
        return augmented_images
    
    def denoise_image(self, image: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Remove noise from image using various denoising methods.
        
        Args:
            image: Input image array
            method: Denoising method ('bilateral', 'gaussian', 'median')
            
        Returns:
            Denoised image
        """
        image_uint8 = image.astype(np.uint8)
        
        if method == 'bilateral':
            # Bilateral filter preserves edges while removing noise
            denoised = cv2.bilateralFilter(image_uint8, 9, 75, 75)
        elif method == 'gaussian':
            # Gaussian blur
            denoised = cv2.GaussianBlur(image_uint8, (5, 5), 0)
        elif method == 'median':
            # Median filter
            denoised = cv2.medianBlur(image_uint8, 5)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        
        return denoised.astype(np.float32)
    
    def get_image_statistics(self, image: np.ndarray) -> dict:
        """
        Calculate basic statistics for an image.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary containing image statistics
        """
        stats = {
            'shape': image.shape,
            'dtype': str(image.dtype),
            'min_value': float(np.min(image)),
            'max_value': float(np.max(image)),
            'mean_value': float(np.mean(image)),
            'std_value': float(np.std(image)),
            'median_value': float(np.median(image))
        }
        
        return stats
    
    def visualize_preprocessing_steps(self, image: np.ndarray) -> dict:
        """
        Apply preprocessing steps individually for visualization.
        
        Args:
            image: Input image array
            
        Returns:
            Dictionary containing images at each preprocessing step
        """
        steps = {}
        
        # Original image
        steps['original'] = image.copy()
        
        # Resize
        resized = self.resize_image(image)
        steps['resized'] = resized
        
        # Remove illumination variations
        illumination_corrected = self.remove_illumination_variations(resized)
        steps['illumination_corrected'] = illumination_corrected
        
        # Histogram equalization
        if self.enable_histogram_equalization:
            hist_eq = self.apply_histogram_equalization(illumination_corrected)
            steps['histogram_equalized'] = hist_eq
            current_image = hist_eq
        else:
            current_image = illumination_corrected
        
        # Gaussian blur
        if self.enable_gaussian_blur:
            blurred = self.apply_gaussian_blur(current_image)
            steps['gaussian_blur'] = blurred
            current_image = blurred
        
        # Edge enhancement
        if self.enable_edge_enhancement:
            edge_enhanced = self.enhance_edges(current_image)
            steps['edge_enhanced'] = edge_enhanced
            current_image = edge_enhanced
        
        # Final normalized image
        final = self.normalize_pixel_values(current_image)
        steps['final'] = final
        
        return steps