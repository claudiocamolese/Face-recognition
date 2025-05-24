"""
Utility functions and helper classes for the face recognition system.
"""

import os
import time
import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import warnings


class Logger:
    """Custom logger class for the face recognition system."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (optional)
            log_dir = Path('logs')
            if not log_dir.exists():
                log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / 'face_recognition.log')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str = "Operation"):
        """
        Initialize timer.
        
        Args:
            operation_name: Name of the operation being timed
        """
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        print(f"{self.operation_name} completed in {elapsed_time:.2f} seconds")
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time."""
        if self.start_time is None:
            return 0.0
        end_time = self.end_time or time.time()
        return end_time - self.start_time


def validate_image_array(images: np.ndarray, expected_shape: Optional[Tuple] = None) -> bool:
    """
    Validate image array format and shape.
    
    Args:
        images: Image array to validate
        expected_shape: Expected shape (optional)
    
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(images, np.ndarray):
        print(f"Error: Expected numpy array, got {type(images)}")
        return False
    
    if images.ndim < 2:
        print(f"Error: Expected at least 2D array, got {images.ndim}D")
        return False
    
    if expected_shape and images.shape != expected_shape:
        print(f"Error: Expected shape {expected_shape}, got {images.shape}")
        return False
    
    # Check for invalid values
    if np.isnan(images).any():
        print("Warning: Array contains NaN values")
        return False
    
    if np.isinf(images).any():
        print("Warning: Array contains infinite values")
        return False
    
    return True


def normalize_data(data: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize data using different methods.
    
    Args:
        data: Data to normalize
        method: Normalization method ('standard', 'minmax', 'l2')
    
    Returns:
        Normalized data
    """
    if method == 'standard':
        # Z-score normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1, std)
        return (data - mean) / std
    
    elif method == 'minmax':
        # Min-max normalization
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        return (data - min_val) / range_val
    
    elif method == 'l2':
        # L2 normalization
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return data / norms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_explained_variance_ratio(singular_values: np.ndarray) -> np.ndarray:
    """
    Calculate explained variance ratio from singular values.
    
    Args:
        singular_values: Singular values from SVD
    
    Returns:
        Explained variance ratio for each component
    """
    # Square singular values to get variances
    variances = singular_values ** 2
    total_variance = np.sum(variances)
    
    # Calculate explained variance ratio
    explained_variance_ratio = variances / total_variance
    
    return explained_variance_ratio


def find_optimal_components(singular_values: np.ndarray, 
                          energy_threshold: float = 0.95) -> int:
    """
    Find optimal number of components based on energy threshold.
    
    Args:
        singular_values: Singular values from SVD
        energy_threshold: Energy retention threshold (0-1)
    
    Returns:
        Optimal number of components
    """
    explained_variance_ratio = calculate_explained_variance_ratio(singular_values)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Find first index where cumulative variance exceeds threshold
    optimal_components = np.argmax(cumulative_variance >= energy_threshold) + 1
    
    return min(optimal_components, len(singular_values))


def create_directory_structure(base_path: Union[str, Path], 
                             subdirs: List[str]) -> None:
    """
    Create directory structure.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectory names to create
    """
    base_path = Path(base_path)
    base_path.mkdir(exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                default_value: float = 0.0) -> np.ndarray:
    """
    Safely divide arrays, handling division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        default_value: Value to use when denominator is zero
    
    Returns:
        Result of safe division
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.divide(numerator, denominator, 
                          out=np.full_like(numerator, default_value, dtype=float),
                          where=(denominator != 0))
    return result


def memory_usage_info() -> str:
    """
    Get current memory usage information.
    
    Returns:
        Memory usage information string
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB"
    except ImportError:
        return "Memory usage information not available (psutil not installed)"


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    else:
        hours = seconds // 3600
        remaining_seconds = seconds % 3600
        minutes = remaining_seconds // 60
        remaining_seconds = remaining_seconds % 60
        return f"{int(hours)}h {int(minutes)}m {remaining_seconds:.2f}s"


def print_system_info():
    """Print system information."""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Python version: {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn: Not available")
    
    try:
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib: Not available")
    
    print(memory_usage_info())
    print("=" * 50)


class ProgressBar:
    """Simple progress bar for long-running operations."""
    
    def __init__(self, total: int, prefix: str = "", length: int = 50):
        """
        Initialize progress bar.
        
        Args:
            total: Total number of iterations
            prefix: Prefix string
            length: Length of progress bar
        """
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self.start_time = time.time()
    
    def update(self, current: Optional[int] = None):
        """
        Update progress bar.
        
        Args:
            current: Current iteration (if None, increment by 1)
        """
        if current is not None:
            self.current = current
        else:
            self.current += 1
        
        # Calculate progress
        percent = (self.current / self.total) * 100
        filled_length = int(self.length * self.current / self.total)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        # Calculate ETA
        elapsed_time = time.time() - self.start_time
        if self.current > 0:
            eta = (elapsed_time / self.current) * (self.total - self.current)
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        # Print progress bar
        print(f'\r{self.prefix} |{bar}| {percent:.1f}% ({self.current}/{self.total}) ETA: {eta_str}', 
              end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


def handle_exception(func):
    """
    Decorator for handling exceptions in functions.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = Logger(func.__name__)
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            raise
    return wrapper


# Constants for common operations
DEFAULT_RANDOM_SEED = 42
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tga']
DEFAULT_IMAGE_SIZE = (112, 92)  # Default AT&T face database size