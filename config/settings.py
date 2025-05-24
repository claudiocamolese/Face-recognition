"""Configuration settings for the face recognition system."""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    """Configuration class containing all system parameters."""
    
    # Dataset settings
    DATASET_PATH: str = "Kaggle/"
    IMAGE_SHAPE: Tuple[int, int] = (112, 92)
    
    # Data splitting
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 30
    
    # Image preprocessing
    NORMALIZE_PIXELS: bool = True
    PIXEL_RANGE: Tuple[float, float] = (0.0, 1.0)
    
    # SVD settings
    SVD_COMPONENTS: int = 90
    ENERGY_THRESHOLD: float = 0.90
    SVD_TOLERANCE: float = 1e-10
    MAX_ITERATIONS: int = 1000
    
    # PCA settings
    PCA_COMPONENTS: int = 2
    
    # KNN settings
    KNN_NEIGHBORS: int = 3
    KNN_WEIGHTS: str = 'distance'
    KNN_METRIC: str = 'euclidean'
    
    # Logging
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Visualization
    ENABLE_VISUALIZATION: bool = True
    FIGURE_SIZE: Tuple[int, int] = (15, 5)
    DPI: int = 100
    
    # Performance
    N_JOBS: int = -1  # Use all available cores
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not os.path.exists(self.DATASET_PATH):
            raise FileNotFoundError(f"Dataset path not found: {self.DATASET_PATH}")
        
        if not 0 < self.TEST_SIZE < 1:
            raise ValueError("TEST_SIZE must be between 0 and 1")
        
        if self.SVD_COMPONENTS <= 0:
            raise ValueError("SVD_COMPONENTS must be positive")
        
        if self.PCA_COMPONENTS <= 0:
            raise ValueError("PCA_COMPONENTS must be positive")
        
        if self.KNN_NEIGHBORS <= 0:
            raise ValueError("KNN_NEIGHBORS must be positive")
        
        if not 0 < self.ENERGY_THRESHOLD <= 1:
            raise ValueError("ENERGY_THRESHOLD must be between 0 and 1")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to file."""
        import json
        with open(filepath, 'w') as f:
            # Convert tuples to lists for JSON serialization
            config_dict = {}
            for key, value in self.to_dict().items():
                if isinstance(value, tuple):
                    config_dict[key] = list(value)
                else:
                    config_dict[key] = value
            json.dump(config_dict, f, indent=2)