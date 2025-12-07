"""Core utilities for generative model evaluation."""

import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: The best available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        DictConfig: Loaded configuration.
    """
    return OmegaConf.load(config_path)


def save_config(config: DictConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save.
        config_path: Path to save configuration.
    """
    OmegaConf.save(config, config_path)


def normalize_images(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Normalize images to [-1, 1] range.
    
    Args:
        images: Input images tensor.
        mean: Mean value for normalization.
        std: Standard deviation for normalization.
        
    Returns:
        torch.Tensor: Normalized images.
    """
    return (images - mean) / std


def denormalize_images(images: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """Denormalize images from [-1, 1] range to [0, 1].
    
    Args:
        images: Normalized images tensor.
        mean: Mean value used for normalization.
        std: Standard deviation used for normalization.
        
    Returns:
        torch.Tensor: Denormalized images.
    """
    return images * std + mean


def resize_images(images: torch.Tensor, size: int = 299) -> torch.Tensor:
    """Resize images to specified size.
    
    Args:
        images: Input images tensor.
        size: Target size for resizing.
        
    Returns:
        torch.Tensor: Resized images.
    """
    return torch.nn.functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False
    )


def create_sample_grid(images: torch.Tensor, n_samples: int = 16, n_cols: int = 4) -> torch.Tensor:
    """Create a grid of sample images for visualization.
    
    Args:
        images: Input images tensor.
        n_samples: Number of samples to display.
        n_cols: Number of columns in the grid.
        
    Returns:
        torch.Tensor: Grid of images.
    """
    n_samples = min(n_samples, images.shape[0])
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Select samples
    selected_images = images[:n_samples]
    
    # Pad if necessary
    if n_samples < n_rows * n_cols:
        padding = n_rows * n_cols - n_samples
        padding_images = torch.zeros_like(selected_images[0:1]).repeat(padding, 1, 1, 1)
        selected_images = torch.cat([selected_images, padding_images], dim=0)
    
    # Reshape to grid
    grid_images = selected_images.view(n_rows, n_cols, *selected_images.shape[1:])
    
    # Create grid
    grid = torch.cat([torch.cat([img for img in row], dim=2) for row in grid_images], dim=1)
    
    return grid
