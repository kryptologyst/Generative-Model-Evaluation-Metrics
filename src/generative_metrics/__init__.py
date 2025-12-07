"""Generative Metrics - Comprehensive evaluation metrics for generative models."""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@example.com"

from .core.config import Config
from .core.utils import set_seed, get_device, normalize_images, denormalize_images, resize_images, create_sample_grid
from .core.feature_extractors import get_feature_extractor, InceptionV3FeatureExtractor, ResNetFeatureExtractor
from .metrics.evaluation import (
    InceptionScore,
    FrechetInceptionDistance,
    PrecisionRecall,
    calculate_all_metrics,
)
from .datasets.loaders import (
    load_dataset,
    create_dataloader,
    generate_toy_dataset,
    get_dataset_info,
)
from .evaluate import GenerativeModelEvaluator

__all__ = [
    # Core
    "Config",
    "set_seed",
    "get_device",
    "normalize_images",
    "denormalize_images",
    "resize_images",
    "create_sample_grid",
    
    # Feature extractors
    "get_feature_extractor",
    "InceptionV3FeatureExtractor",
    "ResNetFeatureExtractor",
    
    # Metrics
    "InceptionScore",
    "FrechetInceptionDistance",
    "PrecisionRecall",
    "calculate_all_metrics",
    
    # Datasets
    "load_dataset",
    "create_dataloader",
    "generate_toy_dataset",
    "get_dataset_info",
    
    # Evaluation
    "GenerativeModelEvaluator",
]
