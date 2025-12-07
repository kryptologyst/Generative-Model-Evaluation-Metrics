"""Unit tests for generative metrics evaluation."""

import pytest
import torch
import numpy as np

from src.generative_metrics.core.utils import set_seed, get_device, normalize_images, denormalize_images, resize_images, create_sample_grid
from src.generative_metrics.core.feature_extractors import get_feature_extractor
from src.generative_metrics.metrics.evaluation import InceptionScore, FrechetInceptionDistance, PrecisionRecall, calculate_all_metrics
from src.generative_metrics.datasets.loaders import generate_toy_dataset, get_dataset_info


class TestCoreUtils:
    """Test core utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # Test that random numbers are deterministic
        torch.manual_seed(42)
        a = torch.randn(10)
        set_seed(42)
        b = torch.randn(10)
        assert torch.allclose(a, b)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_normalize_images(self):
        """Test image normalization."""
        images = torch.rand(5, 3, 32, 32)
        normalized = normalize_images(images)
        assert normalized.min() >= -1
        assert normalized.max() <= 1
    
    def test_denormalize_images(self):
        """Test image denormalization."""
        images = torch.rand(5, 3, 32, 32)
        normalized = normalize_images(images)
        denormalized = denormalize_images(normalized)
        assert torch.allclose(images, denormalized, atol=1e-6)
    
    def test_resize_images(self):
        """Test image resizing."""
        images = torch.rand(5, 3, 32, 32)
        resized = resize_images(images, size=64)
        assert resized.shape == (5, 3, 64, 64)
    
    def test_create_sample_grid(self):
        """Test sample grid creation."""
        images = torch.rand(16, 3, 32, 32)
        grid = create_sample_grid(images, n_samples=16, n_cols=4)
        assert grid.shape[1] == 3  # 3 channels
        assert grid.shape[2] == 32 * 4  # 4 columns
        assert grid.shape[3] == 32 * 4  # 4 rows


class TestFeatureExtractors:
    """Test feature extractors."""
    
    def test_get_feature_extractor(self):
        """Test feature extractor creation."""
        extractor = get_feature_extractor("inception_v3", pretrained=False)
        assert extractor is not None
        assert hasattr(extractor, "forward")
        assert hasattr(extractor, "get_feature_dim")
    
    def test_inception_v3_extractor(self):
        """Test InceptionV3 feature extractor."""
        from src.generative_metrics.core.feature_extractors import InceptionV3FeatureExtractor
        
        extractor = InceptionV3FeatureExtractor(pretrained=False)
        images = torch.rand(2, 3, 64, 64)
        features = extractor(images)
        assert features.shape[0] == 2
        assert features.shape[1] == 2048


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_inception_score(self):
        """Test Inception Score calculation."""
        metric = InceptionScore(device=torch.device("cpu"))
        images = torch.rand(10, 3, 64, 64)
        metric.update(images)
        score = metric.compute()
        assert isinstance(score, torch.Tensor)
        assert score.item() > 0
    
    def test_frechet_inception_distance(self):
        """Test FID calculation."""
        metric = FrechetInceptionDistance(device=torch.device("cpu"))
        real_images = torch.rand(10, 3, 64, 64)
        fake_images = torch.rand(10, 3, 64, 64)
        metric.update(real_images, fake_images)
        fid = metric.compute()
        assert isinstance(fid, torch.Tensor)
        assert fid.item() >= 0
    
    def test_precision_recall(self):
        """Test Precision and Recall calculation."""
        metric = PrecisionRecall(device=torch.device("cpu"))
        real_images = torch.rand(10, 3, 32, 32)
        fake_images = torch.rand(10, 3, 32, 32)
        metric.update(real_images, fake_images)
        precision, recall = metric.compute()
        assert isinstance(precision, torch.Tensor)
        assert isinstance(recall, torch.Tensor)
        assert precision.item() >= 0
        assert recall.item() >= 0
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        real_images = torch.rand(20, 3, 64, 64)
        fake_images = torch.rand(20, 3, 64, 64)
        
        results = calculate_all_metrics(
            real_images=real_images,
            fake_images=fake_images,
            metrics=["fid", "is", "precision_recall"],
            device=torch.device("cpu"),
        )
        
        assert "fid" in results
        assert "is" in results
        assert "precision" in results
        assert "recall" in results
        
        for metric, value in results.items():
            assert isinstance(value, float)
            assert not np.isnan(value)


class TestDatasets:
    """Test dataset utilities."""
    
    def test_generate_toy_dataset(self):
        """Test toy dataset generation."""
        dataset = generate_toy_dataset(n_samples=100, image_size=32, n_channels=3)
        assert dataset.shape == (100, 3, 32, 32)
        assert dataset.min() >= 0
        assert dataset.max() <= 1
    
    def test_get_dataset_info(self):
        """Test dataset info retrieval."""
        info = get_dataset_info("cifar10")
        assert "n_classes" in info
        assert "image_size" in info
        assert "n_channels" in info
        assert info["n_classes"] == 10
        assert info["image_size"] == 32
        assert info["n_channels"] == 3


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_evaluation(self):
        """Test end-to-end evaluation pipeline."""
        # Generate test data
        real_images = generate_toy_dataset(n_samples=50, image_size=64, n_channels=3)
        fake_images = generate_toy_dataset(n_samples=50, image_size=64, n_channels=3)
        
        # Calculate metrics
        results = calculate_all_metrics(
            real_images=real_images,
            fake_images=fake_images,
            metrics=["fid", "is"],
            device=torch.device("cpu"),
        )
        
        # Verify results
        assert len(results) == 2
        assert all(isinstance(v, float) for v in results.values())
        assert all(not np.isnan(v) for v in results.values())


if __name__ == "__main__":
    pytest.main([__file__])
