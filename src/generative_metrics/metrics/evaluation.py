"""Comprehensive evaluation metrics for generative models."""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from sklearn.metrics import pairwise_distances
from torchmetrics import Metric

from ..core.feature_extractors import get_feature_extractor
from ..core.utils import get_device


class InceptionScore(Metric):
    """Inception Score (IS) metric for generative models.
    
    The Inception Score measures both the quality and diversity of generated images
    by evaluating how well they can be classified by a pre-trained InceptionV3 model.
    """
    
    def __init__(
        self,
        feature_extractor: str = "inception_v3",
        splits: int = 10,
        device: Optional[torch.device] = None,
    ):
        """Initialize Inception Score metric.
        
        Args:
            feature_extractor: Name of the feature extractor to use.
            splits: Number of splits for IS calculation.
            device: Device to run computations on.
        """
        super().__init__()
        
        self.splits = splits
        self.device = device or get_device()
        
        # Load feature extractor
        self.feature_extractor = get_feature_extractor(
            feature_extractor, pretrained=True, device=self.device
        )
        
        # Add state for accumulating predictions
        self.add_state("predictions", default=[], dist_reduce_fx="cat")
    
    def update(self, images: torch.Tensor) -> None:
        """Update metric with new images.
        
        Args:
            images: Generated images tensor of shape (N, C, H, W).
        """
        # Ensure images are on the correct device
        images = images.to(self.device)
        
        # Extract features and get predictions
        with torch.no_grad():
            features = self.feature_extractor(images)
            # For InceptionV3, we need to get the logits from the original model
            if hasattr(self.feature_extractor, 'model'):
                # Temporarily restore the classification head
                original_fc = self.feature_extractor.model.fc
                self.feature_extractor.model.fc = torch.nn.Linear(2048, 1000).to(self.device)
                
                # Get predictions
                predictions = self.feature_extractor.model(
                    self.feature_extractor.transform(images)
                )
                
                # Restore the identity layer
                self.feature_extractor.model.fc = original_fc
            else:
                # Fallback: use features directly
                predictions = features
        
        # Convert to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Store predictions
        self.predictions.append(predictions.cpu())
    
    def compute(self) -> torch.Tensor:
        """Compute the Inception Score.
        
        Returns:
            torch.Tensor: Inception Score value.
        """
        if not self.predictions:
            return torch.tensor(0.0)
        
        # Concatenate all predictions
        all_predictions = torch.cat(self.predictions, dim=0).numpy()
        
        # Calculate IS
        is_score = self._calculate_is(all_predictions)
        
        return torch.tensor(is_score, dtype=torch.float32)
    
    def _calculate_is(self, predictions: np.ndarray) -> float:
        """Calculate Inception Score from predictions.
        
        Args:
            predictions: Predictions array of shape (N, num_classes).
            
        Returns:
            float: Inception Score.
        """
        # Calculate marginal distribution
        marginal = np.mean(predictions, axis=0)
        
        # Calculate KL divergence for each sample
        kl_divs = []
        for pred in predictions:
            kl_div = np.sum(pred * (np.log(pred + 1e-16) - np.log(marginal + 1e-16)))
            kl_divs.append(kl_div)
        
        # Calculate IS
        is_score = np.exp(np.mean(kl_divs))
        
        return float(is_score)


class FrechetInceptionDistance(Metric):
    """Fréchet Inception Distance (FID) metric for generative models.
    
    FID measures the distance between the feature distributions of real and generated images.
    Lower FID indicates better performance.
    """
    
    def __init__(
        self,
        feature_extractor: str = "inception_v3",
        device: Optional[torch.device] = None,
    ):
        """Initialize FID metric.
        
        Args:
            feature_extractor: Name of the feature extractor to use.
            device: Device to run computations on.
        """
        super().__init__()
        
        self.device = device or get_device()
        
        # Load feature extractor
        self.feature_extractor = get_feature_extractor(
            feature_extractor, pretrained=True, device=self.device
        )
        
        # Add state for accumulating features
        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")
    
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        """Update metric with real and generated images.
        
        Args:
            real_images: Real images tensor of shape (N, C, H, W).
            fake_images: Generated images tensor of shape (N, C, H, W).
        """
        # Ensure images are on the correct device
        real_images = real_images.to(self.device)
        fake_images = fake_images.to(self.device)
        
        # Extract features
        with torch.no_grad():
            real_feats = self.feature_extractor(real_images)
            fake_feats = self.feature_extractor(fake_images)
        
        # Store features
        self.real_features.append(real_feats.cpu())
        self.fake_features.append(fake_feats.cpu())
    
    def compute(self) -> torch.Tensor:
        """Compute the Fréchet Inception Distance.
        
        Returns:
            torch.Tensor: FID value.
        """
        if not self.real_features or not self.fake_features:
            return torch.tensor(float('inf'))
        
        # Concatenate all features
        real_feats = torch.cat(self.real_features, dim=0).numpy()
        fake_feats = torch.cat(self.fake_features, dim=0).numpy()
        
        # Calculate FID
        fid_score = self._calculate_fid(real_feats, fake_feats)
        
        return torch.tensor(fid_score, dtype=torch.float32)
    
    def _calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """Calculate FID from feature arrays.
        
        Args:
            real_features: Real image features of shape (N, feature_dim).
            fake_features: Generated image features of shape (N, feature_dim).
            
        Returns:
            float: FID value.
        """
        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        # Handle numerical issues
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            )
            print(msg)
            offset = np.eye(sigma_real.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
        
        # Calculate FID
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
        
        return float(fid)


class PrecisionRecall(Metric):
    """Precision and Recall metrics for generative models.
    
    Precision measures the quality of generated samples, while Recall measures
    the diversity/coverage of the generated distribution.
    """
    
    def __init__(self, k: int = 3, device: Optional[torch.device] = None):
        """Initialize Precision and Recall metrics.
        
        Args:
            k: Number of nearest neighbors to consider.
            device: Device to run computations on.
        """
        super().__init__()
        
        self.k = k
        self.device = device or get_device()
        
        # Add state for accumulating features
        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")
    
    def update(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> None:
        """Update metric with real and generated images.
        
        Args:
            real_images: Real images tensor of shape (N, C, H, W).
            fake_images: Generated images tensor of shape (N, C, H, W).
        """
        # Flatten images to use as features
        real_feats = real_images.view(real_images.size(0), -1).cpu()
        fake_feats = fake_images.view(fake_images.size(0), -1).cpu()
        
        # Store features
        self.real_features.append(real_feats)
        self.fake_features.append(fake_feats)
    
    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Precision and Recall.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Precision, Recall) values.
        """
        if not self.real_features or not self.fake_features:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        # Concatenate all features
        real_feats = torch.cat(self.real_features, dim=0).numpy()
        fake_feats = torch.cat(self.fake_features, dim=0).numpy()
        
        # Calculate Precision and Recall
        precision, recall = self._calculate_precision_recall(real_feats, fake_feats)
        
        return torch.tensor(precision, dtype=torch.float32), torch.tensor(recall, dtype=torch.float32)
    
    def _calculate_precision_recall(
        self, real_features: np.ndarray, fake_features: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate Precision and Recall from feature arrays.
        
        Args:
            real_features: Real image features of shape (N, feature_dim).
            fake_features: Generated image features of shape (N, feature_dim).
            
        Returns:
            Tuple[float, float]: (Precision, Recall) values.
        """
        # Calculate pairwise distances
        real_distances = pairwise_distances(real_features, metric='euclidean')
        fake_distances = pairwise_distances(fake_features, metric='euclidean')
        cross_distances = pairwise_distances(real_features, fake_features, metric='euclidean')
        
        # Calculate Precision
        precision_scores = []
        for i in range(len(fake_features)):
            # Find k nearest neighbors in fake features
            fake_neighbors = np.argsort(fake_distances[i])[:self.k + 1][1:]  # +1 to exclude self
            
            # Check if any of these neighbors are close to real features
            min_distances = np.min(cross_distances[:, i])
            precision_scores.append(min_distances)
        
        precision = np.mean(precision_scores)
        
        # Calculate Recall
        recall_scores = []
        for i in range(len(real_features)):
            # Find k nearest neighbors in real features
            real_neighbors = np.argsort(real_distances[i])[:self.k + 1][1:]  # +1 to exclude self
            
            # Check if any of these neighbors are close to fake features
            min_distances = np.min(cross_distances[i, :])
            recall_scores.append(min_distances)
        
        recall = np.mean(recall_scores)
        
        return float(precision), float(recall)


def calculate_all_metrics(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    metrics: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Calculate all evaluation metrics.
    
    Args:
        real_images: Real images tensor.
        fake_images: Generated images tensor.
        metrics: List of metrics to calculate. If None, calculates all.
        device: Device to run computations on.
        
    Returns:
        dict: Dictionary containing metric values.
    """
    if metrics is None:
        metrics = ["fid", "is", "precision_recall"]
    
    device = device or get_device()
    results = {}
    
    # Calculate FID
    if "fid" in metrics:
        fid_metric = FrechetInceptionDistance(device=device)
        fid_metric.update(real_images, fake_images)
        results["fid"] = fid_metric.compute().item()
    
    # Calculate IS
    if "is" in metrics:
        is_metric = InceptionScore(device=device)
        is_metric.update(fake_images)
        results["is"] = is_metric.compute().item()
    
    # Calculate Precision and Recall
    if "precision_recall" in metrics:
        pr_metric = PrecisionRecall(device=device)
        pr_metric.update(real_images, fake_images)
        precision, recall = pr_metric.compute()
        results["precision"] = precision.item()
        results["recall"] = recall.item()
    
    return results
