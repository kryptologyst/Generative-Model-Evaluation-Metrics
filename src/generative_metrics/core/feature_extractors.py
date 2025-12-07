"""Feature extractors for evaluation metrics."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import Inception_V3_Weights


class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 feature extractor for FID and IS calculations."""
    
    def __init__(self, pretrained: bool = True, device: Optional[torch.device] = None):
        """Initialize InceptionV3 feature extractor.
        
        Args:
            pretrained: Whether to use pretrained weights.
            device: Device to load the model on.
        """
        super().__init__()
        
        # Load pretrained InceptionV3
        if pretrained:
            self.model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        else:
            self.model = models.inception_v3(weights=None)
        
        # Remove the final classification layer
        self.model.fc = nn.Identity()
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to device
        if device is not None:
            self.model = self.model.to(device)
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input images tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Extracted features of shape (B, 2048).
        """
        # Ensure input is in [0, 1] range
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Apply transforms
        x = self.transform(x)
        
        # Extract features
        with torch.no_grad():
            features = self.model(x)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features.
        
        Returns:
            int: Feature dimension.
        """
        return 2048


class ResNetFeatureExtractor(nn.Module):
    """ResNet feature extractor for evaluation metrics."""
    
    def __init__(
        self, 
        model_name: str = "resnet50", 
        pretrained: bool = True, 
        device: Optional[torch.device] = None
    ):
        """Initialize ResNet feature extractor.
        
        Args:
            model_name: Name of ResNet model (resnet18, resnet34, resnet50, etc.).
            pretrained: Whether to use pretrained weights.
            device: Device to load the model on.
        """
        super().__init__()
        
        # Load ResNet model
        if hasattr(models, model_name):
            model_class = getattr(models, model_name)
            if pretrained:
                self.model = model_class(weights="IMAGENET1K_V1")
            else:
                self.model = model_class(weights=None)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Remove the final classification layer
        self.model.fc = nn.Identity()
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to device
        if device is not None:
            self.model = self.model.to(device)
        
        # Define preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.
        
        Args:
            x: Input images tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Extracted features.
        """
        # Ensure input is in [0, 1] range
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Apply transforms
        x = self.transform(x)
        
        # Extract features
        with torch.no_grad():
            features = self.model(x)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features.
        
        Returns:
            int: Feature dimension.
        """
        return self.model.fc.in_features if hasattr(self.model.fc, 'in_features') else 2048


def get_feature_extractor(
    name: str, 
    pretrained: bool = True, 
    device: Optional[torch.device] = None
) -> nn.Module:
    """Get a feature extractor by name.
    
    Args:
        name: Name of the feature extractor.
        pretrained: Whether to use pretrained weights.
        device: Device to load the model on.
        
    Returns:
        nn.Module: Feature extractor model.
    """
    if name.lower() == "inception_v3":
        return InceptionV3FeatureExtractor(pretrained=pretrained, device=device)
    elif name.lower().startswith("resnet"):
        return ResNetFeatureExtractor(model_name=name, pretrained=pretrained, device=device)
    else:
        raise ValueError(f"Unknown feature extractor: {name}")
