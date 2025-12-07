"""Dataset utilities for generative model evaluation."""

import os
from typing import Callable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


class ImageDataset(Dataset):
    """Generic image dataset wrapper."""
    
    def __init__(
        self,
        images: torch.Tensor,
        transform: Optional[Callable] = None,
    ):
        """Initialize dataset.
        
        Args:
            images: Tensor of images of shape (N, C, H, W).
            transform: Optional transform to apply to images.
        """
        self.images = images
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item by index."""
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image


def get_cifar10_dataset(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    image_size: int = 32,
    normalize: bool = True,
) -> torchvision.datasets.CIFAR10:
    """Get CIFAR-10 dataset.
    
    Args:
        root: Root directory for dataset.
        train: Whether to use training set.
        download: Whether to download dataset.
        image_size: Target image size.
        normalize: Whether to normalize images to [-1, 1].
        
    Returns:
        CIFAR-10 dataset.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    
    transform = transforms.Compose(transform_list)
    
    return CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def get_mnist_dataset(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    image_size: int = 28,
    normalize: bool = True,
) -> torchvision.datasets.MNIST:
    """Get MNIST dataset.
    
    Args:
        root: Root directory for dataset.
        train: Whether to use training set.
        download: Whether to download dataset.
        image_size: Target image size.
        normalize: Whether to normalize images to [-1, 1].
        
    Returns:
        MNIST dataset.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    transform = transforms.Compose(transform_list)
    
    return MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def get_fashion_mnist_dataset(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    image_size: int = 28,
    normalize: bool = True,
) -> torchvision.datasets.FashionMNIST:
    """Get Fashion-MNIST dataset.
    
    Args:
        root: Root directory for dataset.
        train: Whether to use training set.
        download: Whether to download dataset.
        image_size: Target image size.
        normalize: Whether to normalize images to [-1, 1].
        
    Returns:
        Fashion-MNIST dataset.
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    transform = transforms.Compose(transform_list)
    
    return FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the dataset.
    
    Args:
        dataset: Dataset to create loader for.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.
        
    Returns:
        DataLoader for the dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def generate_toy_dataset(
    n_samples: int = 1000,
    image_size: int = 64,
    n_channels: int = 3,
    distribution: str = "gaussian",
) -> torch.Tensor:
    """Generate a toy dataset for testing.
    
    Args:
        n_samples: Number of samples to generate.
        image_size: Size of each image.
        n_channels: Number of channels.
        distribution: Distribution to sample from ('gaussian', 'uniform').
        
    Returns:
        torch.Tensor: Generated images of shape (n_samples, n_channels, image_size, image_size).
    """
    if distribution == "gaussian":
        images = torch.randn(n_samples, n_channels, image_size, image_size)
    elif distribution == "uniform":
        images = torch.rand(n_samples, n_channels, image_size, image_size)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Normalize to [0, 1] range
    images = (images - images.min()) / (images.max() - images.min())
    
    return images


def load_dataset(
    dataset_name: str,
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    image_size: int = 32,
    normalize: bool = True,
) -> Dataset:
    """Load a dataset by name.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'mnist', 'fashion_mnist').
        root: Root directory for dataset.
        train: Whether to use training set.
        download: Whether to download dataset.
        image_size: Target image size.
        normalize: Whether to normalize images to [-1, 1].
        
    Returns:
        Dataset object.
    """
    if dataset_name.lower() == "cifar10":
        return get_cifar10_dataset(root, train, download, image_size, normalize)
    elif dataset_name.lower() == "mnist":
        return get_mnist_dataset(root, train, download, image_size, normalize)
    elif dataset_name.lower() == "fashion_mnist":
        return get_fashion_mnist_dataset(root, train, download, image_size, normalize)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_dataset_info(dataset_name: str) -> dict:
    """Get information about a dataset.
    
    Args:
        dataset_name: Name of the dataset.
        
    Returns:
        dict: Dataset information.
    """
    info = {
        "cifar10": {
            "n_classes": 10,
            "image_size": 32,
            "n_channels": 3,
            "n_train": 50000,
            "n_test": 10000,
            "description": "CIFAR-10 dataset with 10 classes of natural images",
        },
        "mnist": {
            "n_classes": 10,
            "image_size": 28,
            "n_channels": 1,
            "n_train": 60000,
            "n_test": 10000,
            "description": "MNIST dataset with handwritten digits",
        },
        "fashion_mnist": {
            "n_classes": 10,
            "image_size": 28,
            "n_channels": 1,
            "n_train": 60000,
            "n_test": 10000,
            "description": "Fashion-MNIST dataset with clothing items",
        },
    }
    
    return info.get(dataset_name.lower(), {})
