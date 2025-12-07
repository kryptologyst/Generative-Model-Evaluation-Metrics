"""Main evaluation script for generative models."""

import argparse
import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from .core.config import Config
from .core.utils import set_seed, get_device, create_sample_grid
from .datasets.loaders import load_dataset, create_dataloader, generate_toy_dataset
from .metrics.evaluation import calculate_all_metrics


class GenerativeModelEvaluator:
    """Main evaluator class for generative models."""
    
    def __init__(self, config: Config):
        """Initialize evaluator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.device = get_device()
        
        # Set seed for reproducibility
        set_seed(self.config.get("seed", 42))
        
        # Create output directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create necessary output directories."""
        os.makedirs(self.config.get("output.sample_dir", "./assets/samples"), exist_ok=True)
        os.makedirs(self.config.get("logging.log_dir", "./logs"), exist_ok=True)
    
    def load_real_data(self) -> DataLoader:
        """Load real dataset.
        
        Returns:
            DataLoader: DataLoader for real images.
        """
        dataset_name = self.config.get("data.dataset", "cifar10")
        data_dir = self.config.get("data.data_dir", "./data")
        image_size = self.config.get("data.image_size", 64)
        normalize = self.config.get("data.normalize", True)
        batch_size = self.config.get("batch_size", 32)
        num_workers = self.config.get("num_workers", 4)
        
        # Load dataset
        dataset = load_dataset(
            dataset_name=dataset_name,
            root=data_dir,
            train=True,
            download=True,
            image_size=image_size,
            normalize=normalize,
        )
        
        # Create DataLoader
        dataloader = create_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        
        return dataloader
    
    def generate_fake_data(self, n_samples: int) -> torch.Tensor:
        """Generate fake data for evaluation.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            torch.Tensor: Generated images.
        """
        # For demonstration, we'll generate random images
        # In practice, this would be replaced with actual generated images
        image_size = self.config.get("data.image_size", 64)
        n_channels = 3 if self.config.get("data.dataset", "cifar10") != "mnist" else 1
        
        fake_images = generate_toy_dataset(
            n_samples=n_samples,
            image_size=image_size,
            n_channels=n_channels,
            distribution="gaussian",
        )
        
        return fake_images.to(self.device)
    
    def evaluate_metrics(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate metrics on real and fake images.
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
            metrics: List of metrics to calculate.
            
        Returns:
            dict: Dictionary containing metric values.
        """
        print("Calculating evaluation metrics...")
        
        results = calculate_all_metrics(
            real_images=real_images,
            fake_images=fake_images,
            metrics=metrics,
            device=self.device,
        )
        
        return results
    
    def run_evaluation(
        self,
        real_dataloader: Optional[DataLoader] = None,
        fake_images: Optional[torch.Tensor] = None,
        n_samples: int = 1000,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run complete evaluation pipeline.
        
        Args:
            real_dataloader: DataLoader for real images.
            fake_images: Pre-generated fake images.
            n_samples: Number of samples to evaluate.
            metrics: List of metrics to calculate.
            
        Returns:
            dict: Dictionary containing metric values.
        """
        # Load real data if not provided
        if real_dataloader is None:
            real_dataloader = self.load_real_data()
        
        # Collect real images
        print("Loading real images...")
        real_images_list = []
        for batch in tqdm(real_dataloader, desc="Loading real images"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Take images, ignore labels
            real_images_list.append(batch)
            if len(torch.cat(real_images_list, dim=0)) >= n_samples:
                break
        
        real_images = torch.cat(real_images_list, dim=0)[:n_samples].to(self.device)
        
        # Generate fake images if not provided
        if fake_images is None:
            print("Generating fake images...")
            fake_images = self.generate_fake_data(n_samples)
        else:
            fake_images = fake_images[:n_samples].to(self.device)
        
        # Save sample images
        if self.config.get("output.save_samples", True):
            self._save_sample_images(real_images, fake_images)
        
        # Evaluate metrics
        results = self.evaluate_metrics(real_images, fake_images, metrics)
        
        # Print results
        self._print_results(results)
        
        return results
    
    def _save_sample_images(
        self, real_images: torch.Tensor, fake_images: torch.Tensor
    ) -> None:
        """Save sample images for visualization.
        
        Args:
            real_images: Real images tensor.
            fake_images: Generated images tensor.
        """
        import matplotlib.pyplot as plt
        
        sample_dir = self.config.get("output.sample_dir", "./assets/samples")
        n_samples = self.config.get("output.n_samples", 16)
        
        # Create sample grids
        real_grid = create_sample_grid(real_images, n_samples)
        fake_grid = create_sample_grid(fake_images, n_samples)
        
        # Save grids
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Real images
        axes[0].imshow(real_grid.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Real Images")
        axes[0].axis("off")
        
        # Fake images
        axes[1].imshow(fake_grid.permute(1, 2, 0).cpu().numpy())
        axes[1].set_title("Generated Images")
        axes[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "sample_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()
        
        print(f"Sample images saved to {sample_dir}")
    
    def _print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results.
        
        Args:
            results: Dictionary containing metric values.
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        for metric, value in results.items():
            print(f"{metric.upper():<20}: {value:.4f}")
        
        print("="*50)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(description="Evaluate generative models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--metrics", nargs="+", default=["fid", "is", "precision_recall"], help="Metrics to calculate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = Config()
        config.update({"data.dataset": args.dataset, "device": args.device})
    else:
        config = Config()
        config.update({"data.dataset": args.dataset, "device": args.device})
    
    # Create evaluator
    evaluator = GenerativeModelEvaluator(config)
    
    # Run evaluation
    results = evaluator.run_evaluation(
        n_samples=args.n_samples,
        metrics=args.metrics,
    )
    
    return results


if __name__ == "__main__":
    main()
