#!/usr/bin/env python3
"""Script to run evaluation with different configurations."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generative_metrics import GenerativeModelEvaluator, Config, generate_toy_dataset, calculate_all_metrics
from generative_metrics.core.utils import set_seed, get_device


def main():
    """Main function for running evaluations."""
    parser = argparse.ArgumentParser(description="Run generative model evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to evaluate")
    parser.add_argument("--metrics", nargs="+", default=["fid", "is"], help="Metrics to calculate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = Config()
    config.update({
        "data.dataset": args.dataset,
        "device": args.device,
        "seed": args.seed,
        "output.sample_dir": os.path.join(args.output_dir, "samples"),
    })
    
    # Create evaluator
    evaluator = GenerativeModelEvaluator(config)
    
    # Generate toy data for demonstration
    print("Generating toy data for evaluation...")
    device = get_device()
    n_channels = 3 if args.dataset != "mnist" else 1
    
    real_images = generate_toy_dataset(
        n_samples=args.n_samples,
        image_size=64,
        n_channels=n_channels,
        distribution="gaussian",
    ).to(device)
    
    fake_images = generate_toy_dataset(
        n_samples=args.n_samples,
        image_size=64,
        n_channels=n_channels,
        distribution="uniform",
    ).to(device)
    
    # Run evaluation
    print(f"Running evaluation on {args.n_samples} samples...")
    results = evaluator.run_evaluation(
        fake_images=fake_images,
        n_samples=args.n_samples,
        metrics=args.metrics,
    )
    
    # Save results
    import json
    results_file = os.path.join(args.output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
