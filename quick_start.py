#!/usr/bin/env python3
"""Quick start script for generative model evaluation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generative_metrics import (
    calculate_all_metrics,
    generate_toy_dataset,
    set_seed,
    get_device,
    GenerativeModelEvaluator,
    Config
)


def main():
    """Quick start demonstration."""
    print("🚀 Generative Model Evaluation - Quick Start")
    print("=" * 50)
    
    # Set up
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    n_samples = 500
    real_images = generate_toy_dataset(n_samples, 64, 3, "gaussian").to(device)
    fake_images = generate_toy_dataset(n_samples, 64, 3, "uniform").to(device)
    
    print(f"Real images: {real_images.shape}")
    print(f"Fake images: {fake_images.shape}")
    
    # Calculate metrics
    print("\n🔍 Calculating evaluation metrics...")
    results = calculate_all_metrics(
        real_images=real_images,
        fake_images=fake_images,
        metrics=["fid", "is", "precision_recall"],
        device=device
    )
    
    # Display results
    print("\n📈 Results:")
    print("-" * 30)
    for metric, value in results.items():
        print(f"{metric.upper():<15}: {value:.4f}")
    print("-" * 30)
    
    print("\n✅ Evaluation complete!")
    print("\nNext steps:")
    print("1. Run 'streamlit run demo/streamlit_app.py' for interactive demo")
    print("2. Check 'notebooks/evaluation_demo.ipynb' for detailed examples")
    print("3. Use 'python scripts/run_evaluation.py --help' for CLI options")


if __name__ == "__main__":
    main()
