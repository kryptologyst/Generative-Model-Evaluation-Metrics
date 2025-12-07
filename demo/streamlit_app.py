"""Streamlit demo for generative model evaluation."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from src.generative_metrics.core.config import Config
from src.generative_metrics.core.utils import set_seed, get_device, create_sample_grid
from src.generative_metrics.datasets.loaders import load_dataset, generate_toy_dataset
from src.generative_metrics.metrics.evaluation import calculate_all_metrics


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Generative Model Evaluation",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 Generative Model Evaluation Metrics")
    st.markdown("Comprehensive evaluation of generative models using FID, IS, Precision/Recall, and more.")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset = st.sidebar.selectbox(
        "Dataset",
        ["cifar10", "mnist", "fashion_mnist"],
        index=0,
    )
    
    # Number of samples
    n_samples = st.sidebar.slider(
        "Number of samples",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )
    
    # Metrics selection
    st.sidebar.subheader("Metrics")
    calculate_fid = st.sidebar.checkbox("FID", value=True)
    calculate_is = st.sidebar.checkbox("Inception Score", value=True)
    calculate_pr = st.sidebar.checkbox("Precision/Recall", value=True)
    
    # Device selection
    device = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda", "mps"],
        index=0,
    )
    
    # Seed
    seed = st.sidebar.number_input("Random seed", value=42, min_value=0)
    
    # Generate button
    if st.sidebar.button("🚀 Run Evaluation", type="primary"):
        run_evaluation(dataset, n_samples, calculate_fid, calculate_is, calculate_pr, device, seed)
    
    # Display sample images
    display_sample_images(dataset)


def run_evaluation(
    dataset: str,
    n_samples: int,
    calculate_fid: bool,
    calculate_is: bool,
    calculate_pr: bool,
    device: str,
    seed: int,
):
    """Run the evaluation pipeline."""
    
    # Set up progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Set seed and device
        set_seed(seed)
        device_obj = get_device() if device == "auto" else torch.device(device)
        
        # Load real data
        status_text.text("Loading real images...")
        progress_bar.progress(10)
        
        real_dataset = load_dataset(
            dataset_name=dataset,
            root="./data",
            train=True,
            download=True,
            image_size=64,
            normalize=True,
        )
        
        # Sample real images
        real_images = []
        for i in range(min(n_samples, len(real_dataset))):
            real_images.append(real_dataset[i])
        real_images = torch.stack(real_images).to(device_obj)
        
        # Generate fake images
        status_text.text("Generating fake images...")
        progress_bar.progress(30)
        
        n_channels = 3 if dataset != "mnist" else 1
        fake_images = generate_toy_dataset(
            n_samples=n_samples,
            image_size=64,
            n_channels=n_channels,
            distribution="gaussian",
        ).to(device_obj)
        
        # Calculate metrics
        status_text.text("Calculating metrics...")
        progress_bar.progress(50)
        
        metrics = []
        if calculate_fid:
            metrics.append("fid")
        if calculate_is:
            metrics.append("is")
        if calculate_pr:
            metrics.append("precision_recall")
        
        results = calculate_all_metrics(
            real_images=real_images,
            fake_images=fake_images,
            metrics=metrics,
            device=device_obj,
        )
        
        progress_bar.progress(100)
        status_text.text("Evaluation complete!")
        
        # Display results
        display_results(results)
        
        # Display sample comparison
        display_sample_comparison(real_images, fake_images)
        
    except Exception as e:
        st.error(f"Error during evaluation: {str(e)}")
        progress_bar.progress(0)
        status_text.text("Evaluation failed.")


def display_results(results: Dict[str, float]):
    """Display evaluation results."""
    st.header("📊 Evaluation Results")
    
    # Create columns for metrics
    cols = st.columns(len(results))
    
    for i, (metric, value) in enumerate(results.items()):
        with cols[i]:
            st.metric(
                label=metric.upper(),
                value=f"{value:.4f}",
                help=f"{metric} metric value",
            )
    
    # Detailed results table
    st.subheader("Detailed Results")
    results_df = {
        "Metric": list(results.keys()),
        "Value": [f"{v:.6f}" for v in results.values()],
    }
    st.table(results_df)


def display_sample_comparison(real_images: torch.Tensor, fake_images: torch.Tensor):
    """Display sample image comparison."""
    st.header("🖼️ Sample Comparison")
    
    # Create sample grids
    real_grid = create_sample_grid(real_images, n_samples=16)
    fake_grid = create_sample_grid(fake_images, n_samples=16)
    
    # Convert to numpy for display
    real_np = real_grid.permute(1, 2, 0).cpu().numpy()
    fake_np = fake_grid.permute(1, 2, 0).cpu().numpy()
    
    # Normalize to [0, 1] for display
    real_np = (real_np + 1) / 2
    fake_np = (fake_np + 1) / 2
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Real Images")
        st.image(real_np, use_column_width=True)
    
    with col2:
        st.subheader("Generated Images")
        st.image(fake_np, use_column_width=True)


def display_sample_images(dataset: str):
    """Display sample images from the dataset."""
    st.header("📁 Dataset Samples")
    
    try:
        # Load a small sample
        sample_dataset = load_dataset(
            dataset_name=dataset,
            root="./data",
            train=True,
            download=True,
            image_size=64,
            normalize=True,
        )
        
        # Create sample grid
        sample_images = []
        for i in range(16):
            sample_images.append(sample_dataset[i])
        sample_images = torch.stack(sample_images)
        
        # Create grid
        grid = create_sample_grid(sample_images, n_samples=16)
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        grid_np = (grid_np + 1) / 2  # Normalize for display
        
        st.image(grid_np, use_column_width=True, caption=f"Sample images from {dataset.upper()}")
        
    except Exception as e:
        st.warning(f"Could not load sample images: {str(e)}")


if __name__ == "__main__":
    main()
