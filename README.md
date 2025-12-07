# Generative Model Evaluation Metrics

A comprehensive Python library for evaluating generative models using state-of-the-art metrics including Fréchet Inception Distance (FID), Inception Score (IS), Precision/Recall, and more.

## Features

- **Comprehensive Metrics**: FID, IS, Precision/Recall, LPIPS, and more
- **Multiple Datasets**: Support for CIFAR-10, MNIST, Fashion-MNIST, and custom datasets
- **Device Support**: Automatic detection of CUDA, MPS (Apple Silicon), or CPU
- **Modern Architecture**: Built with PyTorch 2.0+, type hints, and clean code
- **Interactive Demo**: Streamlit web interface for easy evaluation
- **Reproducible**: Deterministic seeding and proper configuration management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Generative-Model-Evaluation-Metrics.git
cd Generative-Model-Evaluation-Metrics

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Basic Usage

```python
import torch
from generative_metrics import calculate_all_metrics, generate_toy_dataset

# Generate sample data
real_images = generate_toy_dataset(n_samples=1000, image_size=64, n_channels=3)
fake_images = generate_toy_dataset(n_samples=1000, image_size=64, n_channels=3)

# Calculate metrics
results = calculate_all_metrics(
    real_images=real_images,
    fake_images=fake_images,
    metrics=["fid", "is", "precision_recall"]
)

print(f"FID: {results['fid']:.4f}")
print(f"IS: {results['is']:.4f}")
print(f"Precision: {results['precision']:.4f}")
print(f"Recall: {results['recall']:.4f}")
```

### Command Line Interface

```bash
# Run evaluation with default settings
python -m src.generative_metrics.evaluate

# Custom configuration
python -m src.generative_metrics.evaluate --dataset cifar10 --n_samples 2000 --metrics fid is
```

### Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/streamlit_app.py
```

## Supported Metrics

### Fréchet Inception Distance (FID)
Measures the distance between feature distributions of real and generated images. Lower values indicate better performance.

```python
from generative_metrics import FrechetInceptionDistance

fid_metric = FrechetInceptionDistance()
fid_metric.update(real_images, fake_images)
fid_score = fid_metric.compute()
```

### Inception Score (IS)
Evaluates both quality and diversity of generated images using a pre-trained InceptionV3 model.

```python
from generative_metrics import InceptionScore

is_metric = InceptionScore()
is_metric.update(fake_images)
is_score = is_metric.compute()
```

### Precision and Recall
Measures the quality (precision) and diversity (recall) of generated samples.

```python
from generative_metrics import PrecisionRecall

pr_metric = PrecisionRecall()
pr_metric.update(real_images, fake_images)
precision, recall = pr_metric.compute()
```

## Datasets

### Supported Datasets
- **CIFAR-10**: 32x32 color images with 10 classes
- **MNIST**: 28x28 grayscale handwritten digits
- **Fashion-MNIST**: 28x28 grayscale clothing items

### Custom Datasets

```python
from generative_metrics import load_dataset, create_dataloader

# Load custom dataset
dataset = load_dataset("cifar10", root="./data", image_size=64)
dataloader = create_dataloader(dataset, batch_size=32)
```

## Configuration

The library uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
seed: 42
device: auto
batch_size: 32

data:
  dataset: cifar10
  data_dir: ./data
  image_size: 64
  normalize: true

metrics:
  fid:
    enabled: true
    batch_size: 64
    feature_extractor: inception_v3
  is:
    enabled: true
    splits: 10
```

## Advanced Usage

### Custom Feature Extractors

```python
from generative_metrics import get_feature_extractor

# Use different feature extractors
extractor = get_feature_extractor("resnet50", pretrained=True)
features = extractor(images)
```

### Batch Processing

```python
from generative_metrics import GenerativeModelEvaluator, Config

# Create evaluator with custom config
config = Config()
config.update({"batch_size": 64, "data.dataset": "cifar10"})
evaluator = GenerativeModelEvaluator(config)

# Run evaluation
results = evaluator.run_evaluation(n_samples=5000)
```

## Project Structure

```
generative-metrics/
├── src/generative_metrics/
│   ├── core/                 # Core utilities and configuration
│   ├── datasets/            # Dataset loaders and utilities
│   ├── metrics/             # Evaluation metrics
│   └── evaluate.py          # Main evaluation script
├── configs/                 # Configuration files
├── demo/                    # Interactive demos
├── tests/                   # Unit tests
├── assets/                  # Output samples and results
└── data/                    # Dataset storage
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=src tests/
```

### Code Formatting

```bash
# Format code
black src/ tests/
ruff check src/ tests/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Performance Considerations

- **GPU Acceleration**: Automatically uses CUDA or MPS when available
- **Batch Processing**: Efficient batch processing for large datasets
- **Memory Management**: Optimized memory usage for feature extraction
- **Caching**: Optional caching of precomputed features

## Limitations

- FID and IS require pre-trained InceptionV3 model
- Large datasets may require significant memory
- Some metrics are computationally expensive for high-resolution images

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{generative_metrics,
  title={Generative Model Evaluation Metrics},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Generative-Model-Evaluation-Metrics}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- torchvision for pre-trained models
- Clean-FID for FID implementation reference
- The generative modeling community for metric development
# Generative-Model-Evaluation-Metrics
