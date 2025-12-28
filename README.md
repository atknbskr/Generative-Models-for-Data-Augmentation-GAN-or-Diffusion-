# GAN-Based Data Augmentation for Imbalanced Classification

This project implements a comprehensive study on using Generative Adversarial Networks (GANs) to improve classification performance on imbalanced datasets through synthetic data augmentation.

## Project Overview

The project addresses the problem of class imbalance in image classification by:
1. Training a conditional GAN (cGAN) to generate synthetic images for minority classes
2. Comparing classifier performance with and without synthetic data augmentation
3. Providing detailed analysis of the impact on per-class metrics

## Dataset Information

### CIFAR-10 (Recommended)
- **Size**: 60,000 32x32 color images (50,000 train, 10,000 test)
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Download**: Automatically downloaded by PyTorch
- **Imbalance Setup**: We artificially reduce minority classes (bird, cat, deer) to 10% of original samples

### Alternative Datasets
- **CIFAR-100**: 100 classes, more challenging
- **Fashion-MNIST**: 10 classes of fashion items, grayscale
- **Custom datasets**: Modify data loading in `utils/data_utils.py`

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Conditional GAN

```bash
python train_gan.py
```

This will:
- Create an imbalanced version of CIFAR-10
- Train a conditional GAN for 200 epochs
- Save checkpoints and sample images
- Generate training curves

**Configuration** (in `train_gan.py`):
- `minority_classes`: [2, 3, 4] (bird, cat, deer)
- `minority_ratio`: 0.1 (10% of samples)
- `num_epochs`: 200
- `batch_size`: 64

### 2. Train Classifiers

```bash
python train_classifier.py
```

This will:
- Train a baseline classifier (no augmentation)
- Train an augmented classifier (with synthetic data)
- Compare performance metrics
- Generate confusion matrices and training curves

**Configuration** (in `train_classifier.py`):
- `synthetic_samples_per_class`: 4000 (for minority classes)
- `num_epochs`: 100
- `batch_size`: 128

### 3. Launch Web Interface

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

**Features**:
- Generate synthetic images for any class
- Classify uploaded images with both models
- View detailed performance comparisons
- Interactive charts and visualizations

## Project Structure

```
12/
├── models/
│   ├── cgan.py              # Conditional GAN architecture
│   └── classifier.py        # ResNet-style classifier
├── utils/
│   ├── data_utils.py        # Dataset utilities
│   └── synthetic_dataset.py # Synthetic data loader
├── templates/
│   └── index.html           # Web interface HTML
├── static/
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic
├── checkpoints/             # Saved models (created during training)
├── samples/                 # Generated images (created during training)
├── data/                    # CIFAR-10 data (auto-downloaded)
├── train_gan.py             # GAN training script
├── train_classifier.py      # Classifier training script
├── app.py                   # Flask web server
└── requirements.txt         # Dependencies
```

## Expected Results

Based on similar studies, you should expect:
- **Baseline Accuracy**: 70-75% (due to imbalance)
- **Augmented Accuracy**: 75-80% (improvement from synthetic data)
- **Minority Class Improvement**: 5-15% accuracy increase
- **F1-Score Improvement**: 0.05-0.10

## Evaluation Metrics

The project evaluates:
1. **Overall Accuracy**: Total correct predictions
2. **Weighted F1-Score**: Accounts for class imbalance
3. **Per-Class Accuracy**: Individual class performance
4. **Confusion Matrix**: Detailed error analysis

## Key Findings to Report

Your research should address:
1. Does synthetic data improve minority class performance?
2. What is the quality/diversity of generated images?
3. Are there cases where synthetic data hurts performance?
4. How does the amount of synthetic data affect results?

## Customization

### Change Minority Classes
Edit in `train_gan.py` and `train_classifier.py`:
```python
'minority_classes': [0, 1, 2],  # Different classes
'minority_ratio': 0.05,          # More severe imbalance
```

### Adjust Synthetic Data Amount
Edit in `train_classifier.py`:
```python
'synthetic_samples_per_class': 2000,  # Less synthetic data
```

### Use Different Dataset
Modify `utils/data_utils.py` to load your custom dataset.

## Troubleshooting

**CUDA Out of Memory**:
- Reduce `batch_size` in training scripts
- Reduce `synthetic_samples_per_class`

**Poor GAN Quality**:
- Train for more epochs (300-500)
- Adjust learning rate
- Try different architectures

**No Improvement from Augmentation**:
- Check GAN sample quality
- Increase synthetic samples
- Verify data preprocessing

## Citation

If you use this code for your research, please cite:
```
@misc{gan-data-augmentation-2025,
  title={GAN-Based Data Augmentation for Imbalanced Classification},
  year={2025},
  author={Your Name}
}
```

## License

MIT License
