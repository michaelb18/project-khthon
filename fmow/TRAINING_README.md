# Training Script for fMoW-Sentinel Classifier

This script trains a classifier on the fMoW-Sentinel dataset using foundation models.

## Features

- Uses `satlaspretrain_models` foundation model (with fallback to `timm` ViT or torchvision ResNet)
- Tracks comprehensive metrics: accuracy, F1, precision, recall
- Saves loss curves (without displaying)
- Creates ROC and Precision-Recall curves using sklearn
- Handles 13-channel Sentinel-2 images
- Automatically infills null/black pixels

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Note: If `satlaspretrain_models` is not available, the script will automatically fall back to `timm` (Vision Transformer) or torchvision ResNet.

## Usage

### Basic Usage

1. **Create or edit the configuration file** (`config.yaml`):

```yaml
root_dir: /home/michael/project_khthon/fmow
output_dir: ./outputs
batch_size: 32
epochs: 20
lr: 0.0001
num_workers: 4
model_name: Sentinel2_SwinT_SI_MS
infill_nulls: true
image_size: 224
categories: null  # Set to null for all categories, or provide a list
```

2. **Run the training script**:

```bash
# Use default config.yaml
python train_classifier.py

# Or specify a custom config file
python train_classifier.py my_config.yaml
```

### Configuration Parameters

- `root_dir`: Root directory containing the fmow-sentinel folder and CSV files
- `output_dir`: Output directory for models and results
- `batch_size`: Batch size for training
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `num_workers`: Number of data loading workers
- `model_name`: Foundation model name (e.g., 'Sentinel2_SwinT_SI_MS' for satlaspretrain_models)
- `infill_nulls`: Enable null pixel infilling (true/false)
- `image_size`: Target image size for resizing
- `categories`: List of category names to include, or `null` to use all categories

## Output Files

The script creates the following files in the output directory:

- `best_model.pth`: Best model checkpoint (based on validation accuracy)
- `metrics.json`: Final metrics (accuracy, F1, precision, recall)
- `classification_report.txt`: Detailed classification report per class
- `loss_curves.png`: Training and validation loss curves
- `accuracy_curves.png`: Training and validation accuracy curves
- `f1_curves.png`: Training and validation F1 score curves
- `all_metrics_curves.png`: Combined metrics visualization
- `roc_curves.png`: ROC curves for all classes
- `pr_curves.png`: Precision-Recall curves for all classes
- `roc_auc.json`: ROC AUC scores per class
- `pr_auc.json`: PR AUC scores per class

## Metrics

The script tracks and reports:

- **Accuracy**: Overall classification accuracy
- **F1 Score**: Weighted F1 score across all classes
- **Precision**: Weighted precision across all classes
- **Recall**: Weighted recall across all classes
- **ROC AUC**: Area under ROC curve (per class and micro-average)
- **PR AUC**: Average precision (per class and micro-average)

## Model Architecture

The script uses a foundation model backbone with a custom classifier head:

1. **Primary**: `satlaspretrain_models` (if available)
2. **Fallback 1**: `timm` Vision Transformer (ViT-Base)
3. **Fallback 2**: torchvision ResNet-50

All models are modified to accept 13-channel Sentinel-2 input.

## Notes

- The script automatically handles null pixel infilling by default
- Loss curves are saved but not displayed (non-interactive matplotlib backend)
- ROC and PR curves use One-vs-Rest strategy for multi-class classification
- The best model is saved based on validation accuracy
- Learning rate scheduling uses ReduceLROnPlateau

