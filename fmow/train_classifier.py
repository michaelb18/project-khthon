"""
Training script for fMoW-Sentinel classifier using foundation models.

This script:
- Uses satlaspretrain_models foundation model (or fallback to timm)
- Trains a classifier on the fMoW-Sentinel dataset
- Tracks metrics: accuracy, F1, recall, precision
- Saves loss curves (without displaying)
- Creates ROC and PR curves using sklearn
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans

import torch.nn as nn
from dataloader import FMoWSentinelDataset, create_dataloader

class Sampling(nn.Module):
    def forward(self, z_mean, z_log_var):
        # get the shape of the tensor for the mean and log variance
        batch, dim = z_mean.shape
        # generate a normal random tensor (epsilon) with the same shape as z_mean
        # this tensor will be used for reparameterization trick
        epsilon = Normal(0, 1).sample((batch, dim)).to(z_mean.device)
        # apply the reparameterization trick to generate the samples in the
        # latent space
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.model = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        self.mean = nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        self.var = nn.Linear(self.model.fc.in_features, self.model.fc.in_features)
        self.cls = nn.Linear(self.model.fc.in_features, num_classes)
        self.sample = Sampling()

    def forward(self, x):
        representations = self.model(x)
        z_mean = self.mean(representations)
        z_log_var = self.var(representations)
        z = self.sample(z_mean, z_log_var)
        x = self.cls(z)

        return z_mean, z_log_var, x
    
    @torch.no_grad()
    def predict(self, x, n_samples=100):
        # 1. Base representations from ResNet
        reps = self.model(x) 
        batch_size = x.size(0)
        
        # 2. Expand representations to simulate n_samples in parallel
        # Shape: [n_samples * batch, features]
        reps_expanded = reps.repeat(n_samples, 1)

        # 3. Stochastic Forward Pass
        z_mean = self.mean(reps_expanded)
        z_log_var = self.var(reps_expanded)
        z = self.sample(z_mean, z_log_var)
        logits = self.cls(z) # [n_samples * batch, num_classes]

        # 4. Reshape and get predictions per sample
        # Shape: [n_samples, batch, num_classes]
        logits = logits.view(n_samples, batch_size, -1)
        sample_preds = torch.argmax(logits, dim=-1) # [n_samples, batch]

        # 5. Calculate the frequency "Buffer" for every class
        # We use one_hot to turn indices into a countable grid
        # Shape: [n_samples, batch, num_classes]
        one_hot_preds = torch.nn.functional.one_hot(sample_preds, num_classes=self.cls.out_features)
        
        # Sum across the sample dimension and convert to percentage
        # Shape: [batch, num_classes]
        class_counts = one_hot_preds.sum(dim=0).float()
        class_probs_buffer = (class_counts / n_samples)

        return class_probs_buffer

def get_foundation_model(num_classes: int, model_name: str = 'Sentinel2_SwinT_SI_MS', pretrained: bool = True, in_channels: int = 3):
    """
    Get foundation model from satlaspretrain_models or fallback to timm.
    
    Args:
        num_classes: Number of output classes
        model_name: Model name for satlaspretrain_models (e.g., 'Sentinel2_SwinT_SI_MS')
                   or timm model name for fallback
        pretrained: Whether to use pretrained weights
        in_channels: Number of input channels (13 for Sentinel-2)
    
    Returns:
        PyTorch model
    """
    import torch
    import torchvision.models as models
    #model = models.resnet152(weights = models.ResNet152_Weights.IMAGENET1K_V2)
    #model.fc = nn.Linear(model.fc.in_features, num_classes)
    #model.to('cuda')
    model = ResNetClassifier(num_classes)
    return model


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_f1s = []
        self.val_f1s = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
    
    def update(self, split: str, loss: float, acc: float, f1: float, precision: float, recall: float):
        """Update metrics for a split."""
        if split == 'train':
            self.train_losses.append(loss)
            self.train_accs.append(acc)
            self.train_f1s.append(f1)
            self.train_precisions.append(precision)
            self.train_recalls.append(recall)
        else:
            self.val_losses.append(loss)
            self.val_accs.append(acc)
            self.val_f1s.append(f1)
            self.val_precisions.append(precision)
            self.val_recalls.append(recall)
    
    def save_curves(self, output_dir: Path):
        """Save loss curves without displaying."""
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_accs, 'b-', label='Train Accuracy', linewidth=2)
        if self.val_accs:
            plt.plot(epochs, self.val_accs, 'r-', label='Val Accuracy', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Training and Validation Accuracy', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # F1 curve
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_f1s, 'b-', label='Train F1', linewidth=2)
        if self.val_f1s:
            plt.plot(epochs, self.val_f1s, 'r-', label='Val F1', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Training and Validation F1 Score', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'f1_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Combined metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        if self.val_losses:
            axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        if self.val_accs:
            axes[0, 1].plot(epochs, self.val_accs, 'r-', label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, self.train_precisions, 'b-', label='Train', linewidth=2)
        if self.val_precisions:
            axes[1, 0].plot(epochs, self.val_precisions, 'r-', label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, self.train_recalls, 'b-', label='Train', linewidth=2)
        if self.val_recalls:
            axes[1, 1].plot(epochs, self.val_recalls, 'r-', label='Val', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'all_metrics_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def kld(mu, logvar):
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)).mean()

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    import time
    
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Timing accumulators
    wait_for_data_times = []  # Time waiting for next batch (data loading bottleneck)
    transfer_times = []
    forward_times = []
    loss_times = []
    backward_times = []
    step_times = []
    batch_times = []  # Total time from batch start to batch end
    iteration_times = []  # Time from one batch start to next batch start (what tqdm measures)
    
    epoch_start = time.time()
    iter_start = time.time()  # Start of iteration (includes data loading wait)
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        # Measure time waiting for this batch (data loading time)
        batch_ready_time = time.time()
        wait_time = batch_ready_time - iter_start
        wait_for_data_times.append(wait_time)
        
        batch_start = time.time()
        
        # Transfer to device
        transfer_start = time.time()
        images = images.to(device)
        labels = labels.to(device)
        transfer_time = time.time() - transfer_start
        transfer_times.append(transfer_time)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        forward_start = time.time()
        mean, var, outputs = model(images)
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)

        # Loss calculation
        loss_start = time.time()
        loss = criterion(outputs, labels) + 0.1 * kld(mean, var)
        loss_time = time.time() - loss_start
        loss_times.append(loss_time)

        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        backward_times.append(backward_time)
        
        # Optimizer step
        step_start = time.time()
        optimizer.step()
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Get predictions
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        batch_time = time.time() - batch_start  # Pure compute time
        batch_times.append(batch_time)
        
        iter_end = time.time()
        iteration_time = iter_end - iter_start  # Total iteration time (what tqdm sees)
        iteration_times.append(iteration_time)
        
        # Print detailed timing for first few batches
        if batch_idx < 5:
            print(f"\n[Batch {batch_idx}] Timing breakdown:")
            print(f"  Wait for data: {wait_time:.4f}s ({100*wait_time/iteration_time:.1f}%)")
            print(f"  Transfer: {transfer_time:.4f}s ({100*transfer_time/iteration_time:.1f}%)")
            print(f"  Forward: {forward_time:.4f}s ({100*forward_time/iteration_time:.1f}%)")
            print(f"  Loss calc: {loss_time:.4f}s ({100*loss_time/iteration_time:.1f}%)")
            print(f"  Backward: {backward_time:.4f}s ({100*backward_time/iteration_time:.1f}%)")
            print(f"  Step: {step_time:.4f}s ({100*step_time/iteration_time:.1f}%)")
            print(f"  Pure compute: {batch_time:.4f}s")
            print(f"  Total iteration: {iteration_time:.4f}s (matches tqdm)")
        
        iter_start = time.time()  # Start timing next iteration
    
    epoch_time = time.time() - epoch_start
    
    # Print summary statistics
    print(f"\n[Epoch Timing Summary]")
    print(f"  Total epoch time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
    print(f"  Number of batches: {len(batch_times)}")
    print(f"  Average iteration time (tqdm): {np.mean(iteration_times):.4f}s")
    print(f"  Average wait for data: {np.mean(wait_for_data_times):.4f}s ({100*np.mean(wait_for_data_times)/np.mean(iteration_times):.1f}%)")
    print(f"  Average pure compute time: {np.mean(batch_times):.4f}s ({100*np.mean(batch_times)/np.mean(iteration_times):.1f}%)")
    print(f"  Average transfer time: {np.mean(transfer_times):.4f}s")
    print(f"  Average forward time: {np.mean(forward_times):.4f}s")
    print(f"  Average backward time: {np.mean(backward_times):.4f}s")
    print(f"  Average step time: {np.mean(step_times):.4f}s")
    print(f"\n  BOTTLENECK ANALYSIS:")
    if np.mean(wait_for_data_times) > np.mean(batch_times):
        print(f"  ⚠️  DATA LOADING is the bottleneck!")
        print(f"     Data loading takes {np.mean(wait_for_data_times)/np.mean(batch_times):.1f}x longer than compute")
        print(f"     Consider: increasing num_workers, disabling null infilling, or using faster storage")
    else:
        print(f"  ✓ Compute is the bottleneck (normal for GPU training)")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(all_preds), np.array(all_labels)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    import time
    
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    data_load_times = []
    batch_times = []
    val_start = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Validating")):
            batch_start = time.time()
            
            if batch_idx > 0:
                data_load_times.append(batch_start - prev_batch_end)
            
            images = images.to(device)
            labels = labels.to(device)
            
            mean, var, outputs = model(images)
            loss = criterion(outputs, labels) + 0.1 * kld(mean, var)
            
            running_loss += loss.item() * images.size(0)
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            prev_batch_end = time.time()
    
    val_time = time.time() - val_start
    
    if len(data_load_times) > 0:
        print(f"\n[Validation Timing]")
        print(f"  Total time: {val_time:.2f}s")
        print(f"  Avg batch time: {np.mean(batch_times):.4f}s")
        print(f"  Avg data load time: {np.mean(data_load_times):.4f}s")
        print(f"  Data load %: {100*np.mean(data_load_times)/np.mean(batch_times):.1f}%")
    
    epoch_loss = running_loss / len(dataloader.dataset)
    all_probs = np.concatenate(all_probs, axis=0)
    
    return epoch_loss, np.array(all_preds), np.array(all_labels), all_probs


def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def plot_roc_curves(y_true, y_probs, class_names, output_path: Path):
    """Plot ROC curves for all classes."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute ROC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Micro-average ROC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot each class
    for i in range(min(n_classes, 20)):  # Limit to 20 classes for readability
        plt.plot(fpr[i], tpr[i], lw=2, alpha=0.7,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro-average
    plt.plot(fpr["micro"], tpr["micro"], color='black', linestyle='--', lw=2,
            label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
    
    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k:', lw=1, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves (One-vs-Rest)', fontsize=14)
    plt.legend(loc="lower right", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc

def plot_confusion_matrix(y_true, y_pred, class_names, output_path: Path):
    """Plot confusion matrix."""
    y_pred = np.argmax(y_pred, axis=1)
    cm = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names)
    plt.figure(figsize=(10, 8))
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return cm

def plot_pr_curves(y_true, y_probs, class_names, output_path: Path):
    """Plot Precision-Recall curves for all classes."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute PR for each class
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
    
    # Micro-average PR
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_probs.ravel())
    pr_auc["micro"] = average_precision_score(y_true_bin, y_probs, average="micro")
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot each class
    for i in range(min(n_classes, 20)):  # Limit to 20 classes for readability
        plt.plot(recall[i], precision[i], lw=2, alpha=0.7,
                label=f'{class_names[i]} (AP = {pr_auc[i]:.3f})')
    
    # Plot micro-average
    plt.plot(recall["micro"], precision["micro"], color='black', linestyle='--', lw=2,
            label=f'Micro-avg (AP = {pr_auc["micro"]:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(loc="lower left", fontsize=9, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return pr_auc


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_anchors(dataloader, n_anchors=8):
    """
    Use KMeans to find canonical bounding boxes for
    each class in dataloader.

    Args:
        dataloader: A pytorch dataloader containing a training set
        n_anchors: the number of anchors per class

    Returns:
        dict: A dictionary where keys are class names and values are
              np.ndarrays containing anchor bounding box sizes.
    """

    from copy import deepcopy
    # Store transform so we can process images without resizing for now but put it back in later
    transform = deepcopy(dataloader.dataset.transform)

    dataloader.dataset.transform = None
    classes = {cl: [] for cl in dataloader.dataset.categories}

    # Iterate through the raw dataset to collect image dimensions for each class
    for idx in tqdm(range(len(dataloader.dataset)), desc="Collecting image dimensions"):
        image, label = dataloader.dataset[idx]
        image = image.numpy()
        label = label.item()
        # Append height and width
        classes[dataloader.dataset.get_category_name(label)].append([image.shape[1], image.shape[2]])

    anchors = {}
    # Perform K-Means clustering on the collected dimensions to find representative anchor boxes
    for cl, sizes in classes.items():
        kmeans = KMeans(n_clusters=n_anchors, random_state=42)
        sizes = np.array(sizes)

        if sizes.size > 0:
            kmeans.fit(sizes)
            anchors[cl] = kmeans.cluster_centers_
        else:
            raise ValueError(f'Class {cl} is missing from the dataset')

    dataloader.dataset.transform = transform
    return anchors

def main():
    import sys
    
    # Check if config file is provided
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default config file
        config_path = 'config.yaml'
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract parameters with defaults
    args = type('Args', (), {
        'root_dir': config.get('root_dir', '/home/michael/project_khthon/fmow'),
        'output_dir': config.get('output_dir', './outputs'),
        'batch_size': config.get('batch_size', 32),
        'epochs': config.get('epochs', 20),
        'lr': config.get('lr', 1e-4),
        'num_workers': config.get('num_workers', 4),
        'model_name': config.get('model_name', 'Sentinel2_SwinT_SI_MS'),
        'infill_nulls': config.get('infill_nulls', True),
        'image_size': config.get('image_size', 224),
        'categories': config.get('categories', None),
        'max_samples_per_category': config.get('max_samples_per_category', None)
    })()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Loading datasets...")
    print(f"Resizing all images to {args.image_size}x{args.image_size} for batching")
    
    if args.categories:
        print(f"Filtering to categories: {args.categories}")
        print(f"Number of specified categories: {len(args.categories)}")
    else:
        print("Using all available categories")
    
    if args.max_samples_per_category:
        print(f"Limiting to {args.max_samples_per_category} samples per category")
    
    train_loader = create_dataloader(
        root_dir=args.root_dir,
        split='train',
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        infill_nulls=args.infill_nulls,
        image_size=args.image_size,
        categories=args.categories,
        max_samples_per_category=args.max_samples_per_category
    )
    
    val_loader = create_dataloader(
        root_dir=args.root_dir,
        split='val',
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        infill_nulls=args.infill_nulls,
        image_size=args.image_size,
        categories=args.categories,
        max_samples_per_category=args.max_samples_per_category
    )
    
    num_classes = train_loader.dataset.get_num_classes()
    class_names = train_loader.dataset.categories
    
    if args.categories:
        print(f"Final categories after filtering: {class_names}")
    
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {args.model_name}")
    # Images are RGB (3 channels: B04, B03, B02)
    model = get_foundation_model(num_classes, model_name=args.model_name, 
                                 pretrained=True, in_channels=3)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Metrics tracker
    metrics = MetricsTracker()
    
    # Training loop
    best_val_acc = 0.0
    best_model_state = None
    
    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_metrics = compute_metrics(train_labels, train_preds)
        
        # Validate
        val_loss, val_preds, val_labels, val_probs = validate(
            model, val_loader, criterion, device
        )
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Update metrics tracker
        metrics.update('train', train_loss, train_metrics['accuracy'],
                      train_metrics['f1'], train_metrics['precision'], train_metrics['recall'])
        metrics.update('val', val_loss, val_metrics['accuracy'],
                      val_metrics['f1'], val_metrics['precision'], val_metrics['recall'])
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, Prec: {train_metrics['precision']:.4f}, "
              f"Rec: {train_metrics['recall']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Prec: {val_metrics['precision']:.4f}, "
              f"Rec: {val_metrics['recall']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, output_dir / 'best_model.pth')
            print(f"Saved best model (Val Acc: {best_val_acc:.4f})")
    
    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation on validation set
    print("\nFinal evaluation on validation set...")
    _, final_preds, final_labels, final_probs = validate(model, val_loader, criterion, device)
    final_metrics = compute_metrics(final_labels, final_preds, final_probs)
    
    print("\nFinal Metrics:")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1 Score: {final_metrics['f1']:.4f}")
    print(f"Precision: {final_metrics['precision']:.4f}")
    print(f"Recall: {final_metrics['recall']:.4f}")
    
    # Save metrics
    metrics_dict = {
        'final_accuracy': float(final_metrics['accuracy']),
        'final_f1': float(final_metrics['f1']),
        'final_precision': float(final_metrics['precision']),
        'final_recall': float(final_metrics['recall']),
        'best_val_accuracy': float(best_val_acc)
    }
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Save classification report
    report = classification_report(final_labels, final_preds, target_names=class_names)
    with open(output_dir / 'classification_report.txt', 'w') as f:
        f.write(report)
    print("\nClassification Report:")
    print(report)
    
    # Save loss curves (without displaying)
    print("\nSaving loss curves...")
    metrics.save_curves(output_dir)
    
    # Create ROC curves
    print("Creating ROC curves...")
    roc_auc = plot_roc_curves(final_labels, final_probs, class_names,
                              output_dir / 'roc_curves.png')
    print(f"Micro-average ROC AUC: {roc_auc['micro']:.4f}")
    
    # Create PR curves
    print("Creating PR curves...")
    pr_auc = plot_pr_curves(final_labels, final_probs, class_names,
                           output_dir / 'pr_curves.png')
    print(f"Micro-average PR AUC: {pr_auc['micro']:.4f}")

    # Create Confusion Matrix
    print("Creating Confusion Matrix...")
    cm = plot_confusion_matrix(final_labels, final_probs, class_names,
                           output_dir / 'confusion_matrix.png')
    
    # Save ROC and PR AUC scores
    roc_auc_dict = {f'class_{i}': float(roc_auc[i]) for i in range(num_classes)}
    roc_auc_dict['micro'] = float(roc_auc['micro'])
    pr_auc_dict = {f'class_{i}': float(pr_auc[i]) for i in range(num_classes)}
    pr_auc_dict['micro'] = float(pr_auc['micro'])
    
    with open(output_dir / 'roc_auc.json', 'w') as f:
        json.dump(roc_auc_dict, f, indent=2)
    with open(output_dir / 'pr_auc.json', 'w') as f:
        json.dump(pr_auc_dict, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")
    print("Done!")


def create_default_config(output_path: str = 'config.yaml'):
    """Create a default configuration file."""
    default_config = {
        'root_dir': '/home/michael/project_khthon/fmow',
        'output_dir': './outputs',
        'batch_size': 32,
        'epochs': 20,
        'lr': 1e-4,
        'num_workers': 4,
        'model_name': 'Sentinel2_SwinT_SI_MS',
        'infill_nulls': True,
        'image_size': 224,
        'categories': None  # Set to None to use all categories, or provide a list like: ['airport', 'hospital', 'stadium']
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Default configuration saved to {output_path}")


if __name__ == '__main__':
    main()

