import os
from tkinter import N
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import rasterio
from typing import Optional, Callable, Tuple, Union, List
from pathlib import Path
from scipy import ndimage
from scipy.interpolate import griddata
import time

class PercentScaler(object):
    def __init__(self, low=2, high=98):
        self.low = low
        self.high = high

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img_np = img.numpy()
        else:
            img_np = img

        min_val, max_val = np.percentile(img_np, (self.low, self.high))

        scaled_band = np.clip(img_np, min_val, max_val)

        if max_val - min_val == 0:
            return torch.from_numpy(scaled_band).float()

        normalized = (scaled_band - min_val) / (max_val - min_val)

        return torch.from_numpy(normalized).float()


class AppendSentinel2Indices:
    """
    Compute Sentinel-2 spectral indices and append as extra channels.
    Formulas from: https://clearsky.vision/knowledge/sentinel2-indices-cheatsheet
    
    Expects input shape (12, H, W) with band order: B1, B2, B3, B4, B5, B6, B7, B8A, B8B, B9, B11, B12.
    B8A (channel 7) is used where formulas reference B8 (NIR).
    Output shape: (18, H, W) = 12 bands + 6 indices [NBR, NFDI, IBI, MSI, RECI, NDVI].
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        # Channels: 0=B1, 1=B2, 2=B3, 3=B4, 4=B5, 5=B6, 6=B7, 7=B8A, 8=B8B, 9=B9, 10=B11, 11=B12
        b2, b3, b4, b5 = img[1], img[2], img[3], img[4]
        b8a, b11, b12 = img[7], img[10], img[11]
        eps = self.eps

        # NDVI = (B8 - B4) / (B8 + B4)
        ndvi = (b8a - b4) / (b8a + b4 + eps)

        # NBR (Normalized Burn Ratio) = (B8 - B12) / (B8 + B12)
        nbr = (b8a - b12) / (b8a + b12 + eps)

        # NFDI = (B3 - B11) / (B3 + B11)  (Green-SWIR, same as MNDWI/NDSI formula)
        nfdi = (b3 - b11) / (b3 + b11 + eps)

        # NDBI = (B11 - B8) / (B11 + B8) for IBI
        ndbi = (b11 - b8a) / (b11 + b8a + eps)
        # IBI (Index-Based Built-up) = (NDBI - NDVI) / (NDBI + NDVI)
        ibi = (ndbi - ndvi) / (ndbi + ndvi + eps)

        # MSI (Moisture Stress Index) = B11 / B8
        msi = b11 / (b8a + eps)

        # RECI (Red Edge Chlorophyll Index) = (B8 - B4) / (B5 - B4)
        reci = (b8a - b4) / (b5 - b4 + eps)

        # Clamp indices to reasonable range to avoid extreme values from noise
        def clamp_idx(x):
            return torch.clamp(x, -2.0, 2.0)
        ndvi = clamp_idx(ndvi)
        nbr = clamp_idx(nbr)
        nfdi = clamp_idx(nfdi)
        ibi = clamp_idx(ibi)
        msi = torch.clamp(msi, 0.0, 10.0)
        reci = clamp_idx(reci)

        indices = torch.stack([nbr, nfdi, ibi, msi, reci, ndvi], dim=0)
        return torch.cat([img, indices], dim=0)  # (18, H, W)


class ResizeMultiBand:
    """
    Resize transform for multi-band images (C, H, W).
    Uses bilinear interpolation to resize all bands.
    """
    def __init__(self, size: Union[int, Tuple[int, int]]):
        """
        Args:
            size: Target size. If int, creates square (size, size).
                  If tuple (H, W), uses those dimensions.
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (C, H, W)
        
        Returns:
            Resized tensor of shape (C, self.size[0], self.size[1])
        """
        start = time.time()
        # Add batch dimension for interpolation: (1, C, H, W)
        img = img.unsqueeze(0)
        # Interpolate: bilinear for continuous values
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        # Remove batch dimension: (C, H, W)
        img = img.squeeze(0)
        end = time.time()
        return img


class FMoWSentinelDataset(Dataset):
    """
    PyTorch Dataset for fMoW-Sentinel dataset.
    
    Outputs 12 Sentinel-2 bands (excluding B10 cirrus):
    B1, B2, B3, B4, B5, B6, B7, B8A, B8B, B9, B11, B12
    Original order: B1, B2, B3, B4, B5, B6, B7, B8A, B8B, B9, B10, B11, B12 (indices 0-12)
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_metadata: bool = False,
        infill_nulls: bool = True,
        null_threshold: float = 0.01,
        categories: Optional[List[str]] = None,
        max_samples_per_category: Optional[int] = None
    ):
        """
        Args:
            root_dir: Root directory containing the fmow-sentinel folder and CSV files
            split: One of 'train', 'val', or 'test_gt'
            transform: Optional transform to be applied on the image
            target_transform: Optional transform to be applied on the category label
            return_metadata: If True, returns additional metadata (location_id, image_id, timestamp, polygon)
            infill_nulls: If True, infill null/black pixels using spatial interpolation
            null_threshold: Threshold below which pixels are considered null (relative to max value)
            categories: Optional list of category names to include. If None, includes all categories.
            max_samples_per_category: Optional maximum number of samples per category. If None, uses all samples.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.return_metadata = return_metadata
        self.infill_nulls = infill_nulls
        self.null_threshold = null_threshold
        self.max_samples_per_category = max_samples_per_category
        
        # Load metadata CSV
        csv_file = self.root_dir / f'{split}.csv'
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        self.metadata = pd.read_csv(csv_file)
        
        # Filter by categories if specified
        if categories is not None:
            # Validate that all specified categories exist
            available_categories = set(self.metadata['category'].unique())
            specified_categories = set(categories)
            missing_categories = specified_categories - available_categories
            
            if missing_categories:
                raise ValueError(
                    f"Categories not found in dataset: {missing_categories}. "
                    f"Available categories: {sorted(available_categories)}"
                )
            
            # Filter metadata to only include specified categories
            self.metadata = self.metadata[self.metadata['category'].isin(categories)].copy()
            
            if len(self.metadata) == 0:
                raise ValueError(f"No samples found for specified categories: {categories}")
            
            # Use specified categories in order
            self.categories = [cat for cat in categories if cat in available_categories]
        else:
            # Get unique categories and create label mapping
            self.categories = sorted(self.metadata['category'].unique())
        
        # Limit samples per category if specified
        if max_samples_per_category is not None:
            limited_samples = []
            for category in self.categories:
                category_samples = self.metadata[self.metadata['category'] == category]
                if len(category_samples) > max_samples_per_category:
                    # Randomly sample if more samples than limit
                    category_samples = category_samples.sample(
                        n=max_samples_per_category,
                        random_state=42  # For reproducibility
                    )
                limited_samples.append(category_samples)
            
            # Combine all limited samples
            self.metadata = pd.concat(limited_samples, ignore_index=True)
            # Shuffle the combined dataframe
            self.metadata = self.metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Create label mapping based on filtered categories
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for cat, idx in self.category_to_idx.items()}
        
        # Filter out rows where image file doesn't exist
        self.valid_indices = []
        for idx, row in self.metadata.iterrows():
            image_path = self._get_image_path(row)
            if image_path.exists():
                self.valid_indices.append(idx)
        
        # Filter metadata to only valid rows
        self.metadata = self.metadata.loc[self.valid_indices].reset_index(drop=True)
        
        # Limit samples per category if specified (after filtering out missing files)
        if max_samples_per_category is not None:
            limited_samples = []
            for category in self.categories:
                category_samples = self.metadata[self.metadata['category'] == category]
                if len(category_samples) > max_samples_per_category:
                    # Randomly sample if more samples than limit
                    category_samples = category_samples.sample(
                        n=max_samples_per_category,
                        random_state=42  # For reproducibility
                    )
                limited_samples.append(category_samples)
            
            # Combine all limited samples
            self.metadata = pd.concat(limited_samples, ignore_index=True)
            # Shuffle the combined dataframe
            self.metadata = self.metadata.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Loaded {split} split: {len(self.metadata)} samples, {len(self.categories)} categories")
        if max_samples_per_category:
            print(f"Limited to {max_samples_per_category} samples per category")
    
    def _get_image_path(self, row: pd.Series) -> Path:
        """Construct image path from metadata row."""
        category = row['category']
        location_id = row['location_id']
        image_id = row['image_id']
        
        # Path format: <split>/<category>/<category>_<location_id>/<category>_<image_id>_<location_id>.tif
        image_path = (
            self.root_dir / 'fmow-sentinel' / self.split / 
            category / f'{category}_{location_id}' / 
            f'{category}_{location_id}_{image_id}.tif'
        )
        return image_path
    
    def _detect_null_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Detect null/black pixels in the image.
        
        Args:
            image: Image array of shape (3, H, W) for RGB or (C, H, W) for multi-band
        
        Returns:
            Boolean mask of shape (H, W) where True indicates null pixels
        """
        # Null pixels are typically all zeros or very low values across all bands
        # Check if sum across bands is below threshold
        band_sum = np.sum(image, axis=0)  # Sum across bands: (H, W)
        
        # Normalize threshold by max value in image
        max_val = np.max(image)
        threshold = max_val * self.null_threshold
        
        # Also check for NaN or Inf values
        has_nan = np.any(np.isnan(image), axis=0)
        has_inf = np.any(np.isinf(image), axis=0)
        
        # Null pixels: very low sum OR NaN OR Inf
        null_mask = (band_sum < threshold) | has_nan | has_inf
        
        return null_mask
    
    def _infill_null_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Infill null pixels using spatial interpolation.
        
        Uses a combination of:
        1. Median filtering for small holes
        2. Spatial interpolation (griddata) for larger regions
        
        Args:
            image: Image array of shape (3, H, W) for RGB or (C, H, W) for multi-band with null pixels
        
        Returns:
            Image array with null pixels infilled
        """
        null_mask = self._detect_null_pixels(image)
        
        # If no null pixels, return original
        if not np.any(null_mask):
            return image
        
        # Create output image
        infilled_image = image.copy()
        H, W = image.shape[1], image.shape[2]
        
        # Process each band independently
        for band_idx in range(image.shape[0]):
            band = infilled_image[band_idx, :, :].copy()  # Work on a copy
            band_null_mask = null_mask.copy()
            
            # Skip if no null pixels in this band
            if not np.any(band_null_mask):
                continue
            
            # Method 1: Use median filter for small isolated null regions
            # This handles small holes well
            if np.sum(band_null_mask) < 0.1 * H * W:  # Less than 10% null
                # Apply median filter only to null regions
                median_filtered = ndimage.median_filter(band, size=5)
                band[band_null_mask] = median_filtered[band_null_mask]
            
            # Method 2: Spatial interpolation using griddata for remaining nulls
            # This handles larger regions
            if np.any(band_null_mask):
                # Create coordinate grids
                y_coords, x_coords = np.mgrid[0:H, 0:W]
                
                # Get valid (non-null) pixel coordinates and values
                valid_mask = ~band_null_mask
                valid_coords = np.column_stack([
                    y_coords[valid_mask],
                    x_coords[valid_mask]
                ])
                valid_values = band[valid_mask]
                
                # Get null pixel coordinates
                null_coords = np.column_stack([
                    y_coords[band_null_mask],
                    x_coords[band_null_mask]
                ])
                
                # Only interpolate if we have valid pixels and null pixels
                if len(valid_coords) > 0 and len(null_coords) > 0:
                    # Use linear interpolation (can also use 'cubic' or 'nearest')
                    try:
                        interpolated_values = griddata(
                            valid_coords,
                            valid_values,
                            null_coords,
                            method='linear',
                            fill_value=np.median(valid_values)  # Fallback for extrapolation
                        )
                        # Handle NaN values from interpolation
                        nan_mask = np.isnan(interpolated_values)
                        if np.any(nan_mask):
                            interpolated_values[nan_mask] = np.median(valid_values)
                        band[band_null_mask] = interpolated_values
                    except:
                        # Fallback: use nearest neighbor if linear fails
                        interpolated_values = griddata(
                            valid_coords,
                            valid_values,
                            null_coords,
                            method='nearest'
                        )
                        band[band_null_mask] = interpolated_values
            
            infilled_image[band_idx, :, :] = band
        
        return infilled_image
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            If return_metadata=False: (image, label)
            If return_metadata=True: (image, label, metadata_dict)
        """
        item_start = time.time()
        
        row = self.metadata.iloc[idx]
        
        # Load image
        image_path = self._get_image_path(row)
        
        load_start = time.time()
        try:
            with rasterio.open(image_path) as src:
                # Read all 13 bands
                # Shape: (13, height, width)
                all_bands = src.read()  # Reads all bands
                all_bands = all_bands.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        load_time = time.time() - load_start
        
        # Extract 12 Sentinel-2 bands (excluding B10 cirrus)
        # Bands: B1, B2, B3, B4, B5, B6, B7, B8A, B8B, B9, B11, B12
        # Original indices: B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6, B8A=7, B8B=8, B9=9, B10=10, B11=11, B12=12
        extract_start = time.time()
        band_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]  # Exclude B10 (index 10)
        image = np.stack([all_bands[i, :, :] for i in band_indices], axis=0)  # Shape: (12, H, W)

        extract_time = time.time() - extract_start
        
        # Infill null pixels if enabled (on RGB image)
        infill_time = 0.0
        if self.infill_nulls:
            infill_start = time.time()
            image = self._infill_null_pixels(image)
            infill_time = time.time() - infill_start
        
        # Convert to torch tensor: (12, H, W)
        tensor_start = time.time()
        image = torch.from_numpy(image)
        tensor_time = time.time() - tensor_start
        
        # Get label
        category = row['category']
        label = self.category_to_idx[category]
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms
        transform_time = 0.0
        if self.transform:
            transform_start = time.time()
            image = self.transform(image)
            transform_time = time.time() - transform_start
        
        if self.target_transform:
            label = self.target_transform(label)
        
        total_item_time = time.time() - item_start
        
        # Store timing info (only for first few items to avoid spam)
        if idx < 5:
            print(f"[Dataset {idx}] Load: {load_time:.4f}s, Extract: {extract_time:.4f}s, "
                    f"Infill: {infill_time:.4f}s, Tensor: {tensor_time:.4f}s, "
                    f"Transform: {transform_time:.4f}s, Total: {total_item_time:.4f}s")
        
        if self.return_metadata:
            metadata = {
                'category': category,
                'location_id': row['location_id'],
                'image_id': row['image_id'],
                'timestamp': row['timestamp'],
                'polygon': row['polygon']
            }
            return image, label, metadata
        else:
            return image, label
    
    def get_num_classes(self) -> int:
        """Return the number of unique categories."""
        return len(self.categories)
    
    def get_category_name(self, idx: int) -> str:
        """Get category name from index."""
        return self.idx_to_category[idx]


def get_default_transform(mode: str = 'train', normalize: bool = True, image_size: int = 224) -> transforms.Compose:
    """
    Get default transforms for the dataset.
    
    Args:
        mode: 'train' for training (with augmentation) or 'val'/'test' for validation/testing
        normalize: Whether to normalize the image values
        image_size: Target image size for resizing (default: 224)
    
    Returns:
        Compose transform
    """
    transform_list = []
    
    # Resize all images to the same size for batching
    #transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Lambda(lambda x: x * 0.0001))
    transform_list.append(transforms.Lambda(lambda x: torch.nan_to_num(x, 0.0)))
    # Append 6 spectral indices: NBR, NFDI, IBI, MSI, RECI, NDVI -> 18 channels total
    transform_list.append(AppendSentinel2Indices(eps=1e-6))
    transform_list.append(ResizeMultiBand(size=image_size))
    #transform_list.append(PercentScaler())
    # 18 channels: 12 bands + 6 indices
    #transform_list.append(transforms.Normalize(mean=[0.5] * 18, std=[0.5] * 18))
    
    return transforms.Compose(transform_list) if transform_list else None


def create_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    transform: Optional[Callable] = None,
    return_metadata: bool = False,
    infill_nulls: bool = True,
    null_threshold: float = 0.01,
    image_size: int = 224,
    categories: Optional[List[str]] = None,
    max_samples_per_category: Optional[int] = None
) -> DataLoader:
    """
    Create a DataLoader for the fMoW-Sentinel dataset.
    
    Args:
        root_dir: Root directory containing the fmow-sentinel folder and CSV files
        split: One of 'train', 'val', or 'test_gt'
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data (typically True for train, False for val/test)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        transform: Optional transform to apply to images. If None, uses default with resize.
        return_metadata: If True, returns additional metadata
        infill_nulls: If True, infill null/black pixels using spatial interpolation
        null_threshold: Threshold below which pixels are considered null (relative to max value)
        image_size: Target image size for resizing (used if transform is None)
        categories: Optional list of category names to include. If None, includes all categories.
        max_samples_per_category: Optional maximum number of samples per category. If None, uses all samples.
    
    Returns:
        DataLoader instance
    """
    # If no transform provided, use default with resize
    if transform is None:
        mode = 'train' if split == 'train' else 'val'
        transform = get_default_transform(mode=mode, normalize=False, image_size=image_size)
    
    dataset = FMoWSentinelDataset(
        root_dir=root_dir,
        split=split,
        transform=transform,
        return_metadata=return_metadata,
        infill_nulls=infill_nulls,
        null_threshold=null_threshold,
        categories=categories,
        max_samples_per_category=max_samples_per_category
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Always set to True for faster GPU transfer
    )
    
    return loader


# Example usage
if __name__ == '__main__':
    # Example: Create dataloaders for train, val, and test
    root_dir = '/home/michael/project_khthon/fmow'
    
    # Training dataloader
    train_transform = get_default_transform(mode='train', normalize=False)
    train_loader = create_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=32,
        shuffle=True,
        transform=train_transform
    )
    
    # Validation dataloader
    val_transform = get_default_transform(mode='val', normalize=False)
    val_loader = create_dataloader(
        root_dir=root_dir,
        split='val',
        batch_size=32,
        shuffle=False,
        transform=val_transform
    )
    
    # Test dataloader
    test_loader = create_dataloader(
        root_dir=root_dir,
        split='test_gt',
        batch_size=32,
        shuffle=False,
        transform=val_transform
    )
    
    # Test the dataloader
    print(f"Number of classes: {train_loader.dataset.get_num_classes()}")
    print(f"Categories: {train_loader.dataset.categories[:10]}...")  # Show first 10
    
    # Get a sample batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")  # Should be (batch_size, 13, H, W)
    print(f"Labels shape: {labels.shape}")  # Should be (batch_size,)
    print(f"Image dtype: {images.dtype}, Label dtype: {labels.dtype}")

