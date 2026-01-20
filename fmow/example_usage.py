"""
Example usage of the fMoW-Sentinel dataloader.

Before running, install dependencies:
    pip install -r requirements.txt
"""

from dataloader import FMoWSentinelDataset, create_dataloader, get_default_transform
import torch


def main():
    # Set root directory
    root_dir = '/home/michael/project_khthon/fmow'
    
    # Option 1: Create dataset directly
    print("=" * 60)
    print("Option 1: Using FMoWSentinelDataset directly")
    print("=" * 60)
    
    train_dataset = FMoWSentinelDataset(
        root_dir=root_dir,
        split='train',
        transform=None,  # Or use get_default_transform(mode='train')
        return_metadata=False
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Sample categories: {train_dataset.categories[:5]}")
    
    # Get a single sample
    image, label = train_dataset[0]
    print(f"\nSample image shape: {image.shape}")  # (13, H, W)
    print(f"Sample label: {label.item()} -> {train_dataset.get_category_name(label.item())}")
    print(f"Image dtype: {image.dtype}, Image range: [{image.min():.2f}, {image.max():.2f}]")
    
    # Option 2: Create DataLoader
    print("\n" + "=" * 60)
    print("Option 2: Using DataLoader")
    print("=" * 60)
    
    train_transform = get_default_transform(mode='train', normalize=False)
    train_loader = create_dataloader(
        root_dir=root_dir,
        split='train',
        batch_size=4,
        shuffle=True,
        num_workers=2,
        transform=train_transform,
        return_metadata=False
    )
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch images shape: {images.shape}")  # (batch_size, 13, H, W)
    print(f"Batch labels shape: {labels.shape}")  # (batch_size,)
    
    # Option 3: Create all splits
    print("\n" + "=" * 60)
    print("Option 3: Create all splits")
    print("=" * 60)
    
    val_loader = create_dataloader(
        root_dir=root_dir,
        split='val',
        batch_size=32,
        shuffle=False,
        transform=get_default_transform(mode='val', normalize=False)
    )
    
    test_loader = create_dataloader(
        root_dir=root_dir,
        split='test_gt',
        batch_size=32,
        shuffle=False,
        transform=get_default_transform(mode='val', normalize=False)
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Option 4: With metadata
    print("\n" + "=" * 60)
    print("Option 4: With metadata")
    print("=" * 60)
    
    dataset_with_meta = FMoWSentinelDataset(
        root_dir=root_dir,
        split='train',
        return_metadata=True
    )
    
    image, label, metadata = dataset_with_meta[0]
    print(f"Metadata keys: {metadata.keys()}")
    print(f"Sample metadata: category={metadata['category']}, "
          f"location_id={metadata['location_id']}, "
          f"image_id={metadata['image_id']}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

