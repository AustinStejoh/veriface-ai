"""
Data utilities for VeriFace AI training
Includes data loading and advanced augmentation pipelines
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
import random
import numpy as np
from PIL import Image, ImageFilter
import io
from typing import Tuple, List
import os

from config import TrainingConfig, AugmentationConfig, DEVICE, DATA_DIR

# ─────────────────────────────────────────
#  CUSTOM AUGMENTATION TRANSFORMS
# ─────────────────────────────────────────

class RandomCompression:
    """Apply random JPEG compression to simulate real-world scenarios"""
    def __init__(self, probability: float = 0.3, quality_range: Tuple[int, int] = (40, 100)):
        self.probability = probability
        self.quality_range = quality_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            img = Image.open(buffer)
            img = img.convert("RGB")
        return img


class RandomGaussianNoise:
    """Add Gaussian noise to the image"""
    def __init__(self, probability: float = 0.2, sigma: float = 0.05):
        self.probability = probability
        self.sigma = sigma
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() < self.probability:
            noise = torch.randn_like(tensor) * self.sigma
            tensor = torch.clamp(tensor + noise, 0, 1)
        return tensor


class RandomResizing:
    """Randomly resize then resize back to original size"""
    def __init__(self, probability: float = 0.3, scale_range: Tuple[float, float] = (0.75, 0.95)):
        self.probability = probability
        self.scale_range = scale_range
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.probability:
            orig_size = img.size
            scale = random.uniform(*self.scale_range)
            new_size = (int(orig_size[0] * scale), int(orig_size[1] * scale))
            img = img.resize(new_size, Image.Resampling.BILINEAR)
            img = img.resize(orig_size, Image.Resampling.BILINEAR)
        return img


# ─────────────────────────────────────────
#  DATA AUGMENTATION PIPELINES
# ─────────────────────────────────────────

def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Advanced training augmentation pipeline
    Includes geometric, color, and advanced augmentations
    """
    aug = AugmentationConfig()
    
    train_transform = transforms.Compose([
        # Geometric transforms
        transforms.RandomResizedCrop(
            img_size, 
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5 if aug.RANDOM_HORIZONTAL_FLIP else 0),
        transforms.RandomVerticalFlip(p=0.1 if aug.RANDOM_VERTICAL_FLIP else 0),
        transforms.RandomRotation(aug.RANDOM_ROTATION, fill=128),
        transforms.RandomAffine(
            degrees=aug.RANDOM_AFFINE_DEGREES,
            translate=aug.RANDOM_AFFINE_TRANSLATE,
            fill=128
        ),
        
        # Color transforms
        transforms.ColorJitter(
            brightness=aug.COLOR_JITTER_BRIGHTNESS,
            contrast=aug.COLOR_JITTER_CONTRAST,
            saturation=aug.COLOR_JITTER_SATURATION,
            hue=aug.COLOR_JITTER_HUE
        ),
        
        # Advanced transforms
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
        
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=aug.RANDOM_ERASING_PROBABILITY,
            scale=aug.RANDOM_ERASING_SCALE,
            ratio=(0.3, 3.0),
            value=0.5
        ),
        
        # Gaussian noise
        RandomGaussianNoise(
            probability=aug.RANDOM_NOISE_PROBABILITY,
            sigma=aug.RANDOM_NOISE_SIGMA
        ),
        
        # Normalization
        transforms.Normalize(
            mean=aug.IMAGENET_MEAN,
            std=aug.IMAGENET_STD
        )
    ])
    
    return train_transform


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Validation augmentation pipeline (minimal augmentation)
    """
    aug = AugmentationConfig()
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=aug.IMAGENET_MEAN,
            std=aug.IMAGENET_STD
        )
    ])
    
    return val_transform


def get_test_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Test augmentation pipeline (no augmentation, only normalization)
    """
    return get_val_transforms(img_size)


# ─────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────

def load_datasets(
    data_dir: str = None,
    train_size: int = None,
    val_size: int = None,
    test_size: int = None,
    batch_size: int = None,
    num_workers: int = None,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load datasets with proper splits and augmentation
    
    Args:
        data_dir: Path to data directory
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples (None = all remaining)
        batch_size: Batch size
        num_workers: Number of worker threads
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info)
    """
    # Use defaults from config
    data_dir = data_dir or str(DATA_DIR)
    train_size = train_size or TrainingConfig.TRAIN_SIZE
    val_size = val_size or TrainingConfig.VAL_SIZE
    batch_size = batch_size or TrainingConfig.BATCH_SIZE
    num_workers = num_workers or TrainingConfig.NUM_WORKERS
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    print("\n" + "="*60)
    print("LOADING DATASETS")
    print("="*60)
    
    # Load full datasets
    print(f"\nLoading from: {data_dir}")
    
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    test_transform = get_test_transforms()
    
    full_train = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transform
    )
    full_val = datasets.ImageFolder(
        os.path.join(data_dir, "valid"),
        transform=val_transform
    )
    full_test = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=test_transform
    )
    
    print(f"Full Train: {len(full_train)} images")
    print(f"Full Val: {len(full_val)} images")
    print(f"Full Test: {len(full_test)} images")
    
    # Create subsets
    train_indices = random.sample(range(len(full_train)), min(train_size, len(full_train)))
    val_indices = random.sample(range(len(full_val)), min(val_size, len(full_val)))
    
    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(full_val, val_indices)
    
    # Use all test data or specified amount
    if test_size and test_size < len(full_test):
        test_indices = random.sample(range(len(full_test)), test_size)
        test_subset = Subset(full_test, test_indices)
    else:
        test_subset = full_test
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    dataset_info = {
        "classes": full_train.classes,
        "num_classes": len(full_train.classes),
        "train_size": len(train_subset),
        "val_size": len(val_subset),
        "test_size": len(test_subset),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "test_batches": len(test_loader),
    }
    
    print(f"\nSubset Train: {len(train_subset)} images → {len(train_loader)} batches")
    print(f"Subset Val: {len(val_subset)} images → {len(val_loader)} batches")
    print(f"Subset Test: {len(test_subset)} images → {len(test_loader)} batches")
    print(f"Classes: {dataset_info['classes']}")
    print("="*60 + "\n")
    
    return train_loader, val_loader, test_loader, dataset_info
