"""
Model utilities for VeriFace AI
Includes multiple architectures and model management utilities
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Tuple, Optional
from config import ModelConfig, DEVICE
import os

# ─────────────────────────────────────────
#  CUSTOM MODEL ARCHITECTURES
# ─────────────────────────────────────────

class ResNet50DeepfakeDetector(nn.Module):
    """ResNet50-based deepfake detector with dropout for regularization"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Remove the original classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Custom classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class EfficientNetB0DeepfakeDetector(nn.Module):
    """EfficientNet-B0 based deepfake detector"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get number of input features for classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class EfficientNetB1DeepfakeDetector(nn.Module):
    """EfficientNet-B1 based deepfake detector (improved over B0)"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained EfficientNet-B1
        self.backbone = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get number of input features for classifier
        num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ViTDeepfakeDetector(nn.Module):
    """Vision Transformer based deepfake detector"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        
        # Load pretrained ViT-B16
        self.backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Get hidden size
        hidden_size = self.backbone.heads[0].in_features
        
        # Replace classification head with custom head
        self.backbone.heads = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ─────────────────────────────────────────
#  MODEL FACTORY
# ─────────────────────────────────────────

def create_model(
    architecture: str = "efficientnet_b1",
    num_classes: int = 2,
    pretrained: bool = True,
    device: torch.device = None,
    dropout: float = 0.3
) -> nn.Module:
    """
    Create a model instance based on architecture name
    
    Args:
        architecture: Name of the architecture (resnet50, efficientnet_b0, efficientnet_b1, vit)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        device: Device to place model on
        dropout: Dropout rate for regularization
    
    Returns:
        Model instance
    """
    device = device or DEVICE
    
    model_map = {
        "resnet50": ResNet50DeepfakeDetector,
        "efficientnet_b0": EfficientNetB0DeepfakeDetector,
        "efficientnet_b1": EfficientNetB1DeepfakeDetector,
        "efficientnet_b2": lambda nc, p, d: EfficientNetB1DeepfakeDetector(nc, p, d),  # Use B1 as proxy for now
        "vit": ViTDeepfakeDetector,
    }
    
    if architecture not in model_map:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(model_map.keys())}")
    
    print(f"Creating model: {architecture}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Dropout: {dropout}")
    print(f"  Classes: {num_classes}")
    
    model = model_map[architecture](
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    model = model.to(device)
    return model


# ─────────────────────────────────────────
#  MODEL UTILITIES
# ─────────────────────────────────────────

def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """
    Count model parameters
    
    Returns:
        (total_params, trainable_params, non_trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return total_params, trainable_params, non_trainable_params


def freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone parameters (transfer learning)"""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("✓ Backbone parameters frozen")


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters (fine-tuning)"""
    if hasattr(model, 'backbone'):
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone parameters unfrozen")


def get_model_summary(model: nn.Module) -> Dict:
    """Get model summary statistics"""
    total, trainable, non_trainable = count_parameters(model)
    
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": non_trainable,
        "model_size_mb": total * 4 / 1024 / 1024,  # Approximate size in MB
    }


def save_model(
    model: nn.Module,
    save_path: str,
    optimizer: torch.optim.Optimizer = None,
    epoch: int = None,
    metrics: Dict = None
) -> None:
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        save_path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_architecture": model.__class__.__name__,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to {save_path}")


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    device: torch.device = None
) -> Tuple[nn.Module, Optional[torch.optim.Optimizer], dict]:
    """
    Load model from checkpoint
    
    Args:
        model: Model to load state into
        checkpoint_path: Path to checkpoint
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint to
    
    Returns:
        Tuple of (model, optimizer, checkpoint_info)
    """
    device = device or DEVICE
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    info = {
        "model_architecture": checkpoint.get("model_architecture", "Unknown"),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    print(f"✓ Model loaded from {checkpoint_path}")
    if info["epoch"] > 0:
        print(f"  Resuming from epoch {info['epoch']}")
    
    return model, optimizer, info
