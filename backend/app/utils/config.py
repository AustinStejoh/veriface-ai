"""
Configuration module for VeriFace AI training and inference
"""
import torch
import os
from pathlib import Path

# ─────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/real_vs_fake/real-vs-fake"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
#  TRAINING CONFIG
# ─────────────────────────────────────────
class TrainingConfig:
    # Dataset
    BATCH_SIZE = 32
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    IMG_SIZE = 224
    
    # Training
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    WARMUP_EPOCHS = 2
    
    # Data splits
    TRAIN_SIZE = 18000
    VAL_SIZE = 2000
    TEST_SIZE = None  # Use all remaining test data
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 1e-4
    
    # Learning rate scheduler
    SCHEDULER_TYPE = "cosine"  # "cosine", "step", "exponential"
    SCHEDULER_STEP_SIZE = 10
    SCHEDULER_GAMMA = 0.5
    
    # Model
    MODEL_ARCHITECTURES = ["resnet50", "efficientnet_b0", "efficientnet_b1"]
    DEFAULT_ARCHITECTURE = "efficientnet_b1"
    PRETRAINED = True
    
    # Checkpointing
    SAVE_FREQUENCY = 5  # Save checkpoint every N epochs
    SAVE_BEST_ONLY = True

# ─────────────────────────────────────────
#  DATA AUGMENTATION CONFIG
# ─────────────────────────────────────────
class AugmentationConfig:
    # Geometric transforms
    RANDOM_CROP_SIZE = 224
    RANDOM_HORIZONTAL_FLIP = True
    RANDOM_VERTICAL_FLIP = False
    RANDOM_ROTATION = 15
    RANDOM_AFFINE_DEGREES = 10
    RANDOM_AFFINE_TRANSLATE = (0.1, 0.1)
    RANDOM_PERSPECTIVE = True
    
    # Color transforms
    COLOR_JITTER_BRIGHTNESS = 0.3
    COLOR_JITTER_CONTRAST = 0.3
    COLOR_JITTER_SATURATION = 0.2
    COLOR_JITTER_HUE = 0.1
    
    # Advanced augmentation
    GAUSSIAN_BLUR_PROBABILITY = 0.3
    GAUSSIAN_BLUR_KERNEL_SIZE = (3, 7)
    GAUSSIAN_BLUR_SIGMA = (0.1, 2.0)
    
    RANDOM_ERASING_PROBABILITY = 0.2
    RANDOM_ERASING_SCALE = (0.02, 0.3)
    
    RANDOM_COMPRESSION_PROBABILITY = 0.2
    RANDOM_COMPRESSION_QUALITY = (40, 100)
    
    RANDOM_NOISE_PROBABILITY = 0.15
    RANDOM_NOISE_SIGMA = 0.05
    
    # Normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────
#  MODEL CONFIG
# ─────────────────────────────────────────
class ModelConfig:
    NUM_CLASSES = 2
    DROPOUT_RATE = 0.3
    
    # Architecture-specific configs
    RESNET50 = {
        "name": "resnet50",
        "pretrained": True,
        "dropout": 0.3,
    }
    
    EFFICIENTNET_B0 = {
        "name": "efficientnet_b0",
        "pretrained": True,
        "dropout": 0.3,
    }
    
    EFFICIENTNET_B1 = {
        "name": "efficientnet_b1",
        "pretrained": True,
        "dropout": 0.3,
    }

# ─────────────────────────────────────────
#  INFERENCE CONFIG
# ─────────────────────────────────────────
class InferenceConfig:
    CONFIDENCE_THRESHOLD = 0.7
    CLASSES = ["fake", "real"]
    DEFAULT_MODEL = "best_model.pth"
