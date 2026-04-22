"""
Quick validation script to test all new modules
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

print("Validating VeriFace AI modules...")
print("="*60)

try:
    print("\n1. Testing config module...")
    from config import TrainingConfig, AugmentationConfig, DEVICE
    print(f"   ✓ Config loaded")
    print(f"     - Device: {DEVICE}")
    print(f"     - Batch size: {TrainingConfig.BATCH_SIZE}")
    print(f"     - Epochs: {TrainingConfig.EPOCHS}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("\n2. Testing data_utils module...")
    from data_utils import get_train_transforms, get_val_transforms
    print(f"   ✓ Data utils loaded")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    print(f"     - Train augmentation pipeline: OK")
    print(f"     - Val augmentation pipeline: OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("\n3. Testing model_utils module...")
    from model_utils import create_model, count_parameters, get_model_summary
    print(f"   ✓ Model utils loaded")
    
    # Try creating a model
    model = create_model("efficientnet_b0", num_classes=2)
    total, trainable, non_trainable = count_parameters(model)
    print(f"     - Created EfficientNet-B0 model: OK")
    print(f"     - Total parameters: {total:,}")
    print(f"     - Trainable parameters: {trainable:,}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("\n4. Testing metrics module...")
    from metrics import MetricsCalculator, plot_confusion_matrix
    import torch
    print(f"   ✓ Metrics module loaded")
    
    # Test metrics calculator
    calc = MetricsCalculator(["fake", "real"])
    outputs = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))
    calc.update(outputs, targets)
    metrics = calc.compute()
    print(f"     - MetricsCalculator: OK")
    print(f"     - Computed metrics: {list(metrics.keys())[:5]}...")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

try:
    print("\n5. Testing train_advanced module...")
    from train_advanced import DeepfakeDetectorTrainer, EarlyStopping
    print(f"   ✓ Training module loaded")
    print(f"     - DeepfakeDetectorTrainer: OK")
    print(f"     - EarlyStopping: OK")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ All modules validated successfully!")
print("="*60)
print("\nYou can now use:")
print("  python train_cli.py --architecture efficientnet_b1")
print("  python train_advanced.py (for all models)")
print("\nFor detailed guide, see: TRAINING_GUIDE.md")
