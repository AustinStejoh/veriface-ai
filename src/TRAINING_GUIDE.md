# VeriFace AI - Advanced Training Guide

## Overview

VeriFace AI has been upgraded to a production-grade, research-level deepfake detection system with:

- **Multiple Model Architectures**: ResNet50, EfficientNet-B0, EfficientNet-B1, Vision Transformer
- **Advanced Data Augmentation**: Geometric, color, and advanced augmentations with real-world scenarios
- **Comprehensive Evaluation**: Precision, Recall, F1, AUC-ROC, Confusion Matrix, Classification Reports
- **Training Optimization**: Early Stopping, Learning Rate Scheduling, Gradient Clipping, Warmup
- **Experiment Tracking**: Automatic checkpoint saving, training history, model comparisons

## Project Structure

```
src/
├── config.py              # Centralized configuration
├── data_utils.py          # Data loading and augmentation pipelines
├── model_utils.py         # Model architectures and utilities
├── metrics.py             # Comprehensive evaluation metrics
├── train_advanced.py      # Advanced training framework
├── train_cli.py           # Simple CLI for training
└── TRAINING_GUIDE.md      # This file

logs/                       # Training logs and results
├── {model_name}/          # Per-model logs
│   ├── confusion_matrix_test.png
│   ├── roc_curve_test.png
│   └── ...
└── training_results.json  # Overall results summary

models/
├── best_model.pth         # Best model (used by app.py)
├── {arch}_final.pth       # Final trained models
└── checkpoints/           # Intermediate checkpoints
    ├── {arch}_best.pth
    └── {arch}_epoch_N.pth
```

## Installation

Install additional dependencies for training:

```bash
pip install scikit-learn matplotlib seaborn python-dotenv
```

Or update all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Option 1: Train Single Model (Recommended for Quick Testing)

```bash
cd src
python train_cli.py --architecture efficientnet_b1 --epochs 10
```

### Option 2: Train Multiple Models (Comprehensive Comparison)

```bash
cd src
python train_advanced.py
```

This trains all architectures (ResNet50, EfficientNet-B0, EfficientNet-B1) and generates comparison plots.

## Configuration

All configuration is centralized in `src/config.py`. Key configurations:

### Training Config

```python
TrainingConfig.BATCH_SIZE = 32           # Batch size
TrainingConfig.EPOCHS = 50               # Max epochs
TrainingConfig.LEARNING_RATE = 1e-4      # Learning rate
TrainingConfig.WEIGHT_DECAY = 1e-5       # L2 regularization
TrainingConfig.WARMUP_EPOCHS = 2         # Warmup before LR scheduling
TrainingConfig.EARLY_STOPPING_PATIENCE = 10  # Patience for early stopping
TrainingConfig.SCHEDULER_TYPE = "cosine" # "cosine", "step", "exponential"
```

### Augmentation Config

```python
AugmentationConfig.RANDOM_HORIZONTAL_FLIP = True
AugmentationConfig.RANDOM_ROTATION = 15
AugmentationConfig.COLOR_JITTER_BRIGHTNESS = 0.3
AugmentationConfig.GAUSSIAN_BLUR_PROBABILITY = 0.3
AugmentationConfig.RANDOM_ERASING_PROBABILITY = 0.2
AugmentationConfig.RANDOM_NOISE_PROBABILITY = 0.15
# ... and more
```

## Advanced Usage

### Train with Custom Hyperparameters

```bash
python train_cli.py \
  --architecture efficientnet_b1 \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 5e-5 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --early-stopping-patience 15 \
  --warmup-epochs 3 \
  --output-model models/custom_model.pth
```

### Train Without Test Evaluation

```bash
python train_cli.py --no-evaluation
```

### Programmatic Training

```python
from train_advanced import load_datasets, create_model, DeepfakeDetectorTrainer
from config import TrainingConfig, DEVICE

# Load data
train_loader, val_loader, test_loader, info = load_datasets(batch_size=32)

# Create model
model = create_model("efficientnet_b1", num_classes=2, pretrained=True)

# Train
trainer = DeepfakeDetectorTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=DEVICE,
    model_name="my_model"
)

results = trainer.train(
    num_epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-5,
    scheduler_type="cosine"
)

# Evaluate
metrics = trainer.evaluate(test_loader)
```

## Data Augmentation

### Training Augmentation

The training pipeline includes:

1. **Geometric Transforms**
   - Random Resized Crop (0.8-1.0 scale)
   - Random Horizontal Flip (50%)
   - Random Rotation (±15°)
   - Random Affine Transform

2. **Color Transforms**
   - Color Jitter (brightness, contrast, saturation, hue)
   - Random Grayscale (5%)

3. **Advanced Augmentation**
   - Gaussian Blur (30%)
   - Random Erasing (20%)
   - Gaussian Noise (15%)
   - JPEG Compression (20%) - simulates real-world scenarios

### Validation/Test Augmentation

Minimal augmentation (only normalization) for validation and test sets to provide fair evaluation.

## Model Architectures

### Available Models

1. **ResNet50** - Classic baseline, 50 layers
   - Parameters: ~25M
   - Inference Speed: Moderate
   - Accuracy: Good

2. **EfficientNet-B0** - Efficient architecture, optimized for mobile
   - Parameters: ~5M
   - Inference Speed: Fast
   - Accuracy: Good

3. **EfficientNet-B1** - Larger EfficientNet variant (recommended)
   - Parameters: ~7M
   - Inference Speed: Fast
   - Accuracy: Excellent

4. **Vision Transformer (ViT)** - Modern transformer-based approach
   - Parameters: ~87M
   - Inference Speed: Moderate
   - Accuracy: Excellent
   - Note: More data/compute required

### Custom Model Head

All models include a custom classification head:

```
Input → Backbone → Feature Extraction → Dropout → Linear(hidden) → 
BatchNorm → ReLU → Dropout → Linear(2) → Output
```

This allows for better adaptation to the deepfake detection task.

## Training Features

### Early Stopping

Training stops automatically if validation loss doesn't improve for N epochs:

```python
EarlyStopping(patience=10, min_delta=1e-4)
```

- Saves best model automatically
- Prevents overfitting
- Configurable via `--early-stopping-patience`

### Learning Rate Scheduling

Three options available:

1. **Cosine Annealing** (recommended)
   - Smooth decay from initial LR to 0
   - Good generalization
   - `scheduler_type="cosine"`

2. **Step Decay**
   - Reduce LR by factor every N epochs
   - `scheduler_type="step"`

3. **Exponential Decay**
   - Exponential reduction
   - `scheduler_type="exponential"`

### Gradient Clipping

Prevents gradient explosion during training:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Warmup

Optional learning rate warmup for first N epochs helps stabilize training:
```bash
--warmup-epochs 2
```

## Evaluation Metrics

### Computed Metrics

- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curve
- **Confusion Matrix**: True positives, false positives, etc.
- **Classification Report**: Per-class detailed metrics

### Output Visualizations

For each model, the following plots are generated:

1. **Confusion Matrix Heatmap**
   - Shows true vs predicted labels
   - Saved as: `logs/{model_name}/confusion_matrix_test.png`

2. **ROC Curve**
   - Plots true positive rate vs false positive rate
   - Includes AUC score
   - Saved as: `logs/{model_name}/roc_curve_test.png`

### Example Output

```
============================================================
METRICS SUMMARY
============================================================
Accuracy:  0.9523
Precision (macro): 0.9487
Recall (macro):    0.9523
F1 Score (macro):  0.9505
AUC-ROC:   0.9876

Per-class metrics:
  fake    : P=0.9456, R=0.9512, F1=0.9484
  real    : P=0.9518, R=0.9534, F1=0.9526
============================================================
```

## Model Comparison

### Automatic Comparison

Running `train_advanced.py` trains all models and generates comparison plots showing:

- Accuracy comparison
- F1 Score comparison
- Precision/Recall comparison
- AUC-ROC comparison

Results are saved in `logs/training_results.json`:

```json
{
  "resnet50": {
    "test_accuracy": 0.945,
    "test_f1": 0.943,
    "test_precision": 0.941,
    "test_recall": 0.945,
    "test_auc": 0.981
  },
  "efficientnet_b1": {
    "test_accuracy": 0.962,
    "test_f1": 0.960,
    "test_precision": 0.958,
    "test_recall": 0.962,
    "test_auc": 0.987
  }
}
```

## Checkpointing

### Model Saving

Models are saved in multiple ways:

1. **Best Model** (during training)
   - Saved automatically when validation loss improves
   - Path: `models/checkpoints/{model_name}_best.pth`

2. **Periodic Checkpoints**
   - Saved every N epochs (configurable)
   - Path: `models/checkpoints/{model_name}_epoch_N.pth`

3. **Final Model**
   - Saved after training completes
   - Path: `models/{architecture}_final.pth` or custom path

### Checkpoint Format

```python
{
  "model_state_dict": {...},
  "optimizer_state_dict": {...},
  "epoch": 25,
  "metrics": {
    "val_loss": 0.234,
    "val_accuracy": 0.952
  },
  "model_architecture": "EfficientNetB1DeepfakeDetector"
}
```

## Performance Tips

### Faster Training

1. **Use EfficientNet-B0** instead of B1 (faster, still accurate)
2. **Reduce batch size** if GPU memory is limited
3. **Use smaller dataset** for quick experimentation
4. **Enable GPU** (automatically detected)

### Better Accuracy

1. **Train longer** (increase `--epochs`)
2. **Use larger batch sizes** (if GPU memory allows)
3. **Lower learning rate** gradually with longer training
4. **Use EfficientNet-B1** or **ViT** (more parameters)
5. **Ensemble multiple models** (advanced)

### GPU Memory Optimization

```bash
# Reduce batch size for limited GPU memory
python train_cli.py --batch-size 16

# Use gradient accumulation in code
for i in range(accumulation_steps):
    outputs = model(images[i*batch:i*batch+batch])
    loss = criterion(outputs, labels[i*batch:i*batch+batch])
    loss.backward()
if i % accumulation_steps == 0:
    optimizer.step()
    optimizer.zero_grad()
```

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_cli.py --batch-size 16

# Use EfficientNet-B0 instead
python train_cli.py --architecture efficientnet_b0
```

### Training is Too Slow

```bash
# Reduce dataset size in config.py
TrainingConfig.TRAIN_SIZE = 5000  # Default: 18000
TrainingConfig.VAL_SIZE = 500

# Use EfficientNet-B0
python train_cli.py --architecture efficientnet_b0 --batch-size 64
```

### Model Not Improving

```bash
# Increase training time
python train_cli.py --epochs 100 --early-stopping-patience 20

# Try different learning rate
python train_cli.py --learning-rate 5e-5

# Use warmup
python train_cli.py --warmup-epochs 5
```

### Poor Test Performance

1. **Check data quality**: Verify training/validation data
2. **Increase augmentation**: More diverse training data
3. **Use pre-trained weights**: Always use `--pretrained`
4. **Train longer**: More epochs may help

## Next Steps

### Deployment

Once you have a trained model:

1. **Copy model to app.py directory**
   ```bash
   cp models/best_model.pth models/best_model_v2.pth
   ```

2. **Update app.py** to use new model:
   ```python
   MODEL_PATH = "models/best_model_v2.pth"
   ```

3. **Test API** before deployment

### Improvements

1. **Robustness Testing**: Add adversarial examples
2. **Explainability**: Use Grad-CAM for visualization
3. **Ensemble Methods**: Combine multiple models
4. **Data Collection**: Collect more real-world deepfake samples
5. **Model Compression**: Quantization or distillation for mobile deployment

## References

- EfficientNet: https://arxiv.org/abs/1905.11946
- Vision Transformer: https://arxiv.org/abs/2010.11929
- Data Augmentation: https://arxiv.org/abs/1512.04412
- Early Stopping: https://en.wikipedia.org/wiki/Early_stopping

## Citation

If you use this training framework in your research, please cite:

```bibtex
@software{veriface_ai,
  title={VeriFace AI: Production-Grade Deepfake Detection},
  year={2024},
}
```

---

For questions or issues, refer to the main project README.md
