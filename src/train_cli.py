"""
Simple entry point for training VeriFace AI models
Run this script to train the deepfake detector
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_advanced import (
    load_datasets, create_model, DeepfakeDetectorTrainer,
    DEVICE, MODELS_DIR, LOGS_DIR
)
from config import TrainingConfig
from model_utils import save_model, get_model_summary

def main():
    parser = argparse.ArgumentParser(
        description="Train VeriFace AI deepfake detection models"
    )
    
    parser.add_argument(
        "--architecture",
        type=str,
        default="efficientnet_b1",
        choices=["resnet50", "efficientnet_b0", "efficientnet_b1", "vit"],
        help="Model architecture to train"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainingConfig.EPOCHS,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainingConfig.BATCH_SIZE,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainingConfig.LEARNING_RATE,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=TrainingConfig.WEIGHT_DECAY,
        help="Weight decay for AdamW optimizer"
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TrainingConfig.EARLY_STOPPING_PATIENCE,
        help="Patience for early stopping"
    )
    
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "exponential"],
        help="Learning rate scheduler type"
    )
    
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of warmup epochs"
    )
    
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Path to save the trained model (defaults to models/best_model.pth)"
    )
    
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="Skip test set evaluation"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("VeriFace AI - Deepfake Detector Training")
    print("="*70)
    print(f"Architecture:        {args.architecture}")
    print(f"Epochs:              {args.epochs}")
    print(f"Batch Size:          {args.batch_size}")
    print(f"Learning Rate:       {args.learning_rate}")
    print(f"Weight Decay:        {args.weight_decay}")
    print(f"LR Scheduler:        {args.scheduler}")
    print(f"Early Stopping:      {args.early_stopping_patience} epochs")
    print(f"Warmup Epochs:       {args.warmup_epochs}")
    print(f"Device:              {DEVICE}")
    print("="*70 + "\n")
    
    # Load datasets
    train_loader, val_loader, test_loader, dataset_info = load_datasets(
        batch_size=args.batch_size
    )
    
    # Create model
    model = create_model(
        architecture=args.architecture,
        num_classes=2,
        pretrained=True,
        device=DEVICE
    )
    
    # Print model info
    summary = get_model_summary(model)
    print(f"\nModel Summary:")
    print(f"  Total Parameters: {summary['total_parameters']:,}")
    print(f"  Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"  Model Size: {summary['model_size_mb']:.2f} MB\n")
    
    # Train
    trainer = DeepfakeDetectorTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=DEVICE,
        model_name=args.architecture,
        dataset_info=dataset_info
    )
    
    train_info = trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        early_stopping_patience=args.early_stopping_patience,
        warmup_epochs=args.warmup_epochs
    )
    
    print(f"\nTraining Summary:")
    print(f"  Best Val Loss: {train_info['best_val_loss']:.4f}")
    print(f"  Best Val Accuracy: {train_info['best_val_accuracy']:.4f}")
    print(f"  Best Epoch: {train_info['best_epoch']}")
    print(f"  Total Time: {train_info['total_time_minutes']:.2f} minutes\n")
    
    # Evaluate on test set
    if not args.no_evaluation:
        test_metrics = trainer.evaluate(test_loader, set_name="Test")
        
        print(f"\nFinal Test Performance:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision_macro']:.4f}")
        print(f"  Recall: {test_metrics['recall_macro']:.4f}")
        print(f"  F1 Score: {test_metrics['f1_macro']:.4f}")
        if test_metrics['auc']:
            print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
    
    # Save model
    output_path = args.output_model or str(MODELS_DIR / "best_model.pth")
    save_model(trainer.model, output_path)
    print(f"\n✓ Model saved to: {output_path}")
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
