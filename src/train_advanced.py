"""
Advanced training script for VeriFace AI
Includes multi-model training, early stopping, learning rate scheduling, and comprehensive evaluation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import time
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import json
from tqdm import tqdm
import numpy as np

from config import TrainingConfig, DEVICE, LOGS_DIR, MODELS_DIR, CHECKPOINTS_DIR
from data_utils import load_datasets
from model_utils import (
    create_model, count_parameters, get_model_summary,
    save_model, load_model, freeze_backbone
)
from metrics import MetricsCalculator, print_classification_report, plot_confusion_matrix, plot_roc_curve

# ─────────────────────────────────────────
#  EARLY STOPPING
# ─────────────────────────────────────────

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        checkpoint_path: str = None
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.best_metric = None
        self.best_epoch = None
        self.early_stop = False
    
    def __call__(
        self,
        val_loss: float,
        val_metric: float = None,
        model: nn.Module = None,
        optimizer: optim.Optimizer = None,
        epoch: int = None
    ):
        """
        Check if training should stop
        
        Args:
            val_loss: Validation loss
            val_metric: Validation metric (e.g., accuracy)
            model: Model to save checkpoint
            optimizer: Optimizer state to save
            epoch: Current epoch
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_metric = val_metric
            self.best_epoch = epoch
            self._save_checkpoint(model, optimizer, epoch, val_loss, val_metric)
        
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_metric = val_metric
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model, optimizer, epoch, val_loss, val_metric)
            print(f"✓ Best model updated (Loss: {val_loss:.4f})")
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered after {self.patience} epochs without improvement")
    
    def _save_checkpoint(self, model, optimizer, epoch, val_loss, val_metric):
        """Save checkpoint"""
        if model is not None and self.checkpoint_path:
            metrics = {"val_loss": val_loss, "val_metric": val_metric}
            save_model(model, self.checkpoint_path, optimizer, epoch, metrics)


# ─────────────────────────────────────────
#  TRAINER CLASS
# ─────────────────────────────────────────

class DeepfakeDetectorTrainer:
    """Main trainer class for model training and evaluation"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: torch.device = DEVICE,
        model_name: str = "model",
        dataset_info: Dict = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.dataset_info = dataset_info or {}
        
        # Training state
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "learning_rate": [],
        }
        
        # Best metrics
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: optim.lr_scheduler._LRScheduler = None
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            metrics_calc.update(outputs.detach(), labels)
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = metrics_calc.compute()
        train_accuracy = metrics["accuracy"]
        
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            current_lr = optimizer.param_groups[0]["lr"]
        
        return avg_loss, train_accuracy, current_lr
    
    def validate(self, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                metrics_calc.update(outputs, labels)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = metrics_calc.compute()
        val_accuracy = metrics["accuracy"]
        
        return avg_loss, val_accuracy, metrics
    
    def train(
        self,
        num_epochs: int = None,
        learning_rate: float = None,
        weight_decay: float = None,
        scheduler_type: str = "cosine",
        early_stopping_patience: int = None,
        warmup_epochs: int = 0,
    ) -> Dict:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            scheduler_type: Type of LR scheduler
            early_stopping_patience: Patience for early stopping
            warmup_epochs: Number of warmup epochs
        
        Returns:
            Training history and best metrics
        """
        # Use config defaults
        num_epochs = num_epochs or TrainingConfig.EPOCHS
        learning_rate = learning_rate or TrainingConfig.LEARNING_RATE
        weight_decay = weight_decay or TrainingConfig.WEIGHT_DECAY
        early_stopping_patience = early_stopping_patience or TrainingConfig.EARLY_STOPPING_PATIENCE
        
        print("\n" + "="*60)
        print(f"TRAINING: {self.model_name}")
        print("="*60)
        print(f"Device:         {self.device}")
        print(f"Epochs:         {num_epochs}")
        print(f"Learning rate:  {learning_rate}")
        print(f"Weight decay:   {weight_decay}")
        print(f"LR Scheduler:   {scheduler_type}")
        print("="*60 + "\n")
        
        # Setup optimizer and loss
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        
        # Setup LR scheduler
        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
        elif scheduler_type == "step":
            scheduler = StepLR(optimizer, step_size=TrainingConfig.SCHEDULER_STEP_SIZE, gamma=TrainingConfig.SCHEDULER_GAMMA)
        elif scheduler_type == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None
        
        # Setup early stopping
        checkpoint_path = CHECKPOINTS_DIR / f"{self.model_name}_best.pth"
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=TrainingConfig.EARLY_STOPPING_MIN_DELTA,
            checkpoint_path=str(checkpoint_path)
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc, lr = self.train_epoch(optimizer, criterion, scheduler if epoch >= warmup_epochs else None)
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(criterion)
            
            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_accuracy"].append(train_acc)
            self.history["val_accuracy"].append(val_acc)
            self.history["learning_rate"].append(lr)
            
            # Update best metrics
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
            
            # Print metrics
            epoch_time = (time.time() - epoch_start) / 60
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            print(f"LR: {lr:.6f} | Time: {epoch_time:.2f}m")
            
            # Early stopping
            early_stopper(val_loss, val_acc, self.model, optimizer, epoch)
            if early_stopper.early_stop:
                break
            
            # Regular checkpoint
            if (epoch + 1) % TrainingConfig.SAVE_FREQUENCY == 0 and not TrainingConfig.SAVE_BEST_ONLY:
                ckpt_path = CHECKPOINTS_DIR / f"{self.model_name}_epoch_{epoch+1}.pth"
                save_model(self.model, str(ckpt_path), optimizer, epoch)
        
        total_time = (time.time() - start_time) / 60
        print(f"\n✓ Training completed in {total_time:.2f} minutes")
        
        # Load best model
        if os.path.exists(checkpoint_path):
            self.model, _, _ = load_model(self.model, str(checkpoint_path), device=self.device)
        
        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": early_stopper.best_epoch,
            "total_epochs_trained": epoch + 1,
            "total_time_minutes": total_time,
        }
    
    def evaluate(self, data_loader, set_name: str = "Test") -> Dict:
        """
        Evaluate model on a dataset
        
        Args:
            data_loader: Data loader
            set_name: Name of the dataset (for logging)
        
        Returns:
            Metrics dictionary
        """
        print(f"\n{set_name} Evaluation")
        print("-" * 60)
        
        self.model.eval()
        metrics_calc = MetricsCalculator()
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc=f"Evaluating on {set_name}"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                metrics_calc.update(outputs, labels)
        
        metrics = metrics_calc.compute()
        
        # Print summary
        print(metrics_calc.get_summary_string())
        
        # Print classification report
        print_classification_report(
            metrics["predictions"],
            metrics["targets"],
            class_names=["fake", "real"]
        )
        
        # Plot confusion matrix
        save_dir = LOGS_DIR / self.model_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            class_names=["fake", "real"],
            save_path=str(save_dir / f"confusion_matrix_{set_name.lower()}.png")
        )
        
        if metrics["auc"] is not None:
            plot_roc_curve(
                metrics["targets"],
                metrics["probabilities"],
                save_path=str(save_dir / f"roc_curve_{set_name.lower()}.png")
            )
        
        return metrics


# ─────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ─────────────────────────────────────────

def main():
    """Main training pipeline"""
    
    # Load datasets
    train_loader, val_loader, test_loader, dataset_info = load_datasets()
    
    # Models to train
    architectures = ["resnet50", "efficientnet_b0", "efficientnet_b1"]
    results = {}
    
    for arch in architectures:
        print(f"\n\n{'='*60}")
        print(f"TRAINING: {arch.upper()}")
        print('='*60)
        
        # Create model
        model = create_model(
            architecture=arch,
            num_classes=2,
            pretrained=True,
            device=DEVICE,
            dropout=0.3
        )
        
        # Print model info
        total, trainable, non_trainable = count_parameters(model)
        print(f"\nModel Parameters:")
        print(f"  Total: {total:,}")
        print(f"  Trainable: {trainable:,}")
        print(f"  Non-trainable: {non_trainable:,}")
        
        # Train
        trainer = DeepfakeDetectorTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=DEVICE,
            model_name=arch,
            dataset_info=dataset_info
        )
        
        train_info = trainer.train(
            num_epochs=TrainingConfig.EPOCHS,
            learning_rate=TrainingConfig.LEARNING_RATE,
            weight_decay=TrainingConfig.WEIGHT_DECAY,
            scheduler_type=TrainingConfig.SCHEDULER_TYPE
        )
        
        # Evaluate
        test_metrics = trainer.evaluate(test_loader, set_name="Test")
        
        # Save final model
        final_model_path = MODELS_DIR / f"{arch}_final.pth"
        save_model(trainer.model, str(final_model_path), metrics=test_metrics)
        
        results[arch] = {
            "train_info": train_info,
            "test_metrics": test_metrics,
            "model_info": get_model_summary(model),
        }
    
    # Save results
    results_path = LOGS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        # Convert to serializable format
        serializable_results = {}
        for model_name, data in results.items():
            serializable_results[model_name] = {
                "train_info": data["train_info"],
                "test_accuracy": float(data["test_metrics"]["accuracy"]),
                "test_f1": float(data["test_metrics"]["f1_macro"]),
                "test_precision": float(data["test_metrics"]["precision_macro"]),
                "test_recall": float(data["test_metrics"]["recall_macro"]),
                "test_auc": float(data["test_metrics"]["auc"]) if data["test_metrics"]["auc"] else None,
                "model_info": data["model_info"],
            }
        json.dump(serializable_results, f, indent=2)
        print(f"\n✓ Results saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)


if __name__ == "__main__":
    main()
