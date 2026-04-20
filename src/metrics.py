"""
Metrics module for VeriFace AI evaluation
Provides comprehensive metrics for model evaluation
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────
#  METRICS CALCULATION
# ─────────────────────────────────────────

class MetricsCalculator:
    """Calculate and track training/validation metrics"""
    
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ["fake", "real"]
        self.reset()
    
    def reset(self):
        """Reset metric accumulators"""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Update metrics with batch predictions"""
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        self.probabilities.extend(probs.cpu().detach().numpy().tolist())
    
    def compute(self) -> Dict:
        """Compute all metrics"""
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        probabilities = np.array(self.probabilities)
        
        # Basic metrics
        accuracy = np.mean(predictions == targets)
        
        # Per-class metrics
        precision = precision_score(targets, predictions, average=None, zero_division=0)
        recall = recall_score(targets, predictions, average=None, zero_division=0)
        f1 = f1_score(targets, predictions, average=None, zero_division=0)
        
        # Macro averages
        precision_macro = np.mean(precision)
        recall_macro = np.mean(recall)
        f1_macro = np.mean(f1)
        
        # Weighted averages
        precision_weighted = precision_score(targets, predictions, average="weighted", zero_division=0)
        recall_weighted = recall_score(targets, predictions, average="weighted", zero_division=0)
        f1_weighted = f1_score(targets, predictions, average="weighted", zero_division=0)
        
        # AUC for binary classification
        auc = None
        if len(np.unique(targets)) == 2:
            auc = roc_auc_score(targets, probabilities[:, 1])
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # True positives, false positives, false negatives
        tp = cm[1, 1] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        fn = cm[1, 0] if cm.shape == (2, 2) else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "auc": auc,
            "confusion_matrix": cm,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "predictions": predictions,
            "targets": targets,
            "probabilities": probabilities,
        }
        
        return metrics
    
    def get_summary_string(self) -> str:
        """Get formatted summary of metrics"""
        metrics = self.compute()
        
        summary = "\n" + "="*60 + "\n"
        summary += "METRICS SUMMARY\n"
        summary += "="*60 + "\n"
        summary += f"Accuracy:  {metrics['accuracy']:.4f}\n"
        summary += f"Precision (macro): {metrics['precision_macro']:.4f}\n"
        summary += f"Recall (macro):    {metrics['recall_macro']:.4f}\n"
        summary += f"F1 Score (macro):  {metrics['f1_macro']:.4f}\n"
        
        if metrics['auc'] is not None:
            summary += f"AUC-ROC:   {metrics['auc']:.4f}\n"
        
        summary += "\nPer-class metrics:\n"
        for i, class_name in enumerate(self.class_names):
            summary += f"  {class_name:8s}: P={metrics['precision'][i]:.4f}, "
            summary += f"R={metrics['recall'][i]:.4f}, F1={metrics['f1'][i]:.4f}\n"
        
        summary += "="*60 + "\n"
        return summary


# ─────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """Plot and optionally save confusion matrix"""
    class_names = class_names or ["fake", "real"]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Confusion matrix saved to {save_path}")
    
    plt.close()


def plot_roc_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    save_path: str = None,
    figsize: Tuple[int, int] = (8, 6)
) -> None:
    """Plot and optionally save ROC curve"""
    if len(np.unique(targets)) != 2:
        print("ROC curve only applicable for binary classification")
        return
    
    fpr, tpr, _ = roc_curve(targets, probabilities[:, 1])
    auc = roc_auc_score(targets, probabilities[:, 1])
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ ROC curve saved to {save_path}")
    
    plt.close()


def plot_training_history(
    history: Dict,
    metrics: List[str] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """Plot training history"""
    metrics = metrics or ["loss", "accuracy"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric in history:
            axes[idx].plot(history[metric]["train"], label="Train", linewidth=2)
            axes[idx].plot(history[metric]["val"], label="Val", linewidth=2)
            axes[idx].set_xlabel("Epoch")
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].set_title(f"Training {metric.capitalize()}")
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training history saved to {save_path}")
    
    plt.close()


# ─────────────────────────────────────────
#  MODEL COMPARISON
# ─────────────────────────────────────────

def compare_models(
    results: Dict,
    metrics_to_compare: List[str] = None,
    save_path: str = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Compare metrics across multiple models"""
    metrics_to_compare = metrics_to_compare or [
        "accuracy", "f1_macro", "precision_macro", "recall_macro", "auc"
    ]
    
    # Filter metrics that exist
    metrics_to_compare = [m for m in metrics_to_compare if m in list(results.values())[0]]
    
    models = list(results.keys())
    
    fig, axes = plt.subplots(1, len(metrics_to_compare), figsize=figsize)
    if len(metrics_to_compare) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics_to_compare):
        values = [results[model].get(metric, 0) for model in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        
        axes[idx].bar(models, values, color=colors)
        axes[idx].set_ylabel(metric.replace("_", " ").title())
        axes[idx].set_title(f"{metric.replace('_', ' ').title()} Comparison")
        axes[idx].set_ylim([0, max(values) * 1.1])
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")
        
        axes[idx].tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Model comparison plot saved to {save_path}")
    
    plt.close()


# ─────────────────────────────────────────
#  CLASSIFICATION REPORT
# ─────────────────────────────────────────

def print_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: List[str] = None
) -> str:
    """Print detailed classification report"""
    class_names = class_names or ["fake", "real"]
    
    report = classification_report(
        targets,
        predictions,
        target_names=class_names,
        digits=4
    )
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)
    print("="*60 + "\n")
    
    return report
