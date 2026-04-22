"""
Standalone evaluation script for VeriFace AI.
Computes accuracy, precision, recall, F1 score, and confusion matrix.
Saves metrics to JSON and generates visualization plots.
"""
import argparse
import json
from pathlib import Path
import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from config import DEVICE, DATA_DIR, LOGS_DIR
from data_utils import get_test_transforms
from model_utils import create_model, load_model
from metrics import MetricsCalculator, plot_confusion_matrix, plot_roc_curve, print_classification_report


def load_test_data(test_dir: Path, batch_size: int = 32, num_workers: int = 0):
    transform = get_test_transforms()
    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return loader, dataset


def serialize_metrics(metrics: dict, class_names: list):
    serialized = {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision_macro": float(metrics.get("precision_macro", 0.0)),
        "recall_macro": float(metrics.get("recall_macro", 0.0)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "precision_weighted": float(metrics.get("precision_weighted", 0.0)),
        "recall_weighted": float(metrics.get("recall_weighted", 0.0)),
        "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
        "auc": float(metrics.get("auc")) if metrics.get("auc") is not None else None,
        "confusion_matrix": metrics.get("confusion_matrix").tolist() if metrics.get("confusion_matrix") is not None else None,
        "tp": int(metrics.get("tp", 0)),
        "fp": int(metrics.get("fp", 0)),
        "fn": int(metrics.get("fn", 0)),
        "class_names": class_names,
        "precision_per_class": [float(x) for x in metrics.get("precision", [])],
        "recall_per_class": [float(x) for x in metrics.get("recall", [])],
        "f1_per_class": [float(x) for x in metrics.get("f1", [])],
    }
    return serialized


def evaluate_model(
    model,
    model_name: str,
    checkpoint_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    num_workers: int = 0
):
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        test_dir = DATA_DIR / "valid"
        if not test_dir.exists():
            raise FileNotFoundError(f"Cannot find test or validation data under {DATA_DIR}")

    test_loader, test_dataset = load_test_data(test_dir, batch_size=batch_size, num_workers=num_workers)
    class_names = test_dataset.classes

    model, _, _ = load_model(model, str(checkpoint_path), device=DEVICE)
    model.eval()

    metrics_calc = MetricsCalculator(class_names=class_names)
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(inputs)
            metrics_calc.update(outputs, targets)

    metrics = metrics_calc.compute()
    summary = metrics_calc.get_summary_string()
    print(summary)
    print_classification_report(metrics["predictions"], metrics["targets"], class_names=class_names)

    metrics_json = serialize_metrics(metrics, class_names)
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion_matrix"], class_names=class_names, save_path=str(cm_path))

    if metrics.get("auc") is not None:
        roc_path = output_dir / "roc_curve.png"
        plot_roc_curve(metrics["targets"], metrics["probabilities"], save_path=str(roc_path))

    return metrics_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a VeriFace AI model checkpoint")
    parser.add_argument(
        "--architecture",
        type=str,
        default="efficientnet_b1",
        choices=["resnet50", "efficientnet_b0", "efficientnet_b1", "vit"],
        help="Model architecture used during training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for metrics and plots"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader worker processes"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else LOGS_DIR / f"evaluation_{checkpoint_path.stem}"

    model = create_model(
        architecture=args.architecture,
        num_classes=2,
        pretrained=False,
        device=DEVICE
    )

    evaluate_model(
        model=model,
        model_name=args.architecture,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

if __name__ == "__main__":
    main()
