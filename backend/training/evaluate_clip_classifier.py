"""Evaluates the zero-shot CLIP crop classifier (app/services/crop_classifier.py)
against a labeled validation set -- no training involved, this just
measures how well CLIP's off-the-shelf pretraining does on the prompts
defined there.

Uses backend/training/data/classify/val/<ClassName>/*.jpg by default (the
same validation split download_classification_dataset.py produces), but
CLIP itself never sees or trains on this data -- it's evaluation-only.

Example:
    python evaluate_clip_classifier.py --data data/classify/val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TRAINING_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.services.crop_classifier import CROP_PROMPTS, CropClassifier  # noqa: E402


def plot_confusion_matrix(matrix: np.ndarray, class_names: list, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CLIP Zero-Shot Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(TRAINING_DIR / "data" / "classify" / "val"))
    args = parser.parse_args()

    val_dir = Path(args.data)
    class_names = sorted(CROP_PROMPTS.keys())
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)
    rows = []  # (path, true, pred, confidence)

    classifier = CropClassifier()

    for class_name in class_names:
        class_dir = val_dir / class_name
        if not class_dir.exists():
            print(f"WARNING: {class_dir} not found, skipping {class_name}")
            continue
        image_paths = sorted(class_dir.glob("*.jpg"))
        true_idx = name_to_idx[class_name]
        for image_path in image_paths:
            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                continue
            pred_label, confidence = classifier.classify(image_bgr)
            pred_idx = name_to_idx[pred_label]
            matrix[true_idx, pred_idx] += 1
            rows.append((image_path.name, class_name, pred_label, confidence))
        print(f"Evaluated {len(image_paths)} {class_name} images.")

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall_acc = correct / total if total else 0.0

    print("\n=== Overall ===")
    print(f"Accuracy: {overall_acc:.4f} ({correct}/{total})")

    print("\n=== Per class ===")
    print(f"{'class':<10}{'correct':>10}{'total':>10}{'accuracy':>10}")
    per_class = []
    for i, name in enumerate(class_names):
        class_total = int(matrix[i].sum())
        class_correct = int(matrix[i, i])
        acc = class_correct / class_total if class_total else 0.0
        print(f"{name:<10}{class_correct:>10}{class_total:>10}{acc:>10.3f}")
        per_class.append((name, class_correct, class_total, acc))

    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("".join(f"{n:>10s}" for n in [""] + class_names))
    for i, name in enumerate(class_names):
        print(f"{name:>10s}" + "".join(f"{v:>10d}" for v in matrix[i]))

    save_dir = TRAINING_DIR / "runs" / "clip_zeroshot_eval"
    save_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(matrix, class_names, confusion_path)
    print(f"\nConfusion matrix plot saved to {confusion_path}")

    lines = [
        "# CLIP Zero-Shot Crop Classifier Evaluation",
        "",
        f"Model: `{CropClassifier().model_name}` -- no fine-tuning, no training data of ours.",
        "Prompts: see `app/services/crop_classifier.py`'s `CROP_PROMPTS`.",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {overall_acc:.4f} ({correct}/{total}) |",
        "",
        "### Per-class accuracy",
        "",
        "| Class | Correct / Total | Accuracy |",
        "|---|---|---|",
    ]
    for name, class_correct, class_total, acc in per_class:
        lines.append(f"| {name} | {class_correct} / {class_total} | {acc:.3f} |")
    lines += ["", "## Plots", "", f"- Confusion matrix: `{confusion_path.relative_to(TRAINING_DIR)}`", ""]

    report_path = TRAINING_DIR / "CLIP_CLASSIFIER_EVAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
