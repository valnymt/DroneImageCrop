"""Evaluation report for the fine-tuned crop-species classifier.

Computes top-1/top-5 accuracy via .val(), builds a per-class confusion
matrix and accuracy breakdown (Ultralytics' ClassifyMetrics doesn't expose
these itself, so this predicts every validation image directly), a
qualitative grid of sample predictions, and writes CLASSIFIER_EVAL_REPORT.md.

Example:
    python report_classifier.py --weights runs/classify_finetune/weights/best.pt --data data/classify
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent


def build_confusion_matrix(model, val_dir: Path, class_names: list) -> tuple:
    """Predicts every validation image and tallies a confusion matrix plus
    per-class accuracy, since ClassifyMetrics only reports overall top1/top5."""
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    n = len(class_names)
    matrix = np.zeros((n, n), dtype=int)  # rows = true, cols = predicted
    samples = []  # (path, true_name, pred_name, confidence)

    for class_dir in sorted(val_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name not in name_to_idx:
            continue
        true_idx = name_to_idx[class_dir.name]
        image_paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.jpeg"))
        for image_path in image_paths:
            result = model.predict(str(image_path), verbose=False)[0]
            pred_idx = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            matrix[true_idx, pred_idx] += 1
            samples.append((image_path, class_dir.name, class_names[pred_idx], confidence))

    return matrix, samples


def build_qualitative_grid(samples: list, n: int, seed: int, save_path: Path) -> None:
    rng = random.Random(seed)
    chosen = rng.sample(samples, min(n, len(samples)))
    cols = min(5, len(chosen))
    rows = max(1, -(-len(chosen) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).reshape(-1)

    from PIL import Image

    for ax, (path, true_name, pred_name, confidence) in zip(axes, chosen):
        with Image.open(path) as img:
            ax.imshow(img)
        correct = true_name == pred_name
        ax.set_title(
            f"true: {true_name}\npred: {pred_name} ({confidence:.0%})",
            fontsize=9,
            color="darkgreen" if correct else "crimson",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(chosen):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def write_report(weights: str, data_dir: Path, class_names: list, top1: float, top5: float, matrix: np.ndarray, confusion_path: Path, qualitative_path: Path) -> Path:
    lines = [
        "# Crop-Species Classifier Evaluation Report",
        "",
        f"Checkpoint: `{weights}`",
        "",
        "## Dataset",
        "",
        "Assembled by `download_classification_dataset.py` from five sources (one per class) -- see that "
        "script's docstring for exactly which dataset backs which class.",
        "",
        "| Class | Train images | Val images |",
        "|---|---|---|",
    ]
    for name in class_names:
        train_n = len(list((data_dir / "train" / name).glob("*.jpg")))
        val_n = len(list((data_dir / "val" / name).glob("*.jpg")))
        lines.append(f"| {name} | {train_n} | {val_n} |")

    lines += [
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Top-1 accuracy | {top1:.4f} |",
        f"| Top-5 accuracy | {top5:.4f} |",
        "",
        "### Per-class accuracy",
        "",
        "| Class | Correct / Total | Accuracy |",
        "|---|---|---|",
    ]
    for i, name in enumerate(class_names):
        total = int(matrix[i].sum())
        correct = int(matrix[i, i])
        acc = correct / total if total else 0.0
        lines.append(f"| {name} | {correct} / {total} | {acc:.3f} |")

    off_diagonal = int(matrix.sum() - np.trace(matrix))
    overall_acc = float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0
    if off_diagonal == 0 and overall_acc >= 0.99:
        lines += [
            "",
            "## Interpretation -- read this before trusting the accuracy number",
            "",
            "Zero confusion across the whole validation set is a red flag here, not a clean "
            "win, and this held even after a deliberate second attempt to fix it. The first "
            "version of this dataset mixed photography styles across classes (grain kernels on "
            "black, studio leaf photos, one field photo) and hit 100% -- unsurprising, since the "
            "background alone gives it away. The dataset was then rebuilt so every class is "
            "genuine in-situ growing-plant/field photography (see "
            "`download_classification_dataset.py`'s docstring). **It still hit 100%.** That "
            "rules out background/style as the sole explanation and points to something deeper: "
            "each class still comes from a different underlying source, and a CNN can separate "
            "sources via artifacts invisible to a human looking at the photo -- JPEG compression "
            "signature, sensor noise, exact resolution/aspect-ratio distribution -- without "
            "learning any real species feature at all. This is the well-documented \"dataset "
            "bias\" / \"name the dataset\" phenomenon in computer vision (Torralba & Efros, "
            "*Unbiased Look at Dataset Bias*, 2011): the only way to actually rule it out is a "
            "test set drawn from a *different* source per class than training, which no amount "
            "of image curation within these five sources can provide. This number should "
            "**not** be read as \"the classifier reliably identifies crop species.\" For "
            "comparison, `evaluate_clip_classifier.py`'s zero-shot CLIP result on this same "
            "data (68.9% accuracy, errors concentrated exactly where a human would expect -- "
            "Wheat/Corn/Rice confused with each other, Soybean/Tomato perfect) never saw any of "
            "these training sources and can't exploit this shortcut, which is precisely why it "
            "scores lower *and* is the more trustworthy number of the two. See "
            "`CLIP_CLASSIFIER_EVAL_REPORT.md`.",
        ]

    lines += [
        "",
        "## Plots",
        "",
        f"- Confusion matrix: `{confusion_path.relative_to(TRAINING_DIR)}`",
        f"- Qualitative sample predictions: `{qualitative_path.relative_to(TRAINING_DIR)}`",
        "",
    ]

    report_path = TRAINING_DIR / "CLASSIFIER_EVAL_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Fine-tuned classifier checkpoint (best.pt)")
    parser.add_argument("--data", default=str(TRAINING_DIR / "data" / "classify"))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--samples", type=int, default=10, help="Number of validation images in the qualitative grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="classify_report", help="Run name under training/runs/")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_dir = Path(args.data)
    val_dir = data_dir / "val"
    class_names = sorted(d.name for d in val_dir.iterdir() if d.is_dir())

    model = YOLO(args.weights)

    metrics = model.val(data=args.data, imgsz=args.imgsz, project=str(TRAINING_DIR / "runs"), name=args.name)
    print(f"\nTop-1 accuracy: {metrics.top1:.4f}")
    print(f"Top-5 accuracy: {metrics.top5:.4f}")

    print("\nBuilding per-class confusion matrix (predicting every validation image)...")
    matrix, samples = build_confusion_matrix(model, val_dir, class_names)
    print("Confusion matrix (rows=true, cols=predicted):")
    print("".join(f"{n:>10s}" for n in [""] + class_names))
    for i, name in enumerate(class_names):
        print(f"{name:>10s}" + "".join(f"{v:>10d}" for v in matrix[i]))

    save_dir = Path(TRAINING_DIR / "runs" / args.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = save_dir / "confusion_matrix.png"
    plot_confusion_matrix(matrix, class_names, confusion_path)
    print(f"\nConfusion matrix plot saved to {confusion_path}")

    qualitative_path = save_dir / "qualitative_predictions.png"
    build_qualitative_grid(samples, args.samples, args.seed, qualitative_path)
    print(f"Qualitative predictions grid saved to {qualitative_path}")

    report_path = write_report(args.weights, data_dir, class_names, metrics.top1, metrics.top5, matrix, confusion_path, qualitative_path)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
