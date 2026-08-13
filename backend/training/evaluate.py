"""Evaluate a fine-tuned YOLO checkpoint: mAP50, mAP50-95, precision, recall.

Also saves the confusion matrix and PR-curve plots Ultralytics generates
during validation.

Example:
    python evaluate.py --weights runs/finetune/weights/best.pt --data data/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
RUNS_DIR = TRAINING_DIR / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Path to the fine-tuned checkpoint (best.pt)")
    parser.add_argument("--data", default=str(TRAINING_DIR / "data" / "data.yaml"), help="Path to the dataset's data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", default="eval", help="Run name under training/runs/")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz, project=str(RUNS_DIR), name=args.name)

    print("\n=== Evaluation results ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"\nConfusion matrix, PR curve, and other plots saved under {metrics.save_dir}")


if __name__ == "__main__":
    main()
