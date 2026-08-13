"""Fine-tunes a pretrained YOLO classification checkpoint (yolo11n-cls.pt)
on the crop-species dataset assembled by download_classification_dataset.py.

This is transfer learning, not training from scratch, same as train.py.

Example:
    python train_classifier.py --data data/classify --epochs 30 --deploy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TRAINING_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
RUNS_DIR = TRAINING_DIR / "runs"
DEPLOY_TARGET = REPO_ROOT / "models" / "yolo" / "classifier.pt"


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(TRAINING_DIR / "data" / "classify"), help="Folder with train/ and val/ subfolders per class")
    parser.add_argument("--weights", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="auto", help="cpu, mps, a CUDA device index, or auto (default)")
    parser.add_argument("--name", default="classify_finetune", help="Run name under training/runs/")
    parser.add_argument("--deploy", action="store_true", help="Copy the resulting best.pt into models/yolo/classifier.pt afterwards")
    args = parser.parse_args()

    from ultralytics import YOLO

    device = resolve_device(args.device)
    print(f"Fine-tuning {args.weights} on {args.data} for {args.epochs} epochs (device={device})")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR),
        name=args.name,
    )

    # Ultralytics auto-increments the run folder on collision, so ask the
    # trainer where it actually saved rather than reconstructing the path.
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but no checkpoint found at {best}")
    print(f"Best checkpoint: {best}")

    if args.deploy:
        DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, DEPLOY_TARGET)
        print(f"Deployed to {DEPLOY_TARGET}")
    else:
        print(f"Run with --deploy, or manually copy {best} to {DEPLOY_TARGET}, to make the API use this checkpoint.")


if __name__ == "__main__":
    main()
