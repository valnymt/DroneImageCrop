"""Fine-tune a pretrained YOLO checkpoint on a downloaded agriculture dataset.

This fine-tunes a COCO-pretrained Ultralytics checkpoint via transfer
learning -- it does not train a model from scratch.

Examples:
    python train.py --data data/data.yaml
    python train.py --data data/data.yaml --weights yolo11s.pt --epochs 80 --deploy
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parent
BACKEND_DIR = TRAINING_DIR.parent
REPO_ROOT = BACKEND_DIR.parent
RUNS_DIR = TRAINING_DIR / "runs"
DEPLOY_TARGET = REPO_ROOT / "models" / "yolo" / "best.pt"


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
    parser.add_argument("--data", default=str(TRAINING_DIR / "data" / "data.yaml"), help="Path to the dataset's data.yaml")
    parser.add_argument("--weights", default="yolo11n.pt", help="Pretrained checkpoint to fine-tune (yolo11n.pt or yolo11s.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="cpu, mps, a CUDA device index, or auto (default)")
    parser.add_argument("--name", default="finetune", help="Run name under training/runs/")
    parser.add_argument("--deploy", action="store_true", help="Copy the resulting best.pt into models/yolo/best.pt afterwards")
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

    # Ultralytics auto-increments the run folder (e.g. finetune2, finetune3)
    # if `name` already exists, so ask the trainer where it actually saved
    # rather than reconstructing the path from args.name.
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"Training finished but no checkpoint found at {best}")
    print(f"Best checkpoint: {best}")

    if args.deploy:
        DEPLOY_TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, DEPLOY_TARGET)
        print(f"Deployed to {DEPLOY_TARGET} -- YOLODetector will pick it up automatically.")
    else:
        print(f"Run with --deploy, or manually copy {best} to {DEPLOY_TARGET}, to make the API use this checkpoint.")


if __name__ == "__main__":
    main()
