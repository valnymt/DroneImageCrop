"""Checks whether YOLO's own confidence scores are internally calibrated --
whether "70% confidence" detections are actually right about 70% of the
time on the validation set, not just whether the model finds/ranks boxes
well (mAP/precision/recall already answer that, see EVAL_REPORT.md).

This is deliberately NOT a real-world accuracy validation -- there's no
ground-truth-labeled photo set from outside this project's own training
data to check that against, and pretending otherwise would be exactly the
kind of fabricated confidence this project avoids elsewhere. What this
script CAN honestly answer: given the merged_retrain validation set's own
labels, does the model's stated confidence track its own hit rate on that
set. That's a real, checkable question -- not a substitute for the
open question of real-world accuracy, and the report this produces says so.

Usage:
    python calibration_eval.py --weights runs/merged_retrain/weights/best.pt --data data/yolo/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
RUNS_DIR = TRAINING_DIR / "runs"

IOU_MATCH_THRESHOLD = 0.5
N_BINS = 10


def _load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Returns [(class_id, (x1, y1, x2, y2))] in pixel coordinates -- empty
    list for an image with no objects (a missing label file is legal in
    YOLO format and means exactly that, not an error)."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id, xc, yc, w, h = int(parts[0]), *[float(v) for v in parts[1:]]
        x1, y1 = (xc - w / 2) * img_w, (yc - h / 2) * img_h
        x2, y2 = (xc + w / 2) * img_w, (yc + h / 2) * img_h
        boxes.append((cls_id, (x1, y1, x2, y2)))
    return boxes


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_predictions(model, image_dir: Path, label_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model over every validation image at a near-zero confidence
    floor (so low-confidence predictions are included in the calibration
    check, not pre-filtered out by the deployed threshold) and greedily
    matches each predicted box to an unclaimed same-class ground-truth box
    at IoU >= IOU_MATCH_THRESHOLD -- standard single-match-per-GT detection
    matching, the same principle mAP itself uses.

    Returns (confidences, is_correct) as parallel arrays, one entry per
    predicted box across the whole validation set.
    """
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    confidences: list[float] = []
    is_correct: list[bool] = []

    for image_path in image_paths:
        result = model.predict(source=str(image_path), conf=0.001, verbose=False)[0]
        img_h, img_w = result.orig_shape
        gt_boxes = _load_yolo_labels(label_dir / f"{image_path.stem}.txt", img_w, img_h)
        gt_claimed = [False] * len(gt_boxes)

        preds = sorted(
            zip(result.boxes.conf.tolist(), result.boxes.cls.tolist(), result.boxes.xyxy.tolist()),
            key=lambda p: -p[0],  # highest confidence first -- it gets first claim on a matching GT box
        )
        for conf, cls_id, xyxy in preds:
            best_iou, best_idx = 0.0, -1
            for i, (gt_cls, gt_box) in enumerate(gt_boxes):
                if gt_claimed[i] or int(gt_cls) != int(cls_id):
                    continue
                iou = _iou(tuple(xyxy), gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            correct = best_iou >= IOU_MATCH_THRESHOLD
            if correct:
                gt_claimed[best_idx] = True
            confidences.append(float(conf))
            is_correct.append(correct)

    return np.array(confidences), np.array(is_correct, dtype=bool)


def reliability_bins(confidences: np.ndarray, is_correct: np.ndarray, n_bins: int = N_BINS):
    """Equal-width confidence bins: (bin_low, bin_high, mean_confidence,
    empirical_precision, count). A bin with count=0 is omitted -- an empty
    bin has nothing to say about calibration in that range, and plotting
    it as 0% precision would misrepresent "no data" as "always wrong"."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidences >= lo) & (confidences < hi if hi < 1.0 else confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        rows.append((lo, hi, float(confidences[in_bin].mean()), float(is_correct[in_bin].mean()), count))
    return rows


def expected_calibration_error(rows, total: int) -> float:
    return sum((count / total) * abs(mean_conf - precision) for _, _, mean_conf, precision, count in rows)


def brier_score(confidences: np.ndarray, is_correct: np.ndarray) -> float:
    return float(np.mean((confidences - is_correct.astype(np.float64)) ** 2))


def plot_reliability_diagram(rows, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    mean_confs = [r[2] for r in rows]
    precisions = [r[3] for r in rows]
    counts = [r[4] for r in rows]
    ax.scatter(mean_confs, precisions, s=[20 + c / 2 for c in counts], alpha=0.8, label="Actual (size = # detections)")
    ax.plot(mean_confs, precisions, alpha=0.5)
    ax.set_xlabel("Predicted confidence (mean per bin)")
    ax.set_ylabel("Empirical precision (fraction actually correct)")
    ax.set_title("YOLO confidence calibration -- merged_retrain validation set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Reliability diagram saved to {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default=str(TRAINING_DIR / "data" / "yolo"), help="Dataset root containing images/valid and labels/valid")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--out", default=str(RUNS_DIR / "calibration"))
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    image_dir = Path(args.data_dir) / "images" / args.split
    label_dir = Path(args.data_dir) / "labels" / args.split

    print(f"Running predictions over {image_dir} at conf=0.001 (this covers the full confidence range, not just the deployed threshold)...")
    confidences, is_correct = collect_predictions(model, image_dir, label_dir)
    total = len(confidences)
    print(f"{total} predicted boxes across {len(list(image_dir.glob('*.jpg')))} validation images.")

    rows = reliability_bins(confidences, is_correct)
    ece = expected_calibration_error(rows, total)
    brier = brier_score(confidences, is_correct)

    print("\n=== Calibration bins (confidence -> empirical precision) ===")
    for lo, hi, mean_conf, precision, count in rows:
        print(f"[{lo:.1f}-{hi:.1f})  n={count:5d}  mean_conf={mean_conf:.3f}  empirical_precision={precision:.3f}  gap={abs(mean_conf - precision):.3f}")

    print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
    print(f"Brier score: {brier:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_reliability_diagram(rows, out_dir / "reliability_diagram.png")


if __name__ == "__main__":
    main()
