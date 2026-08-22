"""Evaluates the deployed YOLO checkpoint against a dataset it has NEVER
seen during training or validation, from a different source entirely --
Voxel51/CottonWeedDet12 (Michigan State University field sites, iPhone/
smartphone photos of cotton fields, 12 weed species; cc-by-nc-4.0, used
here for evaluation only, not redistributed).

Every other accuracy number in this project (EVAL_REPORT.md's mAP/
precision/recall, calibration_eval.py's ECE/Brier) is measured against a
split of the SAME merged training distribution -- a real, honest number,
but one that structurally cannot answer "how does this model do on data
from an entirely different collection effort". This script answers that
question directly, with a dataset that was deliberately kept out of any
merge/training step specifically so it could serve as a true external
holdout.

This is still not "real-world ground truth from an actual farm deployment"
-- it's a different research dataset, still curated, still photographed
with intent. It is a genuinely stronger generalization signal than a
same-distribution validation split, not a substitute for field deployment
data this project doesn't have.

Usage:
    python external_holdout_eval.py --weights ../../models/yolo/best.pt --n-images 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

TRAINING_DIR = Path(__file__).resolve().parent
CACHE_DIR = TRAINING_DIR / "data" / "external_holdout" / "cottonweeddet12"
REPO_ID = "Voxel51/CottonWeedDet12"
IOU_MATCH_THRESHOLD = 0.5
SEED = 7

# This project's data.yaml class scheme (see data/yolo/data.yaml) --
# CottonWeedDet12 has no crop annotations at all (it's a weed-only
# detection dataset), so every one of its ground-truth boxes maps to
# "some kind of plant detection was expected here", matched against ANY
# of this project's three classes -- not just class 2 "weed" -- since
# class 0 "weed-crop-aerial" is itself an ambiguous merged class this
# model was trained with. The per-predicted-class breakdown this script
# prints separately is what's actually diagnostic about class confusion,
# not the headline match rate.
ANY_PLANT_CLASSES = {0, 1, 2}
CLASS_NAMES = {0: "weed-crop-aerial", 1: "crop", 2: "weed"}


def _download_with_retries(repo_id: str, filename: str, attempts: int = 5):
    import time

    from huggingface_hub import hf_hub_download

    last_exc = None
    for attempt in range(attempts):
        try:
            return hf_hub_download(repo_id, filename, repo_type="dataset")
        except Exception as exc:  # this network has shown real mid-transfer drops on both the Xet and plain-HTTP backends -- not hypothetical
            last_exc = exc
            wait = min(2**attempt, 30)
            print(f"    download failed ({exc.__class__.__name__}), retrying in {wait}s [{attempt + 1}/{attempts}]...")
            time.sleep(wait)
    raise last_exc


def _download_subset(n_images: int) -> list[dict]:
    samples_path = _download_with_retries(REPO_ID, "samples.json")
    all_samples = json.load(open(samples_path, encoding="utf-8"))["samples"]
    # Only keep samples with at least one real weed box -- an image with
    # zero ground-truth objects can't contribute to a detection accuracy
    # measurement either way.
    with_boxes = [s for s in all_samples if s["ground_truth"]["detections"]]
    random.Random(SEED).shuffle(with_boxes)
    subset = with_boxes[: n_images + 20]  # a small buffer -- a handful of individual files may fail all retries and get skipped below

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    skipped = 0
    print(f"Downloading up to {n_images} images from {REPO_ID} (this only pulls the images actually used, not the full ~29GB repo)...")
    for sample in subset:
        if len(records) >= n_images:
            break
        try:
            local_path = _download_with_retries(REPO_ID, sample["filepath"])
        except Exception as exc:
            print(f"  giving up on {sample['filepath']} after retries ({exc.__class__.__name__}) -- skipping this one image, not the whole run")
            skipped += 1
            continue
        records.append({"image_path": local_path, "detections": sample["ground_truth"]["detections"]})
        if len(records) % 50 == 0:
            print(f"  {len(records)}/{n_images}")
    if skipped:
        print(f"Skipped {skipped} image(s) that failed to download after retries.")
    return records


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate(model, records: list[dict], conf_threshold: float) -> dict:
    tp = fp = fn = 0
    predicted_class_counts = {0: 0, 1: 0, 2: 0}

    for record in records:
        result = model.predict(source=record["image_path"], conf=conf_threshold, verbose=False)[0]
        img_h, img_w = result.orig_shape

        gt_boxes = []
        for det in record["detections"]:
            x, y, w, h = det["bounding_box"]
            gt_boxes.append((x * img_w, y * img_h, (x + w) * img_w, (y + h) * img_h))
        gt_claimed = [False] * len(gt_boxes)

        preds = sorted(
            zip(result.boxes.conf.tolist(), result.boxes.cls.tolist(), result.boxes.xyxy.tolist()),
            key=lambda p: -p[0],
        )
        for conf, cls_id, xyxy in preds:
            cls_id = int(cls_id)
            if cls_id not in ANY_PLANT_CLASSES:
                continue
            predicted_class_counts[cls_id] += 1
            best_iou, best_idx = 0.0, -1
            for i, gt_box in enumerate(gt_boxes):
                if gt_claimed[i]:
                    continue
                iou = _iou(tuple(xyxy), gt_box)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= IOU_MATCH_THRESHOLD:
                gt_claimed[best_idx] = True
                tp += 1
            else:
                fp += 1
        fn += gt_claimed.count(False)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "images": len(records), "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "predicted_class_counts": predicted_class_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--n-images", type=int, default=500, help="How many CottonWeedDet12 images to evaluate on (capped to bound download/runtime, not to cherry-pick a favorable subset -- selection is a seeded random shuffle).")
    parser.add_argument("--conf", type=float, default=0.39, help="Should match the deployed CONF_THRESHOLD in yolo_detector.py.")
    args = parser.parse_args()

    from ultralytics import YOLO

    records = _download_subset(args.n_images)
    model = YOLO(args.weights)

    print(f"\nRunning {args.weights} against {len(records)} CottonWeedDet12 images at conf={args.conf} (the deployed threshold)...")
    at_deployed = evaluate(model, records, args.conf)
    print(f"\nRunning again at conf=0.01 (recall ceiling -- what the model could find if the threshold weren't filtering) ...")
    at_low_conf = evaluate(model, records, 0.01)

    print("\n=== External holdout: Voxel51/CottonWeedDet12 (never used in training or validation) ===")
    for label, m in (("At deployed threshold (0.39)", at_deployed), ("At conf=0.01 (recall ceiling)", at_low_conf)):
        print(f"\n{label}:")
        print(f"  images={m['images']}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
        print(f"  precision={m['precision']:.3f}  recall={m['recall']:.3f}  f1={m['f1']:.3f}")
        counts = m["predicted_class_counts"]
        total = sum(counts.values()) or 1
        breakdown = ", ".join(f"{CLASS_NAMES[k]}={v} ({100*v/total:.0f}%)" for k, v in counts.items())
        print(f"  predicted-class breakdown: {breakdown}")

    out_path = CACHE_DIR / "results.json"
    out_path.write_text(json.dumps({"at_deployed_threshold": at_deployed, "at_low_conf": at_low_conf, "weights": args.weights}, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
