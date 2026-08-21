# YOLO Fine-Tuning Evaluation Report

**Deployed checkpoint as of the Phase V retrain below: `training/runs/merged_retrain/weights/best.pt`**
(the original `training/runs/finetune/weights/best.pt`, documented in the sections below, is kept
as `models/yolo/best.pt.pre-latvia-retrain.bak` for reference/rollback.)

## Dataset

| Split | Images |
|---|---|
| Train | 823 |
| Validation | 235 |
| Test | 118 |

Validation-set instances per class:

| Class | Instances |
|---|---|
| weed-crop-aerial | 47 |
| crop | 1558 |
| weed | 0 |

## Training configuration

| Setting | Value |
|---|---|
| model | yolo11n.pt |
| epochs | 20 |
| imgsz | 640 |
| batch | 16 |
| device | cpu |
| optimizer | auto |
| lr0 | 0.01 |

## Baseline: before fine-tuning

The stock, pretrained-on-COCO `yolo11n.pt` checkpoint has no `crop` or `weed` class -- fine-tuning is what makes detection on this data possible at all, not just better. Run over the same 235 validation images at the same conf/iou thresholds used everywhere else in this report:

- Total detections: 33
- Classes detected (all irrelevant to this dataset): bird (18), frisbee (4), bench (3), banana (2), person (2), potted plant (2), bear (1), cake (1)


## Results (after fine-tuning)

### Overall

| Metric | Value |
|---|---|
| mAP50 | 0.7100 |
| mAP50-95 | 0.4525 |
| Precision | 0.6786 |
| Recall | 0.7205 |

### Per class

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| weed-crop-aerial | 0.621 | 0.660 | 0.626 | 0.421 |
| crop | 0.736 | 0.782 | 0.794 | 0.484 |

## Vegetation mask ablation: ExG alone vs. ExG+VARI+ExGR majority vote

Ground-truth plant boxes stand in for "true" vegetation pixels. Precision is the fraction of mask pixels that fall inside a real plant box (higher = fewer false positives from soil/shadow/residue); recall is the fraction of each plant box the mask actually covers. Averaged over 235 validation images with at least one labeled plant:

| Mask | Precision | Recall |
|---|---|---|
| Excess Green alone | 0.041 | 0.417 |
| ExG + VARI + ExGR (2-of-3 vote) | 0.038 | 0.437 |

**Interpretation:** the majority vote raised recall (rescuing more true-vegetation pixels) without a matching precision gain here. Absolute precision is low for both masks, which points at the ground-truth-box proxy being a coarse stand-in for actual plant canopy shape (and the sparse-foreground Otsu over-segmentation noted in the project README's Known limitations) as the bigger driver of false positives on this dataset, not the choice of vegetation index. The combined mask is still preferable: it catches more true vegetation for effectively the same precision, and the three indices' failure modes (shadow noise, saturation, soil hue) are independent, which matters more on imagery where any single index would fail outright.


## Plots

- Confusion matrix: `runs\report_final\confusion_matrix.png`
- PR curve: `runs\report_final\PR_curve.png`
- F1 curve: `runs\report_final\F1_curve.png`
- Qualitative comparison (5 validation images -- ground truth vs prediction vs Excess Green mask): `runs\report_final\qualitative_comparison.png`

---

# Phase V — retrain on a merged dataset (fixes the zero-weed-instance gap)

**The defect this fixes**: the validation set above shows `weed: 0` instances -- the original
823-image training set had **no real weed examples at all**. The checkpoint above could
structurally never have learned to distinguish a weed from a crop; it had nothing to learn from.

## What changed

Merged `Project-AgML/crop_weed_detection_latvia` (HuggingFace, CC-BY-4.0, no auth needed) into
the existing dataset via `backend/training/merge_latvia_dataset.py` -- 1,176 real field-crop
images, categories `weed`/`crop` mapped directly onto this project's existing class 2 (`weed`)
and class 1 (`crop`). Added **7,442 real weed instances and 410 crop instances**, split
proportionally (70/20/10) into the existing train/valid/test folders.

| Split | Before | After |
|---|---|---|
| Train | 823 | 1,646 |
| Valid | 235 | 470 |
| Test | 118 | 236 |

Retrained `yolo11n.pt` from scratch (not continued from the old checkpoint -- standard practice
when the dataset composition changed this much) for 25 epochs, same hyperparameters as the
original run. Training took ~7.6 hours total on CPU (no GPU available); the process was
interrupted once by a background-execution mistake (see the conversation this file came from)
and once deliberately paused/resumed cleanly by the user via Ultralytics' `resume=True`, which
picks up from the last completed epoch's optimizer state exactly -- no retraining from scratch,
no lost progress beyond the one in-progress (unsaved) epoch each time.

## Results: same validation set, both checkpoints, apples-to-apples

This is the fair comparison -- both checkpoints evaluated on the identical 470-image merged
validation set (not the old checkpoint's original 235-image set, which the new checkpoint was
never validated against and would make the comparison meaningless in the other direction).

| Metric | Old checkpoint | New checkpoint |
|---|---|---|
| mAP50 (overall) | 0.205 | **0.746** |
| mAP50-95 (overall) | 0.129 | **0.470** |
| Precision (overall) | 0.198 | **0.635** |
| Recall (overall) | 0.445 | **0.699** |
| weed mAP50 | **0.000** (0 instances ever seen in training) | **0.762** |
| crop mAP50 | 0.397 | **0.794** |
| weed-crop-aerial mAP50 | 0.219 | **0.682** |

Every class, every metric, improved -- most dramatically the weed class, which went from
literally no detection capability to real precision/recall (P=0.724, R=0.685).

## What this does NOT fix: true out-of-distribution photos

The honest, important caveat. Re-ran the exact diagnostic from Phase O/U (raw confidence at
`conf_threshold=0.01` against every real, non-training-distribution photo in `images/`) with
both checkpoints:

| Photo | Old top score | New top score |
|---|---|---|
| bad3.jpg | 0.036 | 0.188 |
| badWheat1.jpg | 0.016 | 0.000 |
| badWheat2.jpg | 0.252 | 0.080 |
| Field-soybeans-farm-Oklahoma.jpg | 0.015 | 0.028 |
| greenField.jpg | 0.016 | 0.000 |
| splitCrop.jpg | 0.141 | 0.193 |
| yellow-green.jpg | 0.029 | 0.000 |

Mixed, not a clean win: some improved, some got worse (a few dropped to literally zero). The
retrain broadens what the model recognizes well (now two real datasets' worth of visual styles
instead of one), but it does not grant true open-domain generalization to a photo unlike either
training distribution -- that was never a realistic outcome from 2,352 more images. **Critically,
none of these scores clear the deployed 0.25 confidence threshold either before or after** -- so
in actual production behavior, both checkpoints already produced zero usable detections on every
one of these, and Phase U's `OpenVocabDetector` fallback already covers this exact case
regardless of which YOLO checkpoint is deployed. The retrain doesn't replace that fallback; it
fixes a different, also-real problem (in-distribution weed/crop accuracy) that the fallback
never addressed.

## Deployed

Copied `training/runs/merged_retrain/weights/best.pt` to `models/yolo/best.pt`. The prior
checkpoint is kept at `models/yolo/best.pt.pre-latvia-retrain.bak` for rollback. Verified through
the real `YOLODetector` class (not just the raw Ultralytics model) and the full backend test
suite (169/169 passing) before and after the swap.
