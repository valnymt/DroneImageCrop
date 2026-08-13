# YOLO Fine-Tuning Evaluation Report

Checkpoint: `training/runs/finetune/weights/best.pt`

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
