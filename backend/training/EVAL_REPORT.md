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

## What calibration evidence actually exists (and what still doesn't)

`AnalysisResult.confidence_score`, `texture_uniformity_score`, and `flight_comparator`'s
`inlier_ratio` are all self-reported by their own models -- none of them have ever been checked
against real held-out accuracy. That's still true after this section. There is no ground-truth
-labeled photo set from outside this project's own training/validation data to check real-world
accuracy against, and building one is out of scope here -- so this section does **not** claim to
validate real-world calibration. What it does answer, honestly: on the merged_retrain validation
set's own labels, does YOLO's stated confidence track its own actual hit rate. That's a real,
checkable question distinct from mAP/precision/recall (which measure ranking and coverage, not
whether "70% confidence" boxes are right about 70% of the time) -- and a different, narrower
claim than "this model is accurate in the field."

**Method** (`backend/training/calibration_eval.py`): ran the deployed checkpoint over all 470
validation images at `conf=0.001` (below the deployed 0.39 threshold, so low-confidence
detections are included in the check rather than pre-filtered out), greedily matched each
predicted box to an unclaimed same-class ground-truth box at IoU >= 0.5 (the same matching
principle mAP itself uses), then binned all 36,273 resulting (confidence, correct/incorrect)
pairs into 10 equal-width confidence bins.

| Confidence bin | n | Mean confidence | Empirical precision | Gap |
|---|---|---|---|---|
| 0.0-0.1 | 30,377 | 0.010 | 0.007 | 0.003 |
| 0.1-0.2 | 1,219 | 0.144 | 0.141 | 0.003 |
| 0.2-0.3 | 863 | 0.248 | 0.297 | 0.049 |
| 0.3-0.4 | 768 | 0.349 | 0.415 | 0.066 |
| 0.4-0.5 | 755 | 0.449 | 0.564 | 0.116 |
| 0.5-0.6 | 763 | 0.550 | 0.706 | 0.156 |
| 0.6-0.7 | 733 | 0.648 | 0.804 | 0.156 |
| 0.7-0.8 | 545 | 0.747 | 0.850 | 0.103 |
| 0.8-0.9 | 219 | 0.837 | 0.886 | 0.049 |
| 0.9-1.0 | 31 | 0.926 | 1.000 | 0.074 |

**Expected Calibration Error (ECE): 0.016.** **Brier score: 0.0365.** Both look excellent at
face value, but that's mostly the huge 0.0-0.1 bin (30,377 of 36,273 total predictions --
overwhelmingly correct near-zero-confidence background boxes) pulling the weighted average down;
it says little about calibration in the range that actually matters for deployment.

**The finding that matters**: around and above the deployed `CONF_THRESHOLD = 0.39` (see
`yolo_detector.py`), the model is **consistently underconfident**, not overconfident -- gaps of
0.10-0.16 in the 0.4-0.7 confidence range, all in the same direction (empirical precision higher
than stated confidence; see `runs/calibration/reliability_diagram.png`, where the curve sits
above the diagonal the whole way through the deployed operating range). A detection the model
calls "55% confident" is actually right about 71% of the time on this validation set. This is the
safe direction to be wrong in -- a system that trusts its own stated confidence too little, not
too much -- but it is still a real miscalibration, not the "well-calibrated" result a flat
ECE/Brier number alone would suggest.

**Honest limits of this measurement, stated plainly:**
- This is calibration against the model's *own* validation set, drawn from the same two merged
  training distributions (see Phase V above). It says nothing about calibration on a genuinely
  out-of-distribution photo -- Phase U's `OpenVocabDetector` fallback exists for exactly that
  case and has no calibration measurement of its own here.
- `texture_uniformity_score` and `inlier_ratio` have no equivalent check in this report and
  still don't -- there's no analogous ground-truth "correct texture pattern" or "correct
  alignment" label to check them against, unlike YOLO's boxes which have real annotated
  ground truth to match against. That gap is unresolved, not overlooked.
- No downstream code currently corrects for the underconfidence found here (e.g. no
  temperature-scaling or isotonic recalibration applied to `confidence_score`) -- this section
  is a measurement, not a fix. `AnalysisResult.confidence_score` remains YOLO's raw self-reported
  number.

## Held-out test-set check (never used in any tuning decision until now)

Every number above -- mAP, the confidence threshold recalibration, the calibration check -- was
computed against the **validation** split. `data/yolo/data.yaml` has always defined a separate
**test** split (236 images) that had never actually been evaluated anywhere in this project until
now (confirmed by grep -- no prior mention of `split="test"` or `images/test` in this file).
Running the deployed `merged_retrain` checkpoint against it for the first time:

| Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation (used for every decision above) | 0.746 | 0.470 | 0.635 | 0.699 |
| **Test (never touched before this check)** | **0.743** | **0.456** | **0.671** | **0.726** |

Essentially identical. This is real evidence -- not a re-run of the same numbers -- that the
checkpoint selection and the `CONF_THRESHOLD=0.39` recalibration weren't quietly overfit to the
specific images in the validation set. **Still an important caveat, stated plainly**: this test
split is drawn from the same source datasets as train/valid (see Phase V and the "more datasets"
section below) -- it's genuinely held-out data, but not independent data from a different
collection effort. It answers "did we overfit to validation" (no), not "does this generalize to
the real world" (still unmeasured, see the calibration section's honest limits above).
