# AgriSight — Drone Crop Intelligence

AgriSight is a university-grade computer-vision project for analyzing aerial farm imagery. It uses pretrained models as a base: OpenCV for enhancement and multi-index RGB vegetation analysis, Ultralytics YOLO for detection (fine-tuned via transfer learning, not trained from scratch), MobileSAM for box-prompted segmentation refinement, and a transparent rule-based yield estimator.

## Features

- Responsive dashboard with portfolio metrics, crop-health trend and recent fields
- Drone image upload with crop type, field area and yield calibration
- Interactive analysis result with plant count, density, coverage, health and harvest estimate, backed by a real computer-vision pipeline (not mocked)
- Analysis history persisted in a database, surviving a server restart
- Typed FastAPI service with separated computer-vision modules
- A YOLO fine-tuning workflow (dataset download, training, evaluation, reporting) separate from the running inference service

## Architecture

The frontend and the computer-vision API are two separate processes that talk over HTTP; both need to be running for the full workflow (see "Run it" below).

```text
React/Vinext interface (this repo, app/)
        |
        |── POST /api/analyze ──────────────► FastAPI API (backend/)
        |                                          |
        |                                  OpenCV preprocessing ── YOLO detection (fine-tuned)
        |                                          |                       |
        |                                  vegetation mask ───── MobileSAM refinement
        |                                          |                    (box-prompted)
        |                                   yield estimation
        |                                          |
        |◄───────────── AnalysisResult ────────────┘
        |
        └── POST/GET /api/analyses ─────────► Next.js API route (app/api/analyses/)
                                                          |
                                                  Drizzle ORM ── D1 (SQLite) database
```

The FastAPI service (`backend/`) only runs the computer-vision pipeline — it
does not persist anything itself. After a successful `/api/analyze` call,
the frontend persists the result through the Next.js app's own `/api/analyses`
route (`app/api/analyses/route.ts`), backed by a D1 database defined in
`db/schema.ts`. History and the dashboard read that same route, so results
survive restarting either process.

## Run it

Both the frontend and the backend need to be running for uploads to actually
analyze images; the frontend alone will show clear loading/error states
(including "start the backend") rather than fabricated results if the API
isn't reachable.

### 1. Frontend

```powershell
npm install
npm run db:generate         # first time only, or after editing db/schema.ts
$env:WRANGLER_LOG_PATH = ".wrangler/wrangler.log"
npx vinext dev               # starts the local D1 database on first run
```

`npm run dev` (rather than `npx vinext dev` directly) does not work on
Windows -- npm always runs package.json scripts through `cmd.exe`, which
chokes on that script's `VAR=value command` syntax.

Open the local URL printed by the dev server (`http://localhost:3000` by
default). In a second terminal, once the dev server has started at least
once (that's what creates the local D1 state directory), apply the
migration:

```powershell
npm run db:migrate:local
```

This only needs to be re-run after adding a new migration with
`db:generate`; the local D1 database persists between `npx vinext dev`
restarts.

### 2. Backend (computer-vision API)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.
`POST /api/analyze` accepts `image`, `crop_type`, `field_size_hectares`, and
`average_yield_per_plant_kg`.

Tests (`backend/tests/`) cover the vegetation-index math, the yield formula,
result validation, and the API contract (the CV models are mocked so these
run in seconds, not minutes):

```powershell
pip install -r requirements-dev.txt
pytest
```

### 3. Reproducing the fine-tuning

The YOLO checkpoint the API uses can be reproduced from scratch: download an
agriculture dataset, fine-tune `yolo11n.pt` via transfer learning, evaluate,
and deploy the checkpoint. Full step-by-step instructions, including a
no-GPU/Colab path, are in [`backend/training/README.md`](backend/training/README.md).
Short version:

```powershell
cd backend
python training/download_dataset.py --url <roboflow-universe-url> --api-key <KEY>
python training/train.py --data training/data/data.yaml --epochs 20 --deploy
python training/report.py --weights training/runs/finetune/weights/best.pt --data training/data/data.yaml
```

`--deploy` copies the resulting `best.pt` to `models/yolo/best.pt`, which
`YOLODetector` already checks for and falls back from if it's missing.
`report.py` writes [`backend/training/EVAL_REPORT.md`](backend/training/EVAL_REPORT.md)
with the metrics summarized below.

## Model setup

Put an agriculture-specific fine-tuned YOLO checkpoint at `models/yolo/best.pt`
(see "Reproducing the fine-tuning" above). The service otherwise downloads
the small pretrained YOLO11n checkpoint on first analysis, which has no
crop/weed classes and will detect nothing useful. Put a MobileSAM checkpoint
at `models/sam/mobile_sam.pt` (see [`models/sam/README.md`](models/sam/README.md))
to enable box-prompted segmentation refinement; without it, `SAMSegmenter`
logs a warning and falls back to the unrefined Excess Green mask.

## Evaluation

<!--
Generated by `python backend/training/report.py`. Re-run after any new
fine-tuning run and refresh this section from the new
backend/training/EVAL_REPORT.md.
-->

`yolo11n.pt` fine-tuned for 20 epochs (imgsz 640, batch 16, CPU, ~2.9h) on
[`Francesco/weed-crop-aerial`](https://huggingface.co/datasets/Francesco/weed-crop-aerial)
(823 train / 235 validation / 118 test images, 2 real classes: `crop`,
`weed`). Full detail, plots, and the qualitative/ablation figures are in
[`backend/training/EVAL_REPORT.md`](backend/training/EVAL_REPORT.md).

**Before vs. after fine-tuning**, same 235 validation images: the stock
COCO-pretrained checkpoint (no crop/weed classes at all) produced 33
detections, all irrelevant (`bird`, `frisbee`, `bench`, `banana`, `person`,
`potted plant`, `bear`, `cake` -- it called a patch of soil a banana). The
fine-tuned checkpoint:

| Metric | Value |
|---|---|
| mAP50 | 0.710 |
| mAP50-95 | 0.453 |
| Precision | 0.679 |
| Recall | 0.721 |

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| crop | 0.736 | 0.782 | 0.794 | 0.484 |
| weed-crop-aerial* | 0.621 | 0.660 | 0.626 | 0.421 |

\* an unused placeholder class inherited from the dataset's export format,
not a real category (see `backend/training/EVAL_REPORT.md`).

**Vegetation-mask ablation** (Phase 4): does combining Excess Green, VARI,
and ExGR via a 2-of-3 majority vote actually beat Excess Green alone? Using
ground-truth plant boxes as a proxy for true vegetation pixels, averaged
over 235 validation images: ExG alone scores 0.041 precision / 0.417
recall; the combined mask scores 0.038 precision / 0.437 recall. The result
is honestly mixed, not a clean win: the combined mask trades a negligible
amount of precision for higher recall. Absolute precision is low for both
because ground-truth boxes are a coarse proxy for actual canopy shape, and
because of the sparse-foreground Otsu over-segmentation described below --
that appears to be a bigger driver of false positives here than which
vegetation index is used. The combined mask is kept regardless: its three
indices fail in different, independent ways (shadow noise, saturation,
soil hue), which matters most on exactly the imagery where any single
index would fail outright.

**Crop-type auto-detection (exploratory, not shipped).** Investigated
automatically pre-filling the "Crop type" field from the uploaded photo
instead of requiring manual selection. Two approaches were built and
evaluated head-to-head:

1. A fine-tuned `yolo11n-cls` classifier (`backend/training/train_classifier.py`)
   on a 5-class dataset assembled from public sources (one per species --
   see `backend/training/download_classification_dataset.py`). It scored
   **100% validation accuracy on every attempt**, including after a full
   dataset rebuild specifically to rule out the obvious explanation
   (mismatched photography style per class). It still hit 100%, and every
   single prediction carried exactly 100.00% confidence -- the textbook
   signature of a classifier that learned to fingerprint *which dataset*
   an image came from (compression artifacts, sensor noise, framing)
   rather than any real species feature. This is the well-documented
   "dataset bias" problem (Torralba & Efros, *Unbiased Look at Dataset
   Bias*, 2011), and it isn't fixable by curating these particular sources
   further -- see `backend/training/CLASSIFIER_EVAL_REPORT.md`.
2. Zero-shot classification via a pretrained CLIP model
   (`backend/app/services/crop_classifier.py`, no fine-tuning, no training
   data of ours), evaluated on the same images:
   [**68.9% accuracy**](backend/training/CLIP_CLASSIFIER_EVAL_REPORT.md),
   with errors concentrated exactly where a human would expect them --
   Wheat/Corn/Rice (visually similar grass-family canopies) confused with
   each other, Soybean/Tomato (visually distinctive) both 100% correct.

The lower, honest number is the trustworthy one: CLIP never saw these
training sources, so it cannot exploit the shortcut the fine-tuned
classifier found. **This feature was not wired into the UI** -- the
manual crop-type dropdown remains the only path -- specifically because
neither result clears the bar for a confident auto-fill yet. Reported here
as a real finding, not a shipped capability.

## Known limitations

These are deliberate, disclosed trade-offs, not oversights — they keep the
pipeline explainable and reproducible on a laptop rather than chasing
benchmark numbers with opaque methods.

- **RGB-only health screening can't diagnose *why* vegetation looks stressed.**
  Excess Green / VARI / ExGR (see `backend/app/services/opencv_processor.py`)
  detect low or discolored canopy, but color alone cannot distinguish
  drought, disease, pest damage, natural crop maturity, harvest residue, or
  simple shadow — that requires multispectral/NDVI imagery or a supervised
  disease classifier, neither of which this project has training data or a
  sensor for. The UI states this explicitly next to every health score.
- **The yield estimate is a transparent formula, not a learned model:**
  `plant count × average yield per plant × crop factor × condition factor`
  (see `backend/app/services/yield_estimator.py`). This is a strength for a
  report or demo — every input is visible and auditable, and it degrades
  gracefully with no training data — but it will not match the accuracy of
  a regression model trained on real harvest records, which this project
  does not have access to.
- **YOLO detection quality depends entirely on whether a fine-tuned
  checkpoint is deployed.** Without one at `models/yolo/best.pt`, the API
  falls back to a stock COCO-pretrained checkpoint with no crop or weed
  classes, and detections will be near-zero or wrong (see "Model setup").
- **The Excess Green vegetation mask over-segments on sparse-foreground
  imagery.** On aerial closeups of mostly bare soil with small, scattered
  seedlings, Otsu thresholding can split on soil shadow/texture instead of
  isolating the true (rare) vegetation pixels — visible directly in the
  qualitative comparison figure `report.py` generates. The learned YOLO
  detector is more reliable on exactly this kind of image; the RGB mask is
  best read as a coarse coverage signal, not a precise segmentation.
