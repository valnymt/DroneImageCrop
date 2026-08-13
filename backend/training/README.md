# YOLO fine-tuning

Fine-tunes a pretrained Ultralytics checkpoint (`yolo11n.pt` / `yolo11s.pt`)
on an agriculture dataset — standard transfer learning, not training from
scratch. Lives under `backend/training/`, separate from the running API in
`backend/app/`; nothing here is imported by the inference service.

`app/services/yolo_detector.py` already checks for `models/yolo/best.pt` and
falls back to the stock COCO checkpoint if it's missing, so dropping a
fine-tuned checkpoint in that path is all the API needs to pick it up.

## 0. Install

From `backend/`, with the same virtualenv used for the API:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r training/requirements.txt
```

## 1. Pick a dataset

Any Roboflow Universe project in YOLO format works, matching one of the
`crop_type` options in the UI. Good starting points:

- "Global Wheat Head Detection" (wheat head boxes)
- "PlantDoc" (leaf disease / plant detection)
- Weed-crop detection projects on Roboflow Universe

Get an API key from https://app.roboflow.com/settings/api.

## 2. Download the dataset

```powershell
python training/download_dataset.py --url https://universe.roboflow.com/<workspace>/<project>/dataset/<version> --api-key <YOUR_KEY>
```

Or by workspace/project/version instead of a URL:

```powershell
python training/download_dataset.py --workspace <ws> --project <proj> --version 1 --api-key <YOUR_KEY>
```

Set `ROBOFLOW_API_KEY` in your environment to skip `--api-key`. This writes
YOLO-format images/labels and a `data.yaml` into `backend/training/data/`.

## 3. Fine-tune

```powershell
python training/train.py --data training/data/data.yaml --weights yolo11n.pt --epochs 50 --imgsz 640 --device auto
```

`--device auto` picks CUDA, then Apple MPS, then CPU. On CPU only, keep
`--epochs` in the 20-50 range and use `yolo11n.pt` (the smallest checkpoint)
— a small dataset for 30 epochs is slow but overnight-survivable on CPU, and
minutes on a free Colab GPU (see below). Checkpoints and logs land under
`backend/training/runs/finetune/`.

## 4. Evaluate

```powershell
python training/evaluate.py --weights training/runs/finetune/weights/best.pt --data training/data/data.yaml
```

Prints mAP50, mAP50-95, precision, and recall, and saves the confusion
matrix and PR-curve plots Ultralytics generates automatically under
`backend/training/runs/eval/`.

## 5. Full evaluation report

For a report/slide-deck-ready writeup instead of just console metrics:

```powershell
python training/report.py --weights training/runs/finetune/weights/best.pt --data training/data/data.yaml
```

Beyond what `evaluate.py` prints, this also:

- breaks mAP50/mAP50-95/precision/recall down per class
- runs the stock, pre-fine-tuning checkpoint (`--baseline-weights`, default
  `yolo11n.pt`) over the same validation images, to show concretely what
  fine-tuning fixed -- COCO has no crop/weed classes, so whatever it
  detects is off-topic (`--skip-baseline` to skip)
- quantifies the Phase 4 vegetation-mask claim: precision/recall of the
  Excess-Green-alone mask vs. the ExG+VARI+ExGR majority vote, using
  ground-truth plant boxes as a proxy for true vegetation pixels
  (`--skip-ablation` to skip)
- builds a side-by-side qualitative figure (ground truth boxes vs predicted
  boxes vs the production Excess Green vegetation mask from
  `app/services/opencv_processor.py`) for a handful of validation images

and writes `training/EVAL_REPORT.md` with dataset size, the training config
pulled from the run's `args.yaml`, and all of the above in markdown tables.

## 6. Deploy the checkpoint

Either re-run training with `--deploy` to copy `best.pt` into
`models/yolo/best.pt` automatically:

```powershell
python training/train.py --data training/data/data.yaml --deploy
```

or copy it manually:

```powershell
Copy-Item training/runs/finetune/weights/best.pt ..\models\yolo\best.pt
```

Restart the API (`uvicorn app.main:app --reload`) so `YOLODetector` picks up
the new weights.

## Crop-type classification (exploratory -- not wired into the app)

Two approaches to auto-detecting crop species from a photo were built and
evaluated; neither is used by the running API (the UI's crop-type dropdown
stays manual). See the root README's "Evaluation" section for the full
result and why. Reproduce either:

**Fine-tuned classifier** (scored 100% -- confirmed to be dataset-source
fingerprinting, not real species detection; kept for the record, not
recommended):

```powershell
python training/download_classification_dataset.py --per-class 400
python training/train_classifier.py --data training/data/classify --epochs 30 --deploy
python training/report_classifier.py --weights training/runs/classify_finetune/weights/best.pt --data training/data/classify
```

**Zero-shot CLIP** (`app/services/crop_classifier.py`, no training step --
the more trustworthy of the two, scored 68.9%):

```powershell
python training/evaluate_clip_classifier.py --data training/data/classify/val
```

## No GPU? Use Colab

Run the same steps in a notebook on Colab's free GPU, then bring
`best.pt` back to `models/yolo/best.pt` in this repo:

```python
!pip install ultralytics roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
dataset = rf.workspace("<workspace>").project("<project>").version(1).download("yolov11")

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data=f"{dataset.location}/data.yaml", epochs=50, imgsz=640)
model.val(data=f"{dataset.location}/data.yaml")
# download runs/detect/train/weights/best.pt from the Colab file browser
```
