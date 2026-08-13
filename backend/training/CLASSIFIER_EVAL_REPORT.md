# Crop-Species Classifier Evaluation Report

Checkpoint: `training/runs/classify_finetune/weights/best.pt`

## Dataset

Assembled by `download_classification_dataset.py` from five sources (one per class) -- see that script's docstring for exactly which dataset backs which class.

| Class | Train images | Val images |
|---|---|---|
| Corn | 289 | 72 |
| Rice | 320 | 80 |
| Soybean | 320 | 80 |
| Tomato | 320 | 80 |
| Wheat | 320 | 80 |

## Results

| Metric | Value |
|---|---|
| Top-1 accuracy | 1.0000 |
| Top-5 accuracy | 1.0000 |

### Per-class accuracy

| Class | Correct / Total | Accuracy |
|---|---|---|
| Corn | 72 / 72 | 1.000 |
| Rice | 80 / 80 | 1.000 |
| Soybean | 80 / 80 | 1.000 |
| Tomato | 80 / 80 | 1.000 |
| Wheat | 80 / 80 | 1.000 |

## Interpretation -- read this before trusting the accuracy number

Zero confusion across the whole validation set is a red flag here, not a clean win, and this held even after a deliberate second attempt to fix it. The first version of this dataset mixed photography styles across classes (grain kernels on black, studio leaf photos, one field photo) and hit 100% -- unsurprising, since the background alone gives it away. The dataset was then rebuilt so every class is genuine in-situ growing-plant/field photography (see `download_classification_dataset.py`'s docstring). **It still hit 100%.** That rules out background/style as the sole explanation and points to something deeper: each class still comes from a different underlying source, and a CNN can separate sources via artifacts invisible to a human looking at the photo -- JPEG compression signature, sensor noise, exact resolution/aspect-ratio distribution -- without learning any real species feature at all. This is the well-documented "dataset bias" / "name the dataset" phenomenon in computer vision (Torralba & Efros, *Unbiased Look at Dataset Bias*, 2011): the only way to actually rule it out is a test set drawn from a *different* source per class than training, which no amount of image curation within these five sources can provide. This number should **not** be read as "the classifier reliably identifies crop species." For comparison, `evaluate_clip_classifier.py`'s zero-shot CLIP result on this same data (68.9% accuracy, errors concentrated exactly where a human would expect -- Wheat/Corn/Rice confused with each other, Soybean/Tomato perfect) never saw any of these training sources and can't exploit this shortcut, which is precisely why it scores lower *and* is the more trustworthy number of the two. See `CLIP_CLASSIFIER_EVAL_REPORT.md`.

## Plots

- Confusion matrix: `runs\classify_report\confusion_matrix.png`
- Qualitative sample predictions: `runs\classify_report\qualitative_predictions.png`
