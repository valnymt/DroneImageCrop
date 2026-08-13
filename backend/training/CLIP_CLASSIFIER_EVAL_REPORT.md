# CLIP Zero-Shot Crop Classifier Evaluation

Model: `openai/clip-vit-base-patch32` -- no fine-tuning, no training data of ours.
Prompts: see `app/services/crop_classifier.py`'s `CROP_PROMPTS`.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.6888 (270/392) |

### Per-class accuracy

| Class | Correct / Total | Accuracy |
|---|---|---|
| Corn | 43 / 72 | 0.597 |
| Rice | 27 / 80 | 0.338 |
| Soybean | 80 / 80 | 1.000 |
| Tomato | 80 / 80 | 1.000 |
| Wheat | 40 / 80 | 0.500 |

## Plots

- Confusion matrix: `runs\clip_zeroshot_eval\confusion_matrix.png`
