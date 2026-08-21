"""Downloads Project-AgML/crop_weed_detection_latvia (HuggingFace, CC-BY-4.0,
no auth needed) and merges it into training/data/yolo/{images,labels}/
{train,valid,test} -- fixes a real gap: the current checkpoint's
validation set has 0 "weed" instances (see EVAL_REPORT.md), so it has
never had a real weed example to learn from. This dataset's categories
(0=weed, 1=crop) map directly onto data.yaml's existing classes 2 (weed)
and 1 (crop) -- no data.yaml change needed.

Run once from backend/: python training/merge_latvia_dataset.py
"""

import io
import random

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

TRAINING_DIR = __import__("pathlib").Path(__file__).resolve().parent
YOLO_DIR = TRAINING_DIR / "data" / "yolo"

# Latvia dataset category -> this project's data.yaml class index.
CATEGORY_MAP = {0: 2, 1: 1}  # weed -> 2, crop -> 1
SPLIT_RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}
SEED = 42


def main() -> None:
    print("Downloading Project-AgML/crop_weed_detection_latvia ...")
    parquet_path = hf_hub_download(
        repo_id="Project-AgML/crop_weed_detection_latvia", filename="data/train-00000-of-00001.parquet", repo_type="dataset",
    )
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} images.")

    indices = list(range(len(df)))
    random.Random(SEED).shuffle(indices)
    n_train = int(len(indices) * SPLIT_RATIOS["train"])
    n_valid = int(len(indices) * SPLIT_RATIOS["valid"])
    split_for_index = {}
    for i in indices[:n_train]:
        split_for_index[i] = "train"
    for i in indices[n_train : n_train + n_valid]:
        split_for_index[i] = "valid"
    for i in indices[n_train + n_valid :]:
        split_for_index[i] = "test"

    counts = {"train": 0, "valid": 0, "test": 0}
    box_counts = {"weed": 0, "crop": 0}
    skipped = 0

    for i in range(len(df)):
        row = df.iloc[i]
        split = split_for_index[i]
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size

        bboxes = row["objects"]["bbox"]
        categories = row["objects"]["categories"]
        lines = []
        for bbox, category in zip(bboxes, categories):
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue
            class_id = CATEGORY_MAP.get(int(category))
            if class_id is None:
                continue
            x_center, y_center = (x + w / 2) / width, (y + h / 2) / height
            norm_w, norm_h = w / width, h / height
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            box_counts["weed" if class_id == 2 else "crop"] += 1

        if not lines:
            skipped += 1
            continue

        stem = f"latvia_{i:04d}"
        image_path = YOLO_DIR / "images" / split / f"{stem}.jpg"
        label_path = YOLO_DIR / "labels" / split / f"{stem}.txt"
        image.save(image_path, format="JPEG", quality=92)
        label_path.write_text("\n".join(lines) + "\n")
        counts[split] += 1

    print(f"Wrote {sum(counts.values())} images ({counts}), skipped {skipped} with no valid boxes.")
    print(f"New box instances added: weed={box_counts['weed']}, crop={box_counts['crop']}")


if __name__ == "__main__":
    main()
