"""Assembles a 5-class crop-species image classification dataset (Wheat,
Corn, Rice, Soybean, Tomato -- matching the app's crop_type options).

No single dataset covers all five, so this combines five sources -- chosen
specifically so every class shows the actual growing plant/field (canopy,
seedling, or living leaves), not an isolated product photo. An earlier
version of this script mixed leaf-closeup, harvested-grain, and dried-pod
photography across classes; a classifier trained on that hit 100% val
accuracy by learning "which dataset does this look like" instead of any
real species feature (see CLASSIFIER_EVAL_REPORT.md's original run). All
five sources below are in-situ/growing-plant photography instead:

- Wheat:   yinsights8/global-wheat-detection-hf -- aerial wheat field
           canopy (HF mirror of Global Wheat Head Detection). Detection
           boxes are dropped; every image is Wheat.
- Corn:    Project-AgML/MTDC_Maize_Tassels_Detection_Counting_Dataset --
           aerial maize field canopy. Same treatment.
- Rice:    Project-AgML/rice_seedling_classification -- ground-level
           growing rice seedlings in soil. The dataset's own labels
           describe growth pattern (clustered/single), not species, so
           they're ignored -- every image is Rice.
- Soybean: Project-AgML/soybean_leaf_disease_classification -- living
           soybean leaves/pods (its "Root_images" class is excluded since
           roots aren't visually comparable to the other four classes).
- Tomato:  Project-AgML/tomato_disease_detection -- tomato plants
           in-field/greenhouse. Detection/disease metadata is dropped.

Each source is capped at --per-class images (default 400) and split
train/val by --val-fraction (default 0.2).

Example:
    python download_classification_dataset.py --per-class 400
"""

from __future__ import annotations

import argparse
import io
import random
import shutil
from pathlib import Path

from PIL import Image

TRAINING_DIR = Path(__file__).resolve().parent
DATA_DIR = TRAINING_DIR / "data" / "classify"

SOURCES = {
    "Wheat": "yinsights8/global-wheat-detection-hf",
    "Corn": "Project-AgML/MTDC_Maize_Tassels_Detection_Counting_Dataset",
    "Rice": "Project-AgML/rice_seedling_classification",
    "Soybean": "Project-AgML/soybean_leaf_disease_classification",
    "Tomato": "Project-AgML/tomato_disease_detection",
}
SOYBEAN_EXCLUDE_LABEL = "Root_images"


def save_image(image: Image.Image, dest_dir: Path, stem: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(dest_dir / f"{stem}.jpg", quality=90)


def split_and_place(images: list, class_name: str, val_fraction: float, seed: int) -> int:
    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_fraction)) if images else 0
    val_images, train_images = images[:n_val], images[n_val:]
    for i, image in enumerate(train_images):
        save_image(image, DATA_DIR / "train" / class_name, f"{class_name}_{i:04d}")
    for i, image in enumerate(val_images):
        save_image(image, DATA_DIR / "val" / class_name, f"{class_name}_{i:04d}")
    return len(images)


def collect(class_name: str, repo: str, per_class: int) -> list:
    from datasets import load_dataset

    print(f"Streaming {repo} for {per_class} {class_name} images...")
    ds = load_dataset(repo, split="train", streaming=True)
    images = []
    for row in ds:
        if class_name == "Soybean" and ds.features.get("label") is not None:
            label_name = ds.features["label"].names[row["label"]]
            if label_name == SOYBEAN_EXCLUDE_LABEL:
                continue
        # Some sources (e.g. Tomato, several-MP originals) crash `datasets`
        # streaming with a MemoryError -- it materializes a whole Arrow
        # row-group of raw image bytes before yielding individual rows, and
        # a row-group of multi-MB images can be too much at once. Shrink
        # each image immediately so the accumulated list stays small
        # regardless of source resolution.
        image = row["image"]
        image.thumbnail((640, 640))
        images.append(image.convert("RGB"))
        if len(images) >= per_class:
            break
    print(f"Collected {len(images)} images for {class_name}.")
    return images


def retrying_urlretrieve(url: str, dest: Path, attempts: int = 5) -> None:
    """urlretrieve has no retry logic of its own, unlike `datasets`
    streaming's built-in retries -- this environment's connection to HF
    drops mid-download often enough that large shards need the same
    resilience."""
    import urllib.request

    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            dest.unlink(missing_ok=True)
            print(f"  Download attempt {attempt}/{attempts} failed ({exc}); retrying...")
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts") from last_error


def collect_large_image_source(class_name: str, repo: str, shards: list, per_class: int, tmp_dir: Path) -> list:
    """Row-by-row pyarrow reader for sources whose native resolution is too
    large for `datasets` streaming's row-group batching (see `collect`'s
    docstring note) -- downloads shards directly (stopping as soon as
    per_class is reached), then iterates each with batch_size=1 so at most
    one full-resolution image is ever decoded at a time, instead of a whole
    row-group's worth at once."""
    import pyarrow.parquet as pq

    tmp_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for shard in shards:
        if len(images) >= per_class:
            break
        shard_path = tmp_dir / shard
        if not shard_path.exists():
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/data/{shard}"
            print(f"Downloading {shard} for {class_name} (large-image source, row-by-row read)...")
            retrying_urlretrieve(url, shard_path)

        parquet_file = pq.ParquetFile(shard_path)
        for batch in parquet_file.iter_batches(batch_size=1, columns=["image"]):
            image_struct = batch.to_pylist()[0]["image"]
            with Image.open(io.BytesIO(image_struct["bytes"])) as image:
                image.thumbnail((640, 640))
                images.append(image.convert("RGB"))
            if len(images) >= per_class:
                break
    print(f"Collected {len(images)} images for {class_name}.")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=400, help="Max images per class")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True)
    tmp_dir = DATA_DIR / "_tmp_large_image"

    counts = {}
    for class_name, repo in SOURCES.items():
        if class_name == "Tomato":
            shards = [f"train-0000{i}-of-00006.parquet" for i in range(6)]
            images = collect_large_image_source(class_name, repo, shards, args.per_class, tmp_dir)
        else:
            images = collect(class_name, repo, args.per_class)
        counts[class_name] = split_and_place(images, class_name, args.val_fraction, args.seed)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("\n=== Final class counts ===")
    for name, n in counts.items():
        print(f"{name:10s} {n}")
    print(f"\nDataset written to {DATA_DIR} (train/ and val/ subfolders per class).")


if __name__ == "__main__":
    main()
