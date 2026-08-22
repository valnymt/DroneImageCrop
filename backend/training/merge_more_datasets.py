"""Downloads three more real, freely-licensed HuggingFace crop/weed
detection datasets (no auth needed) and merges them into
training/data/yolo/{images,labels}/{train,valid,test}, the same pattern
merge_latvia_dataset.py already established -- growing the training set
further and adding real aerial-domain images specifically (see
ImageWeeds_aerial below), since Phase V's own honest finding was that a
~2,400-image expansion wasn't going to solve open-domain generalization on
its own.

Datasets merged (all Project-AgML, CC-BY-4.0):
- maize_weed_detection (500 images): categories {maize, weed} map directly
  onto this project's existing class 1 (crop) and class 2 (weed).
- ImageWeeds_aerial_weed_detection (551 images): genuinely aerial
  photography (unlike the mostly ground-level Latvia set) -- 5 weed
  species, no crop annotations, all mapped onto class 2 (weed). The most
  domain-relevant addition here: real drone/aerial-angle imagery, not just
  more images.
- weed_crop_detection (1,120 images): named crop species (Corn, Soybean,
  Canola, ...) and named weed species (Kochia, Waterhemp, Ragweed, ...) --
  the crop names map onto class 1, weed names onto class 2 (see
  CROP_NAMES/WEED_NAMES below; anything not recognized is skipped, not
  guessed).

Run once from backend/: python training/merge_more_datasets.py
"""

import random
import time

from datasets import load_dataset


def _load_dataset_with_retries(repo_id: str, attempts: int = 4):
    # This network has shown real mid-download drops (both HF's Xet
    # backend and plain HTTP) on large multi-shard datasets -- retrying
    # the whole load_dataset call (not just one file) is the level this
    # library actually exposes; load_dataset itself has no per-shard retry.
    last_exc = None
    for attempt in range(attempts):
        try:
            return load_dataset(repo_id, split="train")
        except Exception as exc:
            last_exc = exc
            wait = min(2 ** (attempt + 2), 60)
            print(f"  load_dataset({repo_id}) failed ({exc.__class__.__name__}), retrying in {wait}s [{attempt + 1}/{attempts}]...")
            time.sleep(wait)
    raise last_exc

TRAINING_DIR = __import__("pathlib").Path(__file__).resolve().parent
YOLO_DIR = TRAINING_DIR / "data" / "yolo"
SPLIT_RATIOS = {"train": 0.7, "valid": 0.2, "test": 0.1}
SEED = 42

# This project's data.yaml class scheme (see data/yolo/data.yaml):
# 0 = weed-crop-aerial (ambiguous/mixed), 1 = crop, 2 = weed.
CROP_CLASS, WEED_CLASS = 1, 2

# weed_crop_detection's category list mixes named crop and named weed
# species in one label space -- classified by real botanical identity, not
# guessed. Anything not in either set is skipped (not silently dropped
# into the nearest bucket).
CROP_NAMES = {"corn", "soybean", "canola", "field pea", "flax", "lentil", "sugar beet", "blackbean"}
WEED_NAMES = {"horseweed", "kochia", "ragweed", "redroot pigweed", "waterhemp"}

DATASETS = [
    {
        "repo_id": "Project-AgML/maize_weed_detection",
        "prefix": "maize",
        "category_map": lambda name: CROP_CLASS if name == "maize" else WEED_CLASS if name == "weed" else None,
    },
    {
        "repo_id": "Project-AgML/ImageWeeds_aerial_weed_detection",
        "prefix": "aerialweed",
        "category_map": lambda name: WEED_CLASS,  # every category in this dataset is a weed species
    },
    {
        "repo_id": "Project-AgML/weed_crop_detection",
        "prefix": "weedcrop",
        "category_map": lambda name: (
            CROP_CLASS if name.lower() in CROP_NAMES else WEED_CLASS if name.lower() in WEED_NAMES else None
        ),
    },
]


def merge_dataset(repo_id: str, prefix: str, category_map) -> None:
    print(f"\nDownloading {repo_id} ...")
    ds = _load_dataset_with_retries(repo_id)
    print(f"Loaded {len(ds)} images.")
    # ds.features["objects"] is a plain dict of feature specs (bbox, categories),
    # not a Sequence wrapper -- the ClassLabel with the actual names lives one
    # level down, inside the "categories" List's own .feature.
    label_names = ds.features["objects"]["categories"].feature.names

    indices = list(range(len(ds)))
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
    box_counts = {"crop": 0, "weed": 0}
    skipped = 0

    for i, row in enumerate(ds):
        split = split_for_index[i]
        image = row["image"].convert("RGB")
        width, height = image.size

        bboxes = row["objects"]["bbox"]
        categories = row["objects"]["categories"]
        lines = []
        for bbox, category in zip(bboxes, categories):
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue
            class_id = category_map(label_names[int(category)])
            if class_id is None:
                continue
            x_center, y_center = (x + w / 2) / width, (y + h / 2) / height
            norm_w, norm_h = w / width, h / height
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            box_counts["crop" if class_id == CROP_CLASS else "weed"] += 1

        if not lines:
            skipped += 1
            continue

        stem = f"{prefix}_{i:04d}"
        image_path = YOLO_DIR / "images" / split / f"{stem}.jpg"
        label_path = YOLO_DIR / "labels" / split / f"{stem}.txt"
        image.save(image_path, format="JPEG", quality=92)
        label_path.write_text("\n".join(lines) + "\n")
        counts[split] += 1

    print(f"Wrote {sum(counts.values())} images ({counts}), skipped {skipped} with no valid mapped boxes.")
    print(f"New box instances: crop={box_counts['crop']}, weed={box_counts['weed']}")


def main() -> None:
    failed = []
    for entry in DATASETS:
        try:
            merge_dataset(entry["repo_id"], entry["prefix"], entry["category_map"])
        except Exception as exc:
            # A transient network failure on one dataset (seen in practice:
            # HF's CDN actively refusing a connection mid-download) shouldn't
            # discard whatever earlier datasets in this run already merged
            # successfully -- surfaced plainly at the end instead, not
            # silently swallowed.
            print(f"\n!! Failed to merge {entry['repo_id']}: {exc}")
            failed.append(entry["repo_id"])
    if failed:
        print(f"\n{len(failed)} dataset(s) failed to merge and were skipped: {failed}")
        print("Re-run this script (it appends, doesn't overwrite) to retry just by fixing the underlying issue -- already-merged datasets will re-download but existing files are simply overwritten with the same content.")


if __name__ == "__main__":
    main()
