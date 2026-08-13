Place the MobileSAM checkpoint here as `mobile_sam.pt` to enable real
SAM refinement of the vegetation mask. Without it, `SAMSegmenter` logs a
warning and falls back to the unrefined Excess Green mask -- the app still
runs fine on a laptop with no checkpoint downloaded.

## Download

The checkpoint is ~38.8 MB and hosted directly in the MobileSAM repo, no
account or API key needed:

```powershell
curl -L -o mobile_sam.pt https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt
```

(or download that URL in a browser and save it here as `mobile_sam.pt`)

Resulting path: `models/sam/mobile_sam.pt`

## Why MobileSAM, not full SAM

`SAMSegmenter` uses `sam_model_registry["vit_t"]` -- MobileSAM's distilled
ViT-tiny encoder, ~40 MB and fast enough for CPU inference (the MobileSAM
authors report ~3s per image on a Mac i5 CPU). Full SAM's ViT-H checkpoint
is ~2.4 GB and considerably slower on CPU; swap `model_type="vit_h"` and
point `checkpoint` at a ViT-H checkpoint in `sam_segmenter.py` if a GPU is
available and finer masks are wanted.
