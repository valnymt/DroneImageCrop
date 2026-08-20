import logging
from pathlib import Path
from typing import Any

import numpy as np

from app.services.schemas import Detection

logger = logging.getLogger(__name__)

# Resolved relative to this file rather than the working directory, since
# the API is documented to run from backend/ (`uvicorn app.main:app`) while
# models/ lives at the repo root -- a CWD-relative path would never match.
REPO_ROOT = Path(__file__).resolve().parents[3]


class SAMSegmenter:
    """Box-prompted MobileSAM refinement of the Excess Green vegetation mask.

    YOLO's detection boxes are used as box prompts, one per detected plant,
    and the resulting per-plant masks are unioned into the refined mask --
    this is more accurate than prompting SAM with the whole image. Falls
    back to the unrefined Excess Green mask if no checkpoint is installed.
    """

    def __init__(self, checkpoint: Path = REPO_ROOT / "models" / "sam" / "mobile_sam.pt", model_type: str = "vit_t") -> None:
        self.checkpoint = checkpoint
        self.model_type = model_type
        self._predictor: Any | None = None
        self._warned_missing = False

    def _load(self) -> Any | None:
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint.exists():
            if not self._warned_missing:
                logger.warning(
                    "SAM checkpoint not found at %s -- falling back to the unrefined Excess "
                    "Green mask. See models/sam/README.md to enable SAM refinement.",
                    self.checkpoint,
                )
                self._warned_missing = True
            return None

        import torch
        from mobile_sam import SamPredictor, sam_model_registry

        model = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        model.to(device="cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        self._predictor = SamPredictor(model)
        return self._predictor

    def segment_instances(self, image: np.ndarray, detections: list[Detection] | None) -> list[np.ndarray] | None:
        """One boolean mask per detected plant (box-prompted SAM, same
        predictor call `refine` already made) -- None when SAM isn't
        available or there's nothing to prompt it with, never an empty
        list standing in for "no plants" vs. "couldn't segment", which
        would be ambiguous to callers.
        """
        predictor = self._load()
        if predictor is None or not detections:
            return None
        predictor.set_image(image, image_format="BGR")
        instances = []
        for detection in detections:
            box = np.array([detection.x1, detection.y1, detection.x2, detection.y2])
            masks, _, _ = predictor.predict(box=box, multimask_output=False)
            instances.append(masks[0])
        return instances

    def refine(self, image: np.ndarray, initial_mask: np.ndarray, detections: list[Detection] | None = None) -> np.ndarray:
        instances = self.segment_instances(image, detections)
        if instances is None:
            return initial_mask
        return self.union_masks(instances, initial_mask.shape[:2])

    @staticmethod
    def union_masks(instances: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
        union = np.zeros(shape, dtype=np.uint8)
        for mask in instances:
            union[mask] = 255
        return union
