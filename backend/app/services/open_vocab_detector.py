import logging
from typing import Any

import numpy as np
from PIL import Image

from app.services.schemas import Detection

logger = logging.getLogger(__name__)

# Confirmed empirically (see HANDOFF.md's Phase O): the fine-tuned YOLO
# checkpoint (trained on ~823 images of one Roboflow dataset's visual
# style) produces near-zero raw confidence -- not "borderline misses",
# genuinely nothing -- on any photo stylistically unlike its training set.
# More training data is the real fix for that; this is a fallback for when
# that isn't available yet. OWL-ViT was never fine-tuned on this project's
# narrow dataset at all -- it inherits CLIP's web-scale pretraining (the
# same reasoning already used for crop-type classification in
# crop_classifier.py), so it has a broad, if less precise, prior for "what
# does a plant look like" that generalizes far better outside that one
# training distribution, at the cost of being slower and less accurate
# than the fine-tuned model when the fine-tuned model actually applies.
MODEL_NAME = "google/owlvit-base-patch32"
PLANT_PROMPTS = ["a plant", "a small green plant", "a crop seedling", "a leaf"]
DEFAULT_SCORE_THRESHOLD = 0.1
FALLBACK_LABEL = "plant (general detector)"
# Confirmed empirically: OWL-ViT's scores collapse toward zero on a full
# aerial photo where individual plants are a tiny fraction of the frame
# (0.03 top score on a 1600x1067 field photo) but recover once cropped
# down to a tile where a plant actually occupies a meaningful fraction of
# the input (0.15 top score, several real detections, on a 400x400 crop
# of the same photo) -- the same small-object-at-full-resolution problem
# SAHI tiling already solves for the fine-tuned YOLO path, not a
# coincidence: both are just object detectors losing small instances when
# downscaled too far.
TILE_SIZE = 400
TILE_OVERLAP_RATIO = 0.2


class OpenVocabDetector:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor

            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name)
            self._model.eval()
        return self._model, self._processor

    def detect(self, image_bgr: np.ndarray, score_threshold: float = DEFAULT_SCORE_THRESHOLD) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        if max(h, w) <= TILE_SIZE:
            return _suppress_overlaps(self._detect_tile(image_bgr, score_threshold))

        detections: list[Detection] = []
        stride = int(TILE_SIZE * (1 - TILE_OVERLAP_RATIO))
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2, x2 = min(y + TILE_SIZE, h), min(x + TILE_SIZE, w)
                y1, x1 = max(0, y2 - TILE_SIZE), max(0, x2 - TILE_SIZE)  # keeps edge tiles full-sized instead of truncated
                tile = image_bgr[y1:y2, x1:x2]
                for detection in self._detect_tile(tile, score_threshold):
                    detections.append(Detection(
                        x1=detection.x1 + x1, y1=detection.y1 + y1,
                        x2=detection.x2 + x1, y2=detection.y2 + y1,
                        confidence=detection.confidence, label=detection.label,
                    ))
                if x2 >= w:
                    break
            if y2 >= h:
                break
        return _suppress_overlaps(detections)

    def _detect_tile(self, image_bgr: np.ndarray, score_threshold: float) -> list[Detection]:
        import torch

        model, processor = self._load()
        pil_image = Image.fromarray(image_bgr[:, :, ::-1])  # BGR -> RGB

        inputs = processor(text=[PLANT_PROMPTS], images=pil_image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]])  # (height, width)
        results = processor.post_process_grounded_object_detection(
            outputs, threshold=score_threshold, target_sizes=target_sizes,
        )[0]

        return [
            Detection(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2), confidence=float(score), label=FALLBACK_LABEL)
            for (x1, y1, x2, y2), score in zip(results["boxes"].tolist(), results["scores"].tolist())
        ]


def _iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return intersection / (area_a + area_b - intersection)


def _suppress_overlaps(detections: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """OWL-ViT is queried with several overlapping plant-describing prompts
    (a plant / a seedling / a leaf / ...) so the same real plant commonly
    fires multiple boxes -- greedy NMS by confidence collapses those back
    down to one box per plant, the same role IOU_THRESHOLD/postprocess
    plays for the fine-tuned YOLO path."""
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for candidate in ordered:
        if not any(_iou(candidate, existing) > iou_threshold for existing in kept):
            kept.append(candidate)
    return kept
