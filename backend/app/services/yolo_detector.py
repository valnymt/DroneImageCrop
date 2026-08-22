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

# Empirically recalibrated for the Phase V merged-dataset checkpoint, not
# Ultralytics' generic 0.25 default: its own F1-confidence curve
# (backend/training/runs/merged_retrain/F1_curve.png) peaks at 0.394, not
# 0.25 -- the old value was tuned (implicitly, by never being revisited)
# for the pre-retrain checkpoint's very different confidence distribution.
# Leaving it at 0.25 after the retrain would silently keep accepting
# lower-confidence boxes than the new model's own precision/recall
# tradeoff actually supports.
CONF_THRESHOLD = 0.39
IOU_THRESHOLD = 0.5
# The fine-tuned checkpoint was trained exclusively on 640x640 exports (see
# backend/training/EVAL_REPORT.md) -- tiling at this exact size means each
# slice matches the training resolution instead of being resized down
# further, which is what let small plants vanish on large aerial uploads.
TILE_SIZE = 640
TILE_OVERLAP_RATIO = 0.2


class YOLODetector:
    def __init__(self, weights: Path = REPO_ROOT / "models" / "yolo" / "best.pt") -> None:
        self.weights = weights
        self._model: Any | None = None
        self._sahi_model: Any | None = None

    def _weights_path(self) -> str:
        # Supply a pretrained agriculture checkpoint at this path. No training occurs.
        return str(self.weights if self.weights.exists() else "yolo11n.pt")

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._weights_path())
        return self._model

    def _load_sahi(self) -> Any:
        if self._sahi_model is None:
            from sahi import AutoDetectionModel
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=self._weights_path(),
                confidence_threshold=CONF_THRESHOLD,
            )
        return self._sahi_model

    def detect(self, image: np.ndarray, conf_threshold: float = CONF_THRESHOLD) -> list[Detection]:
        h, w = image.shape[:2]
        # Small images already match (or are close to) the model's native
        # input size, so a single pass is both sufficient and cheaper --
        # tiling only pays off once downscaling to TILE_SIZE would actually
        # shrink content.
        if max(h, w) <= TILE_SIZE:
            return self._detect_single(image, conf_threshold)
        return self._detect_tiled(image, conf_threshold)

    def _detect_single(self, image: np.ndarray, conf_threshold: float) -> list[Detection]:
        h, w = image.shape[:2]
        result = self._load().predict(image, conf=conf_threshold, iou=IOU_THRESHOLD, verbose=False)[0]
        names = result.names
        detections = [
            Detection(
                x1=float(box.xyxy[0][0]), y1=float(box.xyxy[0][1]),
                x2=float(box.xyxy[0][2]), y2=float(box.xyxy[0][3]),
                confidence=float(box.conf[0]), label=names[int(box.cls[0])],
            )
            for box in result.boxes
        ]
        logger.info("Single-pass detection on %dx%d image: %d detections", w, h, len(detections))
        return detections

    def _detect_tiled(self, image: np.ndarray, conf_threshold: float) -> list[Detection]:
        from sahi.predict import get_sliced_prediction
        from sahi.slicing import get_slice_bboxes

        h, w = image.shape[:2]
        result = get_sliced_prediction(
            image,
            self._load_sahi(),
            slice_height=TILE_SIZE,
            slice_width=TILE_SIZE,
            overlap_height_ratio=TILE_OVERLAP_RATIO,
            overlap_width_ratio=TILE_OVERLAP_RATIO,
            postprocess_type="NMS",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=IOU_THRESHOLD,
            confidence_threshold=conf_threshold,
            verbose=0,
        )
        detections = [
            Detection(
                x1=op.bbox.to_xyxy()[0], y1=op.bbox.to_xyxy()[1],
                x2=op.bbox.to_xyxy()[2], y2=op.bbox.to_xyxy()[3],
                confidence=op.score.value, label=op.category.name,
            )
            for op in result.object_prediction_list
        ]
        n_tiles = len(get_slice_bboxes(
            h, w, slice_height=TILE_SIZE, slice_width=TILE_SIZE,
            overlap_height_ratio=TILE_OVERLAP_RATIO, overlap_width_ratio=TILE_OVERLAP_RATIO,
        ))
        logger.info(
            "Tiled detection on %dx%d image: %d tiles, %d merged detections",
            w, h, n_tiles, len(detections),
        )
        return detections
