from pathlib import Path
from typing import Any

import numpy as np

from app.services.schemas import Detection


class YOLODetector:
    def __init__(self, weights: Path = Path("models/yolo/best.pt")) -> None:
        self.weights = weights
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            # Supply a pretrained agriculture checkpoint at this path. No training occurs.
            self._model = YOLO(str(self.weights if self.weights.exists() else "yolo11n.pt"))
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        result = self._load().predict(image, conf=0.25, iou=0.5, verbose=False)[0]
        names = result.names
        return [
            Detection(
                x1=float(box.xyxy[0][0]), y1=float(box.xyxy[0][1]),
                x2=float(box.xyxy[0][2]), y2=float(box.xyxy[0][3]),
                confidence=float(box.conf[0]), label=names[int(box.cls[0])],
            )
            for box in result.boxes
        ]
