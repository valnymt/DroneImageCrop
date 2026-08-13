from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Multiple prompts per class (averaged at load time -- "prompt ensembling",
# a standard CLIP zero-shot technique) rather than one prompt each, to
# reduce sensitivity to any single prompt's exact wording.
CROP_PROMPTS: dict[str, list[str]] = {
    "Wheat": [
        "an aerial drone photo of a wheat field",
        "a top-down view of a wheat crop canopy",
        "a photo of wheat plants growing in a field",
    ],
    "Corn": [
        "an aerial drone photo of a corn field",
        "a top-down view of a maize crop canopy",
        "a photo of corn plants growing in a field",
    ],
    "Rice": [
        "an aerial photo of a rice paddy field",
        "a photo of rice seedlings growing in flooded soil",
        "a top-down view of a rice crop",
    ],
    "Soybean": [
        "an aerial photo of a soybean field",
        "a photo of soybean plants with green leaves and pods",
        "a top-down view of a soybean crop canopy",
    ],
    "Tomato": [
        "a photo of tomato plants growing in a field or greenhouse",
        "a photo of tomato vines with green leaves and fruit",
        "an aerial view of a tomato crop",
    ],
}


class CropClassifier:
    """Zero-shot crop-species classification via CLIP.

    No fine-tuning and no training data of our own -- deliberately, after a
    fine-tuned classifier on assembled scraped data turned out to be
    unreliable (see backend/training/CLASSIFIER_EVAL_REPORT.md: it hit 100%
    validation accuracy by learning "which source dataset" rather than
    "which species", since each class's images came from a differently
    -styled source). CLIP was never trained on our data, so that specific
    failure mode doesn't apply here; accuracy is instead whatever CLIP's
    general web-scale pretraining supports -- see
    backend/training/evaluate_clip_classifier.py for a real measurement of
    that, not an assumption.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._processor: Any | None = None
        self._labels: list[str] = list(CROP_PROMPTS.keys())
        self._text_features: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._model = CLIPModel.from_pretrained(self.model_name)
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
            self._model.eval()

            prompts = [p for label in self._labels for p in CROP_PROMPTS[label]]
            n_per_label = len(next(iter(CROP_PROMPTS.values())))
            with torch.no_grad():
                text_inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
                # This transformers version returns a BaseModelOutputWithPooling,
                # not a plain tensor -- the projected embedding is .pooler_output.
                text_features = self._model.get_text_features(**text_inputs).pooler_output
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_features = text_features.reshape(len(self._labels), n_per_label, -1).mean(dim=1)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            self._text_features = text_features
        return self._model, self._processor

    def classify(self, image: np.ndarray) -> tuple[str, float]:
        """image: BGR numpy array (OpenCV convention, matching the rest of
        the pipeline). Returns (predicted_label, confidence)."""
        import torch

        model, processor = self._load()
        pil_image = Image.fromarray(image[:, :, ::-1])  # BGR -> RGB

        with torch.no_grad():
            image_inputs = processor(images=pil_image, return_tensors="pt")
            image_features = model.get_image_features(**image_inputs).pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * (image_features @ self._text_features.T)
            probs = logits.softmax(dim=-1).squeeze(0)

        best_idx = int(probs.argmax())
        return self._labels[best_idx], float(probs[best_idx])
