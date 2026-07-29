import numpy as np


class SAMSegmenter:
    """Optional pretrained SAM refinement with an OpenCV-mask fallback."""

    def refine(self, image: np.ndarray, initial_mask: np.ndarray) -> np.ndarray:
        # The initial Excess Green mask keeps laptop inference deterministic.
        # A local MobileSAM/SAM checkpoint can be injected here for refinement.
        return initial_mask
