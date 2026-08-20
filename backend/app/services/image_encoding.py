import base64

import cv2
import numpy as np


def encode_png_data_url(image: np.ndarray) -> str:
    """BGR numpy array -> a `data:image/png;base64,...` string, ready to
    drop directly into an <img src>."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image as PNG.")
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")
