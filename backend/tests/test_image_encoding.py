import base64

import numpy as np

from app.services.image_encoding import encode_png_data_url


class TestEncodePngDataUrl:
    def test_returns_a_data_url_prefix(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        url = encode_png_data_url(image)
        assert url.startswith("data:image/png;base64,")

    def test_decodes_back_to_a_valid_png(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        url = encode_png_data_url(image)
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded.startswith(b"\x89PNG\r\n\x1a\n")
