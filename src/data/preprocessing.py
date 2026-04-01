from __future__ import annotations

import numpy as np


def normalize_images(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0
