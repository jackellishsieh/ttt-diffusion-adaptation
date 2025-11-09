"""
Utilities for
- Saving images
- Reproducibility
"""

import os
from PIL import Image
import numpy as np
import random
import torch

def save_image(image: Image.Image, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path)


def save_image_batch(images: list[Image.Image], base_dir: str, name_prefix: str):
    os.makedirs(base_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(images):
        p = os.path.join(base_dir, f"{name_prefix}_{i}.png")
        img.save(p)
        paths.append(p)
    return paths


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
