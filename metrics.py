"""
This file contains a class for computing and tracking metrics throughout test-time training.

Metrics recorded per image:
- prompt_id, round, batch_idx, image_idx_in_batch, score (from scoring), clip_sim (if available), brightness, width, height, chosen (bool)
"""

from typing import Any
import pandas as pd
import numpy as np
from PIL import Image
import os


def image_metrics_dict(
    prompt_id: str,
    rnd: int,
    batch_idx: int,
    image: Image.Image,
    score: float,
    chosen: bool,
    extra: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Create a dictionary of image metrics.
    """
    d = {
        "prompt_id": prompt_id, # the prompt identifier
        "round": int(rnd), # the round number
        "batch_idx": int(batch_idx), # the image index within the batch
        "score": float(score), # the image's score
        "width": int(image.width), # the image's width
        "height": int(image.height), # the image's height
        "mean_pixel": float(np.asarray(image.convert("RGB")).mean() / 255.0), # the image's mean pixel value (as an example metric)
        "chosen": chosen, # whether the image was chosen
    }
    if extra:
        d.update(extra)
    return d

class MetricsTracker:
    def __init__(self):
        self._rows: list[dict[str, Any]] = []

    def append(self, row: dict[str, Any]):
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if len(self._rows) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self._rows)

    def save_csv(self, path: str):
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)

    def save_parquet(self, path: str):
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_parquet(path, index=False)


# Test stub
if __name__ == "__main__":
    tracker = MetricsTracker()
    image = Image.new("RGB", (32, 32), color=(200, 10, 10))

    tracker.append(image_metrics_dict(
        prompt_id="p0",
        rnd=0,
        batch_idx=0,
        image_idx=0,
        image=image,
        score=0.45,
        extra={"clip_sim": 0.45, "chosen": False}
    ))
    print(tracker.to_dataframe())