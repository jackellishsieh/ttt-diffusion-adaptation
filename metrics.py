"""
This file contains a class for computing and tracking metrics throughout feedback rounds and trials.
Each entry (row) corresponds to a single generated image.

Metrics recorded per image:
- prompt_id, round, batch_idx, image_idx_in_batch, score (from scoring), clip_sim (if available), brightness, width, height, chosen (bool)
"""

from __future__ import annotations
from typing import Any, Callable, list, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import os

ImageMetric = Callable[
    [Image.Image, dict[str, Any]], Any
]  # accepts the image and extra information, and returns a scalar value


def compute_mean_pixel(image: Image.image, extra_info: dict[str, Any]) -> float:
    """
    Compute the mean pixel value of the image.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0
    return float(arr.mean())


class MetricsTracker:
    """
    Tracks per-image metrics and metadata across prompts, trials, and rounds.
    """

    """
    Registry of metric names to metric functions.
    """
    DEFAULT_METRIC_REGISTRY: dict[str, ImageMetric] = {
        "mean_pixel": compute_mean_pixel,
    }

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []  # list of dictionaries, each one is a single record
        self._metric_funcs: dict[str, ImageMetric] = {}

        for metric_name, metric in self.DEFAULT_METRIC_REGISTRY.items():
            self.register_metric(metric_name, metric)

    def register_metric(self, metric_name: str, metric: ImageMetric) -> None:
        """
        Register a metric, that accepts (image, extra_info) and returns a scalar (or otherwise serializable) value.
        """
        if not callable(metric):
            raise TypeError("Metric function must be callable.")
        self._metric_funcs[metric_name] = metric

    def log(
        self,
        prompt_id: str,
        trial_id: str,
        round: int,
        image_idx: int,
        image_path: str,
        user_score: float,
        image: Image.Image,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log one image entry.

        Args:
            prompt_id: Identifier for the prompt.
            trial_id: Identifier for the trial (independent runs)
            round: Feedback round index.
            image_idx: Index of image within batch for this round.
            image_path: Path where image is saved.
            user_score: Scalar score assigned to image.
            image: PIL Image object (not stored; used only for metric computation).
            extra_info: Optional dict of extra fields to include.
        """
        record: dict[str, Any] = {
            "prompt_id": prompt_id,
            "trial_id": trial_id,
            "round": int(round),
            "image_idx": int(image_idx),
            "image_path": str(image_path),
            "user_score": float(user_score),
            "chosen": None,  # not yet determined
        }
        # Add any user-specified extra info
        if extra_info:
            record.update(extra_info)

        # Compute registered metrics
        for name, func in self._metric_funcs.items():
            try:
                value = func(image, extra_info or {})
            except Exception as e:
                value = np.nan
                print(f"[MetricsTracker] Warning: metric '{name}' failed: {e}")
            record[name] = value

        # Append to record list
        self._records.append(record)

    def mark_chosen(
        self,
        prompt_id: str,
        trial_id: str,
        round: int,
        chosen_image_idx: int,  # index of the chosen image within the batch for this round
    ) -> None:
        """
        Mark which image was chosen for the given prompt + trial + round.
        Automatically sets all other images in that (prompt, trial, round) to chosen=False.
        """
        for record in self._records:
            if (
                record["prompt_id"] == prompt_id
                and record["trial_id"] == trial_id
                and record["round"] == round
            ):
                record["chosen"] = record["image_idx"] == chosen_image_idx

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert records to a pandas DataFrame.
        """
        df = pd.DataFrame(self._records)
        if not df.empty and "chosen" in df.columns:
            df["chosen"] = df["chosen"].astype("boolean")
        return df

    def save(self, filepath: str) -> None:
        """
        Save logged metrics to CSV or Parquet based on file extension.
        """
        if not self._records:
            print("[MetricsTracker] No records to save.")
            return
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        df = self.to_dataframe()
        ext = Path(filepath).suffix.lower()
        if ext in [".parquet", ".pq"]:
            df.to_parquet(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        print(f"[MetricsTracker] Saved {len(df)} records to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "MetricsTracker":
        """
        Load metrics from a CSV or Parquet file into a new tracker.
        Metric registry will be empty (must be re-registered if needed).
        """
        ext = Path(filepath).suffix.lower()
        if ext in [".parquet", ".pq"]:
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath)

        tracker = cls()
        tracker._records = df.to_dict(orient="records")
        return tracker

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"<MetricsTracker n_records={len(self._records)}, n_metrics={len(self._metric_funcs)}>"
