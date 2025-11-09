"""
This file contains methods for generating a score for a given image, given relevant scoring information.
The generated score is a surrogate for a hypothetical user's absolute preference of the image.

Design:
- Input: image (PIL.Image) and a score_rubric dict (user-provided)
- Output: single scalar float score

Score rubric format examples:
- For CLIP-based scoring:
  rubric = {
      "type": "clip",
      "text": "A photorealistic painting of a cat",
      "model_name": "openai/clip-vit-base-patch32",  # optional
      "weight": 1.0
  }
"""
from PIL import Image
import numpy as np


def score_image(image: Image.Image, rubric: dict[str, any]) -> float:
    """
    Dispatch to concrete scoring function based on rubric['type'].
    Returns a single scalar score (higher is better).
    """
    t = rubric.get("type", "").lower()

    match (t):
        case "brightness":
            return _score_brightness(image, rubric)
        case "random":
            return _score_random(image, rubric)
        case _:
            raise ValueError(f"Unknown rubric type: {t}")


def _score_random(image: Image.Image, rubric: dict[str, any]) -> float:
    """
    Returns a random score between 0 and 1, independent of the image.

    Args:
        image (Image.Image): The image to score.
        rubric (dict[str, any]): Empty dict

    Returns:
        float: A random score between 0 and 1.
    """
    import random
    return random.random()


def _score_brightness(image: Image.Image, rubric: dict[str, any]) -> float:
    """
    Returns a score based on the brightness of the image.

    Args:
        image (Image.Image): The image to score.
        rubric (dict[str, any]): Empty dict

    Returns:
        float: A score based on the brightness of the image.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float32) / 255.0 # mean brightness of the image
    return float(arr.mean())