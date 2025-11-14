"""
This file contains methods for generating a score for a given image, given relevant scoring information.
The generated score is a surrogate for a hypothetical user's absolute preference of the image.

Design:
- Input: image (PIL.Image or list[PIL.Image]) and a score_rubric dict (user-provided)
- Output: single scalar float score or list of float scores

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
import ImageReward as reward

image_reward_model = None


def score_image(images: Image.Image | list[Image.Image], rubric: dict[str, any]) -> float | list[float]:
    """
    Dispatch to concrete scoring function based on rubric['type'].
    Returns a single scalar score (higher is better) for a single image,
    or a list of scores for multiple images.
    
    Args:
        image: A single PIL Image or a list of PIL Images
        rubric: Dictionary containing scoring configuration
        
    Returns:
        A single float score or a list of float scores
    """
    t = rubric.get("type", "").lower()

    match (t):
        case "brightness":
            return _score_brightness(images, rubric)
        case "random":
            return _score_random(images, rubric)
        case "ImageReward":
            return _score_image_reward(images, rubric)
        case _:
            raise ValueError(f"Unknown rubric type: {t}")


def _score_random(images: Image.Image | list[Image.Image], rubric: dict[str, any]) -> float | list[float]:
    """
    Returns a random score between 0 and 1, independent of the image.

    Args:
        image: A single PIL Image or a list of PIL Images
        rubric (dict[str, any]): Empty dict

    Returns:
        A single float score or a list of float scores
    """
    import random
    
    # Check if input is a list of images
    is_list = isinstance(images, list)
    return [random.random() for _ in images] if is_list else random.random()


def _score_brightness(images: Image.Image | list[Image.Image], rubric: dict[str, any]) -> float | list[float]:
    """
    Returns a score based on the brightness of the image.

    Args:
        image: A single PIL Image or a list of PIL Images
        rubric (dict[str, any]): Empty dict

    Returns:
        A single float score or a list of float scores
    """
    # Check if input is a list of images
    is_list = isinstance(images, list)
    images = images if is_list else [images]
    
    # Calculate brightness for all images
    return [float(np.asarray(img.convert("RGB")).astype(np.float32) / 255.0).mean() for img in images] if is_list else float(np.asarray(images.convert("RGB")).astype(np.float32) / 255.0).mean()


def _score_image_reward(image: Image.Image | list[Image.Image], rubric: dict[str, any]) -> float | list[float]:
    """
    Returns a score based on the image reward of the image.
    
    Args:
        image: A single PIL Image or a list of PIL Images
        rubric (dict[str, any]): Dictionary containing "prompt" key
        
    Returns:
        A single float score or a list of float scores
    """
    global image_reward_model
    
    if image_reward_model is None:
        print("Loading ImageReward model...")
        image_reward_model = reward.load("ImageReward-v1.0")
        print("ImageReward model loaded")
    
    # Check if input is a list of images
    is_list = isinstance(image, list)
    images = image if is_list else [image]
    
    # Score all images
    scores = [image_reward_model.score(rubric["prompt"], img) for img in images]
    
    # Return single score or list based on input type
    return scores if is_list else scores[0]


# Test stub
if __name__ == "__main__":
    # Test with single image
    im = Image.new("RGB", (64, 64), color=(128, 128, 128))
    
    rubric = {}

    print("Testing with single image:")
    for type in ["brightness", "random"]:
        rubric["type"] = type
        print(f"{type} score:", score_image(im, rubric))
    
    # Test with multiple images
    print("\nTesting with multiple images:")
    im1 = Image.new("RGB", (64, 64), color=(50, 50, 50))   # darker
    im2 = Image.new("RGB", (64, 64), color=(200, 200, 200))  # brighter
    images = [im1, im2]
    
    for type in ["brightness", "random"]:
        rubric["type"] = type
        scores = score_image(images, rubric)
        print(f"{type} scores:", scores)