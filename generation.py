"""
This file contains our implemented methods for generating images with a diffusion pipeline, potentially conditioning on feedback.

Interface:
- generate_batch(pipeline, prompt, batch_size, guidance_scale=None, generator=None, **kwargs) -> List[PIL.Image]

Notes:
- We assume `pipeline` is an instantiated DiffusionPipeline (e.g., StableDiffusionPipeline).
- This function is intentionally simple and stateless: it takes the pipeline + inputs and returns images.
- Later you can wrap the pipeline to inject conditioning info or states.
"""

from typing import Any
from PIL import Image
import torch
from diffusers import DiffusionPipeline


def generate_unconditional(
    pipeline: DiffusionPipeline,
    prompt: str,
    batch_size: int = 4,
    num_inference_steps: int = None,
    guidance_scale: float = None,
    generator: torch.Generator = None,
    height: int = 512,
    width: int = 512,
    feedback: dict[str, Any] = None
) -> list[Image.Image]:
    """
    Generate a batch of images for `prompt`.


    - pipeline: DiffusionPipeline (must support __call__)
    - prompt: text prompt
    - batch_size: number of images
    - generator: torch.Generator or list of generators (optional) to control randomness
    """
    # The pipeline call typically accepts `prompt` (or list of prompts), guidance_scale, num_inference_steps, generator,
    # and optionally height/width.
    prompts = [prompt] * batch_size
    call_kwargs = dict(prompt=prompts, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
    
    if generator is not None:
        call_kwargs["generator"] = generator
    if height is not None:
        call_kwargs["height"] = height
    if width is not None:
        call_kwargs["width"] = width
    if num_inference_steps is not None:
        call_kwargs["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        call_kwargs["guidance_scale"] = guidance_scale

    outputs = pipeline(**call_kwargs)
    images = outputs.images if hasattr(outputs, "images") else outputs
    return list(images)
