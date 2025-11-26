"""
This file containsa wrapper for image2text generation that conditions on feedback.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from diffusers import DiffusionPipeline
from PIL import Image
import torch

class FeedbackAwareGenerator(ABC):
    """
    Abstract base class for text-to-image generators that can condition on feedback.

    Args:
        diffusion_pipeline : DiffusionPipeline
            A Hugging Face DiffusionPipeline instance (e.g., StableDiffusion3Pipeline).
        seed_safe: bool
            Whether to force the generator to accept seeds for generation
        **kwargs : dict
            Optional hyperparameters or configuration options
            that subclasses may use for conditional generation.
    """

    def __init__(self, diffusion_pipeline: DiffusionPipeline, seed_safe: bool = False, **kwargs: Any) -> None:
        self.diffusion_pipeline = diffusion_pipeline
        self.seed_safe = seed_safe
        self.hyperparameters = kwargs

    @abstractmethod
    def initialize(self, prompt: str, batch_size: int = 4) -> None:
        """
        Initialize the generator for a new trial.
        """
        self.prompt = prompt
        self.batch_size = batch_size

        self.num_new_images = batch_size # begins at batch_size, decreases by 1 each round
        self.rounds = 0 # how many times update has been called

    @abstractmethod
    def generate(self, sampling_parameters: dict[str, Any], seeds: list[int] | None = None) -> list[Image.Image]:
        """
        Generate images given a text prompt and sampling parameters.

        Args:
            sampling_parameters : dict[str, Any]
                dictionary of keyword arguments forwarded to the diffusion pipeline
                (e.g., num_inference_steps, guidance_scale, etc.)
                Should not include prompt or num_images_per_prompt, which are handled by the generator.
        seeds : list[int] | None
            List of seeds to use for generation. If None, the generator will use its own internal seed.
            Optional if seed_safe is False; required if seed_safe is True.
            Must be the same length as the number of new images to generate.
        Returns:
            list[Image.Image]
            The full output from the DiffusionPipeline call, including images and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, winner_index: int) -> None:
        """
        Update the internal state based on feedback information.

        Args:
        winner_index : int
            Index of the winner in the current batch
        """
        self.rounds += 1
        self.num_new_images = self.batch_size - 1
        # Override in subclasses to update internal state
        pass

    def _get_random_generators(self, seeds: list[int] | None = None) -> list[torch.Generator]:
        """
        Returns a list of random generators, (optionally provided seeds)
        Args:
            seeds : list[int] | None
                List of seeds to use for generation. If None, the generator will use its own internal seed.
                Optional if seed_safe is False; required if seed_safe is True.
                Must be the same length as the number of new images to generate.
        Returns:
            list[torch.Generator]
            A list of random generators from the seeds.
        """
        if seeds is not None:
            assert len(seeds) == self.num_new_images, f"Seeds must be the same length as the number of new images {self.num_new_images} to generate"
            return [torch.Generator(device="cpu").manual_seed(s) for s in seeds]
        else:
            assert not self.seed_safe, "Seeds must be provided if seed_safe is True"
            return [torch.Generator(device="cpu")] * self.num_new_images

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"<{cls_name} with {self.diffusion_pipeline.__class__.__name__}>"
