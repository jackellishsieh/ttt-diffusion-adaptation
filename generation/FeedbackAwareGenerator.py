"""
This file containsa wrapper for image2text generation that conditions on feedback.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
from diffusers import DiffusionPipeline
from PIL import Image


class FeedbackAwareGenerator(ABC):
    """
    Abstract base class for generators that can condition on feedback.

    Args:
        diffusion_pipeline : DiffusionPipeline
            A Hugging Face DiffusionPipeline instance (e.g., StableDiffusion3Pipeline).
        **kwargs : dict
            Optional hyperparameters or configuration options
            that subclasses may use for conditional generation.
    """

    def __init__(self, diffusion_pipeline: DiffusionPipeline, **kwargs: Any) -> None:
        self.diffusion_pipeline = diffusion_pipeline
        self.hyperparameters = kwargs
        self.feedback_state: dict[str, Any] = {}

    @abstractmethod
    def initialize(self, prompt: str, batch_size: int = 4) -> None:
        """
        Initialize the generator for a new trial.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, sampling_parameters: dict[str, Any]) -> list[Image.Image]:
        """
        Generate images given a text prompt and sampling parameters.

        Args:
            sampling_parameters : dict[str, Any]
                dictionary of keyword arguments forwarded to the diffusion pipeline
                (e.g., num_inference_steps, guidance_scale, etc.)
                Should not include prompt or num_images_per_prompt, which are handled by the generator.

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
        raise NotImplementedError

   
    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        return f"<{cls_name} with {self.diffusion_pipeline.__class__.__name__}>"
