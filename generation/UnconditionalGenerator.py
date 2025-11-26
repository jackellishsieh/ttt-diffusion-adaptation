from __future__ import annotations
from typing import Any, override
from .FeedbackAwareGenerator import FeedbackAwareGenerator
from PIL import Image
import torch


class UnconditionalGenerator(FeedbackAwareGenerator):
    """
    Simplest generator: ignores feedback entirely and performs standard generation.
    """

    @override
    def initialize(self, prompt: str, batch_size: int = 4) -> None:
        """
        Initialize the generator for a new trial.
        """
        super().initialize(prompt, batch_size)

    @override
    def generate(self, sampling_parameters: dict[str, any], seeds: list[int] | None = None) -> list[Image.Image]:
        """
        Run standard generation with no conditioning on feedback.

        Uses internal self.prompt. Must be called after initialize(prompt).
        First batch: N fresh latents, returns N images
        Subsequent batches: 1 survivor + N-1 new latents, returns N-1 images
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
        assert self.prompt is not None, "Must call initialize(prompt) before generate()"

        generators = self._get_random_generators(seeds)

        output = self.diffusion_pipeline(prompt=self.prompt, num_images_per_prompt=self.num_new_images, generator=generators, **sampling_parameters)
        return output.images

    @override
    def update(self, winner_index: int) -> None:
        """
        Does nothing other than increment the round counter — this generator is unconditional.
        """
        super().update(winner_index)
        pass
