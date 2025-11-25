from __future__ import annotations
from typing import Any, override
from .FeedbackAwareGenerator import FeedbackAwareGenerator
from PIL import Image


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
    def generate(self, sampling_parameters: dict[str, any]) -> list[Image.Image]:
        """
        Run standard generation with no conditioning on feedback.

        Uses internal self.prompt. Must be called after initialize(prompt).
        First batch: N fresh latents, returns N images
        Subsequent batches: 1 survivor + N-1 new latents, returns N-1 images
        """
        assert self.prompt is not None, "Must call initialize(prompt) before generate()"
        num_new_images = self.batch_size - (0 if self.is_first_batch else 1)

        output = self.diffusion_pipeline(prompt=self.prompt, num_images_per_prompt=num_new_images, **sampling_parameters)
        return output.images

    @override
    def update(self, winner_index: int) -> None:
        """
        Does nothing — this generator is unconditional.
        """
        pass
