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
        self.prompt = prompt

    @override
    def generate(self, sampling_parameters: dict[str, any]) -> list[Image.Image]:
        """
        Run standard generation with no conditioning on feedback.

        Example:
            output = generator.generate(
                sampling_parameters={
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "num_images_per_prompt": 4,
                }
            )
        """
        output = self.diffusion_pipeline(prompt=self.prompt, **sampling_parameters)
        return output.images


    @override
    def update(self, winner_index: int) -> None:
        """
        Does nothing — this generator is unconditional.
        """
        pass