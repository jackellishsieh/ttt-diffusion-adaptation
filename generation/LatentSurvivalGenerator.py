from __future__ import annotations
from typing import Any, override
from .FeedbackAwareGenerator import FeedbackAwareGenerator
from PIL import Image
from diffusers import DiffusionPipeline
import torch

Latent = torch.Tensor

class LatentSurvivalGenerator(FeedbackAwareGenerator):
    """
    Generator with latent survival batches:
    - initialize(prompt, batch_size) sets internal prompt and resets state
    - first generate() produces full batch_size new latents
    - update(winner_index) selects the survivor
    - subsequent generate() produces batch_size - 1 new latents, with survivor in slot 0
    """

    def __init__(
        self,
        diffusion_pipeline: DiffusionPipeline,
        alpha: float = 0.5,
        gamma: float = 1.0,  # default: full replacement
        seed_safe: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(diffusion_pipeline, seed_safe, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        assert 0 <= self.alpha <= 1, "alpha must be between 0 and 1"
        assert 0 <= self.gamma <= 1, "gamma must be between 0 and 1"

        self.running_latent = None  # the running latent
        self.last_winner_latent = None  # the latent of the last winner
        self.batch_latents = None  # this stores the latents for the current batch, only used to track the winner

    def initialize(self, prompt: str, batch_size: int = 4) -> None:
        """
        Initialize for a new trial
        """
        super().initialize(prompt, batch_size)

        self.running_latent = None
        self.last_winner_latent = None
        self.batch_latents = None

    # -------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------
    @override
    def generate(self, sampling_parameters: dict[str, Any], seeds: list[int] | None = None) -> list[Image.Image]:
        """
        Uses internal self.prompt. Must be called after initialize(prompt).
        First batch: N fresh latents, returns N images
        Subsequent batches: 1 survivor + N-1 new latents, returns N-1 images
        """
        assert self.prompt is not None, "Must call initialize(prompt) before generate()"
        num_new_images = self.num_new_images

        width = sampling_parameters.get("width", 512)
        height = sampling_parameters.get("height", 512)
        latent_width = width // self.diffusion_pipeline.vae_scale_factor
        latent_height = height // self.diffusion_pipeline.vae_scale_factor
        latent_channels = getattr(self.diffusion_pipeline.vae.config, "latent_channels", None)
        latents_shape = (num_new_images, latent_channels, latent_height, latent_width)

        dtype = self.diffusion_pipeline.vae.dtype
        device = "cpu"  # where to store latents on

        generators = self._get_random_generators(seeds)

        # Generate and scale the new noise
        noise = torch.cat(
            [
                self.diffusion_pipeline.prepare_latents(
                    batch_size=1,
                    num_channels_latents=self.diffusion_pipeline.vae.config.latent_channels,
                    height=height,
                    width=width,
                    dtype=self.diffusion_pipeline.vae.dtype,
                    device=device,
                    generator=generator,
                )
                for generator in generators
            ],
            dim=0,
        )
        assert noise.shape[0] == num_new_images, f"Expected {num_new_images} latents, got {noise.shape[0]}"

        # Normalize to unit variance for consistent mixing regardless of scheduler
        # (will be rescaled before passing to pipeline)
        noise = noise / self.diffusion_pipeline.scheduler.init_noise_sigma

        # For the first batch, just use these as the latents
        if self.rounds == 0:
            new_latents = noise
            self.batch_latents = new_latents
        # For subsequent batches, mix the noise with the running latent
        else:
            new_latents = LatentSurvivalGenerator.mix(running_latent=self.running_latent, noise=noise, alpha=self.alpha)
            self.batch_latents = torch.cat([self.last_winner_latent.unsqueeze(0), new_latents], dim=0)

        # Run diffusion
        output = self.diffusion_pipeline(prompt=self.prompt, num_images_per_prompt=num_new_images, latents=new_latents, **sampling_parameters)
        return output.images

    # -------------------------------------------------------
    # FEEDBACK UPDATE
    # -------------------------------------------------------
    @override
    def update(self, winner_index: int) -> None:
        """
        Winner_index identifies the winner in the current batch
        """
        super().update(winner_index)
        self.last_winner_latent = self.batch_latents[winner_index]

        # Running latent update using gamma decay
        if self.running_latent is None:
            self.running_latent = self.last_winner_latent
        else:
            self.running_latent = (1 - self.gamma) * self.running_latent + self.gamma * self.last_winner_latent

    @classmethod
    def mix(cls, running_latent: Latent, noise: Latent, alpha: float) -> Latent:
        scale_factor = ((1 - alpha)**2 + alpha**2)**0.5
        mixed_latent = (alpha * noise + (1 - alpha) * running_latent) / scale_factor
        return mixed_latent