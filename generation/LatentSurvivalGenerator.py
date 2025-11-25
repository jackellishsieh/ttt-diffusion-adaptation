from __future__ import annotations
from typing import Any, override
from .FeedbackAwareGenerator import FeedbackAwareGenerator
from PIL import Image
from diffusers import DiffusionPipeline
import torch


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
        **kwargs: Any,
    ) -> None:
        super().__init__(diffusion_pipeline, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        assert 0 <= self.alpha <= 1, "alpha must be between 0 and 1"
        assert 0 <= self.gamma <= 1, "gamma must be between 0 and 1"

        # Will be set by initialize()
        self.prompt = None
        self.batch_size = None

        self.running_latent = None  # the running latent
        self.last_winner_latent = None  # the latent of the last winner
        self.batch_latents = None  # this stores the latents for the current batch, only used to track the winner
        self.is_first_batch = True

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
    def generate(self, sampling_parameters: dict[str, Any]) -> list[Image.Image]:
        """
        Uses internal self.prompt. Must be called after initialize(prompt).
        First batch: N fresh latents, returns N images
        Subsequent batches: 1 survivor + N-1 new latents, returns N-1 images
        """
        assert self.prompt is not None, "Must call initialize(prompt) before generate()"
        num_new_images = self.batch_size - (0 if self.is_first_batch else 1)

        width = sampling_parameters.get("width", 512)
        height = sampling_parameters.get("height", 512)
        latent_width = width // self.diffusion_pipeline.vae_scale_factor
        latent_height = height // self.diffusion_pipeline.vae_scale_factor
        latent_channels = getattr(self.diffusion_pipeline.vae.config, "latent_channels", None)
        latents_shape = (num_new_images, latent_channels, latent_height, latent_width)

        dtype = self.diffusion_pipeline.vae.dtype
        device = "cpu"  # where to store latents on

        # --------------------
        # FIRST BATCH: generate N fresh latents
        # --------------------
        if self.is_first_batch:
            new_latents = self.diffusion_pipeline.prepare_latents(
                batch_size=num_new_images,
                num_channels_latents=self.diffusion_pipeline.vae.config.latent_channels,
                height=height,
                width=width,
                dtype=self.diffusion_pipeline.vae.dtype,
                device=device,
                generator=torch.Generator(),
            )

            self.is_first_batch = False
            self.batch_latents = new_latents

        # --------------------
        # SUBSEQUENT BATCHES: generate N-1 new latents (but store the old survivor for feedback)
        # --------------------
        else:
            # Fill remaining slots with fresh noise-mixed latents
            noise = torch.randn(latents_shape, dtype=dtype, device=device)
            new_latents = (1 - self.alpha) * self.running_latent + self.alpha * noise  # broadcasts
            self.batch_latents = torch.cat([self.last_winner_latent.unsqueeze(0), new_latents], axis=0)  # N latents, including the previous winner

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
        self.last_winner_latent = self.batch_latents[winner_index]

        # Running latent update using gamma decay
        if self.running_latent is None:
            self.running_latent = self.last_winner_latent
        else:
            self.running_latent = (1 - self.gamma) * self.running_latent + self.gamma * self.last_winner_latent
