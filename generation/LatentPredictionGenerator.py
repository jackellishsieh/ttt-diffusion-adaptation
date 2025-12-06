from __future__ import annotations
from typing import Any, override
from .FeedbackAwareGenerator import FeedbackAwareGenerator
from PIL import Image
from diffusers import DiffusionPipeline
import torch
import sys
sys.path.append("..")
from training.LatentPredictor import LatentShiftNetwork

Latent = torch.Tensor

class LatentPredictionGenerator(FeedbackAwareGenerator):
    """
    Generator that uses a trained LatentShiftNetwork to predict base latents:
    - initialize(prompt, batch_size) sets internal prompt and resets state
    - first generate() produces full batch_size new latents
    - update(winner_index) saves the winner and losers from the current batch
    - subsequent generate() uses the model to predict a base latent from winner/loser,
      then mixes noise on top of the predicted base latent
    """

    def __init__(
        self,
        diffusion_pipeline: DiffusionPipeline,
        model_weights_path: str,
        alpha: float = 0.5,
        latent_channels: int = 4,
        seed_safe: bool = False,
        device: str = "cpu",
        **kwargs: Any,
    ) -> None:
        super().__init__(diffusion_pipeline, seed_safe, **kwargs)
        self.alpha = alpha
        assert 0 <= self.alpha <= 1, "alpha must be between 0 and 1"

        # Load the LatentShiftNetwork model
        self.model = LatentShiftNetwork(latent_channels=latent_channels, device=device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=device))
        self.model.eval()
        self.device = device
        print(f"Loaded model from {model_weights_path}")

        # State for tracking latents
        self.batch_latents = None  # stores the latents for the current batch
        self.last_winner_latent = None  # the latent of the last winner
        self.last_loser_latents = None  # the latents of the last losers

    def initialize(self, prompt: str, batch_size: int = 4) -> None:
        """
        Initialize for a new trial
        """
        super().initialize(prompt, batch_size)

        self.batch_latents = None
        self.last_winner_latent = None
        self.last_loser_latents = None

    # -------------------------------------------------------
    # GENERATION
    # -------------------------------------------------------
    @override
    def generate(self, sampling_parameters: dict[str, Any], seeds: list[int] | None = None) -> list[Image.Image]:
        """
        Uses internal self.prompt. Must be called after initialize(prompt).
        First batch: N fresh latents, returns N images
        Subsequent batches: Uses model to predict base latent, then mixes with noise
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

        # For the first batch, just use noise as the latents
        if self.rounds == 0:
            new_latents = noise
            self.batch_latents = new_latents
        # For subsequent batches, use the model to predict a base latent and mix with noise
        else:
            # Use the model to predict the base latent from winner/loser
            with torch.no_grad():
                winner_on_device = self.last_winner_latent.to(self.device, dtype=torch.float32)
                losers_on_device = self.last_loser_latents.to(self.device, dtype=torch.float32)
                
                predicted_base_latent = self.model(winner_on_device, losers_on_device)
                predicted_base_latent = predicted_base_latent.to(device, dtype=torch.float16)  # move back to cpu

                predicted_base_latent /= predicted_base_latent.std() # normalize to unit variance

            # Mix the noise with the predicted base latent
            new_latents = LatentPredictionGenerator.mix(
                running_latent=predicted_base_latent, noise=noise, alpha=self.alpha
            )
            self.batch_latents = torch.cat([self.last_winner_latent.unsqueeze(0), new_latents], dim=0)

        # Run diffusion
        print("new_latents.dtype:", new_latents.dtype)
        output = self.diffusion_pipeline(
            prompt=self.prompt, num_images_per_prompt=num_new_images, latents=new_latents, **sampling_parameters
        )
        return output.images

    # -------------------------------------------------------
    # FEEDBACK UPDATE
    # -------------------------------------------------------
    @override
    def update(self, winner_index: int) -> None:
        """
        Winner_index identifies the winner in the current batch.
        Saves the winner and losers for use in the next generation.
        """
        super().update(winner_index)

        # Extract the winner latent
        self.last_winner_latent = self.batch_latents[winner_index]

        # Extract the loser latents (all latents except the winner)
        num_latents = self.batch_latents.shape[0]
        loser_indices = [i for i in range(num_latents) if i != winner_index]
        self.last_loser_latents = self.batch_latents[loser_indices]  # Shape: [num_losers, C, H, W]

    @classmethod
    def mix(cls, running_latent: Latent, noise: Latent, alpha: float) -> Latent:
        """
        Mix the running latent with noise using alpha blending.
        The result is normalized to maintain unit variance.
        """
        scale_factor = ((1 - alpha) ** 2 + alpha ** 2) ** 0.5
        mixed_latent = (alpha * noise + (1 - alpha) * running_latent) / scale_factor
        return mixed_latent
