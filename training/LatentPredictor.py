import torch
import torch.nn as nn

Latent = torch.Tensor
LatentSet = torch.Tensor

class LatentShiftNetwork(nn.Module):
    """
    Given a winner latent and a list of loser latents, predict a mean shift
    If mean shift is 0, then predicted latent is the same as the winner latent
    """

    def __init__(self, latent_channels=4, device="cpu"):
        super().__init__()
        
        # Input: Winner (4) + Mean_Loser (4) = 8 channels
        self.input_channels = latent_channels * 2
        
        # A simple, efficient CNN that preserves spatial dimensions (padding=1)
        # We don't need to downsample/upsample because latents are already compressed
        self.net = nn.Sequential(
            # Layer 1: Fusion
            nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Layer 2: Processing
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            # Layer 3: Output Delta
            # Initialize weights near zero so initial prediction is close to Identity
            nn.Conv2d(64, latent_channels, kernel_size=3, padding=1) 
        )
        self.device = device
        self.to(device)
        
    def forward(self, winner_latent: Latent, loser_latents: LatentSet) -> Latent:
        """
        winner_latent: [4, H, W]
        loser_latents_list: (Unordered) List of [4, H, W] tensors (usually 3 of them)
        Returns:
            Latent: [4, H, W]
        """
        """
        winner_latent: [B, 4, H, W]
        loser_latents: [B, 3, 4, H, W]  <-- Now a Tensor!
        """
        
        # 1. Auto-Detect Batch Dimension
        # If input is 3D/4D (unbatched), unsqueeze to add Batch dim
        is_unbatched = winner_latent.dim() == 3
        if is_unbatched:
            winner_latent = winner_latent.unsqueeze(0)        # [1, 4, H, W]
            loser_latents = loser_latents.unsqueeze(0) # [1, 3, 4, H, W]

        # 2. Average the Losers
        # We now have a clean tensor. Mean over dim 1 (the '3' dimension)
        # Input: [B, 3, 4, H, W] -> Output: [B, 4, H, W]
        assert winner_latent.dim() == 4 and loser_latents.dim() == 5, f"winner_latent.dim(): {winner_latent.dim()}, loser_latents.dim(): {loser_latents.dim()}"
        
        mean_loser = torch.mean(loser_latents, dim=1)
        
        # 3. Concatenate & Predict
        x = torch.cat([winner_latent, mean_loser], dim=1)
        delta = self.net(x)
        output = winner_latent + delta
        
        # 4. Restore Shape if needed
        if is_unbatched:
            output = output.squeeze(0)
            
        return output