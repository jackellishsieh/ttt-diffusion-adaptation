from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


@dataclass
class LatentTrainerConfig:
    # Optimization
    lr: float = 1e-4
    epochs: int = 20
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss Hyperparameters
    # Weight for Triplet Loss relative to MSE (e.g., 0.1 means MSE is 10x more important)
    triplet_weight: float = 0.2

    # The margin 'm' in Triplet Loss: max(d(a,p) - d(a,n) + m, 0)
    # Higher = requires the Winner to be MUCH closer than the Loser
    triplet_margin: float = 1.0

    # Distance norm (2 = Euclidean/L2)
    triplet_p: int = 2


import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any

class LatentShiftTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, config):
        """
        Args:
            model: LatentShiftNetwork
            train_loader: DataLoader yielding dicts
            val_loader: DataLoader yielding dicts
            config: TrainerConfig object
        """
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        
        # 1. MSE Loss (Geometric)
        self.mse_loss_fn = nn.MSELoss()
        
        # 2. Triplet Loss (Directional Ranking)
        self.triplet_loss_fn = nn.TripletMarginLoss(
            margin=self.cfg.triplet_margin, 
            p=self.cfg.triplet_p, 
            reduction='mean'
        )
        
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_cosine": []
        }

    def compute_hybrid_loss(self, predicted_latent, target_winner, target_losers_tensor):
        """
        Computes Hybrid Loss: MSE + (Lambda * Avg_Triplet_Loss)
        """
        # --- Component 1: MSE ---
        loss_mse = self.mse_loss_fn(predicted_latent, target_winner)
        
        # --- Component 2: Triplet ---
        loss_triplet = torch.tensor(0.0, device=self.cfg.device)
        
        # target_losers_tensor shape: [B, 3, 4, H, W]
        if target_losers_tensor.numel() > 0:
            accumulated_triplet = 0.0
            num_losers = target_losers_tensor.shape[1]
            
            # Iterate over the '3' dimension (the number of losers)
            for i in range(num_losers):
                # Slice out i-th loser for whole batch -> [B, 4, H, W]
                current_loser_batch = target_losers_tensor[:, i, ...]
                
                l = self.triplet_loss_fn(predicted_latent, target_winner, current_loser_batch)
                accumulated_triplet += l
            
            loss_triplet = accumulated_triplet / num_losers

        # --- Combine ---
        total_loss = loss_mse + (self.cfg.triplet_weight * loss_triplet)
        
        return total_loss, loss_mse, loss_triplet

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        # CHANGED: We iterate over a single 'batch' dict
        for batch in pbar:
            # 1. Unpack and Move to Device
            # Keys match your __getitem__ return dict
            winner_in = batch["input_winner_latent"].to(self.cfg.device)
            losers_in = batch["input_loser_latents"].to(self.cfg.device) # [B, 3, 4, H, W]
            
            winner_tgt = batch["target_winner_latent"].to(self.cfg.device)
            losers_tgt = batch["target_loser_latents"].to(self.cfg.device)
            
            self.optimizer.zero_grad()
            
            # 2. Forward Pass
            pred_next = self.model(winner_in, losers_in)
            
            # 3. Hybrid Loss
            loss, l_mse, l_trip = self.compute_hybrid_loss(pred_next, winner_tgt, losers_tgt)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            pbar.set_postfix({
                "Total": f"{loss.item():.3f}",
                "MSE": f"{l_mse.item():.3f}",
                "Trip": f"{l_trip.item():.3f}"
            })
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        cosine_sum = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 1. Unpack
                winner_in = batch["input_winner_latent"].to(self.cfg.device)
                losers_in = batch["input_loser_latents"].to(self.cfg.device)
                winner_tgt = batch["target_winner_latent"].to(self.cfg.device)
                
                # 2. Predict
                pred_next = self.model(winner_in, losers_in)
                
                # 3. Val Loss (MSE only for standard tracking)
                loss = self.mse_loss_fn(pred_next, winner_tgt)
                running_loss += loss.item()
                
                # 4. Directional Accuracy
                true_delta = (winner_tgt - winner_in).flatten(1)
                pred_delta = (pred_next - winner_in).flatten(1)
                
                sim = torch.cosine_similarity(true_delta, pred_delta, dim=1).mean()
                cosine_sum += sim.item()
                
        return running_loss / len(self.val_loader), cosine_sum / len(self.val_loader)

    def fit(self):
        print(f"Starting training on {self.cfg.device}")
        print(f"Config: MSE + ({self.cfg.triplet_weight} * TripletLoss)")
        
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch()
            val_loss, val_sim = self.validate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_cosine'].append(val_sim)
            
            print(f"Epoch {epoch+1:02d} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val MSE: {val_loss:.4f} | "
                  f"Val Direction: {val_sim:.4f}")
            
        self.plot_results()

    def plot_results(self):
        epochs = range(1, len(self.history['train_loss']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss
        ax1.plot(epochs, self.history['train_loss'], label='Train (Hybrid)')
        ax1.plot(epochs, self.history['val_loss'], label='Val (MSE)')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Direction
        ax2.plot(epochs, self.history['val_cosine'], color='green', label='Cosine Sim')
        ax2.axhline(0, color='gray', linestyle='--')
        ax2.set_title('Directional Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()