from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Any
import wandb  # <--- Added

@dataclass
class LatentTrainerConfig:
    # Optimization
    lr: float = 1e-4
    epochs: int = 20
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss Hyperparameters
    triplet_weight: float = 0.2
    triplet_margin: float = 1.0
    triplet_p: int = 2
    
    # Logging
    wandb_project: str = "latent-shift-adaptation"
    run_name: Optional[str] = None


class LatentShiftTrainer:
    def __init__(self, model: nn.Module, train_loader, val_loader, config: LatentTrainerConfig):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        
        # Losses
        self.mse_loss_fn = nn.MSELoss()
        self.triplet_loss_fn = nn.TripletMarginLoss(
            margin=self.cfg.triplet_margin, 
            p=self.cfg.triplet_p, 
            reduction='mean'
        )

    def compute_hybrid_loss(self, predicted_latent, target_winner, target_losers_tensor):
        """
        Computes Hybrid Loss: MSE + (Lambda * Avg_Triplet_Loss)
        """
        loss_mse = self.mse_loss_fn(predicted_latent, target_winner)

        loss_triplet = torch.tensor(0.0, device=self.cfg.device)
        
        if target_losers_tensor.numel() > 0:
            accumulated_triplet = 0.0
            num_losers = target_losers_tensor.shape[1]
            
            for i in range(num_losers):
                current_loser_batch = target_losers_tensor[:, i, ...]
                l = self.triplet_loss_fn(predicted_latent, target_winner, current_loser_batch)
                accumulated_triplet += l
            
            loss_triplet = accumulated_triplet / num_losers

        total_loss = loss_mse + (self.cfg.triplet_weight * loss_triplet)
        return total_loss, loss_mse, loss_triplet

    def compute_metrics(self, pred, target, start_latent):
        """
        Calculates physics metrics: Cosine Similarity and Delta Shift Magnitudes
        """
        # Flatten for vector math: [B, C, H, W] -> [B, Features]
        true_delta = (target - start_latent).flatten(1)
        pred_delta = (pred - start_latent).flatten(1)
        start_latent = start_latent.flatten(1)

        # 1. Cosine Similarity
        cosine = torch.cosine_similarity(true_delta, pred_delta, dim=1).mean()
        
        # 2. Magnitude Statistics (Are we moving enough? Too much?)
        # Calculate L2 norm of the predicted shift per image
        pred_mags = torch.norm(pred_delta, p=2, dim=1) 
        norm_ratio = pred_mags / torch.norm(start_latent, p=2, dim=1)

        # 3. Mean and stdev of the predicted vs. target
        pred_mean = pred.mean().item()
        pred_std = pred.std().item()
        target_mean = target.mean().item()
        target_std = target.std().item()

        return {
            "shift_cosine": cosine.item(),
            "shift_magnitude_mean": pred_mags.mean().item(),
            "shift_magnitude_std": pred_mags.std().item(),
            "norm_ratio_mean": norm_ratio.mean().item(),
            "norm_ratio_std": norm_ratio.std().item(),
            "pred_mean": pred_mean,
            "pred_std": pred_std,
            "target_mean": target_mean,
            "target_std": target_std
        }

    def run_epoch(self, loader, is_train: bool):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        # Accumulators for logging
        acc = {
            "total_loss": 0.0,
            "mse_loss": 0.0,
            "triplet_loss": 0.0,
            "shift_cosine": 0.0,
            "shift_magnitude_mean": 0.0,
            "shift_magnitude_std": 0.0,
            "norm_ratio_mean": 0.0,
            "norm_ratio_std": 0.0,
            "pred_mean": 0.0,
            "pred_std": 0.0,
            "target_mean": 0.0,
            "target_std": 0.0
        }
        
        pbar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
        
        context = torch.enable_grad() if is_train else torch.no_grad()
        
        with context:
            for batch in pbar:
                # 1. Unpack
                winner_in = batch["input_winner_latent"].to(self.cfg.device)
                losers_in = batch["input_loser_latents"].to(self.cfg.device)
                winner_tgt = batch["target_winner_latent"].to(self.cfg.device)
                losers_tgt = batch["target_loser_latents"].to(self.cfg.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                
                # 2. Forward
                pred_next = self.model(winner_in, losers_in)
                
                # 3. Loss
                loss, l_mse, l_trip = self.compute_hybrid_loss(pred_next, winner_tgt, losers_tgt)
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                
                # 4. Metrics
                # We pass 'winner_in' to calculate the delta (pred - start)
                metrics = self.compute_metrics(pred_next, winner_tgt, winner_in)
                
                # 5. Accumulate
                acc["total_loss"] += loss.item()
                acc["mse_loss"] += l_mse.item()
                acc["triplet_loss"] += l_trip.item()
                for k, v in metrics.items():
                    acc[k] += v

                # Update pbar visually
                pbar.set_postfix({"L": f"{loss.item():.3f}", "Cos": f"{metrics['shift_cosine']:.2f}"})

        # Average over loader
        avg_metrics = {k: v / len(loader) for k, v in acc.items()}
        return avg_metrics

    def fit(self):
        print(f"Starting training on {self.cfg.device} for {self.cfg.epochs} epochs.")
        
        # Initialize WandB
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.run_name,
            config=self.cfg.__dict__
        )
        
        for epoch in range(self.cfg.epochs):
            # Run Loops
            train_metrics = self.run_epoch(self.train_loader, is_train=True)
            val_metrics = self.run_epoch(self.val_loader, is_train=False)
            
            # Print Summary
            print(f"Epoch {epoch+1:02d} | "
                  f"Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Val MSE: {val_metrics['mse_loss']:.4f} | "
                  f"Val Cos: {val_metrics['shift_cosine']:.3f} | "
                  f"Val Shift Mag: {val_metrics['shift_magnitude_mean']:.2f}")

            # Log to WandB
            # We prefix keys with 'train/' and 'val/' for clean grouping in the dashboard
            log_dict = {}
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = v
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v
            
            log_dict["epoch"] = epoch + 1
            wandb.log(log_dict)
            
        print("Training Complete.")
        wandb.finish()