import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Callable, NamedTuple
from generation.LatentSurvivalGenerator import LatentSurvivalGenerator


Latent = torch.Tensor  # we'll represent as 3-dimensional, not 4-dimensional. We'll add the first dimension (batch size) only when needed.
SeedToLatent = Callable[[[list[int]]], list[Latent]]  # given a list of seeds, prepare the latents


class DataPoint(NamedTuple):
    round: int
    trial_id: str
    prompt_id: str
    alpha: float
    winner_latent: Latent
    loser_latents: list[Latent]

    def __str__(self):
        s = "DataPoint:"
        s += f"\n\tround: {self.round}"
        s += f"\n\ttrial_id: {self.trial_id}"
        s += f"\n\tprompt_id: {self.prompt_id}"
        s += f"\n\talpha: {self.alpha}"
        s += f"\n\twinner_latent: {self.winner_latent.shape}"
        s += f"\n\tloser_latents: {[l.shape for l in self.loser_latents]}"
        return s
    
    def __repr__(self):
        return self.__str__()


class LatentPredictionDataset(Dataset):
    def __init__(self, metrics_df: pd.DataFrame, seeds_to_latent: SeedToLatent, alpha_range=None):
        """
        Args:
            metrics_df: The raw metrics dataframe.
            seeds_to_latent: Function fn(seeds: list[int]) -> list[Latent] to get latents from seeds
            alpha_range: (Optional) Tuple of (min_alpha, max_alpha).
                         Use this to filter out Unconditional (1.0) or Identity (0.0).
                         Example: (0.1, 0.9)
        """
        self.df = metrics_df
        self.seeds_to_latent = seeds_to_latent

        # --- 1. Filter Data (Crucial for Delta Training) ---
        if alpha_range:
            min_a, max_a = alpha_range
            # Keep only rows within the useful alpha range
            self.df = self.df[(self.df["alpha"] >= min_a) & (self.df["alpha"] <= max_a)]

        # --- 2. Group by the Unique Triplet ---
        # A unique sequence is defined by Trial + Prompt + Alpha
        self.groups = self.df[["trial_id", "prompt_id", "alpha"]].drop_duplicates().reset_index(drop=True)

        # Constants
        self.rounds_per_trial = 4
        self.transitions_per_trial = self.rounds_per_trial - 1

    def __len__(self):
        return len(self.groups) * self.transitions_per_trial

    def __getitem__(self, idx):
        # Map linear index to Group + Transition
        group_idx = idx // self.transitions_per_trial
        transition_idx = idx % self.transitions_per_trial

        # Get Key Identifiers
        group_row = self.groups.iloc[group_idx]
        trial_id = group_row["trial_id"]
        prompt_id = group_row["prompt_id"]
        alpha = group_row["alpha"]

        # Determine Rounds
        input_round = transition_idx
        target_round = transition_idx + 1

        # Fetch DataPoints
        # Note: Updated to pass 'alpha' to your helper function if needed
        dp_input = self._get_data_point(input_round, trial_id, prompt_id, alpha)
        dp_target = self._get_data_point(target_round, trial_id, prompt_id, alpha)

        return dp_input, dp_target

    def _get_data_point(self, round: int, trial_id: str, prompt_id: str, alpha: float) -> tuple[DataPoint, DataPoint]:
        # 1. Filter Context
        rows = self.df[(self.df["trial_id"] == trial_id) & (self.df["prompt_id"] == prompt_id) & (self.df["alpha"] == alpha)].sort_values("round")

        curr_rows = rows[rows["round"] == round]

        # Helper to fetch a single latent from a seed
        def get_latent(seed):
            return self.seeds_to_latent([seed])[0]

        # --- CASE 1: Round 0 (Base Case) ---
        if round == 0:
            winner_seed = curr_rows.loc[curr_rows["chosen"], "seed"].item()
            loser_seeds = curr_rows.loc[~curr_rows["chosen"], "seed"].tolist()

            return DataPoint(round, trial_id, prompt_id, alpha, get_latent(winner_seed), self.seeds_to_latent(loser_seeds))

        # --- CASE 2: Round > 0 (Adaptation) ---

        # 2. Reconstruct the "King" (Winner) from Round 0 up to current Round - 1
        # Start with Round 0 winner
        r0_winner_seed = rows.loc[(rows["round"] == 0) & (rows["chosen"]), "seed"].item()
        king_latent = get_latent(r0_winner_seed)

        # Replay history: Filter for winners of previous rounds (1 to round-1)
        history_winners = rows[(rows["round"] > 0) & (rows["round"] < round) & (rows["chosen"])]

        for _, row in history_winners.iterrows():
            # Only update if the winner wasn't simply carried over (reused)
            if not row["reused_from_previous"]:
                noise = get_latent(row["seed"])
                king_latent = LatentSurvivalGenerator.mix(running_latent=king_latent, noise=noise, alpha=alpha)

        # 3. Generate Latents for Current Round
        # Index 0 is always the existing King; 1-3 are the new Challengers
        latents_by_idx = {0: king_latent}

        challengers = curr_rows[(~curr_rows["reused_from_previous"]) & (curr_rows["image_idx"] > 0)]

        for _, row in challengers.iterrows():
            noise = get_latent(row["seed"])
            latents_by_idx[row["image_idx"]] = LatentSurvivalGenerator.mix(running_latent=king_latent, noise=noise, alpha=alpha)

        # 4. separate Winner vs Losers
        winner_idx = curr_rows.loc[curr_rows["chosen"], "image_idx"].item()

        # Construct final list of 4 to ensure correct indexing
        all_latents = [latents_by_idx[i] for i in range(4)]

        return DataPoint(round, trial_id, prompt_id, alpha, all_latents[winner_idx], [l for i, l in enumerate(all_latents) if i != winner_idx])  # Winner  # Losers


def create_splits(df, group_cols=['trial_id', 'prompt_id', 'alpha'], val_ratio=0.1, seed=42):
    """
    Splits a dataframe into Train/Val ensuring that all rounds for a specific
    (trial, prompt, alpha) group stay together.

    This way, all rounds for a specific trial are either in the train or val set, but not both.
    """
    # 1. Identify unique runs (The "Groups")
    unique_groups = df[group_cols].drop_duplicates()
    
    # 2. Sample groups for the Validation set
    val_groups = unique_groups.sample(frac=val_ratio, random_state=seed)
    
    # 3. The rest are Training groups
    # We simply drop the validation indices from the unique list
    train_groups = unique_groups.drop(val_groups.index)
    
    # 4. Filter the original huge DataFrame using Inner Joins
    # This keeps only the rows belonging to the selected groups
    train_df = df.merge(train_groups, on=group_cols, how='inner')
    val_df = df.merge(val_groups, on=group_cols, how='inner')
    
    print(f"Total Groups: {len(unique_groups)}")
    print(f"Train Rows: {len(train_df)} | Val Rows: {len(val_df)}")
    
    return train_df, val_df