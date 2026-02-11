ACTIONS = [
    "move",
    "shoot",
    "hide",
    "reload",
    "explore",
    "loot",
    "fight",
    "die",
    "win",
    "quit"
]

PAD_TOKEN = 0

ACTION_TO_ID = {action: idx + 1 for idx, action in enumerate(ACTIONS)}
ID_TO_ACTION = {idx + 1: action for idx, action in enumerate(ACTIONS)}

import torch
from torch.utils.data import Dataset
import numpy as np


class PlayerSequenceDataset(Dataset):
    def __init__(self, sequences_path, masks_path):
        self.sequences = torch.tensor(
            np.load(sequences_path), dtype=torch.long
        )
        self.masks = torch.tensor(
            np.load(masks_path), dtype=torch.long
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "mask": self.masks[idx]
        }
