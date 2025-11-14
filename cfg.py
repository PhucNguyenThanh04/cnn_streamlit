from dataclasses import dataclass
from typing import Optional
import torch

@dataclass
class Config:
    root: str = "./Ultrasound_Stone_No_Stone"
    epochs: int = 30
    batch_size: int = 16
    image_size: int = 224
    logging_dir: str = "runs"
    trained_models_dir: str = "trained_models"
    checkpoint: Optional[str] = None

    lr: float = 1e-3
    momentum: float = 0.9

    num_workers: int = 4
    seed: int = 42
    device: Optional[torch.device] = None
    num_classes: int = 2
    use_amp: bool = True
    save_every: int = 1