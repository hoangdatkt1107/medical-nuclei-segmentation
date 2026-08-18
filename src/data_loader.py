from __future__ import annotations
import torch
import numpy as np
import random
from src.prepare_torch import NucleiDataset
from config import setting
from src.prepare_torch import default_transform
from torch.utils.data import DataLoader

def seed_worker(worker_id: int):
    """create worker seed for augmentation"""
    worker_seed = torch.initial_seed() & (2 ** 32 -1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(batch_size: int = 8,
                    num_workers: int = 0,
                    gray_method: str = "blue",
                    norm: str = "fixed",
                    seed: int = 42,
                    augment: bool = True):

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_ds = NucleiDataset(setting.data_path, split="train", 
                             transform=default_transform(train=augment), gray_method=gray_method,
                             norm=norm)

    val_ds = NucleiDataset(setting.data_path, split="val",gray_method=gray_method, norm=norm)
    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers,worker_init_fn=seed_worker, pin_memory=setting.pin_memory,
                              generator=generator)
    
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                            worker_init_fn=seed_worker, pin_memory=setting.pin_memory)
    return train_loader, val_loader

def get_test_loader(batch_size: int = 1, **kwargs):
    """batch_size=1 by default, the end-to-end pipeline emits one JSON record per image, so
    there is nothing worth batching"""
    ds = NucleiDataset(split="test", return_id=True, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)