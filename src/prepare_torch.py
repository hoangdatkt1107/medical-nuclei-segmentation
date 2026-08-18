import numpy as np
from collections.abc import Callable
from torch.utils.data import Dataset
import torch
import random
from config import setting
from pathlib import Path
from data import list_stem, load_image, load_mask, load_labels, load_all
from preprocess import process, resize_if_needed

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self):
        return f"JointCompose([{', '.join(repr(t) for t in self.transforms)}])"

class FlipRot90:
    def __init__(self, p: float = 0.5, seed: int | None = None):
        self.p = p

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        k = random.randint(0, 3)
        if k:
            img, mask = np.rot90(img, k), np.rot90(mask, k)
        if self.rng.random() < self.p:
            img, mask = np.fliplr(img), np.fliplr(mask)
        if self.rng.random() < self.p:
            img, mask = np.flipud(img), np.flipud(mask)
        return img, mask

    def __repr__(self):
        return f"FlipRot90(p={self.p})"

class ToTensor:
    def __call__(self, img, mask):
        img = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).unsqueeze(0)
        return img, mask

    def __repr__(self):
        return "ToTensor()"

def default_transform(train: bool):
    if train:
        list_transform = [FlipRot90(), ToTensor()]
    else:
        list_transform = [ToTensor()]
    return JointCompose(list_transform)

class NucleiDataset(Dataset):
    def __init__(self, root=setting.data_path, split: str = "train", 
                transform: Callable | None = None, gray_method: str = "blue",
                  norm: str = "fixed", return_id: bool = False, with_labels: bool = False):
        
        self.root = Path(root)
        self.transform = default_transform(train=(split == "train")) if transform is None else transform
        self.gray_method = gray_method
        self.split = split
        self.norm = norm
        self.ids = list_stem(split, self.root)
        self.with_labels = with_labels
        self.return_id = return_id

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        img_id = self.ids[index]
        img = process(load_image(img_id, self.split, self.root), self.gray_method, self.norm)
        mask = resize_if_needed(load_mask(img_id, self.split, self.root), is_mask=True).astype(np.float32)
        img, mask = self.transform(img, mask)
        output = (img, mask)
        if self.with_labels:
            output += torch.from_numpy(load_labels(img_id, self.split, self.root).astype(np.int32))
        if self.return_id:
            output += (img_id)
        return output

    def raw(self, index: int):
        """return 3 ndarray of original image, mask and label"""
        return load_all(self.ids[index], self.split, self.root)

    def __repr__(self):
        return (f"NucleiDataset(split={self.split}, n={len(self)},"
                f"gray={self.gray_method}, norm={self.norm}, transform={self.transform})")

        


        
        





        

        