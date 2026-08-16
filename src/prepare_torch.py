import numpy as np
import torch
import random

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


        
        





        

        