from __future__ import annotations
from pathlib import Path
from config import setting
import pandas as pd
import numpy as np
import cv2


data_root = Path(__file__).resolve().parents[1] / "data" / "raw" / "nuclei_dataset"
SPLIT = setting.split
img_size = setting.img_size
cv2.setNumThreads(0)

def split_dir(split: str, root: Path = data_root) -> Path: 
    if split not in SPLIT:
        raise ValueError(f"unknown split, please choose one of {SPLIT}")
    return Path(root) / split

def list_stem(split: str, root: Path = data_root) -> list:
    files = (split_dir(split, root) / "images").glob("*.png")
    sorted_files = sorted(files)
    if not sorted_files:
        raise FileNotFoundError(f"No image under forder {split_dir(split, root) / 'images'}")
    return [stem.stem for stem in sorted_files]

def path_for_each_id(img_id: str, split: str, root: Path = data_root) -> dict:
    path = split_dir(split, root)
    return {"images": path / "images" / f"{img_id}.png",
            "labels": path / "labels" / f"{img_id}.png",
            "masks": path / "masks" / f"{img_id}.png"}

# corrupted data
def corrupted_dir(root=data_root) -> Path:
    return Path(root) / "test_corrupted" / "images"

def list_corrupted(root=data_root):
    out = []
    for f in sorted(corrupted_dir(root).glob("*.png")):
        parts = f.stem.split("_")                 
        out.append((f.stem, "_".join(parts[:2]), "_".join(parts[2:])))
    return out

def corrupted_path(stem: str, root=data_root) -> Path:
    return corrupted_dir(root) / f"{stem}.png"

# reader
def read_img(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cannot find image from {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def read_mask(path: Path) ->np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot find mask from {path}")
    return mask >127

def read_label(path: Path) -> np.ndarray:
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise FileNotFoundError(f"cannot find label from {path}")
    elif label.dtype != np.uint16:
        raise ValueError(f"expected a 16 bit but got {label.dtype}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label

# load ID

def load_image(image_id: str, split: str = "train", root=data_root) -> np.ndarray:
    return read_img(path_for_each_id(image_id, split, root)["images"])

def load_mask(image_id: str, split: str = "train", root=data_root) -> np.ndarray:
    return read_mask(path_for_each_id(image_id, split, root)["labels"])

def load_labels(image_id: str, split: str = "train", root=data_root) -> np.ndarray:
    return read_label(path_for_each_id(image_id, split, root)["labels"])

def load_all(image_id: str, split: str = "train", root=data_root):
    p = path_for_each_id(image_id, split, root)
    return read_img(p["images"]), read_mask(p["masks"]), read_label(p["labels"])

def load_corrupted(stem: str, root=data_root) -> np.ndarray:
    return read_img(corrupted_path(stem, root))

if __name__ == "__main__":
    test = load_all(image_id="test_001", split="test")
    print(test)

