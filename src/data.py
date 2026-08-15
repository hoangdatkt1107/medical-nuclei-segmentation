from __future__ import annotations
from pathlib import Path
from config import setting
import pandas as pd
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
        raise FileNotFoundError(f"No image under forder {split_dir(split, root) / "images"}")
    return [stem.stem for stem in sorted_files]

def path_for_each_id(split: str, root: Path = data_root, img_id: str) -> dict:
    path = split_dir(split)

    

if __name__ == "__main__":
    test = list_stem("test")
    print(test)
