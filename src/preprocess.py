from __future__ import annotations
import numpy as np
import pandas as pd
import cv2
from config import setting

def to_grayscale(rgb: np.ndarray, method: str = "blue") -> np.ndarray:
    """because in DAPI-like nuclei nearly all signal comes from blue channel -> prioritise blue
    3 options including:
    - blue (default)
    - luminance
    - max   """
    rgb = rgb.astype(np.float32)
    if method == "blue":
        return rgb[:,:,2]
    elif method == "luminance":
        return 0.2125 * rgb[:,:, 0] + 0.7154 * rgb[:,:, 1] + 0.0721 * rgb[:,:, 2]
    elif method == "max":
        return rgb.max(axis=-1)
    raise ValueError("unkwon provided greyscale method")

def normalise(gray: np.ndarray, mode:str = "fixed") -> np.ndarray:
    """ 2 method of normalisation including:
    - fixed (default)
    - per_image (min-max per image)"""
    gray = gray.astype(np.float32)
    if mode == "fixed":
        gray = gray / 255.0
        return gray
    elif mode == "per_image":
        hi, lo = float(gray.max()), float(gray.min())
        return np.zeros_like(gray) if hi - lo < 1e-6 else (gray - lo) / (hi - lo)
    raise ValueError("unkown provided normalise method")

def resize_if_needed(img_arr: np.ndarray, is_mask: bool, size: float = setting.img_size):
    if not img_arr:
        raise ValueError("no image is inserted into resize function")
    if img_arr.shape[:2] == (size, size):
        return img_arr
    else:
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        out = cv2.resize(img_arr, size, interpolation=interp)
        return out.astype(img_arr.dtype)

def process(rgb: np.ndarray, gray_method: str = "blue", norm: str = "fixed",
             size: int = setting.img_size) -> np.ndarray:
    gray = resize_if_needed(to_grayscale(rgb, gray_method), size, is_mask=False)
    return normalise(gray, norm)

