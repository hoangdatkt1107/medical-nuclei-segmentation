from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Setting(BaseSettings):
    data_url: str = str("https://github.com/Nickolay-K/Assingnment-3-dataset/blob/main/nuclei_dataset.zip")
    data_path: Path = Path(__file__).resolve().parent / "data" / "raw" / "nuclei_dataset"
    img_size: int = 256
    split: tuple[str, ...] = ("train", "val", "test")

    #training
    pin_memory: bool = False # turn it to True if using CUDA
    batch_size: int = 8
    num_workers: int = 0
    seed : int = 42

    #pre process
    gray_method: str = "blue" # there are 3 options: blue, luminance and max
    norm: str = "fixed" # there are 2 options:  fixed and per_image

    #prompts
    prompt_dir: Path = Path(__file__).resolve().parent / "prompts"
    out_dir: Path = Path(__file__).resolve().parent / "outputs" / "json"
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_healcheck_url: str = "http://localhost:11434/api/tags"
    vision_model: str = "gemma3:4b"
    text_model: str = "llama3:latest"
    num_predict: int = 1024   # headroom for the narrative field

    # feature
    vision_keys: list = ["modality", "tissue_type", "notable_features", "image_quality"]
    feature_keys: list = ["n_objects", "density_class", "shape_regularity", "quality_flag"]
    properties: tuple = ("label", "area", "eccentricity", "solidity", "mean_intensity", "perimeter", "centroid")
    min_size: int = 30          # minimum object area in pixels
    min_distance: int = 2       # watershed seed spacing in pixels
    # unet
    base_channels: int = 16
    epochs: int = 40
    lr: float = 1e-3
    loss: str = "bce_dice"      # there are 3 options: bce, dice and bce_dice
    model_dir: Path = Path(__file__).resolve().parent / "outputs" / "models"
    fig_dir: Path = Path(__file__).resolve().parent / "outputs" / "figures"
    csv_dir: Path = Path(__file__).resolve().parent / "outputs" / "csv"

setting = Setting()

