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
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    out_dir = Path(__file__).resolve().parent / "outputs" / "json"
    ollama_url: str = "http://localhost:11434/api/generate"
    vision_model: str = "llama3.2-vision"
    text_model: str = "llama3"

    # feature
    vision_keys = ["modality", "tissue_type", "notable_features", "image_quality"]
    feature_keys = ["n_objects", "density_class", "shape_regularity", "quality_flag"]
setting = Setting()

