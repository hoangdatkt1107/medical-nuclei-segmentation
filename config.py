from __future__ import annotations
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings

class Setting(BaseSettings):
    data_url: str = str("https://github.com/Nickolay-K/Assingnment-3-dataset/blob/main/nuclei_dataset.zip")
    data_path: Path = Path("/Users/hoangdat/Data/Herts/Analysis_AI/vision_assignment/data")
    img_size: float = 256.0
    split: tuple[str, ...] = ("train", "val", "test")
setting = Setting()

