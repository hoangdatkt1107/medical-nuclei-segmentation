from __future__ import annotations
from config import setting
import pandas as pd
from pathlib import Path

def load_metadata(root=setting.data_path) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "metadata.csv").set_index("image_id")

def ground_truth(image_id: str, meta: Optional[pd.DataFrame] = None) -> dict:
    """Answer-key row for one image. Corrupted variants fall back to their base image"""
    meta = load_metadata() if meta is None else meta
    if image_id not in meta.index:
        image_id = "_".join(image_id.split("_")[:2])
    row = meta.loc[image_id]
    return {"n_objects": int(row["n_objects"]),
            "density": str(row["density"]),
            "mean_intensity": float(row["mean_intensity"]),
            "area_fraction": float(row["area_fraction"])}

def metadata_for_split(split: str, meta: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    meta = load_metadata() if meta is None else meta
    return meta[meta["split"] == split]