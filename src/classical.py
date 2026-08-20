from __future__ import annotations
from src.data import list_stem, load_image, load_mask, load_labels, load_corrupted
from src.meta_data import load_metadata
from src.preprocess import process
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, disk, remove_small_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from config import setting
import numpy as np
import pandas as pd

PROPERTIES = setting.properties
setting.csv_dir.mkdir(parents=True, exist_ok=True)

def otsu_mask(grey: np.ndarray, min_size: int = setting.min_size, closing_radius: int = 1):
    """find a global otsu threshold and apply it to the grey image, then tidy up the result"""
    threshold = threshold_otsu(grey)
    mask = grey > threshold
    mask = closing(mask, disk(closing_radius))
    mask = remove_small_objects(mask, max_size=min_size - 1)
    mask = ndi.binary_fill_holes(mask)
    return mask, float(threshold)

def split_touching(mask: np.ndarray, min_distance: int = setting.min_distance) -> np.ndarray:
    """watershed on the distance transform, so nuclei that touch get separate labels"""
    distance = ndi.distance_transform_edt(mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=mask)

    markers = np.zeros(mask.shape, dtype=int)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    return watershed(-distance, markers, mask=mask)

def segment(img: np.ndarray, use_watershed: bool = True, min_size: int = setting.min_size,
            min_distance: int = setting.min_distance) -> dict:
    """rgb image -> everything the rest of the pipeline needs"""
    grey = process(img, setting.gray_method, setting.norm)
    mask, threshold = otsu_mask(grey, min_size=min_size)
    labels = split_touching(mask, min_distance) if use_watershed else label(mask)

    return {"grey": grey, "mask": mask, "labels": labels, "threshold": threshold,
            "n_objects": int(labels.max()), "use_watershed": use_watershed}

def feature_table(labels: np.ndarray, grey: np.ndarray) -> pd.DataFrame:
    """per-object measurements, one row per nucleus"""
    if labels.max() == 0:
        return pd.DataFrame(columns=[p for p in PROPERTIES])
    props = regionprops_table(labels, intensity_image=grey, properties=PROPERTIES)
    return pd.DataFrame(props)

def nearest_neighbour(df: pd.DataFrame) -> float:
    """median distance from each nucleus to its closest neighbour in pixels"""
    if len(df) < 2:
        return float("nan")
    xy = df[["centroid-0", "centroid-1"]].to_numpy()
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))

def summarise_table(df: pd.DataFrame, mask: np.ndarray, image_id: str = "") -> str:
    if df.empty:
        return f"Image {image_id}: segmentation found no objects."

    area = df["area"]
    lines = [f"Image: {image_id or 'unnamed'} (256 x 256 pixels)",
            f"Objects detected: {len(df)}",
            f"Total area covered by objects: {100 * mask.mean():.1f}% of the image",
            f"Object area in pixels: mean {area.mean():.0f}, median {area.median():.0f}, "
            f"min {area.min():.0f}, max {area.max():.0f}, std {area.std():.0f}",
            f"Eccentricity: mean {df['eccentricity'].mean():.2f}, std {df['eccentricity'].std():.2f} "
            f"(0 is a circle, 1 is a line)",
            f"Solidity: mean {df['solidity'].mean():.2f} (1 means a convex, unbroken shape)",
            f"Mean intensity inside objects: {df['mean_intensity'].mean():.3f} on a 0 to 1 scale",
            f"Median distance to the nearest other object: {nearest_neighbour(df):.1f} pixels"]
    return "\n".join(lines)

def dice(pred: np.ndarray, truth: np.ndarray) -> float:
    total = pred.sum() + truth.sum()
    return 1.0 if total == 0 else float(2 * (pred & truth).sum() / total)

def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    union = (pred | truth).sum()
    return 1.0 if union == 0 else float((pred & truth).sum() / union)

def evaluate_image(image_id: str, split: str = "test", **kwargs) -> dict:
    """segment one image and score it against the label map (see notebook 01 on why not metadata)"""
    img = load_image(image_id, split)
    out = segment(img, **kwargs)

    truth_mask = load_mask(image_id, split)
    truth_labels = load_labels(image_id, split)
    n_visible = len(np.unique(truth_labels)) - 1

    return {"image_id": image_id,
            "density": load_metadata().loc[image_id, "density"],
            "threshold": round(out["threshold"], 3),
            "dice": round(dice(out["mask"], truth_mask), 4),
            "iou": round(iou(out["mask"], truth_mask), 4),
            "n_pred": out["n_objects"],
            "n_visible": n_visible,
            "count_error": out["n_objects"] - n_visible}

def evaluate_split(split: str = "test", **kwargs) -> pd.DataFrame:
    return pd.DataFrame([evaluate_image(i, split, **kwargs) for i in list_stem(split)])

def describe_with_numbers(image_id: str, split: str = "test", **kwargs) -> dict:
    """the full task 2 route: segment, measure, summarise, then ask the LLM"""
    from src.llm import summarise_features

    img = load_corrupted(image_id) if "_blur" in image_id or "_lowcontrast" in image_id \
        else load_image(image_id, split)
    out = segment(img, **kwargs)
    df = feature_table(out["labels"], out["grey"])
    summary = summarise_table(df, out["mask"], image_id)

    answer = summarise_features(summary)
    answer.update({"image_id": image_id, "summary_sent": summary, "n_measured": len(df)})
    return answer

def sweep_min_distance(values=(2, 3, 4, 5, 7), split: str = "test") -> pd.DataFrame:
    """try several watershed seed spacings and report the counting error per regime"""
    rows = []
    for value in values:
        scores = evaluate_split(split, min_distance=value)
        row = {"min_distance": value}
        row.update(scores.groupby("density")["count_error"].mean().round(2).to_dict())
        row["MAE"] = round(scores["count_error"].abs().mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)

def compare_watershed(split: str = "test") -> pd.DataFrame:
    """counting error with watershed against plain connected components"""
    with_ws = evaluate_split(split, use_watershed=True)
    without = evaluate_split(split, use_watershed=False)
    both = pd.DataFrame({"density": with_ws["density"],
                         "watershed": with_ws["count_error"],
                         "connected_components": without["count_error"]})
    return both.groupby("density").mean().round(2)

def describe_split_with_numbers(split: str = "test", **kwargs) -> pd.DataFrame:
    """the numbers-first route over a whole split, saving one json per image and one table.

    Needs ollama running."""
    from src.llm import save_json

    meta = load_metadata()
    rows = []
    for image_id in list_stem(split):
        out = describe_with_numbers(image_id, split, **kwargs)
        save_json(out, f"task2_{image_id}")
        record = out["record"]
        rows.append({"image_id": image_id, "n_measured": out["n_measured"],
                     "n_objects_llm": record.get("n_objects"),
                     "density_llm": record.get("density_class"),
                     "density_true": meta.loc[image_id, "density"],
                     "shape_regularity": record.get("shape_regularity"),
                     "quality_flag": record.get("quality_flag"),
                     "missing_keys": ";".join(out["missing_keys"]),
                     "seconds": out["seconds"]})

    df = pd.DataFrame(rows)
    df.to_csv(setting.csv_dir / "task2_numbers_first.csv", index=False)
    return df

if __name__ == "__main__":
    print("otsu and watershed on the test split")
    with_ws = evaluate_split("test", use_watershed=True)
    print(with_ws.to_string(index=False), "\n")

    print("mean by density:")
    print(with_ws.groupby("density")[["dice", "iou", "count_error"]].mean().round(3).to_string(), "\n")

    print("connected components only, no watershed")
    without = evaluate_split("test", use_watershed=False)
    compare = pd.DataFrame({
        "density": with_ws["density"],
        "count_error_watershed": with_ws["count_error"],
        "count_error_plain": without["count_error"]})
    print(compare.groupby("density").mean().round(2).to_string(), "\n")

    example = "test_001"
    out = segment(load_image(example, "test"))
    df = feature_table(out["labels"], out["grey"])
    print(f"feature table for {example} (first 5 of {len(df)} rows)")
    print(df.head().round(3).to_string(index=False), "\n")
    print("summary text that goes to the LLM")
    print(summarise_table(df, out["mask"], example))