from __future__ import annotations
from src.classical import feature_table, nearest_neighbour, segment, split_touching, summarise_table
from src.data import list_corrupted, list_stem, load_corrupted, load_image, load_labels, load_mask
from src.llm import ask_json, check_keys, load_prompt, save_json
from src.meta_data import load_metadata
from config import setting
import json
import numpy as np
import pandas as pd
import time
from src.preprocess import process
from src.train import load_model, predict_mask

RECORD_KEYS = ["image_id", "n_objects", "mean_area", "density_class", "quality_flag", "narrative"]
JSON_DIR = setting.out_dir / "pipeline"
JSON_DIR.mkdir(parents=True, exist_ok=True)

def get_mask(img: np.ndarray, source: str = "unet", model=None) -> np.ndarray:
    """2 options for where the mask comes from:
    -unet (default), the trained network
    -otsu, the classical route from task 2"""
    if source == "unet":
        return predict_mask(model, img)
    elif source == "otsu":
        return segment(img)["mask"]
    raise ValueError("unknown provided mask source")

def density_from_numbers(n_objects: int, nn_distance: float | None) -> str:
    """the same bands the prompt states, applied in code"""
    if nn_distance is not None and nn_distance < 12:
        return "clustered"
    elif n_objects <= 13:
        return "sparse"
    elif n_objects <= 40:
        return "normal"
    return "dense"

def check_numbers(record: dict, df: pd.DataFrame) -> list:
    """the model is asked to copy n_objects and mean_area rather than judge them"""
    issues = []
    try:
        if int(record.get("n_objects")) != len(df):
            issues.append(f"n_objects {record.get('n_objects')} != measured {len(df)}")
    except (TypeError, ValueError):
        issues.append(f"n_objects not a number: {record.get('n_objects')!r}")

    measured_area = float(df["area"].mean()) if not df.empty else 0.0
    try:
        if abs(float(record.get("mean_area")) - measured_area) > 1.0:
            issues.append(f"mean_area {record.get('mean_area')} != measured {measured_area:.0f}")
    except (TypeError, ValueError):
        issues.append(f"mean_area not a number: {record.get('mean_area')!r}")
    return issues

def run_one(image_id: str, split: str = "test", source: str = "unet", model=None) -> dict:
    """raw image -> mask -> feature table -> json record -> narrative, for one image"""
    corrupted = "_blur" in image_id or "_lowcontrast" in image_id
    img = load_corrupted(image_id) if corrupted else load_image(image_id, split)

    started = time.time()
    mask = get_mask(img, source, model)
    labels = split_touching(mask)
    grey = process(img, setting.gray_method, setting.norm)
    df = feature_table(labels, grey)
    summary = summarise_table(df, mask, image_id)

    prompt = load_prompt("pipeline_record").replace("{{FEATURES}}", summary)
    record, raw = ask_json(prompt)

    return {"image_id": image_id, "split": split, "source": source,
            "seconds": round(time.time() - started, 1),
            "measured": {"n_objects": len(df),
                         "mean_area": round(float(df["area"].mean()), 1) if not df.empty else 0.0,
                         "area_fraction": round(float(mask.mean()), 4),
                         "mean_eccentricity": round(float(df["eccentricity"].mean()), 3) if not df.empty else 0.0,
                         "nearest_neighbour": round(nearest_neighbour(df), 1) if len(df) > 1 else None},
            "missing_keys": check_keys(record, RECORD_KEYS),
            "number_issues": check_numbers(record, df),
            "summary_sent": summary, "record": record, "raw": raw}

def run_split(split: str = "test", source: str = "unet", verbose: bool = True) -> pd.DataFrame:
    """run every image of a split and collect the json records into one table"""
    meta = load_metadata()
    model = load_model() if source == "unet" else None

    rows = []
    for image_id in list_stem(split):
        out = run_one(image_id, split, source, model)
        save_json(out, f"pipeline/{source}_{image_id}")

        record = out["record"]
        n_visible = len(np.unique(load_labels(image_id, split))) - 1
        rows.append({"image_id": image_id,
                     "n_objects": out["measured"]["n_objects"],
                     "mean_area": out["measured"]["mean_area"],
                     "density_llm": record.get("density_class"),
                     "density_rule": density_from_numbers(out["measured"]["n_objects"],
                                                          out["measured"]["nearest_neighbour"]),
                     "quality_flag": record.get("quality_flag"),
                     "narrative": record.get("narrative"),
                     "density_true": meta.loc[image_id, "density"],
                     "n_visible": n_visible,
                     "count_error": out["measured"]["n_objects"] - n_visible,
                     "number_issues": "; ".join(out["number_issues"]),
                     "seconds": out["seconds"]})
        if verbose:
            flag = "ok" if not out["number_issues"] else "NUMBERS"
            print(f"  {image_id}: {rows[-1]['n_objects']} objects, "
                  f"llm={rows[-1]['density_llm']}, rule={rows[-1]['density_rule']}, "
                  f"true={rows[-1]['density_true']}, {flag}")

    df = pd.DataFrame(rows)
    df.to_csv(setting.csv_dir / f"pipeline_{split}_{source}.csv", index=False)
    return df

def run_corrupted(source: str = "unet") -> pd.DataFrame:
    """the robustness extension: same pipeline on the corrupted copies, next to their clean originals"""
    model = load_model() if source == "unet" else None
    rows = []
    for stem, base_id, corruption in list_corrupted():
        for image_id, tag in [(base_id, "clean"), (stem, corruption)]:
            out = run_one(image_id, "test", source, model)
            truth = load_mask(base_id, "test")
            mask = get_mask(load_corrupted(image_id) if tag != "clean"
                            else load_image(image_id, "test"), source, model)
            rows.append({"base_id": base_id, "variant": tag,
                         "n_objects": out["measured"]["n_objects"],
                         "mean_area": out["measured"]["mean_area"],
                         "area_fraction": out["measured"]["area_fraction"],
                         "dice_vs_clean_truth": round(float(2 * (mask & truth).sum() /
                                                            (mask.sum() + truth.sum() + 1e-7)), 4),
                         "density_class": out["record"].get("density_class"),
                         "quality_flag": out["record"].get("quality_flag")})
    df = pd.DataFrame(rows)
    df.to_csv(setting.csv_dir / f"pipeline_corrupted_{source}.csv", index=False)
    return df

def check_density_rule(splits=("train", "val")) -> tuple:
    """how often the band rule agrees with the metadata label, measured on the ground truth label maps rather than on pipeline output"""
    from skimage.measure import regionprops_table

    meta = load_metadata()
    rows = []
    for split in splits:
        for image_id in list_stem(split):
            props = pd.DataFrame(regionprops_table(load_labels(image_id, split),
                                                   properties=("area", "centroid")))
            rows.append({"image_id": image_id, "true": meta.loc[image_id, "density"],
                         "predicted": density_from_numbers(len(props), nearest_neighbour(props))})
    df = pd.DataFrame(rows)
    return df, round((df["true"] == df["predicted"]).mean(), 3)

def ask_llm_to_apply_bands(model: str | None = None) -> pd.DataFrame:
    """give the model four sets of numbers and the bands, and ask only for the class"""
    from src.llm import ask_json

    prompt = ("An image has N_OBJ detected objects and a median nearest-neighbour distance of "
              "NN_VAL pixels.\n Bands: objects <= 13 -> sparse; objects 14 to 40 and nn >= 12 -> "
              "normal; objects >= 41 and nn >= 12 -> dense; nn < 12 -> clustered.\n"
              'Reply as {"density_class": "..."} only.')
    cases = [(78, 15.2, "dense"), (8, 33.9, "sparse"), (24, 22.0, "normal"), (32, 10.1, "clustered")]

    rows = []
    for n, nn, expected in cases:
        record, _ = ask_json(prompt.replace("N_OBJ", str(n)).replace("NN_VAL", str(nn)), model=model)
        rows.append({"n_objects": n, "nearest_neighbour": nn, "expected": expected,
                     "llm_answer": record.get("density_class"),
                     "rule_answer": density_from_numbers(n, nn)})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print("full pipeline on the unseen test images, u-net mask")
    df = run_split("test", "unet")

    print(f"\n density_class from the llm : {(df['density_llm'] == df['density_true']).sum()}/{len(df)} correct")
    print(f"density_class from the rule: {(df['density_rule'] == df['density_true']).sum()}/{len(df)} correct")
    print(f"records with a transcription problem: {(df['number_issues'] != '').sum()}/{len(df)}")
    print(f"quality_flag values: {df['quality_flag'].value_counts().to_dict()}")
    print(f"mean time per image: {df['seconds'].mean():.1f}s")
    print(f"\n saved {setting.csv_dir / 'pipeline_test_unet.csv'}")

    print("\nthe corrupted copies")
    print(run_corrupted("unet").drop_duplicates().to_string(index=False))

    print("one example record")
    example = json.loads((JSON_DIR / "unet_test_005.json").read_text())
    print(json.dumps(example["record"], indent=2)[:900])
