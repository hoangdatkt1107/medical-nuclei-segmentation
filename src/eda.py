from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import setting
from src.data import list_stem, load_image, load_mask, load_labels, load_all, list_corrupted, load_corrupted
from src.meta_data import load_metadata
from src.preprocess import to_grayscale, normalise

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# one colour per density regime, reused in every plot so the figures read together
DENSITY_COLOUR = {"sparse": "green", "normal": "blue", "dense": "orange", "clustered": "red"}


def save(fig, name: str) -> Path:
    """save a figure into outputs/figures and give back the path"""
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    return path

def pick_one_per_density(split: str = "train") -> list:
    """one image id per density regime, so a figure shows all four difficulty levels"""
    meta = load_metadata()
    rows = meta[meta["split"] == split]
    picked = []
    for d in ["sparse", "normal", "dense", "clustered"]:
        ids = rows[rows["density"] == d].index.tolist()
        if ids:
            picked.append(ids[0])
    return picked


def sample_grid(split: str = "train", ids: list | None = None):
    """image on the top row, its mask underneath, one column per density regime"""
    meta = load_metadata()
    ids = pick_one_per_density(split) if ids is None else ids

    fig, axes = plt.subplots(2, len(ids), figsize=(3 * len(ids), 6))
    for col, img_id in enumerate(ids):
        img, mask, _ = load_all(img_id, split)
        row = meta.loc[img_id]

        axes[0, col].imshow(img)
        axes[0, col].set_title(f"{img_id}\n{row['density']}, {row['n_objects']} nuclei", fontsize=9)
        axes[1, col].imshow(mask, cmap="grey")
        axes[1, col].set_title("ground truth mask", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"{split} split, one image per density regime")
    fig.tight_layout()
    return fig

def intensity_hist(split: str = "train", n_images: int = 20):
    """foreground vs background intensity over several images"""
    ids = list_stem(split)[:n_images]
    fg, bg = [], []
    for img_id in ids:
        img, mask, _ = load_all(img_id, split)
        grey = normalise(to_grayscale(img, setting.gray_method))
        fg.append(grey[mask])
        bg.append(grey[~mask])
    fg, bg = np.concatenate(fg), np.concatenate(bg)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 80)
    ax.hist(bg, bins=bins, color="grey", alpha=0.7, label="background")
    ax.hist(fg, bins=bins, color="blue", alpha=0.7, label="nuclei")
    ax.set_yscale("log")  
    ax.set_xlabel(f"intensity after {setting.gray_method} channel + {setting.norm} normalisation")
    ax.set_ylabel("pixel count (log)")
    ax.set_title(f"intensity distribution, {len(ids)} {split} images")
    ax.legend()
    fig.tight_layout()
    return fig

def channel_compare(img_id: str = "train_000", split: str = "train"):
    """blue vs luminance vs max, with the foreground-background gap printed on each panel
    """
    img, mask, _ = load_all(img_id, split)
    methods = ["blue", "luminance", "max"]

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    gaps = {}
    for ax, method in zip(axes, methods):
        grey = normalise(to_grayscale(img, method))
        gap = (grey[mask].mean() - grey[~mask].mean()) * 255
        gaps[method] = gap
        ax.imshow(grey, cmap="grey", vmin=0, vmax=1)
        ax.set_title(f"{method}\nseparation = {gap:.1f}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"grayscale conversion on {img_id}")
    fig.tight_layout()
    return fig, gaps

def split_summary() -> pd.DataFrame:
    """how many images per split and per density, plus the object counts"""
    meta = load_metadata()
    table = meta.groupby(["split", "density"]).agg(
        n_images=("n_objects", "size"),
        objects_mean=("n_objects", "mean"),
        objects_min=("n_objects", "min"),
        objects_max=("n_objects", "max")).round(1)
    return table

def density_plot():
    """left: how many images of each regime. right: how crowded they actually are"""
    meta = load_metadata()
    order = ["sparse", "normal", "dense", "clustered"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    counts = meta["density"].value_counts()
    axes[0].bar(order, [counts.get(d, 0) for d in order],
                color=[DENSITY_COLOUR[d] for d in order])
    axes[0].set_ylabel("number of images")
    axes[0].set_title("images per density regime (all splits)")

    axes[1].boxplot([meta[meta["density"] == d]["n_objects"] for d in order], tick_labels=order)
    axes[1].set_ylabel("nuclei per image")
    axes[1].set_title("object count by regime")

    fig.tight_layout()
    return fig

def gt_audit() -> pd.DataFrame:
    """compare the metadata count against what is actually readable from the label map"""

    meta = load_metadata()
    rows = []
    for split in setting.split:
        for img_id in list_stem(split):
            labels = load_labels(img_id, split)
            visible = len(np.unique(labels)) - 1
            placed = int(meta.loc[img_id, "n_objects"])
            rows.append({"image_id": img_id, "split": split,
                         "density": meta.loc[img_id, "density"],
                         "placed": placed, "visible": visible, "lost": placed - visible})
    return pd.DataFrame(rows)


def gt_audit_plot(df: pd.DataFrame | None = None):
    """placed against visible. Points below the diagonal are images with buried nuclei."""
    df = gt_audit() if df is None else df

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for d, colour in DENSITY_COLOUR.items():
        sub = df[df["density"] == d]
        ax.scatter(sub["placed"], sub["visible"], color=colour, label=d, s=18, alpha=0.8)

    top = df["placed"].max() + 5
    ax.plot([0, top], [0, top], color="black", linewidth=1, linestyle="--", label="perfect agreement")
    ax.set_xlabel("n_objects in metadata (nuclei placed)")
    ax.set_ylabel("distinct ids in the label map (nuclei visible)")
    ax.set_title(f"{(df['lost'] > 0).sum()} of {len(df)} images lose nuclei to overlap")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig

def corrupted_preview():
    entries = list_corrupted()

    fig, axes = plt.subplots(2, len(entries), figsize=(3 * len(entries), 6))
    for col, (stem, base_id, corruption) in enumerate(entries):
        axes[0, col].imshow(load_image(base_id, "test"))
        axes[0, col].set_title(f"{base_id}\noriginal", fontsize=9)
        axes[1, col].imshow(load_corrupted(stem))
        axes[1, col].set_title(corruption, fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("robustness set: clean test images and their corrupted variants")
    fig.tight_layout()
    return fig

def run_all():
    """regenerate every figure in one go and report where they landed"""
    saved = []
    saved.append(save(sample_grid("train"), "01_sample_grid"))
    saved.append(save(intensity_hist("train"), "02_intensity_hist"))
    fig, gaps = channel_compare("train_000")
    saved.append(save(fig, "03_channel_compare"))
    saved.append(save(density_plot(), "04_density"))
    saved.append(save(gt_audit_plot(), "05_gt_audit"))
    saved.append(save(corrupted_preview(), "06_corrupted"))
    plt.close("all")

    print("separation by channel:", {k: round(v, 1) for k, v in gaps.items()})
    for p in saved:
        print("saved", p.relative_to(FIG_DIR.parents[1]))
    return saved

if __name__ == "__main__":
    print(split_summary().to_string(), "\n")
    audit = gt_audit()
    print(f"images losing nuclei to overlap: {(audit['lost'] > 0).sum()} / {len(audit)}")
    print(audit.groupby("density")["lost"].agg(["mean", "max"]).round(2).to_string(), "\n")
    run_all()
