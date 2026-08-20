from __future__ import annotations
from src.classical import dice, iou, split_touching
from src.data import list_corrupted, list_stem, load_corrupted, load_image, load_mask, load_labels
from src.data_loader import get_dataloaders
from src.meta_data import load_metadata
from src.preprocess import process
from src.unet import UNet, count_parameters
from config import setting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

for folder in (setting.model_dir, setting.fig_dir, setting.csv_dir):
    folder.mkdir(parents=True, exist_ok=True)

def get_device() -> torch.device:
    """mps on this mac, cuda if the code ever moves, cpu as the fallback"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """soft dice, computed on probabilities so it stays differentiable"""
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * intersection + eps) / (union + eps)).mean()

def get_loss(name: str = setting.loss):
    """3 options:
    - bce
    - dice
    - bce_dice (default), the sum of the two"""
    bce = nn.BCEWithLogitsLoss()
    if name == "bce":
        return bce
    elif name == "dice":
        return dice_loss
    elif name == "bce_dice":
        return lambda logits, target: bce(logits, target) + dice_loss(logits, target)
    raise ValueError("unknown provided loss name")

def batch_scores(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5,
                 eps: float = 1e-7) -> tuple[float, float]:
    """hard dice and iou of one batch, measured on the thresholded prediction rather than
    on the probabilities, so they match what the pipeline actually uses downstream"""
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * target).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    batch_dice = ((2 * intersection + eps) / (union + eps)).mean().item()
    batch_iou = ((intersection + eps) / (union - intersection + eps)).mean().item()
    return batch_dice, batch_iou

def run_epoch(model, loader, loss_fn, device, optimiser=None) -> dict:
    """one pass over a loader. Passing an optimiser makes it a training pass, leaving it out
    makes it an evaluation pass, so the two never drift apart"""
    training = optimiser is not None
    model.train() if training else model.eval()

    losses, dices, ious = [], [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            batch_dice, batch_iou = batch_scores(logits, y)
            losses.append(loss.item())
            dices.append(batch_dice)
            ious.append(batch_iou)

    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices)),
            "iou": float(np.mean(ious))}

def fit(loss_name: str = setting.loss, epochs: int = setting.epochs, lr: float = setting.lr,
        batch_size: int = setting.batch_size, augment: bool = True, seed: int = setting.seed,
        verbose: bool = True):
    
    torch.manual_seed(seed)
    device = get_device()

    model = UNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = get_loss(loss_name)
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, seed=seed, augment=augment)

    history, best_dice = [], -1.0
    checkpoint = setting.model_dir / f"unet_{loss_name}.pt"

    for epoch in range(1, epochs + 1):
        train_stats = run_epoch(model, train_loader, loss_fn, device, optimiser)
        val_stats = run_epoch(model, val_loader, loss_fn, device)

        history.append({"epoch": epoch,
                        "train_loss": train_stats["loss"], "train_dice": train_stats["dice"],
                        "val_loss": val_stats["loss"], "val_dice": val_stats["dice"],
                        "val_iou": val_stats["iou"]})

        if val_stats["dice"] > best_dice:
            best_dice = val_stats["dice"]
            torch.save(model.state_dict(), checkpoint)

        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            print(f"epoch {epoch}: train loss {train_stats['loss']:.4f}, "
                  f"val loss {val_stats['loss']:.4f}, val dice {val_stats['dice']:.4f}")

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model, pd.DataFrame(history), best_dice

def load_model(loss_name: str = setting.loss) -> nn.Module:
    """bring back the best checkpoint saved for a given loss"""
    device = get_device()
    model = UNet().to(device)
    model.load_state_dict(torch.load(setting.model_dir / f"unet_{loss_name}.pt",
                                     map_location=device))
    model.eval()
    return model

def predict_mask(model, img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    device = next(model.parameters()).device
    grey = process(img, setting.gray_method, setting.norm)
    x = torch.from_numpy(grey)[None, None].float().to(device)

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(x))
    return prob[0, 0].cpu().numpy() > threshold

def evaluate_split(model, split: str = "test", count_objects: bool = True) -> pd.DataFrame:
    """run the model on every image in a split, compare the predicted mask to the label map"""
    meta = load_metadata()
    rows = []
    for image_id in list_stem(split):
        pred = predict_mask(model, load_image(image_id, split))
        truth = load_mask(image_id, split)

        row = {"image_id": image_id, "density": meta.loc[image_id, "density"],
               "dice": round(dice(pred, truth), 4), "iou": round(iou(pred, truth), 4)}
        if count_objects:
            n_pred = int(split_touching(pred).max())
            n_visible = len(np.unique(load_labels(image_id, split))) - 1
            row.update({"n_pred": n_pred, "n_visible": n_visible,
                        "count_error": n_pred - n_visible})
        rows.append(row)
    return pd.DataFrame(rows)

def plot_history(history: pd.DataFrame, loss_name: str = setting.loss):
    """loss on the left, dice on the right, the pair of curves the report needs"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["epoch"], history["train_loss"], color="blue", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], color="orange", label="validation")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel(f"{loss_name} loss")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["train_dice"], color="blue", label="train")
    axes[1].plot(history["epoch"], history["val_dice"], color="orange", label="validation")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("dice")
    axes[1].legend()

    fig.suptitle(f"u-net training, loss = {loss_name}")
    fig.tight_layout()
    fig.savefig(setting.fig_dir / f"07_training_{loss_name}.png", dpi=120, bbox_inches="tight")
    return fig

def prediction_panel(model, split: str = "test", image_ids: list | None = None,
                     loss_name: str = setting.loss):
    """input, ground truth and prediction side by side, one row per density regime.
        The dice printed on each row is for that image alone, not the split average"""
    meta = load_metadata()
    if image_ids is None:
        rows = meta[meta["split"] == split]
        image_ids = [rows[rows["density"] == d].index[0]
                     for d in ["sparse", "normal", "dense", "clustered"]
                     if not rows[rows["density"] == d].empty]

    fig, axes = plt.subplots(len(image_ids), 3, figsize=(9, 3 * len(image_ids)))
    for row, image_id in enumerate(image_ids):
        img = load_image(image_id, split)
        grey = process(img, setting.gray_method, setting.norm)
        truth = load_mask(image_id, split)
        pred = predict_mask(model, img)

        axes[row, 0].imshow(grey, cmap="grey")
        axes[row, 0].set_title(f"{image_id}, {meta.loc[image_id, 'density']}", fontsize=9)
        axes[row, 1].imshow(truth, cmap="grey")
        axes[row, 1].set_title("ground truth", fontsize=9)
        axes[row, 2].imshow(pred, cmap="grey")
        axes[row, 2].set_title(f"u-net prediction, dice {dice(pred, truth):.3f}", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"u-net input, ground truth and prediction (loss = {loss_name})")
    fig.tight_layout()
    fig.savefig(setting.fig_dir / f"08_prediction_panel_{loss_name}.png", dpi=120,
                bbox_inches="tight")
    return fig

def compare_losses(names: tuple[str, ...] = ("bce", "dice", "bce_dice")) -> pd.DataFrame:
    """train one model per loss with everything else pinned, then score them on test"""
    rows = []
    for name in names:
        print(f"--- training with {name} ---")
        model, history, best_dice = fit(loss_name=name, verbose=False)
        history.to_csv(setting.csv_dir / f"history_{name}.csv", index=False)
        plot_history(history, name)

        scores = evaluate_split(model, "test")
        rows.append({"loss": name, "val_dice": round(best_dice, 4),
                     "test_dice": round(scores["dice"].mean(), 4),
                     "test_iou": round(scores["iou"].mean(), 4),
                     "count_MAE": round(scores["count_error"].abs().mean(), 2)})
    return pd.DataFrame(rows)

def compare_with_classical(model, split: str = "test") -> pd.DataFrame:
    """otsu against the u-net, same scoring and same watershed, so only the mask source differs"""
    from src.classical import evaluate_split as classical_split

    otsu = classical_split(split)
    unet = evaluate_split(model, split)
    both = pd.DataFrame({"density": otsu["density"],
                         "dice_otsu": otsu["dice"], "dice_unet": unet["dice"],
                         "count_error_otsu": otsu["count_error"],
                         "count_error_unet": unet["count_error"]})
    summary = both.groupby("density").mean().round(3)
    summary.loc["all images"] = both[both.columns[1:]].mean().round(3)
    return summary

def robustness_table(model) -> pd.DataFrame:
    """both segmenters on the corrupted copies, scored against the clean ground truth"""
    from src.classical import segment

    rows = []
    for stem, base_id, corruption in list_corrupted():
        truth = load_mask(base_id, "test")
        for image_id, tag in [(base_id, "clean"), (stem, corruption)]:
            img = load_corrupted(image_id) if tag != "clean" else load_image(image_id, "test")
            rows.append({"base_id": base_id, "variant": tag,
                         "dice_otsu": round(dice(segment(img)["mask"], truth), 4),
                         "dice_unet": round(dice(predict_mask(model, img), truth), 4)})
    return pd.DataFrame(rows).drop_duplicates()

def compare_normalisation(model) -> pd.DataFrame:
    """the same corrupted images through the u-net, once with fixed normalisation and once with per-image min-max"""
    rows = []
    for stem, base_id, corruption in list_corrupted():
        truth = load_mask(base_id, "test")
        row = {"image": stem}
        for mode in ("fixed", "per_image"):
            grey = process(load_corrupted(stem), setting.gray_method, mode)
            x = torch.from_numpy(grey)[None, None].float().to(next(model.parameters()).device)
            with torch.no_grad():
                pred = torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5
            row[mode] = round(dice(pred, truth), 4)
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    print(f"device: {get_device()}, parameters: {count_parameters(UNet()):,}\n")

    model, history, best_dice = fit()
    print(f"\n best validation dice: {best_dice:.4f}")
    history.to_csv(setting.csv_dir / f"history_{setting.loss}.csv", index=False)
    plot_history(history)

    prediction_panel(model, "test")

    scores = evaluate_split(model, "test")
    scores.to_csv(setting.csv_dir / f"unet_test_{setting.loss}.csv", index=False)
    print("\n u-net on the test split")
    print(scores.to_string(index=False))
    print("\n mean by density:")
    print(scores.groupby("density")[["dice", "iou", "count_error"]].mean().round(3).to_string())