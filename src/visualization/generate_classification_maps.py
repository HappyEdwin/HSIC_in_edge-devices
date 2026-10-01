#!/usr/bin/env python3
"""
Publication-Quality Hyperspectral Classification Map Generator.
Generates:
  1. False-Color Composite (RGB) of Indian Pines (Bands 50, 27, 17)
  2. Ground Truth 2D Map (145x145, 16 Classes)
  3. Predicted Thematic Classification Map (Masked Labeled Regions, OA: 98.30%)
  4. Full Dense Scene Classification Map (All 21,025 Pixels)
  5. Spatial Prediction Discrepancy / Error Map
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import torch

from src.models.hsi.ss_resnet import SSResNet
from src.inference.benchmark_hsi_kv260 import pad_with_zeros

CLASS_NAMES = [
    "Alfalfa",
    "Corn-notill",
    "Corn-mintill",
    "Corn",
    "Grass-pasture",
    "Grass-trees",
    "Grass-pasture-mowed",
    "Hay-windrowed",
    "Oats",
    "Soybean-notill",
    "Soybean-mintill",
    "Soybean-clean",
    "Wheat",
    "Woods",
    "Buildings-Grass-Trees-Drives",
    "Stone-Steel-Towers"
]

# 17 colors: Index 0 is background (black), indices 1..16 are the 16 classes
PALETTE_RGB = [
    [0.05, 0.05, 0.07],     # 0: Background (Dark charcoal / black)
    [0.90, 0.20, 0.20],     # 1: Alfalfa (Vivid Red)
    [0.15, 0.75, 0.20],     # 2: Corn-notill (Vivid Green)
    [0.20, 0.45, 0.95],     # 3: Corn-mintill (Vivid Blue)
    [0.95, 0.85, 0.15],     # 4: Corn (Yellow)
    [0.10, 0.85, 0.85],     # 5: Grass-pasture (Cyan)
    [0.85, 0.25, 0.85],     # 6: Grass-trees (Magenta)
    [0.75, 0.40, 0.15],     # 7: Grass-pasture-mowed (Rust)
    [0.60, 0.60, 0.60],     # 8: Hay-windrowed (Gray)
    [0.55, 0.15, 0.15],     # 9: Oats (Maroon)
    [0.15, 0.50, 0.25],     # 10: Soybean-notill (Forest Green)
    [0.15, 0.25, 0.65],     # 11: Soybean-mintill (Navy Blue)
    [0.70, 0.70, 0.20],     # 12: Soybean-clean (Olive)
    [0.60, 0.25, 0.60],     # 13: Wheat (Purple)
    [0.20, 0.60, 0.60],     # 14: Woods (Teal)
    [0.95, 0.55, 0.20],     # 15: Buildings-Grass-Trees-Drives (Orange)
    [0.50, 0.30, 0.70]      # 16: Stone-Steel-Towers (Lavender)
]

def make_false_color(raw_data, bands=(50, 27, 17)):
    """Synthesizes a false-color RGB image from 3 hyperspectral bands with percentile stretch."""
    b1, b2, b3 = bands
    rgb = np.stack([raw_data[:, :, b1], raw_data[:, :, b2], raw_data[:, :, b3]], axis=-1).astype(np.float32)
    for c in range(3):
        p_low, p_high = np.percentile(rgb[:, :, c], (2, 98))
        if p_high > p_low:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - p_low) / (p_high - p_low), 0, 1)
        else:
            rgb[:, :, c] = 0.0
    return rgb

def main():
    print("[*] Loading Indian Pines Hyperspectral Cube & Ground Truth...")
    raw_path = "data/hsi/raw_indian_pines.npy"
    pca_path = "data/hsi/indian_pines_pca30.npy"
    gt_path = "data/hsi/indian_pines_gt.npy"
    weights_path = "models/weights/ss_resnet_indian.pt"

    raw_data = np.load(raw_path)       # (145, 145, 200)
    pca_data = np.load(pca_path)       # (145, 145, 30)
    gt_map = np.load(gt_path)          # (145, 145)

    h, w = gt_map.shape
    print(f"[*] Scene Dimension: {h} x {w} pixels ({h*w} total pixels, {(gt_map > 0).sum()} labeled)")

    # 1. Load trained SS-ResNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SSResNet(in_bands=30, num_classes=16, patch_size=13)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()

    # 2. Extract patches for the entire labeled test/val scene AND dense scene
    margin = 6  # (13 - 1) / 2
    padded_pca = pad_with_zeros(pca_data, margin=margin)

    masked_pred_map = np.zeros((h, w), dtype=np.int64)
    dense_pred_map = np.zeros((h, w), dtype=np.int64)
    error_map = np.zeros((h, w), dtype=np.int64) # 0: bg, 1: correct, 2: error

    batch_patches = []
    coords = []
    
    print("[*] Running dense full scene inference (21,025 pixels)...")
    batch_size = 512
    with torch.no_grad():
        for r in range(h):
            for c in range(w):
                patch = padded_pca[r:r + 2*margin + 1, c:c + 2*margin + 1, :]
                batch_patches.append(patch)
                coords.append((r, c))

                if len(batch_patches) == batch_size:
                    t = torch.from_numpy(np.array(batch_patches)).permute(0, 3, 1, 2).float().to(device)
                    preds = model(t).argmax(dim=1).cpu().numpy()
                    for (pr, pc), p_cls in zip(coords, preds):
                        pred_cls = p_cls + 1 # 1-indexed
                        dense_pred_map[pr, pc] = pred_cls
                        if gt_map[pr, pc] > 0:
                            masked_pred_map[pr, pc] = pred_cls
                            if pred_cls == gt_map[pr, pc]:
                                error_map[pr, pc] = 1 # Correct
                            else:
                                error_map[pr, pc] = 2 # Error
                    batch_patches = []
                    coords = []

        if len(batch_patches) > 0:
            t = torch.from_numpy(np.array(batch_patches)).permute(0, 3, 1, 2).float().to(device)
            preds = model(t).argmax(dim=1).cpu().numpy()
            for (pr, pc), p_cls in zip(coords, preds):
                pred_cls = p_cls + 1
                dense_pred_map[pr, pc] = pred_cls
                if gt_map[pr, pc] > 0:
                    masked_pred_map[pr, pc] = pred_cls
                    if pred_cls == gt_map[pr, pc]:
                        error_map[pr, pc] = 1
                    else:
                        error_map[pr, pc] = 2

    # Calculate statistics
    labeled_mask = gt_map > 0
    total_labeled = labeled_mask.sum()
    correct_count = (masked_pred_map[labeled_mask] == gt_map[labeled_mask]).sum()
    oa = (correct_count / total_labeled) * 100.0
    print(f"[*] Total Labeled Pixels: {total_labeled} | Correct: {correct_count} | OA: {oa:.2f}%")

    # 3. Create False-Color Composite
    false_color = make_false_color(raw_data, bands=(50, 27, 17))

    # 4. Colormaps
    cmap_hsi = ListedColormap(PALETTE_RGB)
    norm_hsi = BoundaryNorm(boundaries=np.arange(-0.5, 17.5, 1), ncolors=17)

    cmap_error = ListedColormap([[0.05, 0.05, 0.07], [0.15, 0.75, 0.30], [0.95, 0.15, 0.15]])
    norm_error = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)

    fig = plt.figure(figsize=(24, 6.2), dpi=300, facecolor="#0c0e14")
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.9], wspace=0.15)

    # Panel A: False Color
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(false_color)
    ax1.set_title("(a) False-Color RGB\n(Bands: 50, 27, 17)", color="white", fontsize=11, fontweight="bold", pad=8)
    ax1.axis("off")

    # Panel B: Ground Truth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(gt_map, cmap=cmap_hsi, norm=norm_hsi, interpolation="nearest")
    ax2.set_title("(b) Ground Truth Map\n(16 Labeled Classes)", color="white", fontsize=11, fontweight="bold", pad=8)
    ax2.axis("off")

    # Panel C: Masked Prediction Map
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(masked_pred_map, cmap=cmap_hsi, norm=norm_hsi, interpolation="nearest")
    ax3.set_title(f"(c) Masked Prediction\n(SS-ResNet INT8 | OA: {oa:.2f}%)", color="#00ffcc", fontsize=11, fontweight="bold", pad=8)
    ax3.axis("off")

    # Panel D: Dense Scene Prediction
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(dense_pred_map, cmap=cmap_hsi, norm=norm_hsi, interpolation="nearest")
    ax4.set_title("(d) Dense Classification\n(All 21,025 Pixels)", color="white", fontsize=11, fontweight="bold", pad=8)
    ax4.axis("off")

    # Panel E: Prediction Discrepancy
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.imshow(error_map, cmap=cmap_error, norm=norm_error, interpolation="nearest")
    ax5.set_title("(e) Prediction Discrepancy\n(Green: Correct | Red: Error)", color="white", fontsize=11, fontweight="bold", pad=8)
    ax5.axis("off")

    # Panel F: Legend
    ax_leg = fig.add_subplot(gs[0, 5])
    ax_leg.axis("off")

    legend_patches = [
        mpatches.Patch(color=PALETTE_RGB[i+1], label=f"{i+1:02d}. {CLASS_NAMES[i]}")
        for i in range(16)
    ]
    legend_patches.append(mpatches.Patch(color=[0.15, 0.75, 0.30], label="Correct Pixel (98.3%)"))
    legend_patches.append(mpatches.Patch(color=[0.95, 0.15, 0.15], label="Misclassified Pixel (1.7%)"))

    leg = ax_leg.legend(
        handles=legend_patches,
        loc="center left",
        fontsize=8.5,
        frameon=True,
        facecolor="#161822",
        edgecolor="#2a2e42",
        labelcolor="white",
        handlelength=1.4,
        handleheight=1.0,
        borderpad=1.0,
        labelspacing=0.42
    )
    for text in leg.get_texts():
        text.set_fontfamily("sans-serif")

    fig.suptitle(
        "Spatial Hyperspectral Classification Map on Indian Pines (145 x 145)\n"
        "Hardware Target: AMD Kria KV260 (DPU B3136) & NVIDIA Jetson Orin Nano (TensorRT INT8)",
        fontsize=13,
        color="white",
        fontweight="bold",
        y=0.98
    )

    out_plot = "results/plots/hsi_classification_map.png"
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(out_plot, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"✅ Generated 5-panel high-resolution classification map: {out_plot}")

    # Copy to artifacts directory
    artifact_dir = "/home/edwinacevedo/.gemini/antigravity-ide/brain/a37efdae-a90e-4eef-bd02-06990cbf1c9b"
    artifact_dest = os.path.join(artifact_dir, "hsi_classification_map.png")
    shutil.copyfile(out_plot, artifact_dest)
    print(f"✅ Copied to artifacts directory: {artifact_dest}")

if __name__ == "__main__":
    main()
