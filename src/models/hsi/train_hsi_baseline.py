#!/usr/bin/env python3
"""
Training and ONNX Export Script for DPU-Compatible SS-ResNet on Hyperspectral Data.
Targets:
  - Dataset: Indian Pines (16 classes, 30 PCA bands, 13x13 window)
  - Target Hardware: AMD Kria KV260 (DPUCZDX8G) & NVIDIA Jetson Orin Nano (TensorRT)
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from ss_resnet import SSResNet

def load_data(name, data_dir="models/TGRS_2025_MCTGCL/data"):
    if name == 'Indian':
        data = sio.loadmat(os.path.join(data_dir, 'Indian.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_dir, 'Indian_gt.mat'))['indian_pines_gt']
    elif name == 'Pavia':
        data = sio.loadmat(os.path.join(data_dir, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_dir, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise ValueError(f"Unknown dataset {name}")
    return data, labels

def apply_pca(X, num_components=30):
    orig_shape = X.shape
    flat_X = np.reshape(X, (-1, orig_shape[2]))
    pca = PCA(n_components=num_components, whiten=True, random_state=42)
    pca_X = pca.fit_transform(flat_X)
    pca_X = np.reshape(pca_X, (orig_shape[0], orig_shape[1], num_components))
    return pca_X, pca

def pad_with_zeros(X, margin=2):
    padded = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=X.dtype)
    padded[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return padded

def create_image_cubes(X, y, window_size=13, remove_zeros=True):
    margin = int((window_size - 1) / 2)
    zero_padded = pad_with_zeros(X, margin=margin)
    h, w, c = X.shape
    total_pixels = h * w
    patches_data = np.zeros((total_pixels, window_size, window_size, c), dtype=np.float32)
    patches_labels = np.zeros(total_pixels, dtype=np.int64)

    idx = 0
    for r in range(margin, zero_padded.shape[0] - margin):
        for col in range(margin, zero_padded.shape[1] - margin):
            patch = zero_padded[r - margin:r + margin + 1, col - margin:col + margin + 1, :]
            patches_data[idx] = patch
            patches_labels[idx] = y[r - margin, col - margin]
            idx += 1

    if remove_zeros:
        mask = patches_labels > 0
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1  # 0-indexed classes

    return patches_data, patches_labels

class HSIDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def compute_metrics(y_true, y_pred, dataset_name='Indian'):
    if dataset_name == 'Indian':
        target_names = [
            'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture',
            'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed', 'Oats',
            'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat',
            'Woods', 'Buildings-grass-trees-drives', 'Stone-steel towers'
        ]
    else:
        target_names = [f"Class {i+1}" for i in range(len(np.unique(y_true)))]

    oa = accuracy_score(y_true, y_pred) * 100.0
    cm = confusion_matrix(y_true, y_pred)
    diag = np.diag(cm)
    row_sum = np.sum(cm, axis=1)
    each_acc = np.nan_to_num(truediv(diag, row_sum)) * 100.0
    aa = np.mean(each_acc)
    kappa = cohen_kappa_score(y_true, y_pred) * 100.0
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return oa, aa, kappa, report, cm

def main():
    parser = argparse.ArgumentParser(description="Train DPU-Compatible SS-ResNet for HSI")
    parser.add_argument("--dataset", type=str, default="Indian", choices=["Indian", "Pavia"])
    parser.add_argument("--components", type=int, default=30, help="PCA components")
    parser.add_argument("--patch_size", type=int, default=13, help="Spatial window size")
    parser.add_argument("--test_ratio", type=float, default=0.90, help="Ratio for test split (default 90% test, 10% train)")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 75)
    print(f"🚀 Training DPU-Compatible SS-ResNet on {args.dataset} Pines Dataset")
    print(f"   PCA Bands: {args.components} | Patch Size: {args.patch_size}x{args.patch_size}")
    print(f"   Train Ratio: {(1.0 - args.test_ratio)*100:.1f}% | Test Ratio: {args.test_ratio*100:.1f}%")
    print("=" * 75)

    # 1. Load Data
    raw_data, gt = load_data(args.dataset)
    num_classes = len(np.unique(gt)) - 1
    print(f"[*] Raw HSI shape: {raw_data.shape} | Ground Truth shape: {gt.shape} | Classes: {num_classes}")

    # 2. PCA
    print(f"[*] Reducing spectral bands to {args.components} using PCA...")
    pca_data, _ = apply_pca(raw_data, args.components)

    # 3. Create Cubes
    print(f"[*] Extracting {args.patch_size}x{args.patch_size} spatial-spectral cubes...")
    X_cubes, y_labels = create_image_cubes(pca_data, gt, window_size=args.patch_size)
    # Shape of X_cubes: (N, 13, 13, 30) -> Transpose to NCHW: (N, 30, 13, 13)
    X_cubes = np.transpose(X_cubes, (0, 3, 1, 2))
    print(f"[*] Total valid labeled patches: {X_cubes.shape[0]}, Shape: {X_cubes.shape}")

    # 4. Split Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X_cubes, y_labels, test_size=args.test_ratio, random_state=args.seed, stratify=y_labels
    )
    print(f"[*] Train patches: {len(X_train)} | Test patches: {len(X_test)}")

    # Save calibration set for Vitis AI INT8 quantization
    calib_dir = "models/vitis_ai"
    os.makedirs(calib_dir, exist_ok=True)
    calib_samples = min(256, len(X_test))
    np.save(os.path.join(calib_dir, f"calib_patches_{args.dataset.lower()}.npy"), X_test[:calib_samples])
    print(f"[*] Saved {calib_samples} test patches to {calib_dir}/calib_patches_{args.dataset.lower()}.npy for INT8 calibration.")

    train_loader = DataLoader(HSIDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(HSIDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    # 5. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    model = SSResNet(in_bands=args.components, num_classes=num_classes, patch_size=args.patch_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. Training Loop
    start_train = time.time()
    best_oa = 0.0
    best_weights_path = f"models/weights/ss_resnet_{args.dataset.lower()}.pt"

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        epoch_loss = train_loss / total
        epoch_acc = (correct / total) * 100.0

        # Quick validation every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            val_acc = (val_correct / val_total) * 100.0
            print(f"   Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            if val_acc > best_oa:
                best_oa = val_acc
                torch.save(model.state_dict(), best_weights_path)
        else:
            print(f"   Epoch [{epoch+1:02d}/{args.epochs:02d}] - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    train_time = time.time() - start_train
    print(f"\n✅ Training completed in {train_time:.2f} s. Best Val Acc: {best_oa:.2f}%")
    print(f"[*] Best weights saved to: {best_weights_path}")

    # 7. Final Comprehensive Evaluation
    print("\n[*] Evaluating Best Model on Full Test Set...")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    start_eval = time.time()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
    eval_time = time.time() - start_eval

    y_test_np = np.array(all_targets)
    y_pred_np = np.array(all_preds)
    oa, aa, kappa, report, cm = compute_metrics(y_test_np, y_pred_np, dataset_name=args.dataset)

    print("\n" + "=" * 75)
    print(f"📊 FINAL BASELINE ACCURACY ({args.dataset} Pines - PyTorch FP32):")
    print(f"   Overall Accuracy (OA): {oa:.2f} %")
    print(f"   Average Accuracy (AA): {aa:.2f} %")
    print(f"   Kappa Coefficient (κ): {kappa:.2f} %")
    print(f"   Test Set Evaluation Time: {eval_time:.2f} s ({len(y_test_np)/eval_time:.1f} patches/s)")
    print("=" * 75)
    print("\nPer-Class Classification Report:\n", report)

    # 8. Export ONNX Model
    onnx_path = f"models/onnx/ss_resnet_{args.dataset.lower()}.onnx"
    print(f"\n[*] Exporting to ONNX -> {onnx_path}...")
    model.cpu().eval()
    dummy_input = torch.randn(1, args.components, args.patch_size, args.patch_size, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}}
    )
    print(f"✅ Exported ONNX model ({os.path.getsize(onnx_path)/(1024*1024):.2f} MB): {onnx_path}")

    # Also export static batch=1 ONNX for strict hardware tools
    onnx_static_path = f"models/onnx/ss_resnet_{args.dataset.lower()}_b1.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_static_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"]
    )
    print(f"✅ Exported static Batch=1 ONNX model: {onnx_static_path}")

if __name__ == "__main__":
    main()
