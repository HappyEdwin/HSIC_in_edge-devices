#!/usr/bin/env python3
"""
Stand-alone COCO Object Detection Evaluation Module.
Computes mAP@50 and mAP@50-95 directly from predictions and YOLO format ground-truth labels.
Compatible with standard edge environments without heavy external dependencies.
"""

import os
import glob
from typing import Dict, List, Tuple
import numpy as np

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes IoU between two sets of boxes: boxes1 (N, 4) and boxes2 (M, 4).
    Format: [x1, y1, x2, y2].
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0, a_max=None)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using standard 101-point interpolation (COCO standard).
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point interpolation
    i = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(i, mrec, mpre), i)
    return float(ap)

def load_yolo_labels(label_path: str, img_w: int = 640, img_h: int = 640) -> np.ndarray:
    """
    Loads YOLO format ground truth labels [cls, x_c, y_c, w, h] and converts to [cls, x1, y1, x2, y2].
    """
    if not os.path.exists(label_path):
        return np.empty((0, 5), dtype=np.float32)

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = [float(x) for x in parts[1:5]]
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                boxes.append([cls_id, x1, y1, x2, y2])

    if not boxes:
        return np.empty((0, 5), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)

def evaluate_predictions(
    all_predictions: Dict[str, np.ndarray],
    labels_dir: str,
    img_w: int = 640,
    img_h: int = 640
) -> Dict[str, float]:
    """
    Evaluates predictions against ground truth labels across COCO IoU thresholds (0.50:0.95).
    all_predictions: {image_stem: np.ndarray of shape (K, 6) -> [x1, y1, x2, y2, score, cls]}
    """
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    num_iou = len(iou_thresholds)
    
    unique_classes = set()
    gt_by_img = {}
    for stem in all_predictions.keys():
        lbl_file = os.path.join(labels_dir, f"{stem}.txt")
        gt = load_yolo_labels(lbl_file, img_w, img_h)
        gt_by_img[stem] = gt
        if len(gt) > 0:
            unique_classes.update(gt[:, 0].astype(int).tolist())

    if not unique_classes:
        return {"mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}

    ap_per_class = {c: np.zeros(num_iou, dtype=np.float32) for c in unique_classes}

    for c in unique_classes:
        # Collect detections and targets for class c
        scores = []
        matches = {k: [] for k in range(num_iou)}
        n_targets = 0

        for stem, preds in all_predictions.items():
            gt = gt_by_img[stem]
            c_gt = gt[gt[:, 0] == c][:, 1:] if len(gt) > 0 else np.empty((0, 4))
            n_targets += len(c_gt)

            if len(preds) == 0:
                continue

            c_preds = preds[preds[:, 5] == c]
            if len(c_preds) == 0:
                continue

            # Sort predictions by score descending
            order = np.argsort(-c_preds[:, 4])
            c_preds = c_preds[order]
            p_boxes = c_preds[:, :4]
            p_scores = c_preds[:, 4]

            scores.extend(p_scores.tolist())

            if len(c_gt) == 0:
                for k in range(num_iou):
                    matches[k].extend([0] * len(p_boxes))
                continue

            ious = box_iou(p_boxes, c_gt)

            for k, iou_thresh in enumerate(iou_thresholds):
                matched_gt = set()
                img_matches = []
                for p_idx in range(len(p_boxes)):
                    best_gt = -1
                    best_iou = iou_thresh
                    for gt_idx in range(len(c_gt)):
                        if gt_idx in matched_gt:
                            continue
                        if ious[p_idx, gt_idx] > best_iou:
                            best_iou = ious[p_idx, gt_idx]
                            best_gt = gt_idx
                    if best_gt >= 0:
                        matched_gt.add(best_gt)
                        img_matches.append(1)
                    else:
                        img_matches.append(0)
                matches[k].extend(img_matches)

        if n_targets == 0:
            continue

        scores_np = np.array(scores)
        if len(scores_np) == 0:
            continue

        sort_idx = np.argsort(-scores_np)

        for k in range(num_iou):
            k_matches = np.array(matches[k])[sort_idx]
            tp = np.cumsum(k_matches)
            fp = np.cumsum(1 - k_matches)
            recall = tp / n_targets
            precision = tp / (tp + fp)
            ap_per_class[c][k] = compute_ap(recall, precision)

    all_aps = np.array(list(ap_per_class.values()))
    map50 = float(np.mean(all_aps[:, 0])) if len(all_aps) > 0 else 0.0
    map50_95 = float(np.mean(all_aps)) if len(all_aps) > 0 else 0.0

    return {
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
    }

if __name__ == "__main__":
    print("COCO Evaluator module loaded successfully.")
