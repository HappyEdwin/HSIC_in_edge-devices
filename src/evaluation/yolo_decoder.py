#!/usr/bin/env python3
"""
DPU YOLO Output Tensor Decoder & NMS Postprocessor.
Decodes raw fixed-point multi-scale output tensors from Xilinx DPU into bounding boxes,
class probabilities, and applies Non-Maximum Suppression (NMS).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.65) -> List[int]:
    """
    Standard Non-Maximum Suppression (NMS).
    boxes: (N, 4) in [x1, y1, x2, y2]
    scores: (N,)
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def decode_dfl_boxes(reg_tensor: np.ndarray, stride: int, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Decodes Distribution Focal Loss (DFL) coordinates.
    reg_tensor: (grid_h, grid_w, 64) -> 4 * 16 channels.
    Returns: (grid_h * grid_w, 4) in [x1, y1, x2, y2].
    """
    reg_tensor = reg_tensor.reshape((grid_h * grid_w, 4, 16))
    prob = softmax(reg_tensor, axis=-1)
    dfl_weights = np.arange(16, dtype=np.float32)
    dist = np.sum(prob * dfl_weights, axis=-1)  # (N, 4): [dx1, dy1, dx2, dy2]

    # Build grid coordinates
    yv, xv = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    xc = (xv.flatten() + 0.5) * stride
    yc = (yv.flatten() + 0.5) * stride

    x1 = xc - dist[:, 0] * stride
    y1 = yc - dist[:, 1] * stride
    x2 = xc + dist[:, 2] * stride
    y2 = yc + dist[:, 3] * stride

    return np.stack([x1, y1, x2, y2], axis=-1)

def postprocess_dpu_heads(
    scale_outputs: List[Tuple[np.ndarray, np.ndarray, int]],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    img_size: int = 640
) -> np.ndarray:
    """
    Decodes multi-scale DPU output heads:
    scale_outputs: list of (box_buf_float, cls_buf_float, stride)
      where box_buf_float is (1, H, W, 64) and cls_buf_float is (1, H, W, 80).
    Returns: np.ndarray of shape (K, 6) -> [x1, y1, x2, y2, score, cls_id]
    """
    all_boxes = []
    all_scores = []
    all_cls_ids = []

    for box_head, cls_head, stride in scale_outputs:
        if box_head.ndim == 4:
            box_head = box_head[0]
        if cls_head.ndim == 4:
            cls_head = cls_head[0]

        gh, gw, _ = box_head.shape
        boxes = decode_dfl_boxes(box_head, stride, gh, gw)  # (N, 4)

        cls_probs = sigmoid(cls_head.reshape(gh * gw, -1))  # (N, 80)
        max_cls_ids = np.argmax(cls_probs, axis=-1)
        max_scores = np.max(cls_probs, axis=-1)

        mask = max_scores >= conf_threshold
        if np.any(mask):
            all_boxes.append(boxes[mask])
            all_scores.append(max_scores[mask])
            all_cls_ids.append(max_cls_ids[mask])

    if not all_boxes:
        return np.empty((0, 6), dtype=np.float32)

    boxes_concat = np.concatenate(all_boxes, axis=0)
    scores_concat = np.concatenate(all_scores, axis=0)
    cls_ids_concat = np.concatenate(all_cls_ids, axis=0)

    # Clip coordinates to image boundary
    boxes_concat[:, [0, 2]] = np.clip(boxes_concat[:, [0, 2]], 0, img_size)
    boxes_concat[:, [1, 3]] = np.clip(boxes_concat[:, [1, 3]], 0, img_size)

    keep_indices = nms(boxes_concat, scores_concat, iou_threshold=iou_threshold)
    if not keep_indices:
        return np.empty((0, 6), dtype=np.float32)

    final_boxes = boxes_concat[keep_indices]
    final_scores = scores_concat[keep_indices, None]
    final_cls = cls_ids_concat[keep_indices, None].astype(np.float32)

    return np.concatenate([final_boxes, final_scores, final_cls], axis=-1)
