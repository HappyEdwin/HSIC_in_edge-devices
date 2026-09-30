#!/usr/bin/env python3
"""
Visual Detection & Multi-Precision Comparator (FP32, FP16, INT8).
Supports PyTorch (.pt), TensorRT (.engine), and Xilinx VART (.xmodel).
Annotates detections and generates side-by-side comparison images.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import argparse
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Color palette for 80 classes
np.random.seed(42)
COLORS = [tuple(np.random.randint(50, 255, size=3).tolist()) for _ in range(80)]

def draw_detections(
    image: Image.Image,
    detections: np.ndarray,
    title: str = "",
    conf_threshold: float = 0.25
) -> Image.Image:
    """
    Draws bounding boxes and labels onto the image.
    detections: (N, 6) -> [x1, y1, x2, y2, score, cls_id]
    """
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    
    # Try loading a clean font, fallback to default
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
        title_font = font

    # Draw title banner if specified
    if title:
        draw.rectangle([(0, 0), (img_draw.width, 32)], fill=(20, 24, 33))
        draw.text((12, 6), title, fill=(255, 255, 255), font=title_font)

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        if score < conf_threshold:
            continue

        cls_id = int(cls_id)
        cls_name = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        color = COLORS[cls_id % len(COLORS)]

        # Draw bounding box
        for offset in range(3):
            draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)

        label_txt = f"{cls_name} {score:.2f}"
        
        # Calculate label background
        try:
            bbox = font.getbbox(label_txt)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 80, 16

        ty1 = max(0, y1 - th - 6)
        ty2 = ty1 + th + 6
        tx2 = x1 + tw + 8

        draw.rectangle([x1, ty1, tx2, ty2], fill=color)
        draw.text((x1 + 4, ty1 + 2), label_txt, fill=(255, 255, 255), font=font)

    return img_draw

def run_pytorch_inference(model_path: str, image_path: str, img_size: int = 640) -> Tuple[np.ndarray, float]:
    """Inference for PyTorch .pt model (FP32)."""
    import time
    from ultralytics import YOLO
    model = YOLO(model_path)
    t0 = time.perf_counter()
    results = model(image_path, imgsz=img_size, verbose=False)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000.0

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()[:, None]
        cls = boxes.cls.cpu().numpy()[:, None]
        preds = np.concatenate([xyxy, conf, cls], axis=-1)
        return preds, lat_ms
    return np.empty((0, 6), dtype=np.float32), lat_ms

def run_tensorrt_inference(engine_path: str, image_path: str, img_size: int = 640) -> Tuple[np.ndarray, float]:
    """Inference for TensorRT .engine model (FP16 or INT8 on Jetson)."""
    import time
    from ultralytics import YOLO
    model = YOLO(engine_path, task="detect")
    t0 = time.perf_counter()
    results = model(image_path, imgsz=img_size, verbose=False)
    t1 = time.perf_counter()
    lat_ms = (t1 - t0) * 1000.0

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()[:, None]
        cls = boxes.cls.cpu().numpy()[:, None]
        preds = np.concatenate([xyxy, conf, cls], axis=-1)
        return preds, lat_ms
    return np.empty((0, 6), dtype=np.float32), lat_ms

def run_xmodel_inference(xmodel_path: str, image_path: str, img_size: int = 640) -> Tuple[np.ndarray, float]:
    """Inference for Xilinx DPU .xmodel (INT8 on Kria KV260)."""
    import time
    import xir
    import vart
    from src.evaluation.yolo_decoder import postprocess_dpu_heads

    t0 = time.perf_counter()
    graph = xir.Graph.deserialize(xmodel_path)
    root_subgraph = graph.get_root_subgraph()
    
    if hasattr(root_subgraph, "toposort_child_subgraph"):
        children = root_subgraph.toposort_child_subgraph()
    elif hasattr(root_subgraph, "children_topological_sort"):
        children = root_subgraph.children_topological_sort()
    else:
        children = root_subgraph.get_children()

    dpu_subs = [c for c in children if c.has_attr("device") and c.get_attr("device").upper() == "DPU"]
    runners = []
    first_scale = 1.0
    for sub in dpu_subs:
        r = vart.Runner.create_runner(sub, "run")
        in_t = r.get_input_tensors()
        out_t = r.get_output_tensors()
        if len(runners) == 0 and len(in_t) > 0:
            fp = in_t[0].get_attr("fix_point") or 0
            first_scale = 2.0 ** fp
        in_bufs = [np.zeros(t.dims, dtype=np.int8) for t in in_t]
        out_bufs = [np.empty(t.dims, dtype=np.int8) for t in out_t]
        runners.append((r, in_bufs, out_bufs, out_t))

    # Preprocess
    with Image.open(image_path) as img:
        img_rgb = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        img_np = np.asarray(img_rgb, dtype=np.float32) / 255.0
        q_in = (img_np * first_scale).astype(np.int8)
        runners[0][1][0] = np.expand_dims(q_in, axis=0)

    # DPU Execute
    for r, in_b, out_b, _ in runners:
        jid = r.execute_async(in_b, out_b)
        r.wait(jid)

    # Postprocess
    scale_outputs = []
    for _, _, out_bufs, out_tensors in runners:
        for b_idx, tensor in enumerate(out_tensors):
            dims = tensor.dims
            fp = tensor.get_attr("fix_point") or 0
            scale = 2.0 ** (-fp)
            buf_f = out_bufs[b_idx].astype(np.float32) * scale
            if len(dims) == 4 and dims[3] == 64:
                gh = dims[1]
                stride = img_size // gh
                for _, _, ob2, ot2 in runners:
                    for b2_idx, t2 in enumerate(ot2):
                        if len(t2.dims) == 4 and t2.dims[1] == gh and t2.dims[3] == 80:
                            fp2 = t2.get_attr("fix_point") or 0
                            cls_f = ob2[b2_idx].astype(np.float32) * (2.0 ** -fp2)
                            scale_outputs.append((buf_f, cls_f, stride))
                            break

    preds = postprocess_dpu_heads(scale_outputs, conf_threshold=0.25, iou_threshold=0.65, img_size=img_size)
    t1 = time.perf_counter()
    return preds, (t1 - t0) * 1000.0

def create_comparison_grid(images: List[Image.Image]) -> Image.Image:
    """Concatenates images horizontally into a clean comparison grid."""
    widths, heights = zip(*(i.size for i in images))
    total_w = sum(widths)
    max_h = max(heights)
    grid = Image.new("RGB", (total_w, max_h), color=(30, 30, 30))
    offset = 0
    for im in images:
        grid.paste(im, (offset, 0))
        offset += im.size[0]
    return grid

def main():
    parser = argparse.ArgumentParser(description="Visual Object Detection & Precision Comparator")
    parser.add_argument("--image", type=str, default="data/coco128/images/train2017/000000000009.jpg", help="Path to input test image")
    parser.add_argument("--model-fp32", type=str, default="models/weights/yolo11n.pt", help="PyTorch FP32 model")
    parser.add_argument("--model-fp16", type=str, default="models/engines/yolo11n_640_fp16.engine", help="TensorRT FP16 engine")
    parser.add_argument("--model-int8", type=str, default="models/xmodel/yolo11n_leaky_kv260.xmodel", help="INT8 model (xmodel or engine)")
    parser.add_argument("--output-dir", type=str, default="results/detections", help="Output directory for annotated images")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    img_path = args.image
    if not os.path.exists(img_path):
        # Search fallback
        candidates = glob.glob("data/coco128/images/train2017/*.jpg") + glob.glob("/home/edwinacevedo/VIP/datasets/coco128/images/train2017/*.jpg")
        if candidates:
            img_path = candidates[0]
        else:
            raise FileNotFoundError(f"Input image not found: {args.image}")

    img_name = Path(img_path).stem
    raw_img = Image.open(img_path).convert("RGB")
    rendered_panels = []

    print("=" * 70)
    print(f"🖼️ VISUAL DETECTION COMPARATOR: {img_name}")
    print(f"   Input Image: {img_path}")
    print(f"   Output Directory: {args.output_dir}")
    print("=" * 70)

    # 1. FP32 (PyTorch)
    if os.path.exists(args.model_fp32):
        print(f"[*] Evaluating FP32 model: {args.model_fp32}...")
        try:
            preds, lat_ms = run_pytorch_inference(args.model_fp32, img_path, args.imgsz)
            panel = draw_detections(raw_img, preds, f"FP32 PyTorch ({lat_ms:.1f} ms | {len(preds)} dets)", args.conf)
            panel.save(os.path.join(args.output_dir, f"{img_name}_fp32.jpg"))
            rendered_panels.append(panel)
            print(f"✅ FP32 completed: {len(preds)} detections in {lat_ms:.1f} ms")
        except Exception as e:
            print(f"[-] FP32 note: {e}")

    # 2. FP16 (TensorRT)
    if os.path.exists(args.model_fp16):
        print(f"[*] Evaluating FP16 model: {args.model_fp16}...")
        try:
            preds, lat_ms = run_tensorrt_inference(args.model_fp16, img_path, args.imgsz)
            panel = draw_detections(raw_img, preds, f"FP16 TensorRT ({lat_ms:.1f} ms | {len(preds)} dets)", args.conf)
            panel.save(os.path.join(args.output_dir, f"{img_name}_fp16.jpg"))
            rendered_panels.append(panel)
            print(f"✅ FP16 completed: {len(preds)} detections in {lat_ms:.1f} ms")
        except Exception as e:
            print(f"[-] FP16 note: {e}")

    # 3. INT8 (DPU xmodel or TensorRT INT8)
    if os.path.exists(args.model_int8):
        print(f"[*] Evaluating INT8 model: {args.model_int8}...")
        try:
            if args.model_int8.endswith(".xmodel"):
                preds, lat_ms = run_xmodel_inference(args.model_int8, img_path, args.imgsz)
                tag = f"INT8 Kria DPU ({lat_ms:.1f} ms | {len(preds)} dets)"
            else:
                preds, lat_ms = run_tensorrt_inference(args.model_int8, img_path, args.imgsz)
                tag = f"INT8 TensorRT ({lat_ms:.1f} ms | {len(preds)} dets)"
            panel = draw_detections(raw_img, preds, tag, args.conf)
            panel.save(os.path.join(args.output_dir, f"{img_name}_int8.jpg"))
            rendered_panels.append(panel)
            print(f"✅ INT8 completed: {len(preds)} detections in {lat_ms:.1f} ms")
        except Exception as e:
            print(f"[-] INT8 note: {e}")

    # 4. Generate Composite Side-by-Side Comparison
    if len(rendered_panels) >= 2:
        grid = create_comparison_grid(rendered_panels)
        grid_path = os.path.join(args.output_dir, f"{img_name}_comparison_grid.jpg")
        grid.save(grid_path, quality=95)
        print(f"\n🎉 Side-by-side comparison grid saved to: {grid_path}")

    print("\n✅ Visual detection completed.")

if __name__ == "__main__":
    main()
