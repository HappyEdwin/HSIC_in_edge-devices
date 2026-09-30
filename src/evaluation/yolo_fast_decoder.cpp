#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

struct Detection {
    float x1, y1, x2, y2;
    float score;
    float cls_id;
};

// Anchors for YOLOv4-tiny
static const float ANCHORS_P4[3][2] = {{10.0f, 14.0f}, {23.0f, 27.0f}, {37.0f, 58.0f}};    // stride 16 (26x26)
static const float ANCHORS_P5[3][2] = {{81.0f, 82.0f}, {135.0f, 169.0f}, {344.0f, 319.0f}}; // stride 32 (13x13)

static inline float fast_sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float compute_iou(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter = w * h;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float un = area_a + area_b - inter;
    return (un > 0.0f) ? (inter / un) : 0.0f;
}

extern "C" {

/**
 * High-performance C++ Anchor Decoder and NMS for YOLOv4-tiny INT8 DPU outputs.
 * buf_p4: raw int8 pointer from DPU for 26x26x255
 * buf_p5: raw int8 pointer from DPU for 13x13x255
 * out_dets: preallocated float array of size (max_detections * 6)
 * Returns number of final detections.
 */
int decode_yolov4_tiny_int8(
    const int8_t* buf_p4, float scale_p4,
    const int8_t* buf_p5, float scale_p5,
    float conf_threshold, float iou_threshold,
    int img_size,
    float* out_dets, int max_detections
) {
    std::vector<Detection> candidates;
    candidates.reserve(128);

    struct LayerInfo {
        const int8_t* buf;
        float scale;
        int gh;
        int gw;
        int stride;
        const float (*anchors)[2];
    };

    LayerInfo layers[2] = {
        {buf_p4, scale_p4, 26, 26, 16, ANCHORS_P4},
        {buf_p5, scale_p5, 13, 13, 32, ANCHORS_P5}
    };

    for (int l = 0; l < 2; ++l) {
        if (!layers[l].buf) continue;
        const int8_t* buf = layers[l].buf;
        float scale = layers[l].scale;
        int gh = layers[l].gh;
        int gw = layers[l].gw;
        int stride = layers[l].stride;
        const auto& anchors = layers[l].anchors;

        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                int cell_base = (y * gw + x) * 255;
                for (int a = 0; a < 3; ++a) {
                    int offset = cell_base + a * 85;

                    // 1. Check Objectness first (eliminates 99% of anchors)
                    float raw_obj = (float)buf[offset + 4] * scale;
                    float obj_conf = fast_sigmoid(raw_obj);
                    if (obj_conf < conf_threshold) continue;

                    // 2. Find Best Class
                    int best_cls = 0;
                    float max_cls_raw = (float)buf[offset + 5] * scale;
                    for (int c = 1; c < 80; ++c) {
                        float v = (float)buf[offset + 5 + c] * scale;
                        if (v > max_cls_raw) {
                            max_cls_raw = v;
                            best_cls = c;
                        }
                    }

                    float max_cls_score = fast_sigmoid(max_cls_raw);
                    float total_score = obj_conf * max_cls_score;
                    if (total_score < conf_threshold) continue;

                    // 3. Decode Coordinates only for surviving candidates
                    float raw_bx = (float)buf[offset + 0] * scale;
                    float raw_by = (float)buf[offset + 1] * scale;
                    float raw_bw = (float)buf[offset + 2] * scale;
                    float raw_bh = (float)buf[offset + 3] * scale;

                    float bx = (fast_sigmoid(raw_bx) + (float)x) * (float)stride;
                    float by = (fast_sigmoid(raw_by) + (float)y) * (float)stride;
                    float bw = std::exp(std::min(10.0f, std::max(-10.0f, raw_bw))) * anchors[a][0];
                    float bh = std::exp(std::min(10.0f, std::max(-10.0f, raw_bh))) * anchors[a][1];

                    Detection det;
                    det.x1 = std::max(0.0f, bx - bw * 0.5f);
                    det.y1 = std::max(0.0f, by - bh * 0.5f);
                    det.x2 = std::min((float)img_size, bx + bw * 0.5f);
                    det.y2 = std::min((float)img_size, by + bh * 0.5f);
                    det.score = total_score;
                    det.cls_id = (float)best_cls;

                    candidates.push_back(det);
                }
            }
        }
    }

    if (candidates.empty()) return 0;

    // 4. Sort descending by confidence score
    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    // 5. Fast Greedy NMS
    std::vector<bool> suppressed(candidates.size(), false);
    int count = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        if (count >= max_detections) break;

        const auto& best = candidates[i];
        out_dets[count * 6 + 0] = best.x1;
        out_dets[count * 6 + 1] = best.y1;
        out_dets[count * 6 + 2] = best.x2;
        out_dets[count * 6 + 3] = best.y2;
        out_dets[count * 6 + 4] = best.score;
        out_dets[count * 6 + 5] = best.cls_id;
        count++;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && (int)candidates[j].cls_id == (int)best.cls_id) {
                if (compute_iou(best, candidates[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return count;
}

/**
 * Fallback variant taking float pointers directly.
 */
int decode_yolov4_tiny_float(
    const float* buf_p4,
    const float* buf_p5,
    float conf_threshold, float iou_threshold,
    int img_size,
    float* out_dets, int max_detections
) {
    std::vector<Detection> candidates;
    candidates.reserve(128);

    struct LayerInfo {
        const float* buf;
        int gh;
        int gw;
        int stride;
        const float (*anchors)[2];
    };

    LayerInfo layers[2] = {
        {buf_p4, 26, 26, 16, ANCHORS_P4},
        {buf_p5, 13, 13, 32, ANCHORS_P5}
    };

    for (int l = 0; l < 2; ++l) {
        if (!layers[l].buf) continue;
        const float* buf = layers[l].buf;
        int gh = layers[l].gh;
        int gw = layers[l].gw;
        int stride = layers[l].stride;
        const auto& anchors = layers[l].anchors;

        for (int y = 0; y < gh; ++y) {
            for (int x = 0; x < gw; ++x) {
                int cell_base = (y * gw + x) * 255;
                for (int a = 0; a < 3; ++a) {
                    int offset = cell_base + a * 85;

                    float raw_obj = buf[offset + 4];
                    float obj_conf = fast_sigmoid(raw_obj);
                    if (obj_conf < conf_threshold) continue;

                    int best_cls = 0;
                    float max_cls_raw = buf[offset + 5];
                    for (int c = 1; c < 80; ++c) {
                        float v = buf[offset + 5 + c];
                        if (v > max_cls_raw) {
                            max_cls_raw = v;
                            best_cls = c;
                        }
                    }

                    float max_cls_score = fast_sigmoid(max_cls_raw);
                    float total_score = obj_conf * max_cls_score;
                    if (total_score < conf_threshold) continue;

                    float raw_bx = buf[offset + 0];
                    float raw_by = buf[offset + 1];
                    float raw_bw = buf[offset + 2];
                    float raw_bh = buf[offset + 3];

                    float bx = (fast_sigmoid(raw_bx) + (float)x) * (float)stride;
                    float by = (fast_sigmoid(raw_by) + (float)y) * (float)stride;
                    float bw = std::exp(std::min(10.0f, std::max(-10.0f, raw_bw))) * anchors[a][0];
                    float bh = std::exp(std::min(10.0f, std::max(-10.0f, raw_bh))) * anchors[a][1];

                    Detection det;
                    det.x1 = std::max(0.0f, bx - bw * 0.5f);
                    det.y1 = std::max(0.0f, by - bh * 0.5f);
                    det.x2 = std::min((float)img_size, bx + bw * 0.5f);
                    det.y2 = std::min((float)img_size, by + bh * 0.5f);
                    det.score = total_score;
                    det.cls_id = (float)best_cls;

                    candidates.push_back(det);
                }
            }
        }
    }

    if (candidates.empty()) return 0;

    std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(candidates.size(), false);
    int count = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;
        if (count >= max_detections) break;

        const auto& best = candidates[i];
        out_dets[count * 6 + 0] = best.x1;
        out_dets[count * 6 + 1] = best.y1;
        out_dets[count * 6 + 2] = best.x2;
        out_dets[count * 6 + 3] = best.y2;
        out_dets[count * 6 + 4] = best.score;
        out_dets[count * 6 + 5] = best.cls_id;
        count++;

        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && (int)candidates[j].cls_id == (int)best.cls_id) {
                if (compute_iou(best, candidates[j]) > iou_threshold) {
                    suppressed[j] = true;
                }
            }
        }
    }

    return count;
}

} // extern "C"
