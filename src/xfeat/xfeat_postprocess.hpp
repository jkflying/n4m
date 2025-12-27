#pragma once

#include <nnmatch/types.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <net.h>

namespace nnmatch
{
namespace detail
{

struct RawKeypoint
{
    int x, y;
    float score;
};

// Convert keypoint logits [65, grid_h, grid_w] to heatmap [grid_h*8, grid_w*8].
// Applies softmax over first 64 channels (spatial bins), discards dustbin (channel 64),
// then reshapes 8x8 blocks to full resolution.
inline std::vector<float> logits_to_heatmap(const ncnn::Mat &kpt_logits, int grid_w, int grid_h)
{
    const int hm_h = grid_h * 8;
    const int hm_w = grid_w * 8;
    std::vector<float> heatmap(hm_h * hm_w, 0.0f);

    for (int gy = 0; gy < grid_h; ++gy)
    {
        for (int gx = 0; gx < grid_w; ++gx)
        {
            // Softmax over 65 channels at this spatial location
            float max_val = -1e30f;
            for (int c = 0; c < 65; ++c)
            {
                float v = kpt_logits.channel(c)[gy * grid_w + gx];
                if (v > max_val)
                    max_val = v;
            }

            float sum_exp = 0.0f;
            float exps[65];
            for (int c = 0; c < 65; ++c)
            {
                float v = kpt_logits.channel(c)[gy * grid_w + gx];
                exps[c] = std::exp(v - max_val);
                sum_exp += exps[c];
            }

            // Map first 64 softmax values to 8x8 block in heatmap
            for (int c = 0; c < 64; ++c)
            {
                float prob = exps[c] / sum_exp;
                int by = c / 8; // row within 8x8 block
                int bx = c % 8; // col within 8x8 block
                int hy = gy * 8 + by;
                int hx = gx * 8 + bx;
                heatmap[hy * hm_w + hx] = prob;
            }
        }
    }

    return heatmap;
}

// NMS with configurable kernel size and threshold.
// Returns keypoints sorted by score descending.
inline std::vector<RawKeypoint> nms(const float *hm_data, int hm_w, int hm_h, int kernel_size, float threshold)
{
    std::vector<RawKeypoint> candidates;
    candidates.reserve(4096);

    const int half = kernel_size / 2;

    for (int y = half; y < hm_h - half; ++y)
    {
        for (int x = half; x < hm_w - half; ++x)
        {
            float val = hm_data[y * hm_w + x];
            if (val <= threshold)
                continue;

            bool is_max = true;
            for (int dy = -half; dy <= half && is_max; ++dy)
            {
                for (int dx = -half; dx <= half && is_max; ++dx)
                {
                    if (dy == 0 && dx == 0)
                        continue;
                    if (hm_data[(y + dy) * hm_w + (x + dx)] >= val)
                    {
                        is_max = false;
                    }
                }
            }
            if (is_max)
            {
                candidates.push_back({x, y, val});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const RawKeypoint &a, const RawKeypoint &b) { return a.score > b.score; });

    return candidates;
}

// Legacy 3x3 NMS with threshold=0 (for tests)
inline std::vector<RawKeypoint> nms_3x3(const float *hm_data, int hm_w, int hm_h)
{
    return nms(hm_data, hm_w, hm_h, 3, 0.0f);
}

// Bilinear sample + L2-normalize a descriptor from a CHW descriptor map at (fx, fy) in map coords
inline Descriptor sample_descriptor(const ncnn::Mat &desc_map, float fx, float fy)
{
    const int w = desc_map.w;
    const int h = desc_map.h;

    int x0 = std::max(0, std::min(static_cast<int>(fx), w - 1));
    int y0 = std::max(0, std::min(static_cast<int>(fy), h - 1));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);

    float wx = fx - x0;
    float wy = fy - y0;

    Descriptor desc;
    float norm_sq = 0.0f;
    for (int d = 0; d < XFEAT_DESCRIPTOR_DIM; ++d)
    {
        const float *ch = desc_map.channel(d);
        float val = ch[y0 * w + x0] * (1 - wx) * (1 - wy) + ch[y0 * w + x1] * wx * (1 - wy) +
                    ch[y1 * w + x0] * (1 - wx) * wy + ch[y1 * w + x1] * wx * wy;
        desc[d] = val;
        norm_sq += val * val;
    }
    float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-8f);
    for (int d = 0; d < XFEAT_DESCRIPTOR_DIM; ++d)
    {
        desc[d] *= inv_norm;
    }
    return desc;
}

} // namespace detail
} // namespace nnmatch
