#include <nnmatch/xfeat.hpp>

#include "xfeat_postprocess.hpp"

#include <cmath>

#include <net.h>
#include <spdlog/spdlog.h>

namespace nnmatch
{

static constexpr float DETECTION_THRESHOLD = 0.05f;
static constexpr int NMS_KERNEL_SIZE = 5;

struct XFeat::Impl
{
    XFeatConfig config;
    ncnn::Net net;

    explicit Impl(const XFeatConfig &cfg) : config(cfg)
    {
#ifdef NM_VULKAN_ENABLED
        if (config.use_vulkan)
        {
            net.opt.use_vulkan_compute = true;
            spdlog::info("XFeat: Vulkan enabled on device {}", config.vulkan_device);
        }
#endif
        int ret = net.load_param(config.param_path.c_str());
        if (ret != 0)
        {
            spdlog::error("XFeat: failed to load param from {}", config.param_path);
            throw std::runtime_error("Failed to load XFeat param file");
        }
        ret = net.load_model(config.bin_path.c_str());
        if (ret != 0)
        {
            spdlog::error("XFeat: failed to load model from {}", config.bin_path);
            throw std::runtime_error("Failed to load XFeat model file");
        }
        spdlog::info("XFeat: loaded model ({} max keypoints)", config.max_keypoints);
    }
};

XFeat::XFeat(const XFeatConfig &config) : impl_(std::make_unique<Impl>(config))
{
}
XFeat::~XFeat() = default;
XFeat::XFeat(XFeat &&) noexcept = default;
XFeat &XFeat::operator=(XFeat &&) noexcept = default;

FeatureResult XFeat::extract(const cv::Mat &image) const
{
    FeatureResult result;

    // Pad to nearest multiple of 32
    const int padded_h = (image.rows / 32) * 32;
    const int padded_w = (image.cols / 32) * 32;
    const float rh = static_cast<float>(image.rows) / padded_h;
    const float rw = static_cast<float>(image.cols) / padded_w;

    // Preprocess: BGR->RGB, resize to padded dimensions, normalize to [0,1]
    ncnn::Mat in =
        ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows, padded_w, padded_h);

    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(nullptr, norm_vals);

    // Forward pass - model converts RGB to grayscale internally
    ncnn::Extractor ex = impl_->net.create_extractor();
    ex.input("in0", in);

    ncnn::Mat desc_map;    // [64, H/8, W/8] raw descriptors
    ncnn::Mat kpt_logits;  // [65, H/8, W/8] keypoint logits
    ncnn::Mat reliability; // [1, H/8, W/8] reliability map
    ex.extract("out0", desc_map);
    ex.extract("out1", kpt_logits);
    ex.extract("out2", reliability);

    // Convert keypoint logits to heatmap:
    // softmax over first 64 channels, then reshape 8x8 blocks to full resolution
    const int grid_h = kpt_logits.h; // H/8
    const int grid_w = kpt_logits.w; // W/8
    const int hm_h = grid_h * 8;
    const int hm_w = grid_w * 8;

    auto heatmap = detail::logits_to_heatmap(kpt_logits, grid_w, grid_h);

    // NMS on heatmap
    auto candidates = detail::nms(heatmap.data(), hm_w, hm_h, NMS_KERNEL_SIZE, DETECTION_THRESHOLD);

    // Compute scores: heatmap * reliability
    const float *rel_data = static_cast<const float *>(reliability.data);
    for (auto &c : candidates)
    {
        // Map heatmap coords to reliability grid coords
        int rx = c.x / 8;
        int ry = c.y / 8;
        rx = std::min(rx, grid_w - 1);
        ry = std::min(ry, grid_h - 1);
        c.score *= rel_data[ry * grid_w + rx];
    }

    // Sort by score descending, take top-k
    std::sort(candidates.begin(), candidates.end(),
              [](const detail::RawKeypoint &a, const detail::RawKeypoint &b) { return a.score > b.score; });

    // Filter negative scores (invalid keypoints)
    while (!candidates.empty() && candidates.back().score <= 0.0f)
    {
        candidates.pop_back();
    }

    const int n = std::min(static_cast<int>(candidates.size()), impl_->config.max_keypoints);
    result.keypoints.resize(n);

    for (int i = 0; i < n; ++i)
    {
        const auto &cand = candidates[i];
        auto &kp = result.keypoints[i];

        // Map heatmap coords back to original image coords
        // hm_w == padded_w and hm_h == padded_h, so just scale by rw/rh
        kp.x = static_cast<float>(cand.x) * rw;
        kp.y = static_cast<float>(cand.y) * rh;
        kp.score = cand.score;

        // Sample descriptor at corresponding location in descriptor map
        float desc_fx = static_cast<float>(cand.x) / 8.0f;
        float desc_fy = static_cast<float>(cand.y) / 8.0f;
        kp.descriptor = detail::sample_descriptor(desc_map, desc_fx, desc_fy);
    }

    spdlog::debug("XFeat: extracted {} keypoints from {}x{} image", n, image.cols, image.rows);
    return result;
}

} // namespace nnmatch
