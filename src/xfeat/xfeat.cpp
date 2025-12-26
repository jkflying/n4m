#include <nnmatch/xfeat.hpp>

#include <algorithm>
#include <numeric>

#include <net.h>
#include <spdlog/spdlog.h>

namespace nnmatch
{

static constexpr int XFEAT_INPUT_W = 640;
static constexpr int XFEAT_INPUT_H = 480;

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

    // Preprocess: BGR->RGB, resize, normalize to [0,1]
    // ncnn::Mat::from_pixels_resize handles BGR->RGB + resize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows,
                                                 XFEAT_INPUT_W, XFEAT_INPUT_H);

    // Normalize to [0, 1]
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(nullptr, norm_vals);

    // Forward pass
    ncnn::Extractor ex = impl_->net.create_extractor();
    ex.input("input", in);

    ncnn::Mat heatmap, descriptors_map;
    ex.extract("heatmap", heatmap);
    ex.extract("descriptors", descriptors_map);

    // Decode heatmap: simple NMS + top-k
    const int hm_h = heatmap.h;
    const int hm_w = heatmap.w;

    struct RawKeypoint
    {
        int x, y;
        float score;
    };
    std::vector<RawKeypoint> candidates;
    candidates.reserve(hm_h * hm_w / 4);

    const float *hm_data = static_cast<const float *>(heatmap.data);

    // 3x3 NMS on heatmap
    for (int y = 1; y < hm_h - 1; ++y)
    {
        for (int x = 1; x < hm_w - 1; ++x)
        {
            float val = hm_data[y * hm_w + x];
            if (val <= 0.0f)
                continue;

            bool is_max = true;
            for (int dy = -1; dy <= 1 && is_max; ++dy)
            {
                for (int dx = -1; dx <= 1 && is_max; ++dx)
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

    // Sort by score descending, take top-k
    std::sort(candidates.begin(), candidates.end(),
              [](const RawKeypoint &a, const RawKeypoint &b) { return a.score > b.score; });

    const int n = std::min(static_cast<int>(candidates.size()), impl_->config.max_keypoints);
    result.keypoints.resize(n);

    // Sample descriptors at keypoint locations
    const int desc_h = descriptors_map.h;
    const int desc_w = descriptors_map.w;

    for (int i = 0; i < n; ++i)
    {
        const auto &cand = candidates[i];
        auto &kp = result.keypoints[i];

        // Map heatmap coords back to original image coords
        kp.x = (static_cast<float>(cand.x) / hm_w) * image.cols;
        kp.y = (static_cast<float>(cand.y) / hm_h) * image.rows;
        kp.score = cand.score;

        // Bilinear sample descriptor at corresponding location in descriptor map
        float desc_fx = (static_cast<float>(cand.x) / hm_w) * desc_w;
        float desc_fy = (static_cast<float>(cand.y) / hm_h) * desc_h;

        int dx0 = std::max(0, std::min(static_cast<int>(desc_fx), desc_w - 1));
        int dy0 = std::max(0, std::min(static_cast<int>(desc_fy), desc_h - 1));
        int dx1 = std::min(dx0 + 1, desc_w - 1);
        int dy1 = std::min(dy0 + 1, desc_h - 1);

        float wx = desc_fx - dx0;
        float wy = desc_fy - dy0;

        // L2-normalize descriptor
        float norm_sq = 0.0f;
        for (int d = 0; d < XFEAT_DESCRIPTOR_DIM; ++d)
        {
            const float *ch = descriptors_map.channel(d);
            float val = ch[dy0 * desc_w + dx0] * (1 - wx) * (1 - wy) + ch[dy0 * desc_w + dx1] * wx * (1 - wy) +
                        ch[dy1 * desc_w + dx0] * (1 - wx) * wy + ch[dy1 * desc_w + dx1] * wx * wy;
            kp.descriptor[d] = val;
            norm_sq += val * val;
        }
        float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-8f);
        for (int d = 0; d < XFEAT_DESCRIPTOR_DIM; ++d)
        {
            kp.descriptor[d] *= inv_norm;
        }
    }

    spdlog::debug("XFeat: extracted {} keypoints from {}x{} image", n, image.cols, image.rows);
    return result;
}

} // namespace nnmatch
