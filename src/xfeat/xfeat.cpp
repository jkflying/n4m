#include <nnmatch/xfeat.hpp>

#include "xfeat_postprocess.hpp"

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

    // Decode heatmap: NMS + top-k
    const int hm_h = heatmap.h;
    const int hm_w = heatmap.w;
    const float *hm_data = static_cast<const float *>(heatmap.data);

    auto candidates = detail::nms_3x3(hm_data, hm_w, hm_h);

    const int n = std::min(static_cast<int>(candidates.size()), impl_->config.max_keypoints);
    result.keypoints.resize(n);

    const int desc_w = descriptors_map.w;
    const int desc_h = descriptors_map.h;

    for (int i = 0; i < n; ++i)
    {
        const auto &cand = candidates[i];
        auto &kp = result.keypoints[i];

        kp.x = (static_cast<float>(cand.x) / hm_w) * image.cols;
        kp.y = (static_cast<float>(cand.y) / hm_h) * image.rows;
        kp.score = cand.score;

        float desc_fx = (static_cast<float>(cand.x) / hm_w) * desc_w;
        float desc_fy = (static_cast<float>(cand.y) / hm_h) * desc_h;
        kp.descriptor = detail::sample_descriptor(descriptors_map, desc_fx, desc_fy);
    }

    spdlog::debug("XFeat: extracted {} keypoints from {}x{} image", n, image.cols, image.rows);
    return result;
}

} // namespace nnmatch
