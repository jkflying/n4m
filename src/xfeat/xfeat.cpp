#include <nnmatch/xfeat.hpp>

#include "xfeat_postprocess.hpp"

#include "../ort_env.hpp"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

namespace nnmatch
{

static constexpr float DETECTION_THRESHOLD = 0.05f;
static constexpr int NMS_KERNEL_SIZE = 5;
static constexpr int STRIDE = 8;
static constexpr int PAD_MULTIPLE = 32;
/// Keep only the best keypoint per grid cell.
static std::vector<detail::RawKeypoint> best_per_cell(const std::vector<detail::RawKeypoint> &candidates, int width,
                                                      int height, int cell_size)
{
    const int cols = std::max(1, (width + cell_size - 1) / cell_size);
    const int rows = std::max(1, (height + cell_size - 1) / cell_size);

    std::vector<detail::RawKeypoint> best(cols * rows, {0, 0, -1.0f});

    for (const auto &c : candidates)
    {
        int cx = std::min(c.x / cell_size, cols - 1);
        int cy = std::min(c.y / cell_size, rows - 1);
        auto &b = best[cy * cols + cx];
        if (c.score > b.score)
            b = c;
    }

    std::vector<detail::RawKeypoint> result;
    result.reserve(cols * rows);
    for (const auto &b : best)
    {
        if (b.score > 0.0f)
            result.push_back(b);
    }
    return result;
}

struct XFeat::Impl
{
    XFeatConfig config;
    Ort::Env env{detail::create_ort_env("xfeat")};
    Ort::Session session{nullptr};

    explicit Impl(const XFeatConfig &cfg) : config(cfg)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = detail::create_ort_session(env, config.model_path.c_str(), opts);
        spdlog::info("XFeat: loaded model from {} ({} max keypoints)", config.model_path, config.max_keypoints);
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

    if (image.rows < PAD_MULTIPLE || image.cols < PAD_MULTIPLE)
    {
        throw std::invalid_argument("Image too small: minimum size is " + std::to_string(PAD_MULTIPLE) + "x" +
                                    std::to_string(PAD_MULTIPLE) + ", got " + std::to_string(image.cols) + "x" +
                                    std::to_string(image.rows));
    }

    // Pad to nearest multiple of PAD_MULTIPLE
    const int padded_h = (image.rows / PAD_MULTIPLE) * PAD_MULTIPLE;
    const int padded_w = (image.cols / PAD_MULTIPLE) * PAD_MULTIPLE;
    const float rh = static_cast<float>(image.rows) / padded_h;
    const float rw = static_cast<float>(image.cols) / padded_w;

    // Preprocess: convert to RGB, resize, normalize to [0,1]
    cv::Mat rgb_input;
    switch (image.channels())
    {
    case 1:
        cv::cvtColor(image, rgb_input, cv::COLOR_GRAY2RGB);
        break;
    case 4:
        cv::cvtColor(image, rgb_input, cv::COLOR_BGRA2RGB);
        break;
    default:
        cv::cvtColor(image, rgb_input, cv::COLOR_BGR2RGB);
        break;
    }

    cv::Mat resized;
    cv::resize(rgb_input, resized, cv::Size(padded_w, padded_h));
    cv::Mat rgb;
    resized.convertTo(rgb, CV_32F, 1.0f / 255.0f);

    // HWC -> NCHW
    const int num_pixels = padded_h * padded_w;
    std::vector<float> input_tensor(3 * num_pixels);
    const float *src = reinterpret_cast<const float *>(rgb.data);
    for (int i = 0; i < num_pixels; ++i)
    {
        input_tensor[0 * num_pixels + i] = src[i * 3 + 0];
        input_tensor[1 * num_pixels + i] = src[i * 3 + 1];
        input_tensor[2 * num_pixels + i] = src[i * 3 + 2];
    }

    // Create ORT input
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape = {1, 3, padded_h, padded_w};
    auto input_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(),
                                                     input_shape.data(), input_shape.size());

    // Run inference
    const char *input_names[] = {"input"};
    const char *output_names[] = {"descriptors", "keypoint_logits", "reliability"};
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &input_ort, 1, output_names, 3);

    // Parse outputs — all are NCHW with batch=1
    auto desc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();   // [1, 64, H/8, W/8]
    auto logits_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape(); // [1, 65, H/8, W/8]

    const int grid_h = static_cast<int>(logits_shape[2]);
    const int grid_w = static_cast<int>(logits_shape[3]);
    const int hm_h = grid_h * STRIDE;
    const int hm_w = grid_w * STRIDE;

    const float *logits_data = outputs[1].GetTensorData<float>();
    const float *desc_data = outputs[0].GetTensorData<float>();
    const float *rel_data = outputs[2].GetTensorData<float>();

    // Postprocessing
    auto heatmap = detail::logits_to_heatmap(logits_data, grid_w, grid_h);
    auto candidates = detail::nms(heatmap.data(), hm_w, hm_h, NMS_KERNEL_SIZE, DETECTION_THRESHOLD);

    // Compute scores: heatmap * reliability
    for (auto &c : candidates)
    {
        int rx = c.x / STRIDE;
        int ry = c.y / STRIDE;
        rx = std::min(rx, grid_w - 1);
        ry = std::min(ry, grid_h - 1);
        c.score *= rel_data[ry * grid_w + rx];
    }

    // Remove non-positive scores
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                    [](const detail::RawKeypoint &c) { return c.score <= 0.0f; }),
                     candidates.end());

    // Select keypoints: best-per-cell or top-k by score
    std::vector<detail::RawKeypoint> selected;
    if (impl_->config.cell_size > 0)
    {
        selected = best_per_cell(candidates, hm_w, hm_h, impl_->config.cell_size);
        if (static_cast<int>(selected.size()) > impl_->config.max_keypoints)
        {
            std::sort(selected.begin(), selected.end(),
                      [](const detail::RawKeypoint &a, const detail::RawKeypoint &b) { return a.score > b.score; });
            selected.resize(impl_->config.max_keypoints);
        }
    }
    else
    {
        std::sort(candidates.begin(), candidates.end(),
                  [](const detail::RawKeypoint &a, const detail::RawKeypoint &b) { return a.score > b.score; });
        int k = std::min(static_cast<int>(candidates.size()), impl_->config.max_keypoints);
        selected.assign(candidates.begin(), candidates.begin() + k);
    }

    const int desc_channels = static_cast<int>(desc_shape[1]);
    const int desc_w = static_cast<int>(desc_shape[3]);
    const int desc_h = static_cast<int>(desc_shape[2]);
    const int n = static_cast<int>(selected.size());
    result.keypoints.resize(n);

    for (int i = 0; i < n; ++i)
    {
        const auto &cand = selected[i];
        auto &kp = result.keypoints[i];

        kp.x = static_cast<float>(cand.x) * rw;
        kp.y = static_cast<float>(cand.y) * rh;
        kp.score = cand.score;

        float desc_fx = static_cast<float>(cand.x) / static_cast<float>(STRIDE);
        float desc_fy = static_cast<float>(cand.y) / static_cast<float>(STRIDE);
        kp.descriptor = detail::sample_descriptor(desc_data, desc_channels, desc_w, desc_h, desc_fx, desc_fy);
    }

    result.image_width = image.cols;
    result.image_height = image.rows;

    spdlog::debug("XFeat: extracted {} keypoints from {}x{} image", n, image.cols, image.rows);
    return result;
}

} // namespace nnmatch
