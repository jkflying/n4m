#include <n4m/xfeat.hpp>

#include "xfeat_postprocess.hpp"

#include "../ort_env.hpp"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include <spdlog/spdlog.h>

namespace n4m
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
        opts.SetIntraOpNumThreads(config.intra_op_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = detail::create_ort_session(env, config.model_path.c_str(), opts);
        spdlog::info("XFeat: loaded model from {} ({} max keypoints, {} threads)", config.model_path,
                     config.max_keypoints, config.intra_op_threads);
    }

    /// Convert image to RGB float32, resize to target dims, and write NCHW into dest buffer.
    /// dest must point to 3 * target_h * target_w floats.
    static void preprocess(const cv::Mat &image, int target_w, int target_h, float *dest)
    {
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
        cv::resize(rgb_input, resized, cv::Size(target_w, target_h));
        cv::Mat rgb;
        resized.convertTo(rgb, CV_32F, 1.0f / 255.0f);

        const int num_pixels = target_h * target_w;
        const float *src = reinterpret_cast<const float *>(rgb.data);
        for (int i = 0; i < num_pixels; ++i)
        {
            dest[0 * num_pixels + i] = src[i * 3 + 0];
            dest[1 * num_pixels + i] = src[i * 3 + 1];
            dest[2 * num_pixels + i] = src[i * 3 + 2];
        }
    }

    /// Postprocess single-image outputs to produce a FeatureResult.
    /// logits_data: [65, grid_h, grid_w], desc_data: [64, grid_h, grid_w], rel_data: [1, grid_h, grid_w]
    FeatureResult postprocess(const float *logits_data, const float *desc_data, const float *rel_data, int grid_w,
                              int grid_h, int desc_channels, float rw, float rh, int orig_w, int orig_h) const
    {
        FeatureResult result;

        const int hm_h = grid_h * STRIDE;
        const int hm_w = grid_w * STRIDE;

        auto heatmap = detail::logits_to_heatmap(logits_data, grid_w, grid_h);
        auto candidates = detail::nms(heatmap.data(), hm_w, hm_h, NMS_KERNEL_SIZE, DETECTION_THRESHOLD);

        for (auto &c : candidates)
        {
            int rx = std::min(c.x / STRIDE, grid_w - 1);
            int ry = std::min(c.y / STRIDE, grid_h - 1);
            c.score *= rel_data[ry * grid_w + rx];
        }

        candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                        [](const detail::RawKeypoint &c) { return c.score <= 0.0f; }),
                         candidates.end());

        std::vector<detail::RawKeypoint> selected;
        if (config.cell_size > 0)
        {
            selected = best_per_cell(candidates, hm_w, hm_h, config.cell_size);
            std::sort(selected.begin(), selected.end(),
                      [](const detail::RawKeypoint &a, const detail::RawKeypoint &b) { return a.score > b.score; });
            if (static_cast<int>(selected.size()) > config.max_keypoints)
            {
                selected.resize(config.max_keypoints);
            }
        }
        else
        {
            std::sort(candidates.begin(), candidates.end(),
                      [](const detail::RawKeypoint &a, const detail::RawKeypoint &b) { return a.score > b.score; });
            int k = std::min(static_cast<int>(candidates.size()), config.max_keypoints);
            selected.assign(candidates.begin(), candidates.begin() + k);
        }

        const int desc_w = grid_w;
        const int desc_h = grid_h;
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

        result.image_width = orig_w;
        result.image_height = orig_h;

        spdlog::debug("XFeat: extracted {} keypoints from {}x{} image", n, orig_w, orig_h);
        return result;
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
    if (image.rows < PAD_MULTIPLE || image.cols < PAD_MULTIPLE)
    {
        throw std::invalid_argument("Image too small: minimum size is " + std::to_string(PAD_MULTIPLE) + "x" +
                                    std::to_string(PAD_MULTIPLE) + ", got " + std::to_string(image.cols) + "x" +
                                    std::to_string(image.rows));
    }

    const int padded_h = (image.rows / PAD_MULTIPLE) * PAD_MULTIPLE;
    const int padded_w = (image.cols / PAD_MULTIPLE) * PAD_MULTIPLE;
    const float rh = static_cast<float>(image.rows) / padded_h;
    const float rw = static_cast<float>(image.cols) / padded_w;

    const int num_pixels = padded_h * padded_w;
    std::vector<float> input_tensor(3 * num_pixels);
    Impl::preprocess(image, padded_w, padded_h, input_tensor.data());

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape = {1, 3, padded_h, padded_w};
    auto input_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(),
                                                     input_shape.data(), input_shape.size());

    const char *input_names[] = {"input"};
    const char *output_names[] = {"descriptors", "keypoint_logits", "reliability"};
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &input_ort, 1, output_names, 3);

    auto desc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    auto logits_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();

    const int grid_h = static_cast<int>(logits_shape[2]);
    const int grid_w = static_cast<int>(logits_shape[3]);
    const int desc_channels = static_cast<int>(desc_shape[1]);

    return impl_->postprocess(outputs[1].GetTensorData<float>(), outputs[0].GetTensorData<float>(),
                              outputs[2].GetTensorData<float>(), grid_w, grid_h, desc_channels, rw, rh, image.cols,
                              image.rows);
}

std::vector<FeatureResult> XFeat::extract_batch(const std::vector<cv::Mat> &images) const
{
    if (images.empty())
    {
        throw std::invalid_argument("extract_batch: images must not be empty");
    }

    if (images.size() == 1)
    {
        return {extract(images[0])};
    }

    const int batch = static_cast<int>(images.size());

    // Validate and compute per-image padded dims, then find max across batch
    struct ImageInfo
    {
        int padded_h, padded_w;
        float rh, rw;
    };
    std::vector<ImageInfo> infos(batch);
    int max_padded_h = 0;
    int max_padded_w = 0;

    for (int b = 0; b < batch; ++b)
    {
        const auto &img = images[b];
        if (img.rows < PAD_MULTIPLE || img.cols < PAD_MULTIPLE)
        {
            throw std::invalid_argument("Image " + std::to_string(b) + " too small: minimum size is " +
                                        std::to_string(PAD_MULTIPLE) + "x" + std::to_string(PAD_MULTIPLE) + ", got " +
                                        std::to_string(img.cols) + "x" + std::to_string(img.rows));
        }
        infos[b].padded_h = (img.rows / PAD_MULTIPLE) * PAD_MULTIPLE;
        infos[b].padded_w = (img.cols / PAD_MULTIPLE) * PAD_MULTIPLE;
        infos[b].rh = static_cast<float>(img.rows) / infos[b].padded_h;
        infos[b].rw = static_cast<float>(img.cols) / infos[b].padded_w;

        max_padded_h = std::max(max_padded_h, infos[b].padded_h);
        max_padded_w = std::max(max_padded_w, infos[b].padded_w);
    }

    // All images get resized to max_padded dims for uniform batch tensor
    const int pixels_per_image = max_padded_h * max_padded_w;
    const int channels_per_image = 3 * pixels_per_image;
    std::vector<float> input_tensor(batch * channels_per_image, 0.0f);

    for (int b = 0; b < batch; ++b)
    {
        Impl::preprocess(images[b], max_padded_w, max_padded_h, input_tensor.data() + b * channels_per_image);
        // Update ratios for the shared max padded dims
        infos[b].rh = static_cast<float>(images[b].rows) / max_padded_h;
        infos[b].rw = static_cast<float>(images[b].cols) / max_padded_w;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> input_shape = {batch, 3, max_padded_h, max_padded_w};
    auto input_ort = Ort::Value::CreateTensor<float>(memory_info, input_tensor.data(), input_tensor.size(),
                                                     input_shape.data(), input_shape.size());

    const char *input_names[] = {"input"};
    const char *output_names[] = {"descriptors", "keypoint_logits", "reliability"};
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &input_ort, 1, output_names, 3);

    auto desc_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();   // [B, 64, H/8, W/8]
    auto logits_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape(); // [B, 65, H/8, W/8]

    const int grid_h = static_cast<int>(logits_shape[2]);
    const int grid_w = static_cast<int>(logits_shape[3]);
    const int desc_channels = static_cast<int>(desc_shape[1]);
    const int logits_stride = 65 * grid_h * grid_w;
    const int desc_stride = desc_channels * grid_h * grid_w;
    const int rel_stride = grid_h * grid_w;

    const float *logits_base = outputs[1].GetTensorData<float>();
    const float *desc_base = outputs[0].GetTensorData<float>();
    const float *rel_base = outputs[2].GetTensorData<float>();

    std::vector<FeatureResult> results(batch);
    for (int b = 0; b < batch; ++b)
    {
        results[b] =
            impl_->postprocess(logits_base + b * logits_stride, desc_base + b * desc_stride, rel_base + b * rel_stride,
                               grid_w, grid_h, desc_channels, infos[b].rw, infos[b].rh, images[b].cols, images[b].rows);
    }

    spdlog::debug("XFeat: batch extracted {} images", batch);
    return results;
}

} // namespace n4m
