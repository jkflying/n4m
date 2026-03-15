#include <n4m/lightglue.hpp>

#include "../ort_env.hpp"

#include <spdlog/spdlog.h>

namespace n4m
{

struct LightGlue::Impl
{
    LightGlueConfig config;
    Ort::Env env{detail::create_ort_env("lightglue")};
    Ort::Session session{nullptr};

    explicit Impl(const LightGlueConfig &cfg) : config(cfg)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(config.intra_op_threads);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = detail::create_ort_session(env, config.model_path.c_str(), opts);
        spdlog::info("LightGlue: loaded model (confidence threshold: {}, {} threads)", config.confidence_threshold,
                     config.intra_op_threads);
    }

    /// Extract matches from raw outputs for a single pair within a batch.
    std::vector<Match> extract_matches(const int64_t *matches0_data, const float *scores0_data, int original_n0) const
    {
        std::vector<Match> matches;
        matches.reserve(original_n0);
        for (int i = 0; i < original_n0; ++i)
        {
            int64_t match_idx = matches0_data[i];
            float score = scores0_data[i];
            if (match_idx >= 0 && score >= config.confidence_threshold)
            {
                matches.push_back({i, static_cast<int>(match_idx), score});
            }
        }
        return matches;
    }
};

LightGlue::LightGlue(const LightGlueConfig &config) : impl_(std::make_unique<Impl>(config))
{
}
LightGlue::~LightGlue() = default;
LightGlue::LightGlue(LightGlue &&) noexcept = default;
LightGlue &LightGlue::operator=(LightGlue &&) noexcept = default;

std::vector<Match> LightGlue::match(const FeatureResult &feats0, const FeatureResult &feats1) const
{
    const int n0 = static_cast<int>(feats0.keypoints.size());
    const int n1 = static_cast<int>(feats1.keypoints.size());

    if (n0 == 0 || n1 == 0)
    {
        return {};
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Pack keypoints as 1xNx2
    std::vector<float> kpts0_data(n0 * 2), kpts1_data(n1 * 2);
    for (int i = 0; i < n0; ++i)
    {
        kpts0_data[i * 2 + 0] = feats0.keypoints[i].x;
        kpts0_data[i * 2 + 1] = feats0.keypoints[i].y;
    }
    for (int i = 0; i < n1; ++i)
    {
        kpts1_data[i * 2 + 0] = feats1.keypoints[i].x;
        kpts1_data[i * 2 + 1] = feats1.keypoints[i].y;
    }

    // Pack descriptors as 1xNxD
    std::vector<float> desc0_data(n0 * XFEAT_DESCRIPTOR_DIM), desc1_data(n1 * XFEAT_DESCRIPTOR_DIM);
    for (int i = 0; i < n0; ++i)
    {
        std::copy(feats0.keypoints[i].descriptor.begin(), feats0.keypoints[i].descriptor.end(),
                  desc0_data.begin() + i * XFEAT_DESCRIPTOR_DIM);
    }
    for (int i = 0; i < n1; ++i)
    {
        std::copy(feats1.keypoints[i].descriptor.begin(), feats1.keypoints[i].descriptor.end(),
                  desc1_data.begin() + i * XFEAT_DESCRIPTOR_DIM);
    }

    // Image sizes as 1x2 (width, height) — matches kornia's convention
    std::vector<int64_t> size0_data = {feats0.image_width, feats0.image_height};
    std::vector<int64_t> size1_data = {feats1.image_width, feats1.image_height};

    std::array<int64_t, 3> kpts0_shape = {1, n0, 2};
    std::array<int64_t, 3> kpts1_shape = {1, n1, 2};
    std::array<int64_t, 3> desc0_shape = {1, n0, XFEAT_DESCRIPTOR_DIM};
    std::array<int64_t, 3> desc1_shape = {1, n1, XFEAT_DESCRIPTOR_DIM};
    std::array<int64_t, 2> size_shape = {1, 2};

    std::array<Ort::Value, 6> inputs = {
        Ort::Value::CreateTensor<float>(memory_info, kpts0_data.data(), kpts0_data.size(), kpts0_shape.data(), 3),
        Ort::Value::CreateTensor<float>(memory_info, kpts1_data.data(), kpts1_data.size(), kpts1_shape.data(), 3),
        Ort::Value::CreateTensor<float>(memory_info, desc0_data.data(), desc0_data.size(), desc0_shape.data(), 3),
        Ort::Value::CreateTensor<float>(memory_info, desc1_data.data(), desc1_data.size(), desc1_shape.data(), 3),
        Ort::Value::CreateTensor<int64_t>(memory_info, size0_data.data(), size0_data.size(), size_shape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(memory_info, size1_data.data(), size1_data.size(), size_shape.data(), 2),
    };

    const char *input_names[] = {"kpts0", "kpts1", "desc0", "desc1", "image_size0", "image_size1"};
    const char *output_names[] = {"matches0", "scores0"};
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 6, output_names, 2);

    auto matches = impl_->extract_matches(outputs[0].GetTensorData<int64_t>(), outputs[1].GetTensorData<float>(), n0);

    spdlog::debug("LightGlue: {} matches from {} + {} keypoints ({} after threshold)", n0, n0, n1, matches.size());
    return matches;
}

std::vector<std::vector<Match>> LightGlue::match_batch(
    const std::vector<std::pair<FeatureResult, FeatureResult>> &pairs) const
{
    if (pairs.empty())
    {
        return {};
    }

    if (pairs.size() == 1)
    {
        return {match(pairs[0].first, pairs[0].second)};
    }

    const int batch = static_cast<int>(pairs.size());

    // Find max keypoint counts and track original counts
    int max_n0 = 0;
    int max_n1 = 0;
    std::vector<int> orig_n0(batch), orig_n1(batch);

    for (int b = 0; b < batch; ++b)
    {
        orig_n0[b] = static_cast<int>(pairs[b].first.keypoints.size());
        orig_n1[b] = static_cast<int>(pairs[b].second.keypoints.size());
        max_n0 = std::max(max_n0, orig_n0[b]);
        max_n1 = std::max(max_n1, orig_n1[b]);
    }

    if (max_n0 == 0 || max_n1 == 0)
    {
        return std::vector<std::vector<Match>>(batch);
    }

    // Pack into batched tensors with zero-padding
    std::vector<float> kpts0_data(batch * max_n0 * 2, 0.0f);
    std::vector<float> kpts1_data(batch * max_n1 * 2, 0.0f);
    std::vector<float> desc0_data(batch * max_n0 * XFEAT_DESCRIPTOR_DIM, 0.0f);
    std::vector<float> desc1_data(batch * max_n1 * XFEAT_DESCRIPTOR_DIM, 0.0f);
    std::vector<int64_t> size0_data(batch * 2);
    std::vector<int64_t> size1_data(batch * 2);

    for (int b = 0; b < batch; ++b)
    {
        const auto &f0 = pairs[b].first;
        const auto &f1 = pairs[b].second;

        float *kp0 = kpts0_data.data() + b * max_n0 * 2;
        float *kp1 = kpts1_data.data() + b * max_n1 * 2;
        float *d0 = desc0_data.data() + b * max_n0 * XFEAT_DESCRIPTOR_DIM;
        float *d1 = desc1_data.data() + b * max_n1 * XFEAT_DESCRIPTOR_DIM;

        for (int i = 0; i < orig_n0[b]; ++i)
        {
            kp0[i * 2 + 0] = f0.keypoints[i].x;
            kp0[i * 2 + 1] = f0.keypoints[i].y;
            std::copy(f0.keypoints[i].descriptor.begin(), f0.keypoints[i].descriptor.end(),
                      d0 + i * XFEAT_DESCRIPTOR_DIM);
        }
        for (int i = 0; i < orig_n1[b]; ++i)
        {
            kp1[i * 2 + 0] = f1.keypoints[i].x;
            kp1[i * 2 + 1] = f1.keypoints[i].y;
            std::copy(f1.keypoints[i].descriptor.begin(), f1.keypoints[i].descriptor.end(),
                      d1 + i * XFEAT_DESCRIPTOR_DIM);
        }

        size0_data[b * 2 + 0] = f0.image_width;
        size0_data[b * 2 + 1] = f0.image_height;
        size1_data[b * 2 + 0] = f1.image_width;
        size1_data[b * 2 + 1] = f1.image_height;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> kpts0_shape = {batch, max_n0, 2};
    std::array<int64_t, 3> kpts1_shape = {batch, max_n1, 2};
    std::array<int64_t, 3> desc0_shape = {batch, max_n0, XFEAT_DESCRIPTOR_DIM};
    std::array<int64_t, 3> desc1_shape = {batch, max_n1, XFEAT_DESCRIPTOR_DIM};
    std::array<int64_t, 2> size_shape = {batch, 2};

    std::array<Ort::Value, 6> inputs = {
        Ort::Value::CreateTensor<float>(memory_info, kpts0_data.data(), kpts0_data.size(), kpts0_shape.data(), 3),
        Ort::Value::CreateTensor<float>(memory_info, kpts1_data.data(), kpts1_data.size(), kpts1_shape.data(), 3),
        Ort::Value::CreateTensor<float>(memory_info, desc0_data.data(), desc0_data.size(), desc0_shape.data(), 3),
        Ort::Value::CreateTensor<float>(memory_info, desc1_data.data(), desc1_data.size(), desc1_shape.data(), 3),
        Ort::Value::CreateTensor<int64_t>(memory_info, size0_data.data(), size0_data.size(), size_shape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(memory_info, size1_data.data(), size1_data.size(), size_shape.data(), 2),
    };

    const char *input_names[] = {"kpts0", "kpts1", "desc0", "desc1", "image_size0", "image_size1"};
    const char *output_names[] = {"matches0", "scores0"};
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 6, output_names, 2);

    // matches0: [B, max_n0], scores0: [B, max_n0]
    const int64_t *matches0_base = outputs[0].GetTensorData<int64_t>();
    const float *scores0_base = outputs[1].GetTensorData<float>();

    std::vector<std::vector<Match>> results(batch);
    for (int b = 0; b < batch; ++b)
    {
        const int64_t *m_data = matches0_base + b * max_n0;
        const float *s_data = scores0_base + b * max_n0;

        results[b].reserve(orig_n0[b]);
        for (int i = 0; i < orig_n0[b]; ++i)
        {
            int64_t match_idx = m_data[i];
            float score = s_data[i];
            // Filter: valid match, within original keypoint range, and above threshold
            if (match_idx >= 0 && match_idx < orig_n1[b] && score >= impl_->config.confidence_threshold)
            {
                results[b].push_back({i, static_cast<int>(match_idx), score});
            }
        }
    }

    spdlog::debug("LightGlue: batch matched {} pairs", batch);
    return results;
}

} // namespace n4m
