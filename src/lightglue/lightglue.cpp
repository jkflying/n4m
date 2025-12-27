#include <nnmatch/lightglue.hpp>

#include "../ort_env.hpp"

#include <spdlog/spdlog.h>

namespace nnmatch
{

struct LightGlue::Impl
{
    LightGlueConfig config;
    Ort::Env env{detail::create_ort_env("lightglue")};
    Ort::Session session{nullptr};

    explicit Impl(const LightGlueConfig &cfg) : config(cfg)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session = detail::create_ort_session(env, config.model_path.c_str(), opts);
        spdlog::info("LightGlue: loaded model (confidence threshold: {})", config.confidence_threshold);
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
    std::vector<Match> matches;

    const int n0 = static_cast<int>(feats0.keypoints.size());
    const int n1 = static_cast<int>(feats1.keypoints.size());

    if (n0 == 0 || n1 == 0)
    {
        return matches;
    }

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Pack keypoints as Nx2
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

    // Pack descriptors as NxD
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

    std::array<int64_t, 2> kpts0_shape = {n0, 2};
    std::array<int64_t, 2> kpts1_shape = {n1, 2};
    std::array<int64_t, 2> desc0_shape = {n0, XFEAT_DESCRIPTOR_DIM};
    std::array<int64_t, 2> desc1_shape = {n1, XFEAT_DESCRIPTOR_DIM};

    std::array<Ort::Value, 4> inputs = {
        Ort::Value::CreateTensor<float>(memory_info, kpts0_data.data(), kpts0_data.size(), kpts0_shape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_info, kpts1_data.data(), kpts1_data.size(), kpts1_shape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_info, desc0_data.data(), desc0_data.size(), desc0_shape.data(), 2),
        Ort::Value::CreateTensor<float>(memory_info, desc1_data.data(), desc1_data.size(), desc1_shape.data(), 2),
    };

    const char *input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
    const char *output_names[] = {"matches", "confidence"};
    auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, inputs.data(), 4, output_names, 2);

    auto match_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int num_matches = static_cast<int>(match_shape[0]);
    const int64_t *match_data = outputs[0].GetTensorData<int64_t>();
    const float *conf_data = outputs[1].GetTensorData<float>();

    matches.reserve(num_matches);
    for (int i = 0; i < num_matches; ++i)
    {
        float conf = conf_data[i];
        if (conf >= impl_->config.confidence_threshold)
        {
            matches.push_back({static_cast<int>(match_data[i * 2]), static_cast<int>(match_data[i * 2 + 1]), conf});
        }
    }

    spdlog::debug("LightGlue: {} matches from {} + {} keypoints ({} after threshold)", num_matches, n0, n1,
                  matches.size());
    return matches;
}

} // namespace nnmatch
