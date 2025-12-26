#include <nnmatch/lightglue.hpp>

#include <net.h>
#include <spdlog/spdlog.h>

namespace nnmatch
{

struct LightGlue::Impl
{
    LightGlueConfig config;
    ncnn::Net net;

    explicit Impl(const LightGlueConfig &cfg) : config(cfg)
    {
#ifdef NM_VULKAN_ENABLED
        if (config.use_vulkan)
        {
            net.opt.use_vulkan_compute = true;
            spdlog::info("LightGlue: Vulkan enabled on device {}", config.vulkan_device);
        }
#endif
        int ret = net.load_param(config.param_path.c_str());
        if (ret != 0)
        {
            spdlog::error("LightGlue: failed to load param from {}", config.param_path);
            throw std::runtime_error("Failed to load LightGlue param file");
        }
        ret = net.load_model(config.bin_path.c_str());
        if (ret != 0)
        {
            spdlog::error("LightGlue: failed to load model from {}", config.bin_path);
            throw std::runtime_error("Failed to load LightGlue model file");
        }
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

    // Pack keypoints: Nx2 matrix (x, y)
    ncnn::Mat kpts0(2, n0);
    ncnn::Mat kpts1(2, n1);
    for (int i = 0; i < n0; ++i)
    {
        float *row = kpts0.row(i);
        row[0] = feats0.keypoints[i].x;
        row[1] = feats0.keypoints[i].y;
    }
    for (int i = 0; i < n1; ++i)
    {
        float *row = kpts1.row(i);
        row[0] = feats1.keypoints[i].x;
        row[1] = feats1.keypoints[i].y;
    }

    // Pack descriptors: NxD matrix
    ncnn::Mat desc0(XFEAT_DESCRIPTOR_DIM, n0);
    ncnn::Mat desc1(XFEAT_DESCRIPTOR_DIM, n1);
    for (int i = 0; i < n0; ++i)
    {
        float *row = desc0.row(i);
        for (int d = 0; d < XFEAT_DESCRIPTOR_DIM; ++d)
        {
            row[d] = feats0.keypoints[i].descriptor[d];
        }
    }
    for (int i = 0; i < n1; ++i)
    {
        float *row = desc1.row(i);
        for (int d = 0; d < XFEAT_DESCRIPTOR_DIM; ++d)
        {
            row[d] = feats1.keypoints[i].descriptor[d];
        }
    }

    ncnn::Extractor ex = impl_->net.create_extractor();
    ex.input("kpts0", kpts0);
    ex.input("kpts1", kpts1);
    ex.input("desc0", desc0);
    ex.input("desc1", desc1);

    ncnn::Mat match_indices, match_confidence;
    ex.extract("matches", match_indices);
    ex.extract("confidence", match_confidence);

    const int num_matches = match_indices.h;
    matches.reserve(num_matches);

    for (int i = 0; i < num_matches; ++i)
    {
        const float *idx_row = match_indices.row(i);
        int i0 = static_cast<int>(idx_row[0]);
        int i1 = static_cast<int>(idx_row[1]);
        float conf = (match_confidence.data != nullptr) ? static_cast<const float *>(match_confidence.data)[i] : 1.0f;

        if (conf >= impl_->config.confidence_threshold)
        {
            matches.push_back({i0, i1, conf});
        }
    }

    spdlog::debug("LightGlue: {} matches from {} + {} keypoints ({} after threshold)", num_matches, n0, n1,
                  matches.size());
    return matches;
}

} // namespace nnmatch
