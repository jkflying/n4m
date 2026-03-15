#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <n4m/types.hpp>

namespace n4m
{

struct LightGlueConfig
{
    std::string model_path;
    float confidence_threshold = 0.0f;
    /// ONNX Runtime intra-op thread count (0 = ORT default, 1 = single-threaded).
    int intra_op_threads = 1;
};

class LightGlue
{
  public:
    explicit LightGlue(const LightGlueConfig &config);
    ~LightGlue();

    LightGlue(const LightGlue &) = delete;
    LightGlue &operator=(const LightGlue &) = delete;
    LightGlue(LightGlue &&) noexcept;
    LightGlue &operator=(LightGlue &&) noexcept;

    std::vector<Match> match(const FeatureResult &feats0, const FeatureResult &feats1) const;
    std::vector<std::vector<Match>> match_batch(
        const std::vector<std::pair<FeatureResult, FeatureResult>> &pairs) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace n4m
