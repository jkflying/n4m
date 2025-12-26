#pragma once

#include <memory>
#include <string>
#include <vector>

#include <nnmatch/types.hpp>

namespace nnmatch
{

struct LightGlueConfig
{
    std::string param_path;
    std::string bin_path;
    bool use_vulkan = false;
    int vulkan_device = 0;
    float confidence_threshold = 0.0f;
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

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nnmatch
