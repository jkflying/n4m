#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include <nnmatch/types.hpp>

namespace nnmatch
{

struct XFeatConfig
{
    std::string param_path;
    std::string bin_path;
    int max_keypoints = 4096;
    bool use_vulkan = false;
    int vulkan_device = 0;
};

class XFeat
{
  public:
    explicit XFeat(const XFeatConfig &config);
    ~XFeat();

    XFeat(const XFeat &) = delete;
    XFeat &operator=(const XFeat &) = delete;
    XFeat(XFeat &&) noexcept;
    XFeat &operator=(XFeat &&) noexcept;

    FeatureResult extract(const cv::Mat &image) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nnmatch
