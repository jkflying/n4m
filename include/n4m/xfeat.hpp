#pragma once

#include <memory>
#include <string>

#include <opencv2/core.hpp>

#include <n4m/types.hpp>

namespace n4m
{

struct XFeatConfig
{
    std::string model_path;
    int max_keypoints = 4096;
    /// Grid cell size in pixels for spatial distribution (0 = disabled, pure top-k by score).
    /// Best keypoint per cell is kept, then capped to max_keypoints.
    /// 16 matches opencalibration's NMS density (8px radius, ~70% fill → ~same count at 1600px).
    int cell_size = 0;
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

} // namespace n4m
