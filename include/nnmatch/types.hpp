#pragma once

#include <array>
#include <vector>

namespace nnmatch
{

static constexpr int XFEAT_DESCRIPTOR_DIM = 64;
using Descriptor = std::array<float, XFEAT_DESCRIPTOR_DIM>;

struct Keypoint
{
    float x, y;
    float score;
    Descriptor descriptor;
};

struct FeatureResult
{
    std::vector<Keypoint> keypoints;
    int image_width = 0;
    int image_height = 0;
};

struct Match
{
    int idx0, idx1;
    float confidence;
};

} // namespace nnmatch
