#pragma once

#include <array>
#include <vector>

namespace n4m
{

enum class Backend
{
    cpu,
    cuda,
    tensorrt,
    coreml,
    directml,
    rocm,
    openvino,
};

inline const char *to_string(Backend b)
{
    switch (b)
    {
    case Backend::cpu:
        return "cpu";
    case Backend::cuda:
        return "cuda";
    case Backend::tensorrt:
        return "tensorrt";
    case Backend::coreml:
        return "coreml";
    case Backend::directml:
        return "directml";
    case Backend::rocm:
        return "rocm";
    case Backend::openvino:
        return "openvino";
    }
    return "unknown";
}

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

} // namespace n4m
