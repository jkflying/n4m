#include <gtest/gtest.h>

#include "xfeat_postprocess.hpp"

#include <cmath>
#include <vector>

using namespace nnmatch::detail;

TEST(NMS, SinglePeak)
{
    // 5x5 heatmap with one peak at (2,2)
    // clang-format off
    const float hm[] = {
        0, 0, 0, 0, 0,
        0, 1, 2, 1, 0,
        0, 2, 5, 2, 0,
        0, 1, 2, 1, 0,
        0, 0, 0, 0, 0,
    };
    // clang-format on

    auto result = nms_3x3(hm, 5, 5);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].x, 2);
    EXPECT_EQ(result[0].y, 2);
    EXPECT_FLOAT_EQ(result[0].score, 5.0f);
}

TEST(NMS, MultiplePeaks)
{
    // 7x7 heatmap with two peaks
    std::vector<float> hm(7 * 7, 0.0f);
    hm[1 * 7 + 1] = 3.0f; // peak at (1,1)
    hm[5 * 7 + 5] = 4.0f; // peak at (5,5)

    auto result = nms_3x3(hm.data(), 7, 7);
    ASSERT_EQ(result.size(), 2u);
    // Sorted by score descending
    EXPECT_EQ(result[0].x, 5);
    EXPECT_EQ(result[0].y, 5);
    EXPECT_FLOAT_EQ(result[0].score, 4.0f);
    EXPECT_EQ(result[1].x, 1);
    EXPECT_EQ(result[1].y, 1);
    EXPECT_FLOAT_EQ(result[1].score, 3.0f);
}

TEST(NMS, NoPeaks)
{
    std::vector<float> hm(5 * 5, 0.0f);
    auto result = nms_3x3(hm.data(), 5, 5);
    EXPECT_TRUE(result.empty());
}

TEST(NMS, AllNegative)
{
    std::vector<float> hm(5 * 5, -1.0f);
    auto result = nms_3x3(hm.data(), 5, 5);
    EXPECT_TRUE(result.empty());
}

TEST(NMS, PlateauSuppressed)
{
    // A flat 3x3 region of equal values - no strict maximum, so no detections
    // clang-format off
    const float hm[] = {
        0, 0, 0, 0, 0,
        0, 3, 3, 3, 0,
        0, 3, 3, 3, 0,
        0, 3, 3, 3, 0,
        0, 0, 0, 0, 0,
    };
    // clang-format on

    auto result = nms_3x3(hm, 5, 5);
    EXPECT_TRUE(result.empty());
}

TEST(NMS, EdgePeaksIgnored)
{
    // Peaks on the border row/col are not considered (NMS starts at y=1, x=1)
    std::vector<float> hm(5 * 5, 0.0f);
    hm[0 * 5 + 2] = 10.0f; // top edge
    hm[2 * 5 + 0] = 10.0f; // left edge
    hm[4 * 5 + 2] = 10.0f; // bottom edge
    hm[2 * 5 + 4] = 10.0f; // right edge

    auto result = nms_3x3(hm.data(), 5, 5);
    EXPECT_TRUE(result.empty());
}

TEST(SampleDescriptor, IntegerCoords)
{
    // Create a 2x2 descriptor map with 64 channels, CHW layout
    const int w = 2, h = 2;
    std::vector<float> desc_map(nnmatch::XFEAT_DESCRIPTOR_DIM * h * w, 0.0f);

    // Set channel 0 at position (0,0) to 3.0
    desc_map[0 * h * w + 0] = 3.0f;

    auto desc = sample_descriptor(desc_map.data(), nnmatch::XFEAT_DESCRIPTOR_DIM, w, h, 0.0f, 0.0f);
    // After L2 normalize, channel 0 should be ~1.0 (only nonzero channel)
    EXPECT_NEAR(desc[0], 1.0f, 1e-6f);
    for (int d = 1; d < nnmatch::XFEAT_DESCRIPTOR_DIM; ++d)
    {
        EXPECT_NEAR(desc[d], 0.0f, 1e-6f);
    }
}

TEST(SampleDescriptor, BilinearInterpolation)
{
    // 2x2 map, channel 0: top-left=4, rest=0
    const int w = 2, h = 2;
    std::vector<float> desc_map(nnmatch::XFEAT_DESCRIPTOR_DIM * h * w, 0.0f);

    desc_map[0 * h * w + 0] = 4.0f; // channel 0, (0,0)

    // Sample at (0.5, 0.0) => bilinear: 4*(1-0.5)*(1-0) = 2.0
    auto desc = sample_descriptor(desc_map.data(), nnmatch::XFEAT_DESCRIPTOR_DIM, w, h, 0.5f, 0.0f);
    // After L2 norm, still 1.0 since only one channel nonzero
    EXPECT_NEAR(desc[0], 1.0f, 1e-6f);
}

TEST(LogitsToHeatmap, SingleCell)
{
    // 1x1 grid with 65 channels: first channel has high logit, rest are 0
    // CHW layout: 65 channels, each 1x1
    std::vector<float> logits(65 * 1 * 1, 0.0f);
    logits[0] = 10.0f; // channel 0, strong activation for bin 0

    auto heatmap = logits_to_heatmap(logits.data(), 1, 1);
    ASSERT_EQ(heatmap.size(), 64u); // 8x8

    // Bin 0 maps to position (0,0) in 8x8 block — should have highest value
    float max_val = *std::max_element(heatmap.begin(), heatmap.end());
    EXPECT_FLOAT_EQ(heatmap[0], max_val);
    EXPECT_GT(max_val, 0.5f); // softmax with one high logit

    // Sum should be < 1.0 (dustbin channel absorbs some probability)
    float sum = 0.0f;
    for (float v : heatmap)
        sum += v;
    EXPECT_LT(sum, 1.0f);
}

TEST(LogitsToHeatmap, UniformLogits)
{
    // All channels equal → each of 65 channels gets 1/65 probability
    std::vector<float> logits(65 * 1 * 1, 0.0f);

    auto heatmap = logits_to_heatmap(logits.data(), 1, 1);
    for (float v : heatmap)
    {
        EXPECT_NEAR(v, 1.0f / 65.0f, 1e-5f);
    }
}

TEST(SampleDescriptor, L2Normalized)
{
    // Set two channels to nonzero to verify normalization
    const int w = 1, h = 1;
    std::vector<float> desc_map(nnmatch::XFEAT_DESCRIPTOR_DIM * h * w, 0.0f);

    desc_map[0] = 3.0f; // channel 0
    desc_map[1] = 4.0f; // channel 1

    auto desc = sample_descriptor(desc_map.data(), nnmatch::XFEAT_DESCRIPTOR_DIM, w, h, 0.0f, 0.0f);

    float norm = 0.0f;
    for (int d = 0; d < nnmatch::XFEAT_DESCRIPTOR_DIM; ++d)
    {
        norm += desc[d] * desc[d];
    }
    EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-5f);
    EXPECT_NEAR(desc[0], 3.0f / 5.0f, 1e-5f);
    EXPECT_NEAR(desc[1], 4.0f / 5.0f, 1e-5f);
}
