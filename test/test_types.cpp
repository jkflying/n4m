#include <gtest/gtest.h>

#include <nnmatch/types.hpp>

#include <cmath>
#include <numeric>

TEST(Types, DescriptorDimension)
{
    EXPECT_EQ(nnmatch::XFEAT_DESCRIPTOR_DIM, 64);
    nnmatch::Descriptor desc;
    EXPECT_EQ(desc.size(), 64u);
}

TEST(Types, KeypointDefaultInit)
{
    nnmatch::Keypoint kp{};
    EXPECT_FLOAT_EQ(kp.x, 0.0f);
    EXPECT_FLOAT_EQ(kp.y, 0.0f);
    EXPECT_FLOAT_EQ(kp.score, 0.0f);
}

TEST(Types, FeatureResultPushBack)
{
    nnmatch::FeatureResult result;
    EXPECT_TRUE(result.keypoints.empty());

    nnmatch::Keypoint kp;
    kp.x = 10.0f;
    kp.y = 20.0f;
    kp.score = 0.5f;
    kp.descriptor.fill(0.0f);
    kp.descriptor[0] = 1.0f;
    result.keypoints.push_back(kp);

    EXPECT_EQ(result.keypoints.size(), 1u);
    EXPECT_FLOAT_EQ(result.keypoints[0].x, 10.0f);
    EXPECT_FLOAT_EQ(result.keypoints[0].descriptor[0], 1.0f);
    EXPECT_FLOAT_EQ(result.keypoints[0].descriptor[1], 0.0f);
}

TEST(Types, MatchFields)
{
    nnmatch::Match m{3, 7, 0.85f};
    EXPECT_EQ(m.idx0, 3);
    EXPECT_EQ(m.idx1, 7);
    EXPECT_FLOAT_EQ(m.confidence, 0.85f);
}

TEST(Types, FeatureResultMove)
{
    nnmatch::FeatureResult a;
    nnmatch::Keypoint kp{};
    kp.x = 1.0f;
    kp.y = 2.0f;
    kp.score = 0.9f;
    kp.descriptor.fill(0.1f);
    a.keypoints.push_back(kp);

    nnmatch::FeatureResult b = std::move(a);
    EXPECT_EQ(b.keypoints.size(), 1u);
    EXPECT_FLOAT_EQ(b.keypoints[0].x, 1.0f);
}
