#include <gtest/gtest.h>

#include <n4m/backend.hpp>
#include <n4m/types.hpp>

#include <cmath>
#include <numeric>

TEST(Types, DescriptorDimension)
{
    EXPECT_EQ(n4m::XFEAT_DESCRIPTOR_DIM, 64);
    n4m::Descriptor desc;
    EXPECT_EQ(desc.size(), 64u);
}

TEST(Types, KeypointDefaultInit)
{
    n4m::Keypoint kp{};
    EXPECT_FLOAT_EQ(kp.x, 0.0f);
    EXPECT_FLOAT_EQ(kp.y, 0.0f);
    EXPECT_FLOAT_EQ(kp.score, 0.0f);
}

TEST(Types, FeatureResultPushBack)
{
    n4m::FeatureResult result;
    EXPECT_TRUE(result.keypoints.empty());

    n4m::Keypoint kp;
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
    n4m::Match m{3, 7, 0.85f};
    EXPECT_EQ(m.idx0, 3);
    EXPECT_EQ(m.idx1, 7);
    EXPECT_FLOAT_EQ(m.confidence, 0.85f);
}

TEST(Types, FeatureResultMove)
{
    n4m::FeatureResult a;
    n4m::Keypoint kp{};
    kp.x = 1.0f;
    kp.y = 2.0f;
    kp.score = 0.9f;
    kp.descriptor.fill(0.1f);
    a.keypoints.push_back(kp);

    n4m::FeatureResult b = std::move(a);
    EXPECT_EQ(b.keypoints.size(), 1u);
    EXPECT_FLOAT_EQ(b.keypoints[0].x, 1.0f);
}

TEST(Types, AvailableBackends)
{
    auto backends = n4m::available_backends();

    // CPU must always be available
    ASSERT_FALSE(backends.empty());
    EXPECT_EQ(backends.front(), n4m::Backend::cpu);

    std::cout << "Available backends (" << backends.size() << "):" << std::endl;
    for (auto b : backends)
    {
        std::cout << "  " << n4m::to_string(b) << " (" << n4m::backend_provider_name(b) << ")" << std::endl;
    }
}

TEST(Types, BackendToString)
{
    EXPECT_STREQ(n4m::to_string(n4m::Backend::cpu), "cpu");
    EXPECT_STREQ(n4m::to_string(n4m::Backend::cuda), "cuda");
    EXPECT_STREQ(n4m::to_string(n4m::Backend::tensorrt), "tensorrt");
    EXPECT_STREQ(n4m::backend_provider_name(n4m::Backend::cpu), "CPUExecutionProvider");
    EXPECT_STREQ(n4m::backend_provider_name(n4m::Backend::cuda), "CUDAExecutionProvider");
}
