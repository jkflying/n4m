#include <gtest/gtest.h>

#include <nnmatch/lightglue.hpp>

#include <fstream>
#include <vector>

static std::string reference_data_dir()
{
    return REFERENCE_DATA_DIR;
}

TEST(LightGlue, DISABLED_MatchMatchesReference)
{
    auto dir = reference_data_dir();

    // Load reference features (would need to deserialize from .npy)
    // For now this test is a placeholder until reference data is generated

    nnmatch::LightGlueConfig config;
    config.model_path = std::string(MODELS_DIR) + "/lightglue.onnx";
    config.confidence_threshold = 0.0f;

    nnmatch::LightGlue lg(config);

    // Create synthetic features for smoke test
    nnmatch::FeatureResult feats0, feats1;
    for (int i = 0; i < 10; ++i)
    {
        nnmatch::Keypoint kp;
        kp.x = static_cast<float>(i * 10);
        kp.y = static_cast<float>(i * 10);
        kp.score = 1.0f;
        kp.descriptor.fill(0.0f);
        kp.descriptor[i % nnmatch::XFEAT_DESCRIPTOR_DIM] = 1.0f;
        feats0.keypoints.push_back(kp);
        feats1.keypoints.push_back(kp);
    }

    auto matches = lg.match(feats0, feats1);
    // With identical features, we expect some matches
    EXPECT_GT(matches.size(), 0u);
}

TEST(LightGlue, EmptyInput)
{
    // Can't construct without valid model files, so just test the type API
    nnmatch::FeatureResult empty;
    EXPECT_TRUE(empty.keypoints.empty());

    nnmatch::Match m{0, 1, 0.95f};
    EXPECT_EQ(m.idx0, 0);
    EXPECT_EQ(m.idx1, 1);
    EXPECT_FLOAT_EQ(m.confidence, 0.95f);
}
