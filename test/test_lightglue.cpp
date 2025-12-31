#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <n4m/lightglue.hpp>
#include <n4m/xfeat.hpp>

#include <chrono>
#include <fstream>
#include <iostream>

static std::string image_test_data_dir()
{
    return IMAGE_TEST_DATA_DIR;
}

static bool lightglue_model_exists()
{
    std::ifstream f(std::string(MODELS_DIR) + "/lightglue.onnx");
    return f.good();
}

static bool xfeat_model_exists()
{
    std::ifstream f(std::string(MODELS_DIR) + "/xfeat.onnx");
    return f.good();
}

class LightGlueTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        if (!lightglue_model_exists())
        {
            GTEST_SKIP() << "LightGlue model not found in models/";
        }

        lg_config.model_path = std::string(MODELS_DIR) + "/lightglue.onnx";
        lg_config.confidence_threshold = 0.0f;
    }

    n4m::LightGlueConfig lg_config;
};

class LightGlueWithXFeatTest : public LightGlueTest
{
  protected:
    void SetUp() override
    {
        LightGlueTest::SetUp();

        if (!xfeat_model_exists())
        {
            GTEST_SKIP() << "XFeat model not found in models/";
        }

        xfeat_config.model_path = std::string(MODELS_DIR) + "/xfeat.onnx";
        xfeat_config.max_keypoints = 4096;
    }

    n4m::XFeatConfig xfeat_config;
};

TEST_F(LightGlueTest, MatchSyntheticFeatures)
{
    n4m::LightGlue lg(lg_config);

    // Create identical features — should produce identity matches
    n4m::FeatureResult feats0, feats1;
    feats0.image_width = 640;
    feats0.image_height = 480;
    feats1.image_width = 640;
    feats1.image_height = 480;

    for (int i = 0; i < 50; ++i)
    {
        n4m::Keypoint kp;
        kp.x = static_cast<float>(i * 12);
        kp.y = static_cast<float>(i * 9);
        kp.score = 1.0f;
        kp.descriptor.fill(0.0f);
        kp.descriptor[i % n4m::XFEAT_DESCRIPTOR_DIM] = 1.0f;
        feats0.keypoints.push_back(kp);
        feats1.keypoints.push_back(kp);
    }

    auto matches = lg.match(feats0, feats1);
    EXPECT_GT(matches.size(), 0u);

    // With identical features, matches should be identity (idx0 == idx1)
    for (const auto &m : matches)
    {
        EXPECT_EQ(m.idx0, m.idx1) << "Identical features should produce identity matches";
        EXPECT_GT(m.confidence, 0.0f);
    }
}

TEST_F(LightGlueWithXFeatTest, MatchRealImages)
{
    n4m::XFeat xfeat(xfeat_config);
    n4m::LightGlue lg(lg_config);

    cv::Mat img0 = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(image_test_data_dir() + "/P2540254.JPG");
    if (img0.empty() || img1.empty())
    {
        GTEST_SKIP() << "Test images not found";
    }

    auto feats0 = xfeat.extract(img0);
    auto feats1 = xfeat.extract(img1);

    auto t0 = std::chrono::steady_clock::now();
    auto matches = lg.match(feats0, feats1);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "LightGlue match (full res): " << ms << " ms, " << matches.size() << " matches from "
              << feats0.keypoints.size() << " + " << feats1.keypoints.size() << " keypoints" << std::endl;

    EXPECT_GT(matches.size(), 50u) << "Same-scene images should produce many matches";

    for (const auto &m : matches)
    {
        EXPECT_GE(m.idx0, 0);
        EXPECT_LT(m.idx0, static_cast<int>(feats0.keypoints.size()));
        EXPECT_GE(m.idx1, 0);
        EXPECT_LT(m.idx1, static_cast<int>(feats1.keypoints.size()));
        EXPECT_GT(m.confidence, 0.0f);
    }
}

TEST_F(LightGlueWithXFeatTest, MatchRealImagesDownscaled1600)
{
    n4m::XFeat xfeat(xfeat_config);
    n4m::LightGlue lg(lg_config);

    cv::Mat img0 = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(image_test_data_dir() + "/P2540254.JPG");
    if (img0.empty() || img1.empty())
    {
        GTEST_SKIP() << "Test images not found";
    }

    const int max_dim = 1600;
    double scale0 = static_cast<double>(max_dim) / std::max(img0.rows, img0.cols);
    double scale1 = static_cast<double>(max_dim) / std::max(img1.rows, img1.cols);
    cv::Mat resized0, resized1;
    cv::resize(img0, resized0, cv::Size(), scale0, scale0);
    cv::resize(img1, resized1, cv::Size(), scale1, scale1);

    auto feats0 = xfeat.extract(resized0);
    auto feats1 = xfeat.extract(resized1);

    auto t0 = std::chrono::steady_clock::now();
    auto matches = lg.match(feats0, feats1);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "LightGlue match (1600px): " << ms << " ms, " << matches.size() << " matches from "
              << feats0.keypoints.size() << " + " << feats1.keypoints.size() << " keypoints" << std::endl;

    EXPECT_GT(matches.size(), 50u);
}

TEST(LightGlueBasic, EmptyInput)
{
    n4m::FeatureResult empty;
    EXPECT_TRUE(empty.keypoints.empty());
    EXPECT_EQ(empty.image_width, 0);
    EXPECT_EQ(empty.image_height, 0);

    n4m::Match m{0, 1, 0.95f};
    EXPECT_EQ(m.idx0, 0);
    EXPECT_EQ(m.idx1, 1);
    EXPECT_FLOAT_EQ(m.confidence, 0.95f);
}

TEST_F(LightGlueTest, EmptyFeatures)
{
    n4m::LightGlue lg(lg_config);

    n4m::FeatureResult feats0, feats1;
    feats0.image_width = 640;
    feats0.image_height = 480;
    feats1.image_width = 640;
    feats1.image_height = 480;

    auto matches = lg.match(feats0, feats1);
    EXPECT_TRUE(matches.empty());
}

TEST_F(LightGlueTest, ConfidenceThreshold)
{
    lg_config.confidence_threshold = 0.99f;
    n4m::LightGlue lg(lg_config);

    n4m::FeatureResult feats0, feats1;
    feats0.image_width = 640;
    feats0.image_height = 480;
    feats1.image_width = 640;
    feats1.image_height = 480;

    for (int i = 0; i < 50; ++i)
    {
        n4m::Keypoint kp;
        kp.x = static_cast<float>(i * 12);
        kp.y = static_cast<float>(i * 9);
        kp.score = 1.0f;
        kp.descriptor.fill(0.0f);
        kp.descriptor[i % n4m::XFEAT_DESCRIPTOR_DIM] = 1.0f;
        feats0.keypoints.push_back(kp);
        feats1.keypoints.push_back(kp);
    }

    auto matches_strict = lg.match(feats0, feats1);

    lg_config.confidence_threshold = 0.0f;
    n4m::LightGlue lg_loose(lg_config);
    auto matches_loose = lg_loose.match(feats0, feats1);

    EXPECT_LE(matches_strict.size(), matches_loose.size()) << "Higher threshold should produce fewer or equal matches";
}

TEST_F(LightGlueWithXFeatTest, FeatureResultHasImageSize)
{
    n4m::XFeat xfeat(xfeat_config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat.extract(image);
    EXPECT_EQ(result.image_width, 640);
    EXPECT_EQ(result.image_height, 480);
}

TEST_F(LightGlueWithXFeatTest, FeatureResultHasImageSizeRealImage)
{
    n4m::XFeat xfeat(xfeat_config);

    cv::Mat image = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    if (image.empty())
    {
        GTEST_SKIP() << "Test image not found";
    }

    auto result = xfeat.extract(image);
    EXPECT_EQ(result.image_width, image.cols);
    EXPECT_EQ(result.image_height, image.rows);
}
