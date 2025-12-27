#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nnmatch/xfeat.hpp>

#include <cmath>
#include <fstream>

static std::string models_dir()
{
    return std::string(TEST_DATA_DIR) + "/../../models";
}

static std::string test_data_dir()
{
    return TEST_DATA_DIR;
}

static bool model_files_exist()
{
    std::ifstream p(models_dir() + "/xfeat.param");
    std::ifstream b(models_dir() + "/xfeat.bin");
    return p.good() && b.good();
}

class XFeatTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        if (!model_files_exist())
        {
            GTEST_SKIP() << "XFeat model files not found in models/";
        }

        config.param_path = models_dir() + "/xfeat.param";
        config.bin_path = models_dir() + "/xfeat.bin";
        config.max_keypoints = 4096;
    }

    void validate_result(const nnmatch::FeatureResult &result, const cv::Mat &image)
    {
        EXPECT_GT(result.keypoints.size(), 0u);

        for (const auto &kp : result.keypoints)
        {
            EXPECT_GE(kp.x, 0.0f);
            EXPECT_LE(kp.x, static_cast<float>(image.cols));
            EXPECT_GE(kp.y, 0.0f);
            EXPECT_LE(kp.y, static_cast<float>(image.rows));
            EXPECT_GT(kp.score, 0.0f);

            float norm_sq = 0.0f;
            for (float v : kp.descriptor)
            {
                norm_sq += v * v;
            }
            EXPECT_NEAR(std::sqrt(norm_sq), 1.0f, 0.01f);
        }

        // Scores sorted descending
        for (size_t i = 1; i < result.keypoints.size(); ++i)
        {
            EXPECT_GE(result.keypoints[i - 1].score, result.keypoints[i].score);
        }
    }

    nnmatch::XFeatConfig config;
};

TEST_F(XFeatTest, ExtractSyntheticImage)
{
    nnmatch::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::rectangle(image, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), 2);
    cv::circle(image, cv::Point(400, 200), 50, cv::Scalar(0, 0, 0), 3);

    auto result = xfeat.extract(image);
    validate_result(result, image);
    EXPECT_LE(result.keypoints.size(), 4096u);
}

TEST_F(XFeatTest, ExtractRealImage)
{
    nnmatch::XFeat xfeat(config);

    cv::Mat image = cv::imread(test_data_dir() + "/P2530253.JPG");
    if (image.empty())
    {
        GTEST_SKIP() << "Test image not found";
    }

    auto result = xfeat.extract(image);
    validate_result(result, image);
    // Real image should produce many keypoints
    EXPECT_GT(result.keypoints.size(), 500u);
}

TEST_F(XFeatTest, TwoImagesSameScene)
{
    nnmatch::XFeat xfeat(config);

    cv::Mat img0 = cv::imread(test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(test_data_dir() + "/P2540254.JPG");
    if (img0.empty() || img1.empty())
    {
        GTEST_SKIP() << "Test images not found";
    }

    auto feats0 = xfeat.extract(img0);
    auto feats1 = xfeat.extract(img1);

    validate_result(feats0, img0);
    validate_result(feats1, img1);

    // Both images from same scene should produce similar keypoint counts
    EXPECT_GT(feats0.keypoints.size(), 500u);
    EXPECT_GT(feats1.keypoints.size(), 500u);
}

TEST_F(XFeatTest, MaxKeypointsRespected)
{
    config.max_keypoints = 100;
    nnmatch::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat.extract(image);
    EXPECT_LE(result.keypoints.size(), 100u);
}
