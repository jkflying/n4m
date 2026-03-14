#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <n4m/xfeat.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

static std::string image_test_data_dir()
{
    return IMAGE_TEST_DATA_DIR;
}

static bool model_files_exist()
{
    std::ifstream f(std::string(MODELS_DIR) + "/xfeat.onnx");
    return f.good();
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

        config.model_path = std::string(MODELS_DIR) + "/xfeat.onnx";
        config.max_keypoints = 4096;
    }

    void validate_result(const n4m::FeatureResult &result, const cv::Mat &image)
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

    n4m::XFeatConfig config;
};

TEST_F(XFeatTest, ExtractSyntheticImage)
{
    n4m::XFeat xfeat(config);

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
    n4m::XFeat xfeat(config);

    cv::Mat image = cv::imread(image_test_data_dir() + "/P2530253.JPG");
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
    n4m::XFeat xfeat(config);

    cv::Mat img0 = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(image_test_data_dir() + "/P2540254.JPG");
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
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat.extract(image);
    EXPECT_LE(result.keypoints.size(), 100u);
}

TEST_F(XFeatTest, GrayscaleImage)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC1);
    cv::randu(image, cv::Scalar(0), cv::Scalar(255));
    cv::rectangle(image, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255), 2);

    auto result = xfeat.extract(image);
    validate_result(result, image);
}

TEST_F(XFeatTest, RGBAImage)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC4);
    cv::randu(image, cv::Scalar(0, 0, 0, 0), cv::Scalar(255, 255, 255, 255));

    auto result = xfeat.extract(image);
    validate_result(result, image);
}

TEST_F(XFeatTest, PortraitImage)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(640, 480, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::rectangle(image, cv::Point(100, 100), cv::Point(300, 500), cv::Scalar(255, 255, 255), 2);

    auto result = xfeat.extract(image);
    validate_result(result, image);
}

TEST_F(XFeatTest, SquareImage)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(512, 512, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat.extract(image);
    validate_result(result, image);
}

TEST_F(XFeatTest, NonMultipleOf32)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(479, 637, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat.extract(image);
    validate_result(result, image);
}

TEST_F(XFeatTest, SmallImage)
{
    n4m::XFeat xfeat(config);

    // Image smaller than PAD_MULTIPLE (32) — must not crash
    cv::Mat image(24, 24, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    EXPECT_THROW(xfeat.extract(image), std::invalid_argument);
}

TEST_F(XFeatTest, RealImageDownscaled1600)
{
    n4m::XFeat xfeat(config);

    cv::Mat image = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    if (image.empty())
    {
        GTEST_SKIP() << "Test image not found";
    }

    // Downscale to 1600px on the long side (matches opencalibration's feature extraction size)
    const int max_dim = 1600;
    double scale = static_cast<double>(max_dim) / std::max(image.rows, image.cols);
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(), scale, scale);

    auto t0 = std::chrono::steady_clock::now();
    auto result = xfeat.extract(resized);
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    std::cout << "XFeat extract " << resized.cols << "x" << resized.rows << ": " << ms << " ms, "
              << result.keypoints.size() << " keypoints" << std::endl;

    validate_result(result, resized);
    EXPECT_GT(result.keypoints.size(), 500u);
}

TEST_F(XFeatTest, CellSizeFiltering)
{
    config.cell_size = 16;
    config.max_keypoints = 4096;
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::rectangle(image, cv::Point(100, 100), cv::Point(300, 300), cv::Scalar(255, 255, 255), 2);

    auto result = xfeat.extract(image);
    EXPECT_GT(result.keypoints.size(), 0u);
    EXPECT_LE(result.keypoints.size(), 4096u);
    validate_result(result, image);
}

TEST_F(XFeatTest, CellSizeWithMaxKeypoints)
{
    config.cell_size = 16;
    config.max_keypoints = 10;
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat.extract(image);
    EXPECT_LE(result.keypoints.size(), 10u);
}

TEST_F(XFeatTest, MoveConstruction)
{
    n4m::XFeat xfeat1(config);
    n4m::XFeat xfeat2(std::move(xfeat1));

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat2.extract(image);
    EXPECT_GT(result.keypoints.size(), 0u);
}

TEST_F(XFeatTest, MoveAssignment)
{
    n4m::XFeat xfeat1(config);
    n4m::XFeat xfeat2(config);
    xfeat2 = std::move(xfeat1);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result = xfeat2.extract(image);
    EXPECT_GT(result.keypoints.size(), 0u);
}

TEST_F(XFeatTest, Deterministic)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto result1 = xfeat.extract(image);
    auto result2 = xfeat.extract(image);

    ASSERT_EQ(result1.keypoints.size(), result2.keypoints.size());
    for (size_t i = 0; i < result1.keypoints.size(); ++i)
    {
        EXPECT_FLOAT_EQ(result1.keypoints[i].x, result2.keypoints[i].x);
        EXPECT_FLOAT_EQ(result1.keypoints[i].y, result2.keypoints[i].y);
        EXPECT_FLOAT_EQ(result1.keypoints[i].score, result2.keypoints[i].score);
    }
}
