#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <n4m/xfeat.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
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

// --- Batch tests ---

TEST_F(XFeatTest, ExtractBatchEmpty)
{
    n4m::XFeat xfeat(config);
    EXPECT_THROW(xfeat.extract_batch({}), std::invalid_argument);
}

TEST_F(XFeatTest, ExtractBatchSingleImage)
{
    n4m::XFeat xfeat(config);

    cv::Mat image(480, 640, CV_8UC3);
    cv::randu(image, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto single = xfeat.extract(image);
    auto batch = xfeat.extract_batch({image});

    ASSERT_EQ(batch.size(), 1u);
    ASSERT_EQ(batch[0].keypoints.size(), single.keypoints.size());
    for (size_t i = 0; i < single.keypoints.size(); ++i)
    {
        EXPECT_FLOAT_EQ(batch[0].keypoints[i].x, single.keypoints[i].x);
        EXPECT_FLOAT_EQ(batch[0].keypoints[i].y, single.keypoints[i].y);
        EXPECT_FLOAT_EQ(batch[0].keypoints[i].score, single.keypoints[i].score);
    }
}

TEST_F(XFeatTest, ExtractBatchSameSize)
{
    n4m::XFeat xfeat(config);

    std::vector<cv::Mat> images(3);
    for (auto &img : images)
    {
        img.create(480, 640, CV_8UC3);
        cv::randu(img, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    }

    auto batch_results = xfeat.extract_batch(images);
    ASSERT_EQ(batch_results.size(), 3u);

    // Compare against sequential extraction — should be bit-identical for same-size images
    for (size_t b = 0; b < images.size(); ++b)
    {
        auto single = xfeat.extract(images[b]);
        ASSERT_EQ(batch_results[b].keypoints.size(), single.keypoints.size()) << "Batch " << b;
        for (size_t i = 0; i < single.keypoints.size(); ++i)
        {
            EXPECT_FLOAT_EQ(batch_results[b].keypoints[i].x, single.keypoints[i].x) << "Batch " << b << " kp " << i;
            EXPECT_FLOAT_EQ(batch_results[b].keypoints[i].y, single.keypoints[i].y) << "Batch " << b << " kp " << i;
        }
    }
}

TEST_F(XFeatTest, ExtractBatchDifferentSizes)
{
    n4m::XFeat xfeat(config);

    cv::Mat img1(480, 640, CV_8UC3);
    cv::Mat img2(320, 480, CV_8UC3);
    cv::Mat img3(640, 480, CV_8UC3);
    cv::randu(img1, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::randu(img2, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::randu(img3, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto results = xfeat.extract_batch({img1, img2, img3});
    ASSERT_EQ(results.size(), 3u);

    // Verify results are valid (bounds, descriptors normalized, etc.)
    validate_result(results[0], img1);
    validate_result(results[1], img2);
    validate_result(results[2], img3);
}

TEST_F(XFeatTest, ExtractBatchDeterministic)
{
    n4m::XFeat xfeat(config);

    cv::Mat img1(480, 640, CV_8UC3);
    cv::Mat img2(480, 640, CV_8UC3);
    cv::randu(img1, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::randu(img2, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto r1 = xfeat.extract_batch({img1, img2});
    auto r2 = xfeat.extract_batch({img1, img2});

    ASSERT_EQ(r1.size(), r2.size());
    for (size_t b = 0; b < r1.size(); ++b)
    {
        ASSERT_EQ(r1[b].keypoints.size(), r2[b].keypoints.size());
        for (size_t i = 0; i < r1[b].keypoints.size(); ++i)
        {
            EXPECT_FLOAT_EQ(r1[b].keypoints[i].x, r2[b].keypoints[i].x);
            EXPECT_FLOAT_EQ(r1[b].keypoints[i].y, r2[b].keypoints[i].y);
        }
    }
}

TEST_F(XFeatTest, ExtractBatchMaxKeypoints)
{
    config.max_keypoints = 50;
    n4m::XFeat xfeat(config);

    cv::Mat img1(480, 640, CV_8UC3);
    cv::Mat img2(480, 640, CV_8UC3);
    cv::randu(img1, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    cv::randu(img2, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

    auto results = xfeat.extract_batch({img1, img2});
    for (const auto &r : results)
    {
        EXPECT_LE(r.keypoints.size(), 50u);
    }
}

TEST_F(XFeatTest, DISABLED_ExtractBatchScaling)
{
    cv::Mat img0 = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(image_test_data_dir() + "/P2540254.JPG");
    cv::Mat img2 = cv::imread(image_test_data_dir() + "/P2550255.JPG");
    cv::Mat img3 = cv::imread(image_test_data_dir() + "/P2560256.JPG");
    if (img0.empty() || img1.empty() || img2.empty() || img3.empty())
    {
        GTEST_SKIP() << "Test images not found";
    }

    const int max_dim = 400;
    std::vector<cv::Mat> source_images = {img0, img1, img2, img3};
    for (auto &img : source_images)
    {
        double scale = static_cast<double>(max_dim) / std::max(img.rows, img.cols);
        cv::resize(img, img, cv::Size(), scale, scale);
    }

    const std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 200};
    const std::vector<int> thread_counts = {1, 2, 4, 8};

    // Build images pool by repeating source images
    const int max_batch = batch_sizes.back();
    std::vector<cv::Mat> images;
    images.reserve(max_batch);
    for (int i = 0; i < max_batch; ++i)
    {
        images.push_back(source_images[i % source_images.size()]);
    }

    // Header
    std::cout << "\nXFeat extract_batch scaling (" << source_images[0].cols << "x" << source_images[0].rows << ")"
              << std::endl;
    std::cout << "batch";
    for (int t : thread_counts)
    {
        std::cout << "\t" << t << "thr(ms)\t" << t << "thr/img";
    }
    std::cout << std::endl;

    for (int bs : batch_sizes)
    {
        std::cout << bs;
        std::vector<cv::Mat> batch(images.begin(), images.begin() + bs);

        for (int threads : thread_counts)
        {
            config.intra_op_threads = threads;
            n4m::XFeat xfeat(config);

            // Warm up
            xfeat.extract(batch[0]);

            auto t0 = std::chrono::steady_clock::now();
            if (bs == 1)
            {
                xfeat.extract(batch[0]);
            }
            else
            {
                xfeat.extract_batch(batch);
            }
            auto t1 = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            double per_img = ms / bs;
            std::cout << "\t" << std::fixed << std::setprecision(1) << ms << "\t" << per_img;
        }
        std::cout << std::endl;
    }
}

TEST_F(XFeatTest, ExtractBatchRealImages)
{
    n4m::XFeat xfeat(config);

    cv::Mat img0 = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(image_test_data_dir() + "/P2540254.JPG");
    if (img0.empty() || img1.empty())
    {
        GTEST_SKIP() << "Test images not found";
    }

    // Use same-size images so batch results should match sequential
    const int max_dim = 800;
    double scale0 = static_cast<double>(max_dim) / std::max(img0.rows, img0.cols);
    double scale1 = static_cast<double>(max_dim) / std::max(img1.rows, img1.cols);
    cv::resize(img0, img0, cv::Size(), scale0, scale0);
    cv::resize(img1, img1, cv::Size(), scale1, scale1);

    auto batch_results = xfeat.extract_batch({img0, img1});
    ASSERT_EQ(batch_results.size(), 2u);

    validate_result(batch_results[0], img0);
    validate_result(batch_results[1], img1);

    EXPECT_GT(batch_results[0].keypoints.size(), 100u);
    EXPECT_GT(batch_results[1].keypoints.size(), 100u);
}
