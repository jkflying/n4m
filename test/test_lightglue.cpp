#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <n4m/lightglue.hpp>
#include <n4m/xfeat.hpp>

#include <chrono>
#include <fstream>
#include <iomanip>
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

TEST_F(LightGlueTest, MoveConstruction)
{
    n4m::LightGlue lg1(lg_config);
    n4m::LightGlue lg2(std::move(lg1));

    n4m::FeatureResult feats0, feats1;
    feats0.image_width = 640;
    feats0.image_height = 480;
    feats1.image_width = 640;
    feats1.image_height = 480;

    auto matches = lg2.match(feats0, feats1);
    EXPECT_TRUE(matches.empty());
}

TEST_F(LightGlueTest, MoveAssignment)
{
    n4m::LightGlue lg1(lg_config);
    n4m::LightGlue lg2(lg_config);
    lg2 = std::move(lg1);

    n4m::FeatureResult feats0, feats1;
    feats0.image_width = 640;
    feats0.image_height = 480;
    feats1.image_width = 640;
    feats1.image_height = 480;

    auto matches = lg2.match(feats0, feats1);
    EXPECT_TRUE(matches.empty());
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

// --- Batch tests ---

TEST_F(LightGlueTest, MatchBatchEmpty)
{
    n4m::LightGlue lg(lg_config);
    auto results = lg.match_batch({});
    EXPECT_TRUE(results.empty());
}

TEST_F(LightGlueTest, MatchBatchSinglePair)
{
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

    auto single = lg.match(feats0, feats1);
    auto batch = lg.match_batch({{feats0, feats1}});

    ASSERT_EQ(batch.size(), 1u);
    ASSERT_EQ(batch[0].size(), single.size());
    for (size_t i = 0; i < single.size(); ++i)
    {
        EXPECT_EQ(batch[0][i].idx0, single[i].idx0);
        EXPECT_EQ(batch[0][i].idx1, single[i].idx1);
        EXPECT_FLOAT_EQ(batch[0][i].confidence, single[i].confidence);
    }
}

TEST_F(LightGlueTest, MatchBatchMultiplePairs)
{
    n4m::LightGlue lg(lg_config);

    // Create two different pairs of features
    auto make_feats = [](int n, float offset) {
        n4m::FeatureResult feats;
        feats.image_width = 640;
        feats.image_height = 480;
        for (int i = 0; i < n; ++i)
        {
            n4m::Keypoint kp;
            kp.x = static_cast<float>(i * 12) + offset;
            kp.y = static_cast<float>(i * 9) + offset;
            kp.score = 1.0f;
            kp.descriptor.fill(0.0f);
            kp.descriptor[i % n4m::XFEAT_DESCRIPTOR_DIM] = 1.0f;
            feats.keypoints.push_back(kp);
        }
        return feats;
    };

    auto f0a = make_feats(50, 0.0f);
    auto f0b = make_feats(50, 0.0f);
    auto f1a = make_feats(50, 10.0f);
    auto f1b = make_feats(50, 10.0f);

    std::vector<std::pair<n4m::FeatureResult, n4m::FeatureResult>> pairs = {{f0a, f0b}, {f1a, f1b}};

    auto batch_results = lg.match_batch(pairs);
    ASSERT_EQ(batch_results.size(), 2u);

    // Each pair should produce valid matches
    for (size_t p = 0; p < batch_results.size(); ++p)
    {
        for (const auto &m : batch_results[p])
        {
            EXPECT_GE(m.idx0, 0);
            EXPECT_LT(m.idx0, static_cast<int>(pairs[p].first.keypoints.size()));
            EXPECT_GE(m.idx1, 0);
            EXPECT_LT(m.idx1, static_cast<int>(pairs[p].second.keypoints.size()));
            EXPECT_GT(m.confidence, 0.0f);
        }
    }
}

TEST_F(LightGlueTest, MatchBatchUniformKeypoints)
{
    n4m::LightGlue lg(lg_config);

    // All pairs have exactly the same keypoint count — optimal path (no padding waste)
    auto make_feats = [](int n) {
        n4m::FeatureResult feats;
        feats.image_width = 640;
        feats.image_height = 480;
        for (int i = 0; i < n; ++i)
        {
            n4m::Keypoint kp;
            kp.x = static_cast<float>(i * 12);
            kp.y = static_cast<float>(i * 9);
            kp.score = 1.0f;
            kp.descriptor.fill(0.0f);
            kp.descriptor[i % n4m::XFEAT_DESCRIPTOR_DIM] = 1.0f;
            feats.keypoints.push_back(kp);
        }
        return feats;
    };

    std::vector<std::pair<n4m::FeatureResult, n4m::FeatureResult>> pairs;
    for (int i = 0; i < 3; ++i)
    {
        pairs.push_back({make_feats(40), make_feats(40)});
    }

    auto results = lg.match_batch(pairs);
    ASSERT_EQ(results.size(), 3u);

    for (const auto &matches : results)
    {
        EXPECT_GT(matches.size(), 0u);
    }
}

TEST_F(LightGlueTest, MatchBatchMixedKeypoints)
{
    n4m::LightGlue lg(lg_config);

    auto make_feats = [](int n) {
        n4m::FeatureResult feats;
        feats.image_width = 640;
        feats.image_height = 480;
        for (int i = 0; i < n; ++i)
        {
            n4m::Keypoint kp;
            kp.x = static_cast<float>(i * 12);
            kp.y = static_cast<float>(i * 9);
            kp.score = 1.0f;
            kp.descriptor.fill(0.0f);
            kp.descriptor[i % n4m::XFEAT_DESCRIPTOR_DIM] = 1.0f;
            feats.keypoints.push_back(kp);
        }
        return feats;
    };

    // Different keypoint counts → padding required
    std::vector<std::pair<n4m::FeatureResult, n4m::FeatureResult>> pairs = {
        {make_feats(30), make_feats(50)},
        {make_feats(60), make_feats(20)},
    };

    auto results = lg.match_batch(pairs);
    ASSERT_EQ(results.size(), 2u);

    // Verify no match indices exceed original keypoint counts
    for (size_t p = 0; p < results.size(); ++p)
    {
        for (const auto &m : results[p])
        {
            EXPECT_GE(m.idx0, 0);
            EXPECT_LT(m.idx0, static_cast<int>(pairs[p].first.keypoints.size()));
            EXPECT_GE(m.idx1, 0);
            EXPECT_LT(m.idx1, static_cast<int>(pairs[p].second.keypoints.size()));
        }
    }
}

TEST_F(LightGlueTest, MatchBatchMixedEmpty)
{
    n4m::LightGlue lg(lg_config);

    n4m::FeatureResult empty;
    empty.image_width = 640;
    empty.image_height = 480;

    n4m::FeatureResult feats;
    feats.image_width = 640;
    feats.image_height = 480;
    for (int i = 0; i < 30; ++i)
    {
        n4m::Keypoint kp;
        kp.x = static_cast<float>(i * 12);
        kp.y = static_cast<float>(i * 9);
        kp.score = 1.0f;
        kp.descriptor.fill(0.0f);
        kp.descriptor[i % n4m::XFEAT_DESCRIPTOR_DIM] = 1.0f;
        feats.keypoints.push_back(kp);
    }

    // One empty pair + one valid pair
    // When one pair has 0 keypoints and the other doesn't, max_n0/n1 > 0
    // but the empty pair contributes 0 original keypoints so gets no matches
    auto results = lg.match_batch({{empty, feats}, {feats, feats}});
    ASSERT_EQ(results.size(), 2u);
    EXPECT_TRUE(results[0].empty()) << "Empty source keypoints should produce no matches";
}

TEST_F(LightGlueWithXFeatTest, MatchBatchWithRealImages)
{
    n4m::XFeat xfeat(xfeat_config);
    n4m::LightGlue lg(lg_config);

    cv::Mat img0 = cv::imread(image_test_data_dir() + "/P2530253.JPG");
    cv::Mat img1 = cv::imread(image_test_data_dir() + "/P2540254.JPG");
    if (img0.empty() || img1.empty())
    {
        GTEST_SKIP() << "Test images not found";
    }

    const int max_dim = 800;
    double scale0 = static_cast<double>(max_dim) / std::max(img0.rows, img0.cols);
    double scale1 = static_cast<double>(max_dim) / std::max(img1.rows, img1.cols);
    cv::resize(img0, img0, cv::Size(), scale0, scale0);
    cv::resize(img1, img1, cv::Size(), scale1, scale1);

    auto feats0 = xfeat.extract(img0);
    auto feats1 = xfeat.extract(img1);

    // Sequential timing
    auto t0 = std::chrono::steady_clock::now();
    auto single0 = lg.match(feats0, feats1);
    auto single1 = lg.match(feats1, feats0);
    auto t1 = std::chrono::steady_clock::now();

    // Batch timing
    std::vector<std::pair<n4m::FeatureResult, n4m::FeatureResult>> pairs = {{feats0, feats1}, {feats1, feats0}};
    auto t2 = std::chrono::steady_clock::now();
    auto batch_results = lg.match_batch(pairs);
    auto t3 = std::chrono::steady_clock::now();

    auto seq_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();

    std::cout << "LightGlue match_batch (2 pairs): sequential=" << seq_ms << "ms, batch=" << batch_ms << "ms"
              << std::endl;

    ASSERT_EQ(batch_results.size(), 2u);
    EXPECT_GT(batch_results[0].size(), 50u);
    EXPECT_GT(batch_results[1].size(), 50u);
}

TEST_F(LightGlueWithXFeatTest, DISABLED_MatchBatchScaling)
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

    // Extract features with reduced keypoints for tractable large batches
    xfeat_config.max_keypoints = 512;
    n4m::XFeat xfeat(xfeat_config);
    std::vector<n4m::FeatureResult> source_feats;
    for (const auto &img : source_images)
    {
        source_feats.push_back(xfeat.extract(img));
    }

    // Build pair pool by cycling through source features
    const std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 200};
    const std::vector<int> thread_counts = {1, 2, 4, 8};
    const int max_batch = batch_sizes.back();

    std::vector<std::pair<n4m::FeatureResult, n4m::FeatureResult>> all_pairs;
    all_pairs.reserve(max_batch);
    for (int i = 0; i < max_batch; ++i)
    {
        all_pairs.push_back({source_feats[i % source_feats.size()], source_feats[(i + 1) % source_feats.size()]});
    }

    std::cout << "\nLightGlue match_batch scaling (" << source_feats[0].keypoints.size() << " kpts)" << std::endl;
    std::cout << "batch";
    for (int t : thread_counts)
    {
        std::cout << "\t" << t << "thr(ms)\t" << t << "thr/pair";
    }
    std::cout << std::endl;

    for (int bs : batch_sizes)
    {
        std::cout << bs;
        std::vector<std::pair<n4m::FeatureResult, n4m::FeatureResult>> batch(all_pairs.begin(), all_pairs.begin() + bs);

        for (int threads : thread_counts)
        {
            lg_config.intra_op_threads = threads;
            n4m::LightGlue lg(lg_config);

            // Warm up
            lg.match(batch[0].first, batch[0].second);

            auto t0 = std::chrono::steady_clock::now();
            if (bs == 1)
            {
                lg.match(batch[0].first, batch[0].second);
            }
            else
            {
                lg.match_batch(batch);
            }
            auto t1 = std::chrono::steady_clock::now();
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            double per_pair = ms / bs;
            std::cout << "\t" << std::fixed << std::setprecision(1) << ms << "\t" << per_pair;
        }
        std::cout << std::endl;
    }
}
