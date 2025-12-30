#include <chrono>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nnmatch/lightglue.hpp>
#include <nnmatch/xfeat.hpp>

static double ms_since(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/// BFMatcher with Lowe's ratio test on XFeat descriptors.
static std::vector<nnmatch::Match> bf_match(const nnmatch::FeatureResult &feats0, const nnmatch::FeatureResult &feats1,
                                            float ratio_thresh = 0.8f)
{
    const int n0 = static_cast<int>(feats0.keypoints.size());
    const int n1 = static_cast<int>(feats1.keypoints.size());

    cv::Mat desc0(n0, nnmatch::XFEAT_DESCRIPTOR_DIM, CV_32F);
    cv::Mat desc1(n1, nnmatch::XFEAT_DESCRIPTOR_DIM, CV_32F);
    for (int i = 0; i < n0; ++i)
        std::copy(feats0.keypoints[i].descriptor.begin(), feats0.keypoints[i].descriptor.end(), desc0.ptr<float>(i));
    for (int i = 0; i < n1; ++i)
        std::copy(feats1.keypoints[i].descriptor.begin(), feats1.keypoints[i].descriptor.end(), desc1.ptr<float>(i));

    auto bf = cv::BFMatcher::create(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf->knnMatch(desc0, desc1, knn_matches, 2);

    std::vector<nnmatch::Match> matches;
    for (const auto &knn : knn_matches)
    {
        if (knn.size() >= 2 && knn[0].distance < ratio_thresh * knn[1].distance)
        {
            matches.push_back({knn[0].queryIdx, knn[0].trainIdx, 1.0f - knn[0].distance});
        }
    }
    return matches;
}

static void draw_matches(cv::Mat &canvas, const cv::Mat &img1, const nnmatch::FeatureResult &feats1,
                         const nnmatch::FeatureResult &feats2, const std::vector<nnmatch::Match> &matches, int x_offset)
{
    // All keypoints as small gray dots
    for (const auto &kp : feats1.keypoints)
        cv::circle(canvas, cv::Point2f(kp.x + x_offset, kp.y), 2, cv::Scalar(128, 128, 128), -1);
    for (const auto &kp : feats2.keypoints)
        cv::circle(canvas, cv::Point2f(kp.x + x_offset + img1.cols, kp.y), 2, cv::Scalar(128, 128, 128), -1);

    // Match lines
    for (const auto &m : matches)
    {
        const auto &kp1 = feats1.keypoints[m.idx0];
        const auto &kp2 = feats2.keypoints[m.idx1];
        cv::Point2f pt1(kp1.x + x_offset, kp1.y);
        cv::Point2f pt2(kp2.x + x_offset + img1.cols, kp2.y);

        int green = std::clamp(static_cast<int>(m.confidence * 255), 0, 255);
        cv::line(canvas, pt1, pt2, cv::Scalar(0, green, 255 - green), 1);
        cv::circle(canvas, pt1, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(canvas, pt2, 3, cv::Scalar(0, 255, 0), -1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> <output> [model_dir]\n";
        return 1;
    }

    std::string output_path = argv[3];
    std::string model_dir = (argc >= 5) ? argv[4] : "models";

    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);
    if (img1.empty() || img2.empty())
    {
        std::cerr << "Failed to load images\n";
        return 1;
    }

    // Resize to 1600px on the longest side
    const int max_dim = 800;
    auto resize_to_max = [&](cv::Mat &img) {
        double scale = static_cast<double>(max_dim) / std::max(img.rows, img.cols);
        if (scale < 1.0)
            cv::resize(img, img, cv::Size(), scale, scale);
    };
    resize_to_max(img1);
    resize_to_max(img2);

    // --- Feature extraction ---
    nnmatch::XFeatConfig xfeat_cfg;
    xfeat_cfg.model_path = model_dir + "/xfeat.onnx";
    xfeat_cfg.cell_size = 16;
    nnmatch::XFeat xfeat(xfeat_cfg);

    auto t0 = std::chrono::steady_clock::now();
    auto feats1 = xfeat.extract(img1);
    double ms_extract1 = ms_since(t0);

    t0 = std::chrono::steady_clock::now();
    auto feats2 = xfeat.extract(img2);
    double ms_extract2 = ms_since(t0);

    std::cout << "XFeat extract img1 (" << img1.cols << "x" << img1.rows << "): " << ms_extract1 << " ms, "
              << feats1.keypoints.size() << " keypoints\n";
    std::cout << "XFeat extract img2 (" << img2.cols << "x" << img2.rows << "): " << ms_extract2 << " ms, "
              << feats2.keypoints.size() << " keypoints\n";

    // --- LightGlue matching ---
    nnmatch::LightGlueConfig lg_cfg;
    lg_cfg.model_path = model_dir + "/lightglue.onnx";
    nnmatch::LightGlue lightglue(lg_cfg);

    t0 = std::chrono::steady_clock::now();
    auto lg_matches = lightglue.match(feats1, feats2);
    double ms_lightglue = ms_since(t0);

    std::cout << "LightGlue match: " << ms_lightglue << " ms, " << lg_matches.size() << " matches\n";

    // --- BFMatcher matching ---
    t0 = std::chrono::steady_clock::now();
    auto bf_matches = bf_match(feats1, feats2);
    double ms_bf = ms_since(t0);

    std::cout << "BFMatcher match: " << ms_bf << " ms, " << bf_matches.size() << " matches\n";

    // --- Draw: LightGlue on top, BFMatcher on bottom ---
    int pair_w = img1.cols + img2.cols;
    int pair_h = std::max(img1.rows, img2.rows);
    int label_h = 30;
    cv::Mat canvas(pair_h * 2 + label_h * 2, pair_w, CV_8UC3, cv::Scalar(0));

    // Top row: LightGlue
    img1.copyTo(canvas(cv::Rect(0, label_h, img1.cols, img1.rows)));
    img2.copyTo(canvas(cv::Rect(img1.cols, label_h, img2.cols, img2.rows)));
    cv::Mat top_row = canvas(cv::Rect(0, label_h, pair_w, pair_h));
    draw_matches(top_row, img1, feats1, feats2, lg_matches, 0);

    std::string lg_label = "LightGlue: " + std::to_string(lg_matches.size()) + " matches, " +
                           std::to_string(static_cast<int>(ms_lightglue)) + " ms";
    cv::putText(canvas, lg_label, cv::Point(10, 22), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

    // Bottom row: BFMatcher
    int y_off = label_h + pair_h + label_h;
    img1.copyTo(canvas(cv::Rect(0, y_off, img1.cols, img1.rows)));
    img2.copyTo(canvas(cv::Rect(img1.cols, y_off, img2.cols, img2.rows)));
    cv::Mat bot_row = canvas(cv::Rect(0, y_off, pair_w, pair_h));
    draw_matches(bot_row, img1, feats1, feats2, bf_matches, 0);

    std::string bf_label = "BFMatcher: " + std::to_string(bf_matches.size()) + " matches, " +
                           std::to_string(static_cast<int>(ms_bf)) + " ms";
    cv::putText(canvas, bf_label, cv::Point(10, label_h + pair_h + 22), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2);

    cv::imwrite(output_path, canvas);
    std::cout << "Saved to " << output_path << "\n";

    return 0;
}
