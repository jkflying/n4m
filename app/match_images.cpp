#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <nnmatch/lightglue.hpp>
#include <nnmatch/xfeat.hpp>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> [model_dir]\n";
        return 1;
    }

    std::string model_dir = (argc >= 4) ? argv[3] : "models";

    cv::Mat img1 = cv::imread(argv[1]);
    cv::Mat img2 = cv::imread(argv[2]);
    if (img1.empty() || img2.empty())
    {
        std::cerr << "Failed to load images\n";
        return 1;
    }

    nnmatch::XFeatConfig xfeat_cfg;
    xfeat_cfg.model_path = model_dir + "/xfeat.onnx";
    nnmatch::XFeat xfeat(xfeat_cfg);

    nnmatch::LightGlueConfig lg_cfg;
    lg_cfg.model_path = model_dir + "/lightglue.onnx";
    nnmatch::LightGlue lightglue(lg_cfg);

    auto feats1 = xfeat.extract(img1);
    auto feats2 = xfeat.extract(img2);
    std::cout << "Extracted " << feats1.keypoints.size() << " + " << feats2.keypoints.size() << " keypoints\n";

    auto matches = lightglue.match(feats1, feats2);
    std::cout << "Found " << matches.size() << " matches\n";

    // Draw matches side by side
    int h = std::max(img1.rows, img2.rows);
    int w = img1.cols + img2.cols;
    cv::Mat canvas(h, w, CV_8UC3, cv::Scalar(0));
    img1.copyTo(canvas(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(canvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    for (const auto &m : matches)
    {
        const auto &kp1 = feats1.keypoints[m.idx0];
        const auto &kp2 = feats2.keypoints[m.idx1];
        cv::Point2f pt1(kp1.x, kp1.y);
        cv::Point2f pt2(kp2.x + img1.cols, kp2.y);

        int green = static_cast<int>(m.confidence * 255);
        cv::line(canvas, pt1, pt2, cv::Scalar(0, green, 255 - green), 1);
        cv::circle(canvas, pt1, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(canvas, pt2, 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::imshow("Matches", canvas);
    cv::waitKey(0);

    return 0;
}
