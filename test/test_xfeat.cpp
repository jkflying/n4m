#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include <nnmatch/xfeat.hpp>

#include <cstdio>
#include <fstream>
#include <vector>

static std::string test_data_dir()
{
    return TEST_DATA_DIR;
}

static std::vector<float> load_npy_floats(const std::string &path)
{
    // Minimal .npy loader for 1D/2D float32 arrays
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        return {};

    // Skip header: magic (6) + 2 bytes version + 2 bytes header_len
    char magic[6];
    f.read(magic, 6);
    uint8_t major, minor;
    f.read(reinterpret_cast<char *>(&major), 1);
    f.read(reinterpret_cast<char *>(&minor), 1);

    uint16_t header_len;
    if (major == 1)
    {
        f.read(reinterpret_cast<char *>(&header_len), 2);
    }
    else
    {
        uint32_t hl;
        f.read(reinterpret_cast<char *>(&hl), 4);
        header_len = static_cast<uint16_t>(hl);
    }
    std::string header(header_len, '\0');
    f.read(header.data(), header_len);

    // Read remaining data as float32
    std::vector<float> data;
    float val;
    while (f.read(reinterpret_cast<char *>(&val), sizeof(float)))
    {
        data.push_back(val);
    }
    return data;
}

TEST(XFeat, DISABLED_ExtractMatchesReference)
{
    // This test requires model files and reference data
    auto dir = test_data_dir() + "/reference";

    nnmatch::XFeatConfig config;
    config.param_path = dir + "/../../models/xfeat.param";
    config.bin_path = dir + "/../../models/xfeat.bin";
    config.max_keypoints = 4096;

    nnmatch::XFeat xfeat(config);

    cv::Mat image = cv::imread(dir + "/test_image.jpg");
    ASSERT_FALSE(image.empty());

    auto result = xfeat.extract(image);
    EXPECT_GT(result.keypoints.size(), 0u);
    EXPECT_LE(result.keypoints.size(), 4096u);

    // Compare against reference keypoints
    auto ref_kpts = load_npy_floats(dir + "/xfeat_keypoints.npy");
    if (!ref_kpts.empty())
    {
        // ref_kpts layout: Nx3 (x, y, score)
        int ref_n = static_cast<int>(ref_kpts.size()) / 3;
        EXPECT_EQ(result.keypoints.size(), static_cast<size_t>(ref_n));

        for (int i = 0; i < std::min(ref_n, static_cast<int>(result.keypoints.size())); ++i)
        {
            EXPECT_NEAR(result.keypoints[i].x, ref_kpts[i * 3 + 0], 2.0f) << "keypoint " << i << " x mismatch";
            EXPECT_NEAR(result.keypoints[i].y, ref_kpts[i * 3 + 1], 2.0f) << "keypoint " << i << " y mismatch";
        }
    }

    // Compare descriptors
    auto ref_desc = load_npy_floats(dir + "/xfeat_descriptors.npy");
    if (!ref_desc.empty())
    {
        int ref_n = static_cast<int>(ref_desc.size()) / nnmatch::XFEAT_DESCRIPTOR_DIM;
        for (int i = 0; i < std::min(ref_n, static_cast<int>(result.keypoints.size())); ++i)
        {
            for (int d = 0; d < nnmatch::XFEAT_DESCRIPTOR_DIM; ++d)
            {
                EXPECT_NEAR(result.keypoints[i].descriptor[d], ref_desc[i * nnmatch::XFEAT_DESCRIPTOR_DIM + d], 0.01f)
                    << "keypoint " << i << " descriptor[" << d << "] mismatch";
            }
        }
    }
}
