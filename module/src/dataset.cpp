
#include <iostream>
#include <string>
#include <istream>
#include <fstream>
#include <filesystem>
#include <algorithm>

#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Dense"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

#include "dataset.hpp"
#include "frame.hpp"

namespace myslam // filesystem으로 바꿔야겠다. 
{
    Dataset::Dataset(const std::string &dataset_path)
        : dataset_path_(dataset_path) {}

    bool Dataset::init()
    {
        // read camera intrinsics and extrinsics
        std::ifstream fin(dataset_path_ + "/calib.txt");
        if (!fin)
        {
            LOG(ERROR) << "cannot find " << dataset_path_ << "/calib.txt!";
            return false;
        }

        for (int i = 0; i < 4; ++i)
        {
            char camera_name[3];
            for (int k = 0; k < 3; ++k)
            {
                fin >> camera_name[k];
            }
            double projection_data[12];
            for (int k = 0; k < 12; ++k)
            {
                fin >> projection_data[k];
            }
            Eigen::Matrix<double, 3, 3> K;
            K << projection_data[0], projection_data[1], projection_data[2],
                projection_data[4], projection_data[5], projection_data[6],
                projection_data[8], projection_data[9], projection_data[10];
            Eigen::Matrix<double, 3, 1> t;
            t << projection_data[3], projection_data[7], projection_data[11];
            t = K.inverse() * t;
            K = K * 0.5;
            Camera::Ptr new_camera(new Camera(K(0, 0), K(1, 1), K(0, 2), K(1, 2),
                                              t.norm(), Sophus::SE3d(Sophus::SO3d(), t)));
            cameras_.push_back(new_camera);
            LOG(INFO) << "Camera " << i << " extrinsics: " << t.transpose();
        }
        fin.close();

        files_ = argparse();
        current_image_index_ = 0;
        return true;
    }

    std::pair<std::vector<std::string>, std::vector<std::string>> Dataset::argparse()
    {
        std::string p0 = dataset_path_ + "/image_0";
        std::string p1 = dataset_path_ + "/image_1";

        std::filesystem::path path0(p0);
        std::filesystem::path path1(p1);

        std::filesystem::directory_iterator itr0(std::filesystem::absolute(path0));
        std::filesystem::directory_iterator itr1(std::filesystem::absolute(path1));

        std::vector<std::string> leftFiles;
        std::vector<std::string> rightFiles;

        while(itr0 != std::filesystem::end(itr0))
        {
            const std::filesystem::directory_entry& entry0 = *itr0;
            const std::filesystem::directory_entry& entry1 = *itr1;

            leftFiles.emplace_back(entry0.path());
            rightFiles.emplace_back(entry1.path());

            ++itr0; ++itr1;
        }

        std::sort(leftFiles.begin(), leftFiles.end());
        std::sort(rightFiles.begin(), rightFiles.end());

        return std::make_pair(leftFiles, rightFiles);
    }

    Frame::Ptr Dataset::nextFrame()
    {
        cv::Mat image_left, image_right;
        // read images
        image_left =
            cv::imread(files_.first[current_image_index_],
                       cv::IMREAD_GRAYSCALE);
        image_right =
            cv::imread(files_.second[current_image_index_],
                       cv::IMREAD_GRAYSCALE);

        if (image_left.data == nullptr || image_right.data == nullptr)
        {
            LOG(WARNING) << "cannot find images at index " << current_image_index_;
            return nullptr;
        }

        cv::Mat image_left_resized, image_right_resized;
        cv::resize(image_left, image_left_resized, cv::Size(), 0.5, 0.5,
                   cv::INTER_NEAREST);
        cv::resize(image_right, image_right_resized, cv::Size(), 0.5, 0.5,
                   cv::INTER_NEAREST);

        auto new_frame = Frame::createFrame();
        new_frame->left_img_ = image_left_resized;
        new_frame->right_img_ = image_right_resized;
        ++current_image_index_;
        return new_frame;
    }
}
