#pragma once

#ifndef MYSLAM_DATASET_HPP
#define MYSLAM_DATASET_HPP

#include "common_include.hpp"
#include "camera.hpp"
#include "frame.hpp"

namespace myslam
{
    class Dataset
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Dataset> Ptr;

        Dataset(const std::string& dataset_path);

        bool init();

        std::pair<std::vector<std::string>, std::vector<std::string>> argparse();

        Frame::Ptr nextFrame();

        Camera::Ptr getCamera(int camera_id) const
        {
            return cameras_.at(camera_id);
        }
    private:
        std::string dataset_path_;
        std::pair<std::vector<std::string>, std::vector<std::string>> files_;
        int current_image_index_ = 0;

        std::vector<Camera::Ptr> cameras_;
    };
}

#endif
