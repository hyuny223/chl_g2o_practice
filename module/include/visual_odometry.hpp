#pragma once

#ifndef MYSLAM_VISUAL_ODOMETRY_HPP
#define MYSLAM_VISUAL_ODOMETRY_HPP

#include <iostream>
#include "eigen3/Eigen/Dense"

#include "backend.hpp"
#include "common_include.hpp"
#include "dataset.hpp"
#include "frontend.hpp"
#include "viewer.hpp"

namespace myslam
{
    class VisualOdometry
    {
    private:
        bool inited_ = false;
        std::string config_file_path_;

        Frontend::Ptr frontend_ = nullptr;
        Backend::Ptr backend_ = nullptr;
        Map::Ptr map_ = nullptr;
        Viewer::Ptr viewer_ = nullptr;

        Dataset::Ptr dataset_ = nullptr;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<VisualOdometry> Ptr;

        VisualOdometry(std::string &config_path);

        bool init();
        void run();
        bool step();

        FrontendStatus getFrontendStatus() const
        {
            return frontend_->getStatus();
        }
    };
}

#endif
