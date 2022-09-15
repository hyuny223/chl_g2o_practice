#pragma once

#ifndef MYSLAM_ALGORITHM_HPP
#define MYSLAM_ALGORITHM_HPP

#include "common_include.hpp"

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"


#include "sophus/se3.hpp"

namespace myslam
{
    bool triangulation(const std::vector<Sophus::SE3d> &poses,
                       const std::vector<Eigen::Matrix<double, 3, 1>> points,
                       Eigen::Matrix<double, 3, 1> &pt_world);

    Eigen::Matrix<double, 2, 1> toVec2(const cv::Point2d p);
}

#endif
