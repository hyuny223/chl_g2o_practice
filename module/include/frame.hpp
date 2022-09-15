#pragma once

#ifndef MYSLAM_FRAME_HPP
#define MYSLAM_FRAME_HPP

#include "eigen3/Eigen/Dense"

#include "camera.hpp"
#include "common_include.hpp"

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace myslam
{
    struct MapPoint;
    struct Feature;

    struct Frame
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frame> Ptr;

        unsigned long id_ = 0;
        unsigned long keyframe_id_ = 0;
        bool is_keyframe_ = false;
        double time_stamp_;
        Sophus::SE3d pose_;
        std::mutex pose_mutex_;
        cv::Mat left_img_, right_img_;

        std::vector<std::shared_ptr<Feature>> features_left_, features_right_;

    public:
        Frame(){};
        Frame(long id,
              double time_stamp,
              const Sophus::SE3d &pose,
              const cv::Mat &left,
              const cv::Mat &right);

        Sophus::SE3d pose()
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            return pose_;
        }

        void setPose(const Sophus::SE3d &pose)
        {
            std::unique_lock<std::mutex> lck(pose_mutex_);
            pose_ = pose;
        }

        void setKeyFrame();

        static std::shared_ptr<Frame> createFrame();
    };
} // namespace myslam

#endif
