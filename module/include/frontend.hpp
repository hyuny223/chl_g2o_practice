#pragma once

#ifndef MYSLAM_FRONTEND_HPP
#define MYSLAM_FRONTEND_HPP

#include "opencv2/opencv.hpp"
#include "sophus/se3.hpp"

#include "common_include.hpp"
#include "camera.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace myslam
{
    class Backend;
    class Viewer;

    enum class FrontendStatus
    {
        INITIATING,
        TRACKING_GOOD,
        TRACKING_BAD,
        LOST
    };

    class Frontend
    {

    private:
        FrontendStatus status_ = FrontendStatus::INITIATING;

        Frame::Ptr current_frame_ = nullptr;
        Frame::Ptr last_frame_ = nullptr;
        Camera::Ptr camera_left_ = nullptr;
        Camera::Ptr camera_right_ = nullptr;

        Map::Ptr map_ = nullptr;
        std::shared_ptr<Backend> backend_ = nullptr;
        std::shared_ptr<Viewer> viewer_ = nullptr;

        Sophus::SE3d relative_motion_;

        int tracking_inliers_ = 0;

        int num_features_ = 200;
        int num_features_init_ = 100;
        int num_features_tracking_ = 50;
        int num_features_tracking_bad_ = 20;
        int num_features_needed_for_keyframe_ = 80;

        cv::Ptr<cv::GFTTDetector> gftt_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Frontend> Ptr;

        Frontend();

        bool addFrame(Frame::Ptr frame);

        void setMap(Map::Ptr map)
        {
            map_ = map;
        }

        void setBackend(std::shared_ptr<Backend> backend)
        {
            backend_ = backend;
        }

        void setViewer(std::shared_ptr<Viewer> viewer)
        {
            viewer_ = viewer;
        }

        FrontendStatus getStatus() const
        {
            return status_;
        }

        void setCameras(Camera::Ptr left, Camera::Ptr right)
        {
            camera_left_ = left;
            camera_right_ = right;
        }

    private:
        /**
         * Track in normal mode
         * @return true if success
         */
        bool track();

        /**
         * Reset when lost
         * @return true if success
         */
        bool reset();

        /**
         * Track with last frame
         * @return num of tracked points
         */
        int trackLastFrame();
        /**
         * estimate current frame's pose
         * @return num of inliers
         */
        int estimateCurrentPose();
        /**
         * set current frame as a keyframe and insert it into backend
         * @return true if success
         */
        bool insertKeyFrame();
        /**
         * Try init the frontend with stereo images saved in current_frame_
         * @return true if success
         */
        bool stereoInit();
        /**
         * Detect features in left image in current_frame_
         * keypoints will be saved in current_frame_
         * @return
         */
        int detectFeatures();
        /**
         * Find the corresponding features in right image of current_frame_
         * @return num of features found
         */
        int findFeaturesInRight();
        /**
         * Build the initial map with single image
         * @return true if succeed
         */
        bool buildInitMap();
        /**
         * Triangulate the 2D points in current frame
         * @return num of triangulated points
         */
        int triangulateNewPoints();
        /**
         * Set the features in keyframe as new observation of the map points
         */
        void setObservationsForKeyFrame();
    };
}

#endif
