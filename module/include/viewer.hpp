#pragma once

#ifndef MYSLAM_VIEWER_HPP
#define MYSLAM_VIEWER_HPP

#include <thread>
#include <unordered_map>

#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Dense"
#include "pangolin/pangolin.h"

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace myslam
{
    class Viewer
    {
    private:
        cv::Mat plotFrameImage();

        Frame::Ptr current_frame_ = nullptr;
        Map::Ptr map_ = nullptr;

        std::thread viewer_thread_;
        bool viewer_running_ = true;

        std::unordered_map<unsigned long, Frame::Ptr> active_keyframes_;
        std::unordered_map<unsigned long, MapPoint::Ptr> active_landmarks_;

        bool map_updated_ = false;

        std::mutex viewer_data_mutex_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Viewer> Ptr;

        Viewer();

        void setMap(Map::Ptr map)
        {
            map_ = map;
        }

        void close();

        void addCurrentFrame(Frame::Ptr current_frame);

        void updateMap();

    private:
        void threadLoop();

        void drawFrame(Frame::Ptr frame, const float *color);

        void drawMapPoints();

        void followCurrentFrame(pangolin::OpenGlRenderState &vis_camera);
    };
}

#endif
