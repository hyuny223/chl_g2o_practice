#pragma once

#ifndef MYSLAM_BACKEND_HPP
#define MYSLAM_BACKEND_HPP

#include <thread>

#include "eigen3/Eigen/Dense"

#include "common_include.hpp"
#include "frame.hpp"
#include "map.hpp"

namespace myslam
{
    class Map;

    class Backend
    {
    private:
        std::shared_ptr<Map> map_;
        std::thread backend_thread_;
        std::mutex data_mutex_;

        std::condition_variable map_update_;
        std::atomic<bool> backend_running_;

        Camera::Ptr cam_left_ = nullptr;
        Camera::Ptr cam_right_ = nullptr;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Backend> Ptr;

        Backend();

        void setCameras(Camera::Ptr left, Camera::Ptr right)
        {
            cam_left_ = left;
            cam_right_ = right;
        }

        void setMap(std::shared_ptr<Map> map)
        {
            map_ = map;
        }

        void updateMap();
        void stop();

    private:
        void backendLoop();

        void optimize(Map::KeyFramesType& keyframes, Map::LandmarksType& landmarks);
    };
}
#endif
