#pragma once

#ifndef MYSLAM_MAP_HPP
#define MYSLAM_MAP_HPP

#include <iostream>
#include <unordered_map>

#include "common_include.hpp"
#include "frame.hpp"
#include "mappoint.hpp"

namespace myslam
{
    class Map
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Map> Ptr;
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
        typedef std::unordered_map<unsigned long, Frame::Ptr> KeyFramesType;

    public:
        Map(){};

        void insertKeyFrame(Frame::Ptr frame);
        void insertMapPoint(MapPoint::Ptr map_point);

        KeyFramesType getAllKeyFrames()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return keyframes_;
        }

        LandmarksType getAllMaPoints()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return landmarks_;
        }

        KeyFramesType getActiveKeyFrames()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_keyframes_;
        }

        LandmarksType getActiveMapPoints()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return active_landmarks_;
        }

        void cleanMap();

    private:
        void removeOldKeyFrame();
        std::mutex data_mutex_;
        LandmarksType landmarks_, active_landmarks_;
        KeyFramesType keyframes_, active_keyframes_;

        Frame::Ptr current_frame_ = nullptr;

        int num_active_keyframes_ = 7;
    };
}

#endif
