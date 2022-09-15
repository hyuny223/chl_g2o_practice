#pragma once

#ifndef MYSLAM_MAPPOINT_HPP
#define MYSLAM_MAPPOINT_HPP

#include "common_include.hpp"

namespace myslam
{
    struct Frame;
    struct Feature;

    struct MapPoint
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<MapPoint> Ptr;

        unsigned long id_ = 0;
        bool is_outlier_ = false;

        Eigen::Matrix<double, 3, 1> pos_ = Eigen::Matrix<double, 3, 1>::Zero();
        std::mutex data_mutex_;

        int observed_times_ = 0;
        std::list<std::weak_ptr<Feature>> observations_; // 하나의 맵포인트에는 여러 feature가 들어갈 수 있다. 그러나 하나의 feature에는 하나의 맵포인트만 가능

    public:
        MapPoint(){};
        MapPoint(long id, Eigen::Matrix<double, 3, 1> position);

        Eigen::Matrix<double, 3, 1> pos()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return pos_;
        }

        void setPos(const Eigen::Matrix<double, 3, 1> &pos)
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            pos_ = pos;
        }

        void addObservation(std::shared_ptr<Feature> feature) // 하나의 mp에 여러 features가 들어갈 수 있다.
        {
            std::unique_lock<std::mutex> lck(data_mutex_);

            observations_.push_back(feature);
            ++observed_times_;
        }

        void removeObservation(std::shared_ptr<Feature> feature);

        std::list<std::weak_ptr<Feature>> getObservation()
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            return observations_;
        }

        static MapPoint::Ptr createNewMappoint();
    };
}

#endif
