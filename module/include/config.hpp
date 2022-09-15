#pragma once

#ifndef MYSLAM_CONFIG_HPP
#define MYSLAM_CONFIG_HPP

#include "opencv2/opencv.hpp"
#include "common_include.hpp"

namespace myslam
{
    class Config
    {
    private:
        static std::shared_ptr<Config> config_;
        cv::FileStorage file_;

        Config(){};
    public:
        ~Config();

        static bool setParameterFile(const std::string &filename);

        template <typename T>
        static T get(const std::string &key)
        {
            return T(Config::config_->file_[key]);
        }
    };
}

#endif
