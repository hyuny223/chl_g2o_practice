#include "frame.hpp"
#include "common_include.hpp"

#include <iostream>

#include "opencv2/opencv.hpp"

#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

namespace myslam
{
    Frame::Frame(long id, 
                 double time_stamp, 
                 const Sophus::SE3d &pose, 
                 const cv::Mat &left, 
                 const cv::Mat &right)
        : id_(id), time_stamp_(time_stamp), pose_(pose), left_img_(left), right_img_(right) {}

    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0;
        Frame::Ptr new_frame(new Frame);
        new_frame->id_ = ++factory_id;
        return new_frame;
    }

    void Frame::setKeyFrame()
    {
        static long keyframe_factory_id = 0;
        is_keyframe_ = true;
        keyframe_id_ = ++keyframe_factory_id;
    }
}
