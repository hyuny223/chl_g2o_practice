#pragma once

#ifndef MYSLAM_CAMERA_HPP
#define MYSLAM_CAMERA_HPP

#include "common_include.hpp"

namespace myslam
{

    // Pinhole stereo camera model
    class Camera
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        typedef std::shared_ptr<Camera> Ptr;

        double fx_ = 0, fy_ = 0, cx_ = 0, cy_ = 0, baseline_ = 0; // Camera intrinsics
        Sophus::SE3d pose_;                                                // extrinsic, from stereo camera to single camera
        Sophus::SE3d pose_inv_;                                            // inverse of extrinsics

    public:
        Camera();

        Camera(double fx,
               double fy,
               double cx,
               double cy,
               double baseline,
               const Sophus::SE3d &pose)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose)
        {
            pose_inv_ = pose_.inverse();
        }

        Sophus::SE3d pose() const
        {
            return pose_;
        }

        // return intrinsic matrix
        Mat33 K() const
        {
            Mat33 k;
            k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
            return k;
        }

        // coordinate transform: world, camera, pixel
        Eigen::Matrix<double, 3, 1> world2camera(const Eigen::Matrix<double, 3, 1> &p_w, 
                                                 const Sophus::SE3d &T_c_w);

        Eigen::Matrix<double, 3, 1> camera2world(const Eigen::Matrix<double, 3, 1> &p_c, 
                                                 const Sophus::SE3d &T_c_w);

        Eigen::Matrix<double, 2, 1> camera2pixel(const Eigen::Matrix<double, 3, 1> &p_c);

        Eigen::Matrix<double, 3, 1> pixel2camera(const Eigen::Matrix<double, 2, 1> &p_p, 
                                                 double depth = 1);

        Eigen::Matrix<double, 3, 1> pixel2world(const Eigen::Matrix<double, 2, 1> &p_p, 
                                                const Sophus::SE3d &T_c_w, 
                                                double depth = 1);

        Eigen::Matrix<double, 2, 1> world2pixel(const Eigen::Matrix<double, 3, 1> &p_w, 
                                                const Sophus::SE3d &T_c_w);
    };

} // namespace myslam
#endif // MYSLAM_CAMERA_H
