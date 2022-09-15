#include "common_include.hpp"

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "sophus/se3.hpp"

namespace myslam
{
    bool triangulation(const std::vector<Sophus::SE3d> &poses,
                       const std::vector<Eigen::Matrix<double, 3, 1>> points,
                       Eigen::Matrix<double, 3, 1> &pt_world)
    {
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(2 * poses.size(), 4);
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> b(2 * poses.size(), 1);
        Eigen::MatrixXd A(2 * poses.size(), 4);
        Eigen::MatrixXd b(2 * poses.size(), 1);
        b.setZero();

        for (std::size_t i = 0; i < poses.size(); ++i)
        {
            Eigen::Matrix<double, 3, 4> m = poses[i].matrix3x4();
            A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
            A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeFullU | Eigen::ComputeFullV);
        // auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
        pt_world = (svd.matrixV().col(3) / svd.matrixV()(3,3)).head<3>();

        if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2)
        {
            return true;
        }
        return false;
    }

    Eigen::Matrix<double, 2, 1> toVec2(const cv::Point2d p)
    {
        return Eigen::Matrix<double, 2, 1>(p.x, p.y);
    }
}
