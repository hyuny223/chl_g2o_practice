#include <opencv2/opencv.hpp>

#include "algorithm.hpp"
#include "backend.hpp"
#include "config.hpp"
#include "feature.hpp"
#include "frontend.hpp"
#include "g2o_types.hpp"
#include "map.hpp"
#include "viewer.hpp"

namespace myslam
{

    Frontend::Frontend()
    {
        gftt_ = cv::GFTTDetector::create(Config::get<int>("num_features"), 0.01, 20);
        num_features_init_ = Config::get<int>("num_features_init");
        num_features_ = Config::get<int>("num_features");
    }

    bool Frontend::addFrame(myslam::Frame::Ptr frame)
    {
        current_frame_ = frame;

        switch (status_)
        {
        case FrontendStatus::INITIATING:
            stereoInit();
            break;
        case FrontendStatus::TRACKING_GOOD:
        case FrontendStatus::TRACKING_BAD:
            track();
            break;
        case FrontendStatus::LOST:
            reset();
            break;
        }

        last_frame_ = current_frame_;
        return true;
    }

    bool Frontend::track()
    {
        if (last_frame_)
        {
            current_frame_->setPose(relative_motion_ * last_frame_->pose());
        }

        int num_track_last = trackLastFrame();
        tracking_inliers_ = estimateCurrentPose();

        if (tracking_inliers_ > num_features_tracking_)
        {
            // tracking good
            status_ = FrontendStatus::TRACKING_GOOD;
        }
        else if (tracking_inliers_ > num_features_tracking_bad_)
        {
            // tracking bad
            status_ = FrontendStatus::TRACKING_BAD;
        }
        else
        {
            // lost
            status_ = FrontendStatus::LOST;
        }

        insertKeyFrame();
        relative_motion_ = current_frame_->pose() * last_frame_->pose().inverse();

        if (viewer_)
        {
            viewer_->addCurrentFrame(current_frame_);
        }
        return true;
    }

    bool Frontend::insertKeyFrame()
    {
        if (tracking_inliers_ >= num_features_needed_for_keyframe_)
        {
            // still have enough features, don't insert keyframe
            return false;
        }
        // current frame is a new keyframe
        current_frame_->setKeyFrame();
        map_->insertKeyFrame(current_frame_);

        LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
                  << current_frame_->keyframe_id_;

        setObservationsForKeyFrame();
        detectFeatures(); // detect new features

        // track in right image
        findFeaturesInRight();
        // triangulate map points
        triangulateNewPoints();
        // update backend because we have a new keyframe
        backend_->updateMap();

        if (viewer_)
            viewer_->updateMap();

        return true;
    }

    void Frontend::setObservationsForKeyFrame()
    {
        for (auto &feat : current_frame_->features_left_)
        {
            auto mp = feat->map_point_.lock();
            if (mp)
            {
                mp->addObservation(feat);
            }
        }
    }

    int Frontend::triangulateNewPoints()
    {
        std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
        Sophus::SE3d current_pose_Twc = current_frame_->pose().inverse();
        int cnt_triangulated_pts = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            if (current_frame_->features_left_[i]->map_point_.expired() &&
                current_frame_->features_right_[i] != nullptr)
            {
                std::vector<Eigen::Matrix<double, 3, 1>> points{
                    camera_left_->pixel2camera(
                        Vec2(current_frame_->features_left_[i]->position_.pt.x,
                             current_frame_->features_left_[i]->position_.pt.y)),
                    camera_right_->pixel2camera(
                        Vec2(current_frame_->features_right_[i]->position_.pt.x,
                             current_frame_->features_right_[i]->position_.pt.y))};
                Eigen::Matrix<double, 3, 1> pworld = Eigen::Matrix<double, 3, 1>::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0)
                {
                    auto new_map_point = MapPoint::createNewMappoint();
                    pworld = current_pose_Twc * pworld;
                    new_map_point->setPos(pworld);
                    new_map_point->addObservation(
                        current_frame_->features_left_[i]);
                    new_map_point->addObservation(
                        current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    map_->insertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    int Frontend::estimateCurrentPose()
    {
        // setup g2o
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        // vertex
        VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
        vertex_pose->setId(0);
        vertex_pose->setEstimate(current_frame_->pose());
        optimizer.addVertex(vertex_pose);

        // K
        Eigen::Matrix<double, 3, 3> K = camera_left_->K();

        // edges
        int index = 1;
        std::vector<EdgeProjectionPoseOnly *> edges;
        std::vector<Feature::Ptr> features;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            auto mp = current_frame_->features_left_[i]->map_point_.lock();
            if (mp)
            {
                features.push_back(current_frame_->features_left_[i]);
                EdgeProjectionPoseOnly *edge = new EdgeProjectionPoseOnly(mp->pos_, K);
                edge->setId(index);
                edge->setVertex(0, vertex_pose);
                edge->setMeasurement(toVec2(current_frame_->features_left_[i]->position_.pt));
                edge->setInformation(Eigen::Matrix2d::Identity());
                edge->setRobustKernel(new g2o::RobustKernelHuber);
                edges.push_back(edge);
                optimizer.addEdge(edge);
                ++index;
            }
        }

        // estimate the Pose the determine the outliers
        const double chi2_th = 5.991;
        int cnt_outlier = 0;
        for (int iteration = 0; iteration < 4; ++iteration)
        {
            vertex_pose->setEstimate(current_frame_->pose());
            optimizer.initializeOptimization();
            optimizer.optimize(10);
            cnt_outlier = 0;

            // count the outliers
            for (size_t i = 0; i < edges.size(); ++i)
            {
                auto e = edges[i];
                if (features[i]->is_outlier_)
                {
                    e->computeError();
                }
                if (e->chi2() > chi2_th)
                {
                    features[i]->is_outlier_ = true;
                    e->setLevel(1);
                    cnt_outlier++;
                }
                else
                {
                    features[i]->is_outlier_ = false;
                    e->setLevel(0);
                };

                if (iteration == 2)
                {
                    e->setRobustKernel(nullptr);
                }
            }
        }

        LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
                  << features.size() - cnt_outlier;
        // Set pose and outlier
        current_frame_->setPose(vertex_pose->estimate());

        LOG(INFO) << "Current Pose = \n"
                  << current_frame_->pose().matrix();

        for (auto &feat : features)
        {
            if (feat->is_outlier_)
            {
                feat->map_point_.reset();
                feat->is_outlier_ = false; // maybe we can still use it in future
            }
        }
        return features.size() - cnt_outlier;
    }

    int Frontend::trackLastFrame()
    {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_left_)
        {
            if (kp->map_point_.lock())
            {
                // use project point
                auto mp = kp->map_point_.lock();
                auto px = camera_left_->world2pixel(mp->pos_, current_frame_->pose());
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            }
            else
            {
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        cv::Mat error;
        cv::calcOpticalFlowPyrLK(
            last_frame_->left_img_,
            current_frame_->left_img_,
            kps_last,
            kps_current,
            status,
            error,
            cv::Size(11, 11),
            3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                current_frame_->features_left_.push_back(feature);
                num_good_pts++;
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }

    bool Frontend::stereoInit()
    {
        int num_features_left = detectFeatures();
        int num_coor_features = findFeaturesInRight();
        if (num_coor_features < num_features_init_)
        {
            return false;
        }

        bool build_map_success = buildInitMap();
        if (build_map_success)
        {
            status_ = FrontendStatus::TRACKING_GOOD;
            if (viewer_)
            {
                viewer_->addCurrentFrame(current_frame_);
                viewer_->updateMap();
            }
            return true;
        }
        return false;
    }

    int Frontend::detectFeatures()
    {
        cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
        for (auto &feat : current_frame_->features_left_)
        {
            cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                          feat->position_.pt + cv::Point2f(10, 10), 0, cv::FILLED);
        }

        std::vector<cv::KeyPoint> keypoints;
        gftt_->detect(current_frame_->left_img_, keypoints, mask);
        int cnt_detected = 0;
        for (auto &kp : keypoints)
        {
            current_frame_->features_left_.push_back(Feature::Ptr(new Feature(current_frame_, kp)));
            cnt_detected++;
        }

        LOG(INFO) << "Detect " << cnt_detected << " new features";
        return cnt_detected;
    }

    int Frontend::findFeaturesInRight()
    {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_left, kps_right;
        for (auto &kp : current_frame_->features_left_)
        {
            kps_left.push_back(kp->position_.pt);
            auto mp = kp->map_point_.lock();
            if (mp)
            {
                // use projected points as initial guess
                auto px = camera_right_->world2pixel(mp->pos_, current_frame_->pose());
                kps_right.push_back(cv::Point2f(px[0], px[1]));
            }
            else
            {
                // use same pixel in left iamge
                kps_right.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
            current_frame_->left_img_,
            current_frame_->right_img_,
            kps_left,
            kps_right,
            status,
            error,
            cv::Size(11, 11),
            3,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
            cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;
        for (size_t i = 0; i < status.size(); ++i)
        {
            if (status[i])
            {
                cv::KeyPoint kp(kps_right[i], 7);
                Feature::Ptr feat(new Feature(current_frame_, kp));
                feat->is_on_left_image_ = false;
                current_frame_->features_right_.push_back(feat);
                ++num_good_pts;
            }
            else
            {
                current_frame_->features_right_.push_back(nullptr);
            }
        }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
    }

    bool Frontend::buildInitMap()
    {
        std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
        size_t cnt_init_landmarks = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            if (current_frame_->features_right_[i] == nullptr)
            {
                continue;
            }
            // create map point from triangulation
            std::vector<Eigen::Matrix<double, 3, 1>> points{
                camera_left_->pixel2camera(
                    Vec2(current_frame_->features_left_[i]->position_.pt.x,
                         current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                    Vec2(current_frame_->features_right_[i]->position_.pt.x,
                         current_frame_->features_right_[i]->position_.pt.y))};
            Eigen::Matrix<double, 3, 1> pworld = Eigen::Matrix<double, 3, 1>::Zero();

            if (triangulation(poses, points, pworld) && pworld[2] > 0)
            {
                auto new_map_point = MapPoint::createNewMappoint();
                new_map_point->setPos(pworld);
                new_map_point->addObservation(current_frame_->features_left_[i]);
                new_map_point->addObservation(current_frame_->features_right_[i]);
                current_frame_->features_left_[i]->map_point_ = new_map_point;
                current_frame_->features_right_[i]->map_point_ = new_map_point;
                cnt_init_landmarks++;
                map_->insertMapPoint(new_map_point);
            }
        }
        current_frame_->setKeyFrame();
        map_->insertKeyFrame(current_frame_);
        backend_->updateMap();

        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                  << " map points";

        return true;
    }

    bool Frontend::reset()
    {
        LOG(INFO) << "Reset is not implemented. ";
        return true;
    }

} // namespace myslam
