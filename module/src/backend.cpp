#include <atomic>
#include <thread>
#include <unordered_map>

#include "backend.hpp"
#include "algorithm.hpp"
#include "feature.hpp"
#include "g2o_types.hpp"
#include "map.hpp"
#include "mappoint.hpp"

namespace myslam
{
    Backend::Backend()
    {
        backend_running_.store(true);
        backend_thread_ = std::thread(std::bind(&Backend::backendLoop, this));
    }

    void Backend::updateMap()
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        map_update_.notify_one();
    }

    void Backend::stop()
    {
        backend_running_.store(false);
        map_update_.notify_one();
        backend_thread_.join();
    }

    void Backend::backendLoop()
    {
        while (backend_running_.load())
        {
            std::unique_lock<std::mutex> lck(data_mutex_);
            map_update_.wait(lck);

            Map::KeyFramesType active_kfs = map_->getActiveKeyFrames();
            Map::LandmarksType active_landmarks = map_->getActiveMapPoints();
            optimize(active_kfs, active_landmarks);
        }
    }

    void Backend::optimize(Map::KeyFramesType &keyframes,
                           Map::LandmarksType &landmarks)
    {
        typedef g2o::BlockSolver_6_3 BlockSolverType;
        typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;

        auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                g2o::make_unique<LinearSolverType>()));

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);

        std::map<unsigned long, VertexPose *> vertices;
        unsigned long max_kf_id = 0;
        for (auto &keyframe : keyframes)
        {
            auto kf = keyframe.second;
            VertexPose *vertex_pose = new VertexPose();
            vertex_pose->setId(kf->keyframe_id_);
            vertex_pose->setEstimate(kf->pose());
            optimizer.addVertex(vertex_pose);

            if (kf->keyframe_id_ > max_kf_id)
            {
                max_kf_id = kf->keyframe_id_;
            }

            vertices.insert({kf->keyframe_id_, vertex_pose});
        }

        std::map<unsigned long, VertexXYZ *> vertices_landmarks;

        Eigen::Matrix<double, 3, 3> K = cam_left_->K();
        Sophus::SE3d left_ext = cam_left_->pose();
        Sophus::SE3d right_ext = cam_right_->pose();

        // edges
        int index = 1;
        double chi2_th = 5.991;
        std::map<EdgeProjection *, Feature::Ptr> edges_and_features; // 해당 프레임의 키포인트와 projection값

        for (auto &landmark : landmarks)
        {
            if (landmark.second->is_outlier_)
            {
                continue;
            }

            unsigned long landmark_id = landmark.second->id_;
            auto observations = landmark.second->getObservation();

            for (auto &obs : observations)
            {
                if (obs.lock() == nullptr)
                {
                    continue;
                }

                auto feat = obs.lock(); // weak->shared
                if (feat->is_outlier_ || feat->frame_.lock() == nullptr)
                {
                    continue;
                }

                auto frame = feat->frame_.lock();
                EdgeProjection *edge = nullptr;

                if (feat->is_on_left_image_)
                {
                    edge = new EdgeProjection(K, left_ext);
                }
                else
                {
                    edge = new EdgeProjection(K, right_ext);
                }

                if (!vertices_landmarks.contains(landmark_id))
                {
                    VertexXYZ *v = new VertexXYZ;
                    v->setEstimate(landmark.second->pos());
                    v->setId(landmark_id + max_kf_id + 1);
                    v->setMarginalized(true);
                    vertices_landmarks.insert({landmark_id, v});
                    optimizer.addVertex(v);
                }

                edge->setId(index);
                edge->setVertex(0, vertices.at(frame->keyframe_id_));
                edge->setVertex(1, vertices_landmarks.at(landmark_id));
                edge->setMeasurement(toVec2(feat->position_.pt));
                edge->setInformation(Eigen::Matrix<double, 2, 2>::Identity());
                auto rk = new g2o::RobustKernelHuber();
                rk->setDelta(chi2_th);
                edge->setRobustKernel(rk);
                edges_and_features.insert({edge, feat});

                optimizer.addEdge(edge);

                ++index;
            }
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        int cnt_outlier = 0, cnt_inlier = 0;
        int iteration = 0;
        while (iteration < 5)
        {
            cnt_outlier = 0;
            cnt_inlier = 0;

            for (auto &ef : edges_and_features)
            {
                if (ef.first->chi2() > chi2_th)
                {
                    ++cnt_outlier;
                }
                else
                {
                    ++cnt_inlier;
                }
            }
            double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
            if (inlier_ratio > 0.5)
            {
                break;
            }
            else
            {
                chi2_th *= 2;
                ++iteration;
            }
        }

        for (auto &ef : edges_and_features)
        {
            if (ef.first->chi2() > chi2_th)
            {
                ef.second->is_outlier_ = true;
                ef.second->map_point_.lock()->removeObservation(ef.second);
            }
            else
            {
                ef.second->is_outlier_ = false;
            }
        }
        LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
                  << cnt_inlier;

        for (auto &v : vertices)
        {
            keyframes.at(v.first)->setPose(v.second->estimate());
        }
        for (auto &v : vertices_landmarks)
        {
            landmarks.at(v.first)->setPos(v.second->estimate());
        }
    }
}
