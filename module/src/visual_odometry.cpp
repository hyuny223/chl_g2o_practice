#include <chrono>

#include "visual_odometry.hpp"
#include "config.hpp"

namespace myslam
{
    VisualOdometry::VisualOdometry(std::string &config_path)
        : config_file_path_(config_path) {}
    
    bool VisualOdometry::init()
    {
        if(Config::setParameterFile(config_file_path_) == false)
        {
            return false;
        }

        dataset_ = Dataset::Ptr(new Dataset(Config::get<std::string>("dataset_dir")));
        CHECK_EQ(dataset_->init(), true);

        frontend_ = Frontend::Ptr(new Frontend);
        backend_ = Backend::Ptr(new Backend);
        map_ = Map::Ptr(new Map);
        viewer_ = Viewer::Ptr(new Viewer);

        frontend_->setBackend(backend_);
        frontend_->setMap(map_);
        frontend_->setViewer(viewer_);
        frontend_->setCameras(dataset_->getCamera(0), dataset_->getCamera(1));

        backend_->setMap(map_);
        backend_->setCameras(dataset_->getCamera(0), dataset_->getCamera(1));

        viewer_->setMap(map_);

        return true;
    }

    void VisualOdometry::run()
    {
        while(1)
        {
            LOG(INFO) << "VO is running";
            if(step() == false)
            {
                break;
            }
        }

        backend_->stop();
        viewer_->close();

        LOG(INFO) << "VO exit";
    }

    bool VisualOdometry::step()
    {
        Frame::Ptr new_frame = dataset_->nextFrame();
        if(new_frame == nullptr)
        {
            return false;
        }

        auto t1 = std::chrono::steady_clock::now();
        bool success = frontend_->addFrame(new_frame);
        auto t2 = std::chrono::steady_clock::now();
        auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
        LOG(INFO) << "VO cost time : " << time_used.count() << "s.";
        return success;
    }
}
