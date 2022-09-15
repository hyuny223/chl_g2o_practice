#include "mappoint.hpp"
#include "feature.hpp"

namespace myslam
{
    MapPoint::Ptr MapPoint::createNewMappoint()
    {
        static long factory_id = 0;
        MapPoint::Ptr new_mappoint(new MapPoint);
        new_mappoint->id_ = ++factory_id;
        return new_mappoint;
    }

    void MapPoint::removeObservation(std::shared_ptr<Feature> feature)
    {
        std::unique_lock<std::mutex> lck(data_mutex_);
        for(auto iter = observations_.begin(); iter != observations_.end(); ++iter)
        {
            if (iter->lock() == feature)
            {
                observations_.erase(iter);
                feature->map_point_.reset();
                --observed_times_;
                break;
            }
        }
    }
}
