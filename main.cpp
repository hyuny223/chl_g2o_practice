#include <iostream>

#include "visual_odometry.hpp"
#include "gflags/gflags.h"

DEFINE_string(config_file, "/root/dataset/00/default.yaml", "config file path");


int main(int argc, char **argv)
{
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->init() == true);
    vo->run();

    return 0;
}
