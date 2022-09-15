cmake_minimum_required(VERSION 3.14)
project(module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

set(MODULE_SOURCE_FILES
        src/algorithm.cpp
        src/camera.cpp
        src/frame.cpp
        src/feature.cpp
        src/mappoint.cpp
        src/map.cpp
        src/frontend.cpp
        src/config.cpp
        src/dataset.cpp
        src/viewer.cpp
        src/backend.cpp
        src/visual_odometry.cpp
)

# set(G2O_CSPARSE_EXTENSION ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/lib/libg2o_csparse_extension.so)

find_package(OpenCV REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/opencv/install/lib/cmake/opencv4)
find_package(Eigen3 REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/eigen/install/share/eigen3/cmake)
find_package(g2o REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/lib/cmake/g2o)
find_package(Pangolin REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/pangolin/install/lib/cmake/Pangolin)
find_package(Sophus REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/sophus/install/share/sophus/cmake)
find_package(glog REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/glog/install/lib/cmake/glog)
find_package(GTest REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/gtest/install/lib/cmake/GTest)
find_package(gflags REQUIRED HINTS ${CMAKE_SOURCE_DIR}/Thirdparty/gflags/install/lib/cmake/gflags)
find_package(CSparse REQUIRED)

if(g2o_FOUND)
        set(g2o_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/include)
        set(g2o_LIBRARIES ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/lib/libg2o_core.so)
endif()

if(Eigen3_FOUND)
        set(Eigen3_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Thirdparty/eigen/install/include)
endif()

if(Sophus_FOUND)
        set(Sophus_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Thirdparty/sophus/install/include)
endif()

if(glog_FOUND)
        set(GLOG_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Thirdparty/glog/install/include)
        set(GLOG_LIBRARIES ${CMAKE_SOURCE_DIR}/Thirdparty/glog/install/lib/libglog.so)
endif()

if(gflags_FOUND)
        set(GFLAGS_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Thirdparty/gflags/install/include)
        set(GFLAGS_LIBRARIES ${CMAKE_SOURCE_DIR}/Thirdparty/gflags/install/lib/libgflags.so)
endif()

if(GTest_FOUND)
        set(GTEST_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/Thirdparty/gtest/install/include)
        set(GTEST_LIBRARIES ${CMAKE_SOURCE_DIR}/Thirdparty/gtest/install/lib/libgtest.a)
endif()

add_library(${PROJECT_NAME} SHARED
        ${MODULE_SOURCE_FILES}
        )


target_include_directories(${PROJECT_NAME} PUBLIC
        include
        /usr/include/suitesparse
        ${OpenCV_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${g2o_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${GTEST_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        )

target_link_libraries(${PROJECT_NAME} PUBLIC
        ${OpenCV_LIBS}
        Eigen3::Eigen
        ${g2o_LIBRARIES}
        Sophus::Sophus
        ${Pangolin_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${CSPARSE_LIBRARY}
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/lib/libg2o_csparse_extension.so
        ${CMAKE_SOURCE_DIR}/Thirdparty/g2o/install/lib/libg2o_stuff.so
        )

