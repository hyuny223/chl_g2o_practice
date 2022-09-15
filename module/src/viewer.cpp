
#include <thread>
#include <cassert>

#include "opencv2/opencv.hpp"
#include "pangolin/pangolin.h"
#include "sophus/se3.hpp"

#include "viewer.hpp"
#include "feature.hpp"
#include "frame.hpp"

namespace myslam
{
    Viewer::Viewer()
    {
        viewer_thread_ = std::thread(std::bind(&Viewer::threadLoop, this));
    }

    void Viewer::close()
    {
        viewer_running_ = false;
        viewer_thread_.join();
    }

    void Viewer::addCurrentFrame(Frame::Ptr current_frame)
    {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        current_frame_ = current_frame;
    }

    void Viewer::updateMap()
    {
        std::unique_lock<std::mutex> lck(viewer_data_mutex_);
        assert(map_ != nullptr);

        active_keyframes_ = map_->getActiveKeyFrames();
        active_landmarks_ = map_->getActiveMapPoints();
        map_updated_ = true;
    }

    void Viewer::threadLoop()
    {
        pangolin::CreateWindowAndBind("MySLAM", 1024, 786);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 들어오는 정보와 있는 정보간의 blending 비율

        // 원하는 카메라 정보(우리가 어떻게 이미지를 보고 싶은지)
        pangolin::OpenGlRenderState vis_camera(
            pangolin::ProjectionMatrix(1024, 768, 400, 400, 512, 384, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -5, -10, 0, 0, 0, 0.0, -1.0, 0.0));

        // vis_camera를 실제로 보여준다.
        pangolin::View &vis_display =
            pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                .SetHandler(new pangolin::Handler3D(vis_camera));

        const float blue[3] = {0, 0, 1};
        const float green[3] = {0, 1, 0};

        while (!pangolin::ShouldQuit() && viewer_running_)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 버퍼를 미리 설정된 값으로 지운다. 컬러와 깊이 정보가 있다.
                glClearColor(1.0f, 1.0f, 1.0f, 1.0f);          // 0 ~ 255 -> 0(b) ~ 1(w)
            vis_display.Activate(vis_camera);

            std::unique_lock<std::mutex> lock(viewer_data_mutex_);
            if (current_frame_)
            {
                drawFrame(current_frame_, green); // 현재 프레임을 초록색으로 그려라
                followCurrentFrame(vis_camera);   // 현재 프레임을 따라가라

                cv::Mat img = plotFrameImage();
                cv::imshow("image", img);
                cv::waitKey(1);
            }
            if (map_)
            {
                drawMapPoints();
            }
            pangolin::FinishFrame();
            usleep(5000);
        }
        LOG(INFO) << "Stop viewer";
    }

    cv::Mat Viewer::plotFrameImage()
    {
        cv::Mat img_out;
        cv::cvtColor(current_frame_->left_img_, img_out, cv::COLOR_GRAY2BGR);

        for (std::size_t i = 0; i < current_frame_->features_left_.size(); ++i)
        {
            if (current_frame_->features_left_[i]->map_point_.lock()) // nullptr이 아니라면,
            {
                auto feat = current_frame_->features_left_[i];
                cv::circle(img_out, feat->position_.pt, 2, cv::Scalar(0, 250, 0), 2);
            }
        }
        return img_out;
    }

    void Viewer::followCurrentFrame(pangolin::OpenGlRenderState &vis_camera)
    {
        Sophus::SE3d Twc = current_frame_->pose().inverse();
        pangolin::OpenGlMatrix m(Twc.matrix());
        vis_camera.Follow(m, true);
    }

    void Viewer::drawFrame(Frame::Ptr frame, const float *color)
    {
        Sophus::SE3d Twc = frame->pose().inverse();
        const float sz = 1.0;
        const int line_width = 2.0;
        const float fx = 400;
        const float fy = 400;
        const float cx = 512;
        const float cy = 384;
        const float width = 1080;
        const float height = 768;

        glPushMatrix();

        Sophus::Matrix4f m = Twc.matrix().template cast<float>(); // 무슨 식이지..?
        glMultMatrixf((GLfloat *)m.data());

        if (color == nullptr)
        {
            glColor3f(1, 0, 0); // 색 지정
        }
        else
        {
            glColor3f(color[0], color[1], color[2]);
        }

        glLineWidth(line_width);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(0, 0, 0);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

        glEnd();
        glPopMatrix();
    }

    void Viewer::drawMapPoints()
    {
        const float red[3] = {1.0, 0, 0};
        for (auto &kf : active_keyframes_)
        {
            drawFrame(kf.second, red);
        }
        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &landmark : active_landmarks_)
        {
            auto pos = landmark.second->pos();
            glColor3f(red[0], red[1], red[2]);
            glVertex3d(pos[0], pos[1], pos[2]);
        }
        glEnd();
    }
}
