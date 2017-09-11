
#ifndef OPEN_CV_CONNECTOR
#define OPEN_CV_CONNECTOR

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <camera_info_manager/camera_info_manager.h>
#include <image_transport/image_transport.h>



class OpenCVConnector {

public:
   OpenCVConnector(std::string node_name, std::string camera_name, std::string calib_file_path);
   void WriteToOpenCV(const cv::Mat& img, ros::Time stamp);

   ros::NodeHandle nh;
   image_transport::ImageTransport it;
   image_transport::CameraPublisher pub;
   camera_info_manager::CameraInfoManager cinfo;

   unsigned int counter;
};

void GetFilePath(std::string &left_path, std::string &right_path);



#endif

