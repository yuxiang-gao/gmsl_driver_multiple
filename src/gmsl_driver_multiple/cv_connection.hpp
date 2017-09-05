
#ifndef OPEN_CV_CONNECTOR
#define OPEN_CV_CONNECTOR

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>



class OpenCVConnector {

public:
   OpenCVConnector(std::string topic);

   void WriteToOpenCV(const cv::Mat& img, ros::Time stamp);

   ros::NodeHandle nh;
   image_transport::ImageTransport it;
   image_transport::Publisher pub;

   unsigned int counter;
};



#endif

