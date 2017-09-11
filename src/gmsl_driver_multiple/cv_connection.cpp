#include "cv_connection.hpp"

#include <opencv2/opencv.hpp>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
	
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <camera_info_manager/camera_info_manager.h>

#include <iostream>

OpenCVConnector::OpenCVConnector(std::string node_name, std::string camera_name, std::string calib_file_path) : 
    nh(node_name) ,
    it(nh), cinfo(nh, camera_name, calib_file_path), counter(0)	{
   pub = it.advertiseCamera("image_raw", 1);
}

void GetFilePath(std::string &left_path, std::string &right_path){ 
    ros::param::get("~left_camera_info_path", left_path);
    ros::param::get("~right_camera_info_path", right_path);
}

void OpenCVConnector::WriteToOpenCV(const cv::Mat& img, ros::Time stamp) {
    cv_bridge::CvImage img_bridge;
    sensor_msgs::Image img_msg; // >> message to be sent
    
    sensor_msgs::CameraInfo ci(cinfo.getCameraInfo());

    std_msgs::Header header; // empty header
    header.seq = counter; // user defined counter
    header.stamp = stamp; // time
    ci.header.frame_id = counter;
    ci.header.stamp = stamp;
    
    img_bridge = cv_bridge::CvImage(header, 
        img.type() == CV_8UC1 ? sensor_msgs::image_encodings::MONO8 : sensor_msgs::image_encodings::BGR8, 
        img);
    img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image
    pub.publish(img_msg, ci); // ros::Publisher pub_img = node.advertise<sensor_msgs::Image>("topic", queuesize);
    counter++;
}


