#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "cv_connection/cv_connection_nodelet.h"

namespace cv_connection{
    cv_connection_nodelet::cv_connection_nodelet(){
       
    }
    cv_connection_nodelet::~cv_connection_nodelet(){
        //stop pubThread_
        //release resources here
    }
    void cv_connection_nodelet::onInit()
    {
        NODELET_DEBUG("Init nodelet");
        nh = getNodeHandle();
        private_nh = getPrivateNodeHandle();
        
        //init 
        boost::lock_guard<boost::mutex> lock(connect_mutex_);
        cinfo_.reset(new camera_info_manager::CameraInfoManager(blablabla))
        it_.reset(new image_transport::ImageTransport(nh));
        image_transport::SubscriberStatusCallback cb = boost::bind(&cv_connection_nodelet::connectCb, this);
        ros::SubscriberStatusCallback info_cb = boost::bind(&v_connection_nodelet::connectCb, this);
        it_pub_ = it_->advertiseCamera("image_raw", 3, cb, cb, info_cb, info_cb);
        


    }
    void cv_connection_nodelet::connectCb()
    {
        boost::lock_guard<boost::mutex> lock(connect_mutex_);
        //use these kind of things to stop reading image when no one is subscribing
        if(it_pub_.getNumSubscribers() == 0 && pub_->getPublisher().getNumSubscribers() == 0)
        {
            //stop pubThread
        }
        else if(!pubThread){
            //start pubthread
            pubThread_.reset(new boost::thread(boost::bind(&cv_connection::cv_connection_nodelet::devicePoll, this)));
        }

    }

    void cv_connection_nodelet::devicePoll()
    {
        enum State
        {
            NONE
          , ERROR
          , STOPPED
          , DISCONNECTED
          , CONNECTED
          , STARTED
        };

        //...
        //...
        //I think we could change and use the devicePoll func from: https://github.com/ros-drivers/pointgrey_camera_driver/blob/master/pointgrey_camera_driver/src/nodelet.cpp
        if(it_pub_.getNumSubscribers() > 0)
        {
            //it_pub_ image
        }
    }
}


PLUGINLIB_DECLARE_CLASS(gmsl_driver_multiple, cv_connection_nodelet,
    cv_connection::cv_connection_nodelet, nodelet::Nodelet);