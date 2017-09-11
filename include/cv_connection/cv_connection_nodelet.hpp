#include <nodelet/nodelet.h>
#include <boost/shared_ptr.hpp>
#include "cv_connection.h"

namespace cv_connection{
    class cv_connection_nodelet: public nodelet::Nodelet
    {
        public:
            cv_connection_nodelet(){}
            ~cv_connection_nodelet(){}
        private:
            virtual void onInit();
            void connectCB();
            boost::shared_ptr<image_transport::ImageTransport> it_;
            boost::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_;
            image_transport::CameraPublisher it_pub_;
            boost::shared_ptr<boost::thread> pubThread_;
    };
}
