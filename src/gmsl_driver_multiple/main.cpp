#include <iostream>

#define float16_t float16_cv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#undef float16_t

#include <common/ProgramArguments.hpp>
#include <common/SampleFramework.hpp>

// SDK
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/image/ImageStreamer.h>
#include <dw/image/FormatConverter.h>

#include "cv_connection.hpp"
#include <ros/ros.h>

#define WINDOW_HEIGHT 800
#define WINDOW_WIDTH 1280

#define LOG_VERBOSE

#ifdef LOG_VERBOSE
#define INFO(info) std::cout << "\033[1m\033[34m" << info << "\033[0m" << std::endl;
#else
#define INFO(info) 
#endif

//------------------------------------------------------------------------------
// FUNCS
//------------------------------------------------------------------------------
void captureImageThread(dwSensorHandle_t *cameraSensor, dwImageProperties *baseProp);

void initSdk(dwContextHandle_t *sdk);
void initSensors(
    dwSALHandle_t *sal, dwSensorHandle_t *camera,
    uint32_t *imageWidth, uint32_t *imageHeight, 
    dwImageType *cameraImageType,
    ProgramArguments *arguments,
    dwContextHandle_t context
);

//------------------------------------------------------------------------------
int main(int argc, const char** argv) {
    std::cout << "Compiled at " << __TIME__ << ", " << __DATE__ << std::endl;

    ros::init(argc, (char**)argv, "image_publisher");    

    dwImageProperties baseProp;

    baseProp.type = DW_IMAGE_NVMEDIA;
    baseProp.pxlFormat = DW_IMAGE_RGBA;
    baseProp.pxlType = DW_TYPE_UINT8;
    baseProp.planeCount = 1;

    dwContextHandle_t sdk;
    dwSALHandle_t sal;
    dwSensorHandle_t cameraSensor;

    // Program arguments
    ProgramArguments default_arguments({
        ProgramArguments::Option_t("camera-type", "ar0231-rccb"),
        ProgramArguments::Option_t("csi-port", "ab"),
        ProgramArguments::Option_t("offscreen", "1"),
        ProgramArguments::Option_t("slave", "0"),
        ProgramArguments::Option_t("fifo-size", "3"),
    });

    if(initSampleApp(argc, argv, &default_arguments, NULL, WINDOW_WIDTH, WINDOW_HEIGHT)){
        initSdk(&sdk);
        dwImageType cameraImageType;
        initSensors(&sal, &cameraSensor, &baseProp.width, &baseProp.height, &cameraImageType, &gArguments, sdk);
        captureImageThread(&cameraSensor, &baseProp);
    }

    dwSAL_releaseSensor(&cameraSensor);
    dwSAL_release(&sal);
    dwRelease(&sdk);
    dwLogger_release();
    releaseSampleApp();
    return 0;
}

//------------------------------------------------------------------------------
// THREADS
//------------------------------------------------------------------------------
void captureImageThread_captureCamera(
    uint8_t *yBuffer,
    size_t dataLength,
    dwSensorHandle_t cameraSensor,
    uint32_t index
){
    dwCameraFrameHandle_t frameHandle;
    dwImageNvMedia *frameNVMyuv;
    NvMediaImageSurfaceMap surfaceMap;

    CHECK_DW_ERROR(dwSensorCamera_readFrame(&frameHandle, index, 1000000, cameraSensor));
    CHECK_DW_ERROR(dwSensorCamera_getImageNvMedia(&frameNVMyuv, DW_CAMERA_PROCESSED_IMAGE, frameHandle));

    if(NvMediaImageLock(frameNVMyuv->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK){
        std::memcpy(yBuffer, (uint8_t*)surfaceMap.surface[0].mapping, dataLength);
        NvMediaImageUnlock(frameNVMyuv->img);
    }

    CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frameHandle));
}

void captureImageThread_testCameraAndGetPitch(
    uint32_t *pitch,
    dwSensorHandle_t cameraSensor,
    uint32_t index
){
    dwCameraFrameHandle_t frameHandle;
    dwImageNvMedia *frameNVMyuv;
    NvMediaImageSurfaceMap surfaceMap;

    CHECK_DW_ERROR(dwSensorCamera_readFrame(&frameHandle, index, 1000000, cameraSensor));
    CHECK_DW_ERROR(dwSensorCamera_getImageNvMedia(&frameNVMyuv, DW_CAMERA_PROCESSED_IMAGE, frameHandle));

    if(NvMediaImageLock(frameNVMyuv->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK){
        *pitch = surfaceMap.surface[0].pitch;
        NvMediaImageUnlock(frameNVMyuv->img);
    }

    CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frameHandle));
}

void captureImageThread(dwSensorHandle_t *cameraSensor, dwImageProperties *baseProp){
    std::string pathL = "", pathR = "";
    GetFilePath(pathL, pathR);
    std::cout<<"LLLLLL"<<pathL<<std::endl;
    std::cout<<"RRRRRR"<<pathR<<std::endl;
    
    OpenCVConnector cvcL("left", "narrow_stereo/left", pathL);
    OpenCVConnector cvcR("right", "narrow_stereo/right", pathR);
    dwImageProperties cameraImageProperties;
    dwSensorCamera_getImageProperties(&cameraImageProperties, DW_CAMERA_PROCESSED_IMAGE, *cameraSensor);
    if(cameraImageProperties.pxlFormat!=DW_IMAGE_YUV420 || cameraImageProperties.planeCount != 3) exit(-1);
    CHECK_DW_ERROR(dwSensor_start(*cameraSensor));

    uint32_t pitchL, pitchR;
    captureImageThread_testCameraAndGetPitch(&pitchL, *cameraSensor, 0);
    captureImageThread_testCameraAndGetPitch(&pitchR, *cameraSensor, 1);
    if(pitchL != pitchR) exit(-1);

    size_t dataLength = pitchL * baseProp->height;
    uint8_t *buffer = new uint8_t[dataLength*2];
    uint8_t *bufferL = buffer, *bufferR = buffer + dataLength;

    cv::Mat pitchImgL(baseProp->height, pitchL, CV_8UC1, bufferL);
    cv::Mat pitchImgR(baseProp->height, pitchR, CV_8UC1, bufferR);
    //cv::Mat pitchImgStack(baseProp->height * 2, pitchR, CV_8UC1, bufferR);
    
    cv::Mat imgL = pitchImgL(cv::Rect(0, 0, baseProp->width, baseProp->height));
    cv::Mat imgR = pitchImgR(cv::Rect(0, 0, baseProp->width, baseProp->height));
    //cv::Mat imgStack = pitchImgStack(cv::Rect(0, 0, baseProp->width, baseProp->height * 2));
    
    while(gRun){
        ros::Time pubTime = ros::Time::now();
        captureImageThread_captureCamera(bufferL, dataLength, *cameraSensor, 0);
        captureImageThread_captureCamera(bufferR, dataLength, *cameraSensor, 1);
        cvcL.WriteToOpenCV(imgL, pubTime);
        cvcR.WriteToOpenCV(imgR, pubTime);
    }
}

//------------------------------------------------------------------------------
// INITIALIZE
//------------------------------------------------------------------------------

void initSdk(dwContextHandle_t *sdk){
    CHECK_DW_ERROR(dwLogger_initialize(getConsoleLoggerCallback(true)));
#ifdef LOG_VERBOSE0
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_VERBOSE));
#else
    CHECK_DW_ERROR(dwLogger_setLogLevel(DW_LOG_ERROR));
#endif

    dwContextParameters sdkParams;
    memset(&sdkParams, 0, sizeof(dwContextParameters));
    CHECK_DW_ERROR(dwInitialize(sdk, DW_VERSION, &sdkParams));
}

void initSensors(
    dwSALHandle_t *sal, dwSensorHandle_t *camera,
    uint32_t *imageWidth, uint32_t *imageHeight, 
    dwImageType *cameraImageType,
    ProgramArguments *arguments,
    dwContextHandle_t context
){
    CHECK_DW_ERROR(dwSAL_initialize(sal, context));

    // create GMSL Camera interface
    dwSensorParams params;
    std::string parameterString = arguments->parameterString();
    parameterString             += ",output-format=yuv";
    parameterString             += ",camera-count=4";
    parameterString             += ",camera-mask=0110";
    params.parameters           = parameterString.c_str();
    params.protocol             = "camera.gmsl";
    CHECK_DW_ERROR(dwSAL_createSensor(camera, params, *sal));

    dwImageProperties cameraImageProperties;
    dwSensorCamera_getImageProperties(&cameraImageProperties,
                            DW_CAMERA_PROCESSED_IMAGE,
                        *camera);
    *imageWidth = cameraImageProperties.width;
    *imageHeight = cameraImageProperties.height;
    *cameraImageType = cameraImageProperties.type;

    dwCameraProperties cameraProperties;
    dwSensorCamera_getSensorProperties(&cameraProperties, *camera);

    std::cout << "Camera image with " << *imageWidth << "x" << *imageHeight
    << " at " << cameraProperties.framerate << " FPS" << std::endl;
}
