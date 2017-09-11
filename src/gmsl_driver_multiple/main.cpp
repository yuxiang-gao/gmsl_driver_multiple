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
void captureImageThread(
    dwSensorHandle_t *cameraSensor, 
    dwImageProperties *baseProp,
    bool output_color
);

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
        const bool output_color = default_arguments.has("color");
        if(output_color) std::cout << "output in color mode" << std::endl;
        initSdk(&sdk);
        dwImageType cameraImageType;
        initSensors(&sal, &cameraSensor, &baseProp.width, &baseProp.height, &cameraImageType, &gArguments, sdk);
        captureImageThread(&cameraSensor, &baseProp, sdk, output_color);
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
    uint32_t index,
    dwImageFormatConverterHandle_t yuv2rgba = DW_NULL_HANDLE,
    dwImageNvMedia *frameNVMrgba = nullptr
){
    dwCameraFrameHandle_t frameHandle;
    dwImageNvMedia *frameNVMyuv;
    NvMediaImageSurfaceMap surfaceMap;
    dwImageNvMedia *frameNVMFinal;

    CHECK_DW_ERROR(dwSensorCamera_readFrame(&frameHandle, index, 1000000, cameraSensor));
    CHECK_DW_ERROR(dwSensorCamera_getImageNvMedia(&frameNVMyuv, DW_CAMERA_PROCESSED_IMAGE, frameHandle));

    if(yuv2rgba != DW_NULL_HANDLE) { //Color mode
        CHECK_DW_ERROR(dwImageFormatConverter_copyConvertNvMedia(frameNVMrgba, frameNVMyuv, yuv2rgba));
        frameNVMFinal = frameNVMrgba;
    }
    else{
        frameNVMFinal = frameNVMyuv;
    }

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
    dwImageFormatConverterHandle_t yuv2rgba = DW_NULL_HANDLE
    dwImageNvMedia *frameNVMrgba = nullptr
){
    dwCameraFrameHandle_t frameHandle;
    dwImageNvMedia *frameNVMyuv;
    NvMediaImageSurfaceMap surfaceMap;
    dwImageNvMedia *frameNVMFinal;

    CHECK_DW_ERROR(dwSensorCamera_readFrame(&frameHandle, index, 1000000, cameraSensor));
    CHECK_DW_ERROR(dwSensorCamera_getImageNvMedia(&frameNVMyuv, DW_CAMERA_PROCESSED_IMAGE, frameHandle));

    if(yuv2rgba != DW_NULL_HANDLE) { //Color mode
        CHECK_DW_ERROR(dwImageFormatConverter_copyConvertNvMedia(frameNVMrgba, frameNVMyuv, yuv2rgba));
        frameNVMFinal = frameNVMrgba;
    }
    else{
        frameNVMFinal = frameNVMyuv;
    }

    if(NvMediaImageLock(frameNVMFinal->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK){
        *pitch = surfaceMap.surface[0].pitch;
        NvMediaImageUnlock(frameNVMFinal->img);
    }

    CHECK_DW_ERROR(dwSensorCamera_returnFrame(&frameHandle));
}

void captureImageThread(
    dwSensorHandle_t *cameraSensor, 
    dwImageProperties *baseProp,
    dwContextHandle_t context,
    bool output_color
){
    std::string pathL = "", pathR = "";
    GetFilePath(pathL, pathR);
    std::cout<<"Left camera info file: "<<pathL<<std::endl;
    std::cout<<"Right camera info file:"<<pathR<<std::endl;
    
    OpenCVConnector cvcL("left", "narrow_stereo/left", pathL);
    OpenCVConnector cvcR("right", "narrow_stereo/right", pathR);
    dwImageProperties cameraImageProperties;
    dwSensorCamera_getImageProperties(&cameraImageProperties, DW_CAMERA_PROCESSED_IMAGE, *cameraSensor);
    if(cameraImageProperties.pxlFormat!=DW_IMAGE_YUV420 || cameraImageProperties.planeCount != 3) exit(-1);

    //Init yuv-> rgba converter
    dwImageFormatConverterHandle_t yuv2rgba = DW_NULL_HANDLE
    dwImageNvMedia *frameNVMrgba = nullptr

    if(output_color){
        dwImageProperties rgbaProp = *baseProp;
        rgbaProp.pxlFormat         = DW_IMAGE_RGBA;
        rgbaProp.planeCount        = 1;
        CHECK_DW_ERROR(dwImageFormatConverter_initialize(&yuv2rgba, &baseProp, &rgbaProp, context));
        CHECK_DW_ERROR(dwImageNvMedia_create(&frameNVMrgba, rgbaProp, context));
        if(yuv2rgba == DW_NULL_HANDLE || frameNVMrgba == nullptr){
            yuv2rgba = DW_NULL_HANDLE;
            output_color = false;
        }
    }

    CHECK_DW_ERROR(dwSensor_start(*cameraSensor));

    uint32_t pitchL, pitchR;
    captureImageThread_testCameraAndGetPitch(&pitchL, *cameraSensor, 0, yuv2rgba, frameNVMrgba);
    captureImageThread_testCameraAndGetPitch(&pitchR, *cameraSensor, 1, yuv2rgba, frameNVMrgba);
    if(pitchL != pitchR) exit(-1);

    size_t dataLength = pitchL * baseProp->height;
    uint8_t *buffer = new uint8_t[dataLength*2];
    uint8_t *bufferL = buffer, *bufferR = buffer + dataLength;

    cv::Mat pitchImgL, pitchImgR;

    if(output_color){
        pitchImgL = cv::Mat(baseProp->height, pitchL / 4, CV_8UC4, bufferL);
        pitchImgR = cv::Mat(baseProp->height, pitchR / 4, CV_8UC4, bufferR);
    }
    else{
        pitchImgL = cv::Mat(baseProp->height, pitchL, CV_8UC1, bufferL);
        pitchImgR = cv::Mat(baseProp->height, pitchR, CV_8UC1, bufferR);
    }
    
    cv::Mat imgL = pitchImgL(cv::Rect(0, 0, baseProp->width, baseProp->height));
    cv::Mat imgR = pitchImgR(cv::Rect(0, 0, baseProp->width, baseProp->height));

    cv::Mat imgLFinal, imgRFinal;
    if(output_color){
        pitchImgLFinal = cv::Mat(baseProp->height, baseProp->width, CV_8UC3);
        pitchImgRFinal = cv::Mat(baseProp->height, baseProp->width, CV_8UC3);
    }
    else{
        imgLFinal = imgL;
        imgRFinal = imgR;
    }
    //cv::Mat imgStack = pitchImgStack(cv::Rect(0, 0, baseProp->width, baseProp->height * 2));
    
    while(gRun){
        ros::Time pubTime = ros::Time::now();
        captureImageThread_captureCamera(bufferL, dataLength, *cameraSensor, 0, yuv2rgba, frameNVMrgba);
        captureImageThread_captureCamera(bufferR, dataLength, *cameraSensor, 1, yuv2rgba, frameNVMrgba);

        if(output_color){
            cv::cvtColor(imgL, imgLFinal, cv::COLOR_RGBA2BGR);
            cv::cvtColor(imgR, imgRFinal, cv::COLOR_RGBA2BGR);
        }

        cvcL.WriteToOpenCV(imgLFinal, pubTime);
        cvcR.WriteToOpenCV(imgRFinal, pubTime);
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
