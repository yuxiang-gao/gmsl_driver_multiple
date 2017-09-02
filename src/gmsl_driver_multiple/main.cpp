/////////////////////////////////////////////////////////////////////////////////////////
// This code contains NVIDIA Confidential Information and is disclosed
// under the Mutual Non-Disclosure Agreement.
//
// Notice
// ALL NVIDIA DESIGN SPECIFICATIONS AND CODE ("MATERIALS") ARE PROVIDED "AS IS" NVIDIA MAKES
// NO REPRESENTATIONS, WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ANY IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
//
// NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. No third party distribution is allowed unless
// expressly authorized by NVIDIA.  Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2015-2016 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include <signal.h>
#include <iostream>
#include <cstring>
#include <thread>
#include <queue>
#include <vector>

//#include <lodepng.h>

#include <common/Checks.hpp>
#include <common/WindowGLFW.hpp>
#include <common/WindowEGL.hpp>
#include <common/ProgramArguments.hpp>
#include <common/Grid.hpp>
#include <common/ConsoleColor.hpp>

// SDK
#include <dw/core/Context.h>
#include <dw/core/Logger.h>
#include <dw/renderer/Renderer.h>
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>
#include <dw/image/ImageStreamer.h>
#include <dw/image/FormatConverter.h>

#include "cv_connection.hpp"
#include <ros/ros.h>

//------------------------------------------------------------------------------
// Variables
//------------------------------------------------------------------------------
static bool volatile g_run = true;
static bool gTakeScreenshot = false;
static int gScreenshotCount = 0;

ProgramArguments g_arguments(
    {
        ProgramArguments::Option_t("type-ab", "ar0231-rccb"),
        ProgramArguments::Option_t("type-cd", "ar0231-rccb"),
        ProgramArguments::Option_t("type-ef", "ar0231-rccb"),
        ProgramArguments::Option_t("selector-mask", "0011"),
        ProgramArguments::Option_t("offscreen", "0"),
        ProgramArguments::Option_t("slave", "0"),
        ProgramArguments::Option_t("fifo-size", "3"),
        ProgramArguments::Option_t("cross-csi-sync", "1"),
        ProgramArguments::Option_t("camera-count", "2"),
    });

uint32_t g_imageWidth;
uint32_t g_imageHeight;
uint32_t g_numCameras;
GridData_t g_grid;

// combine by camera sensor, which might have camera siblings
struct Camera {
    dwSensorHandle_t sensor;
    uint32_t numSiblings;
    uint32_t width;
    uint32_t height;
    dwImageStreamerHandle_t streamer; // different streamers to support different resolutions
    dwImageFormatConverterHandle_t yuv2rgba;
    std::queue<dwImageNvMedia *> rgbaPool;
};

//------------------------------------------------------------------------------
// Method declarations
//------------------------------------------------------------------------------
int main(int argc, const char **argv);
void takeScreenshot(dwImageNvMedia *frameNVMrgba, uint8_t group, uint32_t sibling);
void pubImg(dwImageNvMedia *frameNVMrgba, dwImageNvMedia *frameNVMrgba2);
void parseArguments(int argc, const char **argv);
void initGL(WindowBase **window);
void initRenderer(dwRendererHandle_t *renderer,
                  dwContextHandle_t context, WindowBase *window);
void initSdk(dwContextHandle_t *context, WindowBase *window);
void initSAL(dwSALHandle_t *sal, dwContextHandle_t context);
void initSensors(std::vector<Camera> *cameras,
                uint32_t *numCameras,
                dwSALHandle_t sal,
                ProgramArguments &arguments);
dwStatus captureCamera(dwImageNvMedia *frameNVMrgba,
                                 dwSensorHandle_t cameraSensor,
                                 uint32_t sibling,
                                 dwImageFormatConverterHandle_t yuv2rgba);

void renderFrame(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer);

void sig_int_handler(int sig);
void keyPressCallback(int key);
void resizeCallback(int width, int height);


//#######################################################################################
int main(int argc, const char **argv)
{
    ros::init(argc, (char**)argv, "image_publisher");
    //SDK objects
    WindowBase *window            = nullptr;
    dwContextHandle_t sdk         = DW_NULL_HANDLE;
    dwRendererHandle_t renderer   = DW_NULL_HANDLE;
    dwSALHandle_t sal             = DW_NULL_HANDLE;

    struct sigaction action;
    memset(&action, 0, sizeof(action));
    action.sa_handler = sig_int_handler;

    sigaction(SIGHUP, &action, NULL);  // controlling terminal closed, Ctrl-D
    sigaction(SIGINT, &action, NULL);  // Ctrl-C
    sigaction(SIGQUIT, &action, NULL); // Ctrl-\, clean quit with core dump
    sigaction(SIGABRT, &action, NULL); // abort() called.
    sigaction(SIGTERM, &action, NULL); // kill command
    sigaction(SIGSTOP, &action, NULL); // kill command

    g_run = true;

    parseArguments(argc, argv);
    initGL(&window);
    initSdk(&sdk, window);
    initRenderer(&renderer, sdk, window);
    initSAL(&sal, sdk);


    // create GMSL Camera interface, based on the camera selector mask
    std::vector<Camera> cameraSensor;
    initSensors(&cameraSensor, &g_numCameras, sal, g_arguments);

    if (cameraSensor.size() == 0 || cameraSensor.size() > 12) {
        std::cerr << "Need to specify at least 1 at most 12 cameras to be used" << std::endl;
        exit(-1);
    }

    // Grid
    g_imageWidth = cameraSensor[0].width;
    g_imageHeight = cameraSensor[0].height;
    configureGrid(&g_grid, window->width(), window->height(), g_imageWidth, g_imageHeight, g_numCameras);

    // image API translator
    size_t numFramesRGBA = 0;
    for (size_t i = 0; i < cameraSensor.size(); i++) {
        numFramesRGBA += 2*cameraSensor[i].numSiblings;
    }

    dwStatus result;

    std::vector<bool> eof;
    // RGBA image pool for conversion from YUV camera output
    // two RGBA frames per camera per sibling for a pool
    // since image streamer might hold up-to one frame when using egl streams
    std::vector<dwImageNvMedia> frameRGBA;
    frameRGBA.reserve(numFramesRGBA);
    for (size_t csiPort = 0; csiPort < cameraSensor.size() && g_run; csiPort++) {
        dwImageProperties cameraImageProperties;
        dwSensorCamera_getImageProperties(&cameraImageProperties, DW_CAMERA_PROCESSED_IMAGE,
                                          cameraSensor[csiPort].sensor);
        dwImageProperties displayImageProperties = cameraImageProperties;
        displayImageProperties.pxlFormat         = DW_IMAGE_RGBA;
        displayImageProperties.planeCount        = 1;

        // format converter
        result = dwImageFormatConverter_initialize(&cameraSensor[csiPort].yuv2rgba, &cameraImageProperties,
                                                            &displayImageProperties, sdk);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot create pixel format converter : yuv->rgba" <<
                         dwGetStatusName(result) <<  std::endl;
            g_run = false;
            break;
        }

        result = dwImageStreamer_initialize(&cameraSensor[csiPort].streamer,
                                                     &displayImageProperties, DW_IMAGE_GL, sdk);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot init image streamer: " << dwGetStatusName(result) << std::endl;
            g_run = false;
            break;
        }

        // allocate pool
        for (uint32_t cameraIdx = 0; cameraIdx < cameraSensor[csiPort].numSiblings; cameraIdx++) {
            for (int32_t k = 0; k < 2; k++) {
                dwImageNvMedia rgba{};
                result = dwImageNvMedia_create(&rgba, &displayImageProperties, sdk);
                if (result != DW_SUCCESS) {
                    std::cerr << "Cannot create nvmedia image for pool:" <<
                                 dwGetStatusName(result) << std::endl;
                    g_run = false;
                    break;
                }

                frameRGBA.push_back(rgba);
                cameraSensor[csiPort].rgbaPool.push(&frameRGBA.back());
            }
        }

        g_run = g_run && dwSensor_start(cameraSensor[csiPort].sensor) == DW_SUCCESS;
        eof.push_back(false);
    }

    std::vector<std::vector<dwImageNvMedia*>> frameRGBAPtr;

    for (size_t csiPort = 0; csiPort < cameraSensor.size(); csiPort++) {
        std::vector<dwImageNvMedia*> pool;
        for (size_t cameraIdx = 0; cameraIdx < cameraSensor[csiPort].numSiblings; ++cameraIdx) {
            pool.push_back(nullptr);
        }
        frameRGBAPtr.push_back(pool);
    }

    // main loop
    while (g_run && !window->shouldClose()) {

        std::this_thread::yield();

        bool eofAny = false;

        // capture from all csi-ports
        // NOTE if cross-csi-synch is active, all cameras will capture at the same time
        for (size_t csiPort = 0; csiPort < cameraSensor.size(); csiPort++) {
            if (eof[csiPort]) {
                eofAny = true;
                continue;
            }

            // capture from all cameras within a csi port
            for (uint32_t cameraIdx = 0;
                 cameraIdx < cameraSensor[csiPort].numSiblings && !cameraSensor[csiPort].rgbaPool.empty();
                 cameraIdx++) {

                frameRGBAPtr[csiPort][cameraIdx] = cameraSensor[csiPort].rgbaPool.front();
                cameraSensor[csiPort].rgbaPool.pop();

                // capture, convert to rgba and return it
                eof[csiPort] = captureCamera(frameRGBAPtr[csiPort][cameraIdx],
                                             cameraSensor[csiPort].sensor, cameraIdx,
                                             cameraSensor[csiPort].yuv2rgba);

                if (!eof[csiPort]) {
                    cameraSensor[csiPort].rgbaPool.push(frameRGBAPtr[csiPort][cameraIdx]);
                }

                eofAny |= eof[csiPort];
            }
        }

        // retrieve imgs
        //pubImg(frameRGBAPtr[0][0], frameRGBAPtr[0][1]);
/*
        // render all
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            int cellIdx = 0;
            for (size_t csiPort = 0; csiPort < cameraSensor.size(); csiPort++) {
                for (uint32_t cameraIdx = 0;
                     cameraIdx < cameraSensor[csiPort].numSiblings && !cameraSensor[csiPort].rgbaPool.empty();
                     cameraIdx++) {

                    result = dwImageStreamer_postNvMedia(frameRGBAPtr[csiPort][cameraIdx],
                                                         cameraSensor[csiPort].streamer);
                    if (result != DW_SUCCESS) {
                        std::cerr << "Cannot post nvmedia: " << dwGetStatusName(result) << std::endl;
                        g_run = false;
                        break;
                    }

                    //Set area
                    dwRect rect;
                    gridCellRect(&rect, g_grid, cellIdx);
                    dwRenderer_setRect(rect, renderer);

                    renderFrame(cameraSensor[csiPort].streamer, renderer);

                    // return frame which has been lately processed by the streamer
                    dwImageNvMedia *processed = nullptr;
                    result = dwImageStreamer_waitPostedNvMedia(&processed, 60000,
                                                               cameraSensor[csiPort].streamer);
                    if (result != DW_SUCCESS) {
                        std::cerr << "Cannot waitpost nvmedia: " << dwGetStatusName(result) << std::endl;
                        g_run = false;
                        break;
                    }

                    if (processed) {
                        cameraSensor[csiPort].rgbaPool.push(processed);
                    }

                    cellIdx++;
                }
            }

            window->swapBuffers();
            CHECK_GL_ERROR();
        }
*/
        for (size_t i = 0; i < cameraSensor.size(); i++) {
            if (cameraSensor[i].rgbaPool.empty()) {
                g_run = false;
                std::cerr << "Ran out of RGBA buffers, stop" << std::endl;
            }
        }

        g_run = g_run && !eofAny;
    }

    //Clean up
    // release used objects in correct order

    for (size_t i = 0; i < cameraSensor.size(); i++) {
        dwSensor_stop(cameraSensor[i].sensor);
        dwSAL_releaseSensor(&cameraSensor[i].sensor);

        dwImageStreamer_release(&cameraSensor[i].streamer);
        dwImageFormatConverter_release(&cameraSensor[i].yuv2rgba);
    }

    for (dwImageNvMedia& frame : frameRGBA) {
        dwStatus result = dwImageNvMedia_destroy(&frame);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot destroy nvmedia: " << dwGetStatusName(result) << std::endl;
            g_run = false;
            break;
        }
    }

    // release used objects in correct order
    dwSAL_release(&sal);

    // todo - render release code has been commented out, since that one results in a stall
    //        of the GMSL (nvmedia) pipeline. The issue is known and will be fixed in the future.
    //dwRenderer_release(&renderer);

    dwRelease(&sdk);
    dwLogger_release();
    delete window;
    return 0;
}

//------------------------------------------------------------------------------
/*
void takeScreenshot(dwImageNvMedia *frameNVMrgba, uint8_t group, uint32_t sibling)
{
    NvMediaImageSurfaceMap surfaceMap;
    if (NvMediaImageLock(frameNVMrgba->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK)
    {
        char fname[128];
        sprintf(fname, "screenshot_%u_%d_%04d.png", group, sibling, gScreenshotCount);
        lodepng_encode32_file(fname, (unsigned char*)surfaceMap.surface[0].mapping,
                frameNVMrgba->prop.width, frameNVMrgba->prop.height);
        NvMediaImageUnlock(frameNVMrgba->img);
        std::cout << "SCREENSHOT TAKEN to " << fname << "\n";
    }else
    {
        std::cout << "CANNOT LOCK NVMEDIA IMAGE - NO SCREENSHOT\n";
    }
}
*/
void pubImg(dwImageNvMedia *frameNVMrgba, dwImageNvMedia *frameNVMrgba2)
{   

    std::cout<<"a"<<"\n";
    int size = frameNVMrgba->prop.width * frameNVMrgba->prop.height * 4;
    unsigned char* nbuffer = new unsigned char[size * 2];   
    NvMediaImageSurfaceMap surfaceMap;
    NvMediaImageSurfaceMap surfaceMap2;         
    OpenCVConnector cvc;

    std::cout<<"b"<<"\n";
    if (NvMediaImageLock(frameNVMrgba->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK && NvMediaImageLock(frameNVMrgba2->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap2) == NVMEDIA_STATUS_OK)
    {
    std::cout<<"c"<<"\n";
        std::memcpy(nbuffer, (unsigned char*)surfaceMap.surface[0].mapping, size);
 std::cout<<"f"<<"\n";
        std::memcpy(nbuffer+size, (unsigned char*)surfaceMap2.surface[0].mapping, size);
 std::cout<<"g"<<"\n";
        cvc.WriteToOpenCV(nbuffer, frameNVMrgba->prop.width, frameNVMrgba->prop.height*2);
        NvMediaImageUnlock(frameNVMrgba->img);
        NvMediaImageUnlock(frameNVMrgba2->img);
std::cout<<"end"<<std::endl;
    }else
    {
        std::cout << "CANNOT LOCK NVMEDIA IMAGE - NO SCREENSHOT\n";
    }
/*
    int size = frameNVMrgba->prop.width * frameNVMrgba->prop.height * 4;
    std::cout<<"a"<<"\n";         
    OpenCVConnector cvc;
    NvMediaImageSurfaceMap surfaceMap;
    std::cout<<"b"<<"\n";
    if (NvMediaImageLock(frameNVMrgba->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK)
    {
        cvc.WriteToOpenCV((unsigned char*)surfaceMap.surface[0].mapping, frameNVMrgba->prop.width, frameNVMrgba->prop.height);
        NvMediaImageUnlock(frameNVMrgba->img);
std::cout<<"end"<<std::endl;
    }else
    {
        std::cout << "CANNOT LOCK NVMEDIA IMAGE - NO SCREENSHOT\n";
    }*/
}
//------------------------------------------------------------------------------
void parseArguments(int argc, const char **argv)
{
    if (!g_arguments.parse(argc, argv))
        exit(-1); // Exit if not all require arguments are provided

    std::cout << "Program Arguments:\n" << g_arguments.printList() << std::endl;
}

//------------------------------------------------------------------------------
void initGL(WindowBase **window)
{
    //bool offscreen = atoi(g_arguments.get("offscreen").c_str()) != 0;
    bool offscreen = true;
    #ifdef VIBRANTE
        if (offscreen)
            *window = new WindowOffscreenEGL(1280, 800);
    #else
        (void)offscreen;
    #endif

    if(!*window)
        //*window = new WindowGLFW(1280, 800);

    (*window)->makeCurrent();
    (*window)->setOnKeypressCallback(keyPressCallback);
    (*window)->setOnResizeWindowCallback(resizeCallback);
}

//------------------------------------------------------------------------------
void initSdk(dwContextHandle_t *context, WindowBase *window)
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams;
    memset(&sdkParams, 0, sizeof(dwContextParameters));

#ifdef VIBRANTE
    sdkParams.eglDisplay = window->getEGLDisplay();
#else
    (void)window;
#endif

    dwInitialize(context, DW_VERSION, &sdkParams);
}

//------------------------------------------------------------------------------
void initRenderer(dwRendererHandle_t *renderer,
                  dwContextHandle_t context, WindowBase *window)
{
    dwStatus result;

    result = dwRenderer_initialize(renderer, context);
    if (result != DW_SUCCESS)
        throw std::runtime_error(std::string("Cannot init renderer: ") +
                                 dwGetStatusName(result));

    // Set some renderer defaults
    (void)window;
}

//------------------------------------------------------------------------------
void initSAL(dwSALHandle_t *sal, dwContextHandle_t context)
{
    dwStatus result;

    result = dwSAL_initialize(sal, context);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot initialize SAL: "
                    << dwGetStatusName(result) << std::endl;
        exit(1);
    }
}

//------------------------------------------------------------------------------
void initSensors(std::vector<Camera> *cameras,
                uint32_t *numCameras,
                dwSALHandle_t sal,
                ProgramArguments &arguments)
{
    std::string selector = arguments.get("selector-mask");

    dwStatus result;

    // identify active ports
    int idx             = 0;
    int cnt[3]          = {0, 0, 0};
    std::string port[3] = {"ab", "cd", "ef"};
    for (size_t i = 0; i < selector.length() && i < 12; i++, idx++) {
        const char s = selector[i];
        if (s == '1') {
            cnt[idx / 4]++;
        }
    }

    // how many cameras selected in a port
    (*numCameras) = 0;
    for (size_t p = 0; p < 3; p++) {
        if (cnt[p] > 0) {
            std::string params;

            params += std::string("csi-port=") + port[p];
            params += ",camera-type=" + arguments.get((std::string("type-") + port[p]).c_str());
            params += ",camera-count=2"; // when using the mask, just ask for all cameras, mask will select properly

            if (selector.size() >= p*4) {
                params += ",camera-mask="+ selector.substr(p*4, std::min(selector.size() - p*4, size_t{4}));
            }

            params += ",slave="  + arguments.get("slave");
            params += ",cross-csi-sync="  + arguments.get("cross-csi-sync");
            params += ",fifo-size="  + arguments.get("fifo-size");

            dwSensorHandle_t salSensor = DW_NULL_HANDLE;
            dwSensorParams salParams;
            salParams.parameters = params.c_str();
            salParams.protocol = "camera.gmsl";
            result = dwSAL_createSensor(&salSensor, salParams, sal);
            if (result == DW_SUCCESS) {
                Camera cam;
                cam.sensor = salSensor;

                dwImageProperties cameraImageProperties;
                dwSensorCamera_getImageProperties(&cameraImageProperties,
                                DW_CAMERA_PROCESSED_IMAGE,
                                salSensor);

                dwCameraProperties cameraProperties;
                dwSensorCamera_getSensorProperties(&cameraProperties, salSensor);

                cam.width = cameraImageProperties.width;
                cam.height = cameraImageProperties.height;
                cam.numSiblings = cameraProperties.siblings;

                cameras->push_back(cam);

                (*numCameras) += cam.numSiblings;
            }
            else
            {
                std::cerr << "Cannot create driver: " << salParams.protocol
                    << " with params: " << salParams.parameters << std::endl
                    << "Error: " << dwGetStatusName(result) << std::endl;
                if (result == DW_INVALID_ARGUMENT) {
                    std::cerr << "It is possible the given camera is not supported. "
                              << "Please refer to the documentation for this sample."
                              << std::endl;
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
dwStatus captureCamera(dwImageNvMedia *frameNVMrgba,
                                 dwSensorHandle_t cameraSensor,
                                 uint32_t sibling,
                                 dwImageFormatConverterHandle_t yuv2rgba)
{
    dwCameraFrameHandle_t frameHandle;
    dwImageNvMedia *frameNVMyuv = nullptr;

    dwStatus result = DW_FAILURE;
    result = dwSensorCamera_readFrame(&frameHandle, sibling, 300000, cameraSensor);
    if (result != DW_SUCCESS) {
        std::cout << "readFrameNvMedia: " << dwGetStatusName(result) << std::endl;
        return result;
    }

    result = dwSensorCamera_getImageNvMedia(&frameNVMyuv, DW_CAMERA_PROCESSED_IMAGE, frameHandle);
    if( result != DW_SUCCESS ){
        std::cout << "readFrameNvMedia: " << dwGetStatusName(result) << std::endl;

    }

    result = dwImageFormatConverter_copyConvertNvMedia(frameNVMrgba, frameNVMyuv, yuv2rgba);
    if( result != DW_SUCCESS ){
        std::cout << "copyConvertNvMedia: " << dwGetStatusName(result) << std::endl;

    }

    result = dwSensorCamera_returnFrame(&frameHandle);
    if( result != DW_SUCCESS ){
        std::cout << "copyConvertNvMedia: " << dwGetStatusName(result) << std::endl;
    }

    return DW_SUCCESS;
}

//------------------------------------------------------------------------------
void renderFrame(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer)
{
    dwImageGL *frameGL = nullptr;

    if (dwImageStreamer_receiveGL(&frameGL, 60000, streamer) != DW_SUCCESS) {
        std::cerr << "did not received GL frame within 30ms" << std::endl;
    } else {
        // render received texture
        dwRenderer_renderTexture(frameGL->tex, frameGL->target, renderer);

        //std::cout << "received GL: " << frameGL->prop.timestamp_us << std::endl;
        dwImageStreamer_returnReceivedGL(frameGL, streamer);
    }
}

//------------------------------------------------------------------------------
void sig_int_handler(int sig)
{
    (void)sig;

    g_run = false;
}

//------------------------------------------------------------------------------
void keyPressCallback(int key)
{
    // stop application
    if (key == GLFW_KEY_ESCAPE)
        g_run = false;

    // take screenshot
    if (key == GLFW_KEY_S)
         gTakeScreenshot = true;
}

//------------------------------------------------------------------------------
void resizeCallback(int width, int height)
{
    configureGrid(&g_grid, width, height, g_imageWidth, g_imageHeight, g_numCameras);
}
