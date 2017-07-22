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
// Copyright (c) 2017 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software and related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is
// strictly prohibited.
//
/////////////////////////////////////////////////////////////////////////////////////////

#include "common.hpp"
#include "../drivenet_lanenet/cv_connection.hpp"

cudaStream_t g_cudaStream  = 0;

// Driveworks Handles
dwContextHandle_t gSdk                          = DW_NULL_HANDLE;
dwRendererHandle_t gRenderer                    = DW_NULL_HANDLE;
dwRenderBufferHandle_t gLineBuffer              = DW_NULL_HANDLE;
dwSALHandle_t gSal                              = DW_NULL_HANDLE;
dwSensorHandle_t gCameraSensor                  = DW_NULL_HANDLE;
dwRawPipelineHandle_t gRawPipeline              = DW_NULL_HANDLE;

// frame processing
dwImageCUDA gRCBImage{};
dwImageCUDA gRGBAImage{};
dwImageProperties gRCBProperties{};
dwImageProperties rgbaImageProperties{};
dwImageProperties rgbaGLImageProperties{};

dwImageStreamerHandle_t gl2nvm                  = DW_NULL_HANDLE;
dwImageStreamerHandle_t gCuda2gl                = DW_NULL_HANDLE;
dwImageStreamerHandle_t gInput2cuda             = DW_NULL_HANDLE;
dwImageFormatConverterHandle_t gConvert2RGBA    = DW_NULL_HANDLE;

// Sample variables
dwRect gScreenRectangle{};
std::string gInputType;

// Colors for rendering bounding boxes
const uint32_t gMaxBoxColors = 5;
float32_t gBoxColors[5][4] = {{1.0f, 0.0f, 0.0f, 1.0f},
                              {0.0f, 1.0f, 0.0f, 1.0f},
                              {0.0f, 0.0f, 1.0f, 1.0f},
                              {0.0f, 1.0f, 1.0f, 1.0f},
                              {1.0f, 1.0f, 0.0f, 1.0f}};

//------------------------------------------------------------------------------
dwLaneDetectorHandle_t gLaneDetector;
void drawROI(dwRect roi, const float32_t color[4], dwRenderBufferHandle_t renderBuffer, dwRendererHandle_t renderer)
{
    float32_t x_start = static_cast<float32_t>(roi.x) ;
    float32_t x_end   = static_cast<float32_t>(roi.x + roi.width);
    float32_t y_start = static_cast<float32_t>(roi.y);
    float32_t y_end   = static_cast<float32_t>(roi.y + roi.height);

    float32_t *coords     = nullptr;
    uint32_t maxVertices  = 0;
    uint32_t vertexStride = 0;
    dwRenderBuffer_map(&coords, &maxVertices, &vertexStride, renderBuffer);
    coords[0]  = x_start;
    coords[1]  = y_start;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_start;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0]  = x_end;
    coords[1]  = y_end;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_end;
    coords[1] = y_start;
    coords    += vertexStride;
    coords[0] = x_start;
    coords[1] = y_start;
    dwRenderBuffer_unmap(8, renderBuffer);
    dwRenderer_setColor(color, renderer);
    dwRenderer_setLineWidth(2, renderer);
    dwRenderer_renderBuffer(renderBuffer, renderer);
}

//------------------------------------------------------------------------------
bool initPipeline(const dwImageProperties &rawImageProps, const dwCameraProperties &cameraProps, dwContextHandle_t ctx)
{
    dwStatus result = dwRawPipeline_initialize(&gRawPipeline, rawImageProps, cameraProps, ctx);
    if (result != DW_SUCCESS) {
        std::cerr << "Image streamer initialization failed:1 " << dwGetStatusName(result) << std::endl;
        return false;
    }

    // Initialize Raw pipeline
    dwRawPipeline_setCUDAStream(g_cudaStream, gRawPipeline);
    dwRawPipeline_getDemosaicImageProperties(&gRCBProperties, gRawPipeline);

    // RCB image to get output from the RawPipeline
    dwImageCUDA_create(&gRCBImage, &gRCBProperties, DW_IMAGE_CUDA_PITCH);

    // RGBA image to display over GL
    rgbaImageProperties = gRCBProperties;
    rgbaImageProperties.pxlFormat         = DW_IMAGE_RGBA;
    rgbaImageProperties.pxlType           = DW_TYPE_UINT8;
    rgbaImageProperties.planeCount        = 1;
    rgbaGLImageProperties = rgbaImageProperties;
    rgbaGLImageProperties.type = DW_IMAGE_GL;  
    dwImageCUDA_create(&gRGBAImage, &rgbaImageProperties, DW_IMAGE_CUDA_PITCH);

    // Setup streamer to pass input to CUDA and from CUDA to GL
    result = result != DW_SUCCESS ? result : dwImageStreamer_initialize(&gInput2cuda, &rawImageProps, DW_IMAGE_CUDA, ctx);
    result = result != DW_SUCCESS ? result : dwImageStreamer_initialize(&gCuda2gl, &rgbaImageProperties, DW_IMAGE_GL, ctx);
    result = result != DW_SUCCESS ? result : dwImageStreamer_initialize(&gl2nvm, &rgbaGLImageProperties, DW_IMAGE_NVMEDIA, ctx);
    if (result != DW_SUCCESS) {
        std::cerr << "Image streamer initialization failed:2 " << dwGetStatusName(result) << std::endl;
        return false;
    }

    // init format converter to convert from RCB->RGBA
    result = result != DW_SUCCESS ? result : dwImageFormatConverter_initialize(&gConvert2RGBA, &gRCBProperties, &rgbaImageProperties, ctx);
    if (result != DW_SUCCESS) {
        std::cerr << "Image format converter initialization failed: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    return true;
}

//------------------------------------------------------------------------------
void initSdk(dwContextHandle_t *context, WindowBase *window)
{
    // create a Logger to log to console
    // we keep the ownership of the logger at the application level
    dwLogger_initialize(getConsoleLoggerCallback(true));
    dwLogger_setLogLevel(DW_LOG_VERBOSE);

    // instantiate Driveworks SDK context
    dwContextParameters sdkParams{};
    //memset(&sdkParams, 0, sizeof(dwContextParameters));
    //std::string path = DataPath::get();
    //sdkParams.dataPath = path.c_str();

//#ifdef VIBRANTE
  //  sdkParams.eglDisplay = window->getEGLDisplay();
//#else
    (void)window;
//#endif

    dwInitialize(context, DW_VERSION, &sdkParams);
}

//------------------------------------------------------------------------------
void initRenderer(const dwImageProperties& rcbProperties, dwRendererHandle_t *renderer, dwContextHandle_t context, WindowBase *window)
{
    dwStatus result = dwRenderer_initialize(renderer, context);
    if (result != DW_SUCCESS)
        throw std::runtime_error(std::string("Cannot init renderer: ") + dwGetStatusName(result));

    // Set some renderer defaults
    gScreenRectangle.width  = window->width();
    gScreenRectangle.height = window->height();
    gScreenRectangle.x      = 0;
    gScreenRectangle.y      = 0;

    float32_t rasterTransform[9];
    rasterTransform[0] = 1.0f;
    rasterTransform[3] = 0.0f;
    rasterTransform[6] = 0.0f;
    rasterTransform[1] = 0.0f;
    rasterTransform[4] = 1.0f;
    rasterTransform[7] = 0.0f;
    rasterTransform[2] = 0.0f;
    rasterTransform[5] = 0.0f;
    rasterTransform[8] = 1.0f;

    dwRenderer_set2DTransform(rasterTransform, *renderer);
    float32_t boxColor[4] = {0.0f,1.0f,0.0f,1.0f};
    dwRenderer_setColor(boxColor, *renderer);
    dwRenderer_setLineWidth(2.0f, *renderer);

    dwRenderer_setRect(gScreenRectangle, *renderer);

    uint32_t maxLines = 20000;
    {
        dwRenderBufferVertexLayout layout;
        layout.posFormat   = DW_RENDER_FORMAT_R32G32_FLOAT;
        layout.posSemantic = DW_RENDER_SEMANTIC_POS_XY;
        layout.colFormat   = DW_RENDER_FORMAT_NULL;
        layout.colSemantic = DW_RENDER_SEMANTIC_COL_NULL;
        layout.texFormat   = DW_RENDER_FORMAT_NULL;
        layout.texSemantic = DW_RENDER_SEMANTIC_TEX_NULL;
        dwRenderBuffer_initialize(&gLineBuffer, layout, DW_RENDER_PRIM_LINELIST, maxLines, context);
        dwRenderBuffer_set2DCoordNormalizationFactors((float32_t)rcbProperties.width,
                                                      (float32_t)rcbProperties.height, gLineBuffer);
    }
}

//------------------------------------------------------------------------------
bool initSensors(dwSALHandle_t *sal, dwSensorHandle_t *camera, dwImageProperties *cameraImageProperties,
                 dwCameraProperties* cameraProperties, dwContextHandle_t context)
{
    dwStatus result;

    result = dwSAL_initialize(sal, context);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot initialize SAL: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    // create GMSL Camera interface
    dwSensorParams params;
//#ifdef VIBRANTE
    //if (gInputType.compare("camera") == 0) {
        std::string parameterString = "camera-type=" + gArguments.get("camera-type");
        parameterString += ",csi-port=" + gArguments.get("csi-port");
        parameterString += ",slave=" + gArguments.get("slave");
        parameterString += ",serialize=false,output-format=raw,camera-count=4";
        std::string cameraMask[4] = {"0001", "0010", "0100", "1000"};
        uint32_t cameraIdx = std::stoi(gArguments.get("camera-index"));
        if(cameraIdx < 0 || cameraIdx > 3){
            std::cerr << "Error: camera index must be 0, 1, 2 or 3" << std::endl;
            return false;
        }
        parameterString += ",camera-mask=" + cameraMask[cameraIdx];

        params.parameters           = parameterString.c_str();
        params.protocol             = "camera.gmsl";

        result                      = dwSAL_createSensor(camera, params, *sal);
   /* }else
#endif
    {
        std::string parameterString = gArguments.parameterString();
        params.parameters           = parameterString.c_str();
        //params.protocol             = "camera.virtual";
        result                      = dwSAL_createSensor(camera, params, *sal);
    }*/
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot create driver: camera.virtual with params: " << params.parameters << std::endl
                  << "Error: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    dwSensorCamera_getImageProperties(cameraImageProperties, DW_CAMERA_RAW_IMAGE, *camera);
    dwSensorCamera_getSensorProperties(cameraProperties, *camera);

//#ifdef VIBRANTE
    //if(gInputType.compare("camera") == 0){
        if(cameraImageProperties->pxlFormat == DW_IMAGE_RCCB ||
           cameraImageProperties->pxlFormat == DW_IMAGE_BCCR ||
           cameraImageProperties->pxlFormat == DW_IMAGE_CRBC ||
           cameraImageProperties->pxlFormat == DW_IMAGE_CBRC){

           std::cout << "Camera image with " << cameraProperties->resolution.x << "x"
                << cameraProperties->resolution.y << " at " << cameraProperties->framerate << " FPS" << std::endl;

           return true;
        }
        else{
            std::cerr << "Camera is not supported" << std::endl;

            return false;
        }
    //}
//#endif

    return true;
}

//------------------------------------------------------------------------------
void resizeWindowCallback(int width, int height) {
   gScreenRectangle.width = width;
   gScreenRectangle.height = height;
   gScreenRectangle.x = 0;
   gScreenRectangle.y = 0;
   dwRenderer_setRect(gScreenRectangle, gRenderer);
}

bool pubGLImage(dwImageGL* rgbaGLImage)
{   
        std::cout << "receive nvm" << std::endl;
        std::cout << "------NVM IMAGE PROP------" << std::endl;
        std::cout << "width:      " << rgbaGLImage->prop.width     << std::endl;
        std::cout << "heigh:      " << rgbaGLImage->prop.height    << std::endl;
        std::cout << "pxlformat:  " << rgbaGLImage->prop.pxlFormat << std::endl;
        std::cout << "pxlType:    " << rgbaGLImage->prop.pxlType   << std::endl;
        std::cout << "type:       " << rgbaGLImage->prop.type      << std::endl;
    OpenCVConnector cvc;
    unsigned char* myBuffer = new unsigned char[rgbaGLImage->prop.height*rgbaGLImage->prop.width*4];
    //unsigned char* myBuffer = new unsigned char[960*604*4];
    glReadBuffer(GL_BACK);
    glBindBuffer(rgbaGLImage->target, rgbaGLImage->tex);
    glReadPixels(0, 0, rgbaGLImage->prop.width, rgbaGLImage->prop.height, GL_RGBA, GL_UNSIGNED_BYTE, myBuffer);
    //glReadPixels(0, 0, 960, 604, GL_RGBA, GL_UNSIGNED_BYTE, myBuffer);
    cvc.WriteToOpenCV(myBuffer, rgbaGLImage->prop.width, rgbaGLImage->prop.height);
    //cvc.WriteToOpenCV(myBuffer, 960, 604);
}
//------------------------------------------------------------------------------
bool pubGLImage2(dwImageGL* rgbaGLImage)
{

    OpenCVConnector cvc;
        //dwImageStreamer_returnReceivedGL(rgbaGLImage, gCuda2gl);
    std::cout << "return received gl" << std::endl;
    dwStatus result = DW_SUCCESS;


    result = dwImageStreamer_postGL(rgbaGLImage, gl2nvm);
    std::cout << "post gl" << std::endl;
    if (result != DW_SUCCESS) {
        std::cout << "\n Error postGL:" << dwGetStatusName(result) << std::endl;
        return false;
    } else {
        dwImageNvMedia *imageNvMedia = nullptr;
        //dwImageProperties nvmProps = rgbaGLImageProperties;
        //nvmProps.type = DW_IMAGE_NVMEDIA;
        //dwImageNvMedia_create(imageNvMedia, &nvmProps, gSdk);

        result = dwImageStreamer_receiveNvMedia(&imageNvMedia, 60000, gl2nvm);
        std::cout << "receive nvm" << std::endl;
        std::cout << "------NVM IMAGE PROP------" << std::endl;
        std::cout << "width:      " << imageNvMedia->prop.width     << std::endl;
        std::cout << "heigh:      " << imageNvMedia->prop.height    << std::endl;
        std::cout << "pxlformat:  " << imageNvMedia->prop.pxlFormat << std::endl;
        std::cout << "pxlType:    " << imageNvMedia->prop.pxlType   << std::endl;
        std::cout << "type:       " << imageNvMedia->prop.type      << std::endl;
        if ((result == DW_SUCCESS) && imageNvMedia) {
            // Read buffer to cvMat and publish
            NvMediaImageSurfaceMap surfaceMap;
            if(NvMediaImageLock(imageNvMedia->img, NVMEDIA_IMAGE_ACCESS_READ, &surfaceMap) == NVMEDIA_STATUS_OK){
                unsigned char *buffer = (unsigned char*)surfaceMap.surface[0].mapping;
                int s = sizeof((unsigned char*)surfaceMap.surface[0].mapping);
                std::cout<<"buffer length:"<<s<<buffer<<std::endl;
                cvc.WriteToOpenCV((unsigned char*)surfaceMap.surface[0].mapping, imageNvMedia->prop.width, imageNvMedia->prop.height);

                NvMediaImageUnlock(imageNvMedia->img);
                dwImageStreamer_returnReceivedNvMedia(imageNvMedia, gl2nvm);
                std::cout << "returnreceived nvm" << std::endl;
                dwImageStreamer_waitPostedGL(&rgbaGLImage, 33000, gl2nvm);
                //cvc.WriteToOpenCV(buffer, imageNvMedia->prop.width, imageNvMedia->prop.height);
             }else{
                std::cout << "img read fail \n" ;
                return false;
             }
         }
    }
    //dwImageNvMedia *retimg = nullptr;



    return true;

}


bool getNextFrameImages(dwImageCUDA** rcbCudaImageOut, dwImageGL** rgbaGLImageOut, dwCameraFrameHandle_t frameHandle)
{
    dwStatus result = DW_SUCCESS;

    const dwCameraDataLines* dataLines;
    dwImageCPU  *rawImageCPU;
    dwImageCUDA *rawImageCUDA;

#ifdef VIBRANTE
    dwImageNvMedia *rawImageNvMedia = nullptr;

    if (gInputType.compare("camera") == 0) {
        result = dwSensorCamera_getImageNvMedia(&rawImageNvMedia, DW_CAMERA_RAW_IMAGE, frameHandle);
        rawImageNvMedia->prop.pxlFormat = DW_IMAGE_RAW;
    }else
#endif
    {
        result = dwSensorCamera_getImageCPU(&rawImageCPU, DW_CAMERA_RAW_IMAGE, frameHandle);
    }

    if (result != DW_SUCCESS) {
        std::cerr << "Cannot get raw image: " << dwGetStatusName(result) << std::endl;
        return false;
    }
    dwSensorCamera_getDataLines(&dataLines, frameHandle);

    // process
#ifdef VIBRANTE
    if (gInputType.compare("camera") == 0) {
        result = dwImageStreamer_postNvMedia(rawImageNvMedia, gInput2cuda);
        std::cout << "post NvMedia Raw" << std::endl;
    }else
#endif
    {
        result = dwImageStreamer_postCPU(rawImageCPU, gInput2cuda);
    }
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot post raw image: " << dwGetStatusName(result) << std::endl;
        return false;
    }

    // input image was posted, get now CUDA image out of the streamer
    dwImageStreamer_receiveCUDA(&rawImageCUDA, 10000, gInput2cuda);
    {
        // remap CUDA images to the actual data representing the frame
        // the reason is that the image streamer will unfold the embedded lines, if such exists in source image
        dwImageCUDA cudaFrame{};
        {
            dwRect roi;
            dwSensorCamera_getImageROI(&roi, gCameraSensor);
            dwImageCUDA_mapToROI(&cudaFrame, rawImageCUDA, roi);
        }

        // RAW -> RCB
        {
            dwRawPipeline_convertRawToDemosaic(&gRCBImage, &cudaFrame, dataLines, gRawPipeline);
        }

        // return used RAW image, we do not need it anymore, as we now have a copy through the RawPipeline
        dwImageStreamer_returnReceivedCUDA(rawImageCUDA, gInput2cuda);
    }


    // wait
#ifdef VIBRANTE
    if (gInputType.compare("camera") == 0) {
        dwImageStreamer_waitPostedNvMedia(&rawImageNvMedia, 10000, gInput2cuda);
    }else
#endif
    {
        dwImageStreamer_waitPostedCPU(&rawImageCPU, 10000, gInput2cuda);
    }

    // RCB -> RGBA
    {
        dwImageFormatConverter_copyConvertCUDA(&gRGBAImage, &gRCBImage, gConvert2RGBA, g_cudaStream);
    }

    // get GL image
    {
        dwImageStreamer_postCUDA(&gRGBAImage, gCuda2gl);
        dwImageStreamer_receiveGL(rgbaGLImageOut, 10000, gCuda2gl);
    }

    // RCB result
    *rcbCudaImageOut = &gRCBImage;

    return true;
}

//------------------------------------------------------------------------------
void returnNextFrameImages(dwImageCUDA* rcbCudaImageOut, dwImageGL* rgbaGLImage)
{
    (void)rcbCudaImageOut;

    // return GL image
    {
        dwImageStreamer_returnReceivedGL(rgbaGLImage, gCuda2gl);
        dwImageCUDA *returnedFrame;
        dwImageStreamer_waitPostedCUDA(&returnedFrame, 10000, gCuda2gl);
    }
}

//------------------------------------------------------------------------------
void release()
{
    if (gRGBAImage.dptr[0]) dwImageCUDA_destroy(&gRGBAImage);
    if (gRCBImage.dptr[0]) dwImageCUDA_destroy(&gRCBImage);

    if (gConvert2RGBA) {
        dwImageFormatConverter_release(&gConvert2RGBA);
    }
    if (gCuda2gl) {
        dwImageStreamer_release(&gCuda2gl);
    }
    if (gInput2cuda) {
        dwImageStreamer_release(&gInput2cuda);
    }
    if (gl2nvm) {
        dwImageStreamer_release(&gl2nvm);
    }
    if (g_cudaStream) {
        cudaStreamDestroy(g_cudaStream);
    }
    if (gLineBuffer) {
        dwRenderBuffer_release(&gLineBuffer);
    }
    if (gRenderer) {
        dwRenderer_release(&gRenderer);
    }
    if (gCameraSensor) {
        dwSAL_releaseSensor(&gCameraSensor);
    }
    if (gSal) {
        dwSAL_release(&gSal);
    }
    if (gRawPipeline) {
        dwRawPipeline_release(&gRawPipeline);
    }
    dwRelease(&gSdk);
    dwLogger_release();
}
