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

#include "SensorIONvmedia.hpp"

#include <dw/sensors/camera/Camera.h>

#include <iostream>
#include <thread>
#include <cstring>

SensorIONvmedia::SensorIONvmedia(dwContextHandle_t context, cudaStream_t stream,
                         dwSensorHandle_t cameraSensor, int cameraWidth, int cameraHeight)
    : m_cudaStream(stream)
    , m_sensor(cameraSensor)
    , m_yuv2rgbaCUDA(DW_NULL_HANDLE)
    , m_yuv2rgbaNVM(DW_NULL_HANDLE)
    , m_nvm2gl(DW_NULL_HANDLE)
    , m_nvm2cudaYuv(DW_NULL_HANDLE)
    , m_frameHandle(DW_NULL_HANDLE)
    , m_frameNvmYuv(nullptr)
    , m_frameCUDAyuv(nullptr)
    , m_frameGlRgba(nullptr)
{
    dwStatus result = DW_FAILURE;

    dwContext_getNvMediaDevice(&m_nvmedia, context);

    dwImageProperties cameraImageProperties;
    dwSensorCamera_getImageProperties(&cameraImageProperties, DW_CAMERA_PROCESSED_IMAGE, cameraSensor);
    dwImageProperties displayImageProperties = cameraImageProperties;

    displayImageProperties.pxlFormat = DW_IMAGE_RGBA;
    displayImageProperties.planeCount = 1;

    // format converter CUDA
    {
        cameraImageProperties.type = DW_IMAGE_CUDA;     // we convert in CUDA space
        displayImageProperties.type = DW_IMAGE_CUDA;

        result = dwImageFormatConverter_initialize(&m_yuv2rgbaCUDA, &cameraImageProperties, &displayImageProperties, context);
        if (result != DW_SUCCESS) {
            throw std::runtime_error("Cannot create pixel format converter : yuv->rgba");
        }
    }

    // format converter NvMedia
    {
        cameraImageProperties.type = DW_IMAGE_NVMEDIA;
        displayImageProperties.type = DW_IMAGE_NVMEDIA;

        result = dwImageFormatConverter_initialize(&m_yuv2rgbaNVM, &cameraImageProperties, &displayImageProperties, context);
        if (result != DW_SUCCESS) {
            throw std::runtime_error("Cannot create pixel format converter : yuv->rgba");
        }
    }

    // image API translator
    displayImageProperties.type = DW_IMAGE_NVMEDIA;
    result = dwImageStreamer_initialize(&m_nvm2gl, &displayImageProperties, DW_IMAGE_GL, context);
    if (result != DW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot init image streamer nvm-gl: ") + dwGetStatusName(result));
    }

    cameraImageProperties.type = DW_IMAGE_NVMEDIA;
    result = dwImageStreamer_initialize(&m_nvm2cudaYuv, &cameraImageProperties, DW_IMAGE_CUDA, context);
    if (result != DW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot init image streamer nvm-cuda: ") + dwGetStatusName(result));
    }

    /*result = dwImageStreamer_initialize(&m_nvm2cudaRgba, DW_IMAGE_NVMEDIA, DW_IMAGE_CUDA,
                                        DW_IMAGE_RGBA, DW_TYPE_UINT8,
                                        cameraWidth, cameraHeight, context);*/
    if (result != DW_SUCCESS) {
        throw std::runtime_error(std::string("Cannot init image streamer nvm-cuda: ") + dwGetStatusName(result));
    }
    dwImageStreamer_setCUDAStream(m_cudaStream, m_nvm2cudaYuv);
    //dwImageStreamer_setCUDAStream(m_cudaStream, m_nvm2cudaRgba);
    //dwImageStreamer_setCUDAStream(m_cudaStream, m_nvm2gl);

    // RGBA image pool for conversion from YUV camera output
    for (int i = 0; i < POOL_SIZE; ++i) {
        NvMediaImageAdvancedConfig advConfig;
        std::memset(&advConfig, 0, sizeof(advConfig));
        dwImageNvMedia *rgbaImage = new dwImageNvMedia();
        NvMediaImage *rgbaNvMediaImage = NvMediaImageCreate(m_nvmedia,
                                              NvMediaSurfaceType_Image_RGBA,
                                              NVMEDIA_IMAGE_CLASS_SINGLE_IMAGE, 1,
                                              cameraWidth, cameraHeight,
                                              NVMEDIA_IMAGE_ATTRIBUTE_UNMAPPED,
                                              &advConfig);
        dwImageNvMedia_setFromImage(rgbaImage, rgbaNvMediaImage);

        m_rgbaImagePool.push_back(rgbaImage);
    }

    // Setup rgba cuda frame
    m_frameCUDArgba = new dwImageCUDA();
    {
        void *dptr = nullptr;
        size_t pitch = 0;
        cudaMallocPitch(&dptr, &pitch, cameraWidth * 4, cameraHeight);
        dwImageCUDA_setFromPitch(m_frameCUDArgba, dptr, cameraWidth, cameraHeight, pitch, DW_IMAGE_RGBA);
    }
}

SensorIONvmedia::~SensorIONvmedia()
{
    if (m_nvm2gl != DW_NULL_HANDLE)
        dwImageStreamer_release(&m_nvm2gl);
    if (m_nvm2cudaYuv != DW_NULL_HANDLE)
        dwImageStreamer_release(&m_nvm2cudaYuv);
    if (m_yuv2rgbaCUDA != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&m_yuv2rgbaCUDA);
    if (m_yuv2rgbaNVM != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&m_yuv2rgbaNVM);
    if (m_frameCUDArgba != nullptr) {
        cudaFree(m_frameCUDArgba->dptr[0]);
        delete m_frameCUDArgba;
    }

    //Release image pool
    for (auto &image : m_rgbaImagePool) {
        NvMediaImageDestroy(image->img);
        delete image;
    }
}

dwStatus SensorIONvmedia::getFrame()
{
    // try different image types
    dwStatus result = DW_FAILURE;
    result = dwSensorCamera_readFrame(&m_frameHandle, 0, 1000000, m_sensor);
    if (result != DW_SUCCESS)
    {
        m_frameHandle = nullptr;
        return result;
    }

    result = dwSensorCamera_getImageNvMedia(&m_frameNvmYuv, DW_CAMERA_PROCESSED_IMAGE, m_frameHandle);
    if (result != DW_SUCCESS)
    {
        m_frameNvmYuv = nullptr;
        return result;
    }

    // RGBA version of the frame
    if (m_rgbaImagePool.size() > 0) {
        m_frameNvmRgba = m_rgbaImagePool.back();
        m_rgbaImagePool.pop_back();

        // RGBA version of the frame
        result = dwImageFormatConverter_copyConvertNvMedia(m_frameNvmRgba, m_frameNvmYuv, m_yuv2rgbaNVM);
        if (result != DW_SUCCESS) {
            std::cerr << "Cannot convert frame: " << dwGetStatusName(result) << std::endl;
        }
    }

    // Send the YUV frame
    result = dwImageStreamer_postNvMedia(m_frameNvmYuv, m_nvm2cudaYuv);
    if (result != DW_SUCCESS) {
        std::cerr << "cannot post NvMedia YUV frame " << dwGetStatusName(result) << std::endl;
    }

    // Send the RGBA frame
    result = dwImageStreamer_postNvMedia(m_frameNvmRgba, m_nvm2gl);
    if (result != DW_SUCCESS) {
        std::cerr << "cannot post NvMedia RGBA frame " << dwGetStatusName(result) << std::endl;
    }

    return DW_SUCCESS;
}

dwImageCUDA *SensorIONvmedia::getCudaYuv()
{
    // receive YUV frame if not already received
    if (m_frameCUDAyuv == nullptr)
    {
        dwStatus result = dwImageStreamer_receiveCUDA(&m_frameCUDAyuv, 60000, m_nvm2cudaYuv);

        if (result != DW_SUCCESS || m_frameCUDAyuv == 0) {
            std::cerr << "did not received CUDA YUV frame within 60ms" << std::endl;
            m_frameCUDAyuv = nullptr;
        }
    }

    return m_frameCUDAyuv;
}

void SensorIONvmedia::releaseCudaYuv()
{
    if (m_frameCUDAyuv) {
        dwImageStreamer_returnReceivedCUDA(m_frameCUDAyuv, m_nvm2cudaYuv);
        m_frameCUDAyuv = nullptr;
    }
}

dwImageCUDA *SensorIONvmedia::getCudaRgba()
{
    // we need YUV frame, before we can get RGBA out of it
    if (!m_frameCUDAyuv) m_frameCUDAyuv = getCudaYuv();

    // RGBA version of the frame
    dwStatus result = dwImageFormatConverter_copyConvertCUDA(m_frameCUDArgba, m_frameCUDAyuv, m_yuv2rgbaCUDA, m_cudaStream);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot convert frame CUDA YUV->RGBA: " << dwGetStatusName(result) << std::endl;
    }

    return m_frameCUDArgba;
}

void SensorIONvmedia::releaseCudaRgba()
{
}

dwImageGL *SensorIONvmedia::getGlRgbaFrame()
{
    if (!m_frameGlRgba)
    {
        dwStatus result = dwImageStreamer_receiveGL(&m_frameGlRgba, 60000, m_nvm2gl);

        if (result != DW_SUCCESS) {
            std::cerr << "did not received GL frame within 30ms" << std::endl;
            m_frameGlRgba = nullptr;
        }
    }

    return m_frameGlRgba;
}

void SensorIONvmedia::releaseGLRgbaFrame()
{
    if (m_frameGlRgba) {
        dwImageStreamer_returnReceivedGL(m_frameGlRgba, m_nvm2gl);
        m_frameGlRgba = nullptr;
    }
}

void SensorIONvmedia::releaseFrame()
{
    dwTime_t WAIT_TIME = 33000;

    {
        dwImageNvMedia *retimg = nullptr;
        dwStatus result = dwImageStreamer_waitPostedNvMedia(&retimg, WAIT_TIME, m_nvm2cudaYuv);
        if (result == DW_TIME_OUT) {
            std::cerr << "Warning: waitPosted (nvmedia cuda) timed out\n";
        } else if (result != DW_SUCCESS) {
            std::cout << " ERROR waitPostedNvMedia CUDA: " << dwGetStatusName(result) << std::endl;
        }
    }

    // any image returned back, we put back into the pool
    {
        dwImageNvMedia *retimg = nullptr;
        dwStatus result = dwImageStreamer_waitPostedNvMedia(&retimg, WAIT_TIME, m_nvm2gl);
        if (result == DW_TIME_OUT) {
            std::cerr << "Warning: waitPosted (nvmedia gl) timed out\n";
        } else if (result != DW_SUCCESS) {
            std::cout << " ERROR waitPostedNvMedia GL: " << dwGetStatusName(result) << std::endl;
        } else {
            m_rgbaImagePool.push_back(retimg);
        }
    }

    if (m_frameHandle)
    {
        dwSensorCamera_returnFrame(&m_frameHandle);
    }
}
