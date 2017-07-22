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

#include "SensorIOCuda.hpp"

#include <dw/sensors/camera/Camera.h>

#include <iostream>
#include <thread>

SensorIOCuda::SensorIOCuda(dwContextHandle_t context, cudaStream_t stream,
                           dwSensorHandle_t cameraSensor, int cameraWidth, int cameraHeight)
    : m_cudaStream(stream)
    , m_sensor(cameraSensor)
    , m_yuv2rgba(DW_NULL_HANDLE)
    , m_cuda2gl(DW_NULL_HANDLE)
    , m_frameHandle(DW_NULL_HANDLE)
    , m_frameCUDAyuv(nullptr)
    , m_frameCUDArgba(nullptr)
    , m_frameGlRgba(nullptr)
{
    dwStatus result = DW_FAILURE;

    // format converter for GL display
    dwImageProperties cameraProperties;
    dwSensorCamera_getImageProperties(&cameraProperties, DW_CAMERA_PROCESSED_IMAGE, cameraSensor);
    dwImageProperties displayProperties = cameraProperties;
    displayProperties.pxlFormat = DW_IMAGE_RGBA;
    displayProperties.planeCount = 1;
    result = dwImageFormatConverter_initialize(&m_yuv2rgba, &cameraProperties,
                                               &displayProperties, context);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot create pixel format converter : yuv->rgba" << std::endl;
        exit(1);
    }

    // CUDA RGBA image pool
    for (int i = 0; i < POOL_SIZE; ++i) {
        dwImageCUDA *frameCUDArgba = new dwImageCUDA;
        {
            void *dptr   = nullptr;
            size_t pitch = 0;
            cudaMallocPitch(&dptr, &pitch, cameraWidth * 4, cameraHeight);
            dwImageCUDA_setFromPitch(frameCUDArgba, dptr, cameraWidth,
                                     cameraHeight, pitch, DW_IMAGE_RGBA);
        }

        m_rgbaImagePool.push_back(frameCUDArgba);
    }

    // image API translator
    result = dwImageStreamer_initialize(&m_cuda2gl, &displayProperties, DW_IMAGE_GL, context);
    if (result != DW_SUCCESS) {
        std::cerr << "Cannot init image streamer: " << dwGetStatusName(result) << std::endl;
        exit(1);
    }
    dwImageStreamer_setCUDAStream(m_cudaStream, m_cuda2gl);
}

SensorIOCuda::~SensorIOCuda()
{
    if (m_cuda2gl != DW_NULL_HANDLE)
        dwImageStreamer_release(&m_cuda2gl);
    if (m_yuv2rgba != DW_NULL_HANDLE)
        dwImageFormatConverter_release(&m_yuv2rgba);

    //Release image pool
    for (auto &image : m_rgbaImagePool) {
        cudaFree(image->dptr[0]);
        delete image;
    }
}

dwStatus SensorIOCuda::getFrame()
{
    // try different image types
    dwStatus result = DW_FAILURE;
    result = dwSensorCamera_readFrame(&m_frameHandle, 0, 1000000, m_sensor);
    if (result != DW_SUCCESS) {
        m_frameCUDAyuv = nullptr;
        return result;
    }

    result = dwSensorCamera_getImageCUDA(&m_frameCUDAyuv, DW_CAMERA_PROCESSED_IMAGE, m_frameHandle);
    if (result != DW_SUCCESS) {
        m_frameCUDAyuv = nullptr;
        return result;
    }

    // CUDA copy conversion
    if (m_rgbaImagePool.size() > 0) {
        m_frameCUDArgba = m_rgbaImagePool.back();
        m_rgbaImagePool.pop_back();

        // convert CUDA YUV image to RGBA
        result = dwImageFormatConverter_copyConvertCUDA(m_frameCUDArgba,
                                                        m_frameCUDAyuv,
                                                        m_yuv2rgba,
                                                        m_cudaStream);
        if (result != DW_SUCCESS) {
            std::cerr << "cannot convert frame YUV to RGBA"
                      << dwGetStatusName(result) << std::endl;
        }

        result = dwImageStreamer_postCUDA(m_frameCUDArgba, m_cuda2gl);
        if (result != DW_SUCCESS) {
            std::cerr << "cannot post CUDA RGBA image" << dwGetStatusName(result) << std::endl;
        }
    }

    return DW_SUCCESS;
}

dwImageCUDA *SensorIOCuda::getCudaYuv()
{
    return m_frameCUDAyuv;
}

void SensorIOCuda::releaseCudaYuv()
{
    dwSensorCamera_returnFrame(&m_frameHandle);
}

dwImageCUDA *SensorIOCuda::getCudaRgba()
{
    return m_frameCUDArgba;
}

void SensorIOCuda::releaseCudaRgba()
{
    return;
}

dwImageGL *SensorIOCuda::getGlRgbaFrame()
{
    if (dwImageStreamer_receiveGL(&m_frameGlRgba, 30000, m_cuda2gl) != DW_SUCCESS) {
        std::cerr << "did not received GL frame within 30ms" << std::endl;
        m_frameGlRgba = nullptr;
    }
    return m_frameGlRgba;
}

void SensorIOCuda::releaseGLRgbaFrame()
{
    if (m_frameGlRgba) {
        dwImageStreamer_returnReceivedGL(m_frameGlRgba, m_cuda2gl);
        m_frameGlRgba = nullptr;
    }
}

void SensorIOCuda::releaseFrame()
{
    dwStatus result;
    dwImageCUDA *retimg = nullptr;
    result = dwImageStreamer_waitPostedCUDA(&retimg, 33000, m_cuda2gl);
    if (result == DW_SUCCESS && retimg) {
        m_rgbaImagePool.push_back(retimg);
    }
}
