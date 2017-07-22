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

#ifndef SAMPLES_LANEDETECTIONCOMMON_HPP__
#define SAMPLES_LANEDETECTIONCOMMON_HPP__

// SAL
#include <dw/sensors/Sensors.h>
#include <dw/sensors/camera/Camera.h>

// Renderer
#include <dw/renderer/Renderer.h>

// LaneDetector
#include <dw/lanes/LaneDetector.h>

// IMAGE
#include <dw/image/FormatConverter.h>
#include <dw/image/ImageStreamer.h>

#include <string>
#include <common/ProgramArguments.hpp>

namespace laneDetection
{
    // Create sensor and get camera properties
    void createVideoReplay(dwSensorHandle_t &salSensor,
                           uint32_t &cameraWidth,
                           uint32_t &cameraHeight,
                           uint32_t &cameraSiblings,
                           float32_t &cameraFrameRate,
                           dwImageType &imageType,
                           dwSALHandle_t sal,
                           const std::string &videoFName,
                           const std::string &offscreen);

    // Set up renderer
    void setupRenderer(dwRendererHandle_t &renderer, const dwRect &screenRectangle, dwContextHandle_t dwSdk);

    // Set up line buffer
    void setupLineBuffer(dwRenderBufferHandle_t &lineBuffer, unsigned int maxLines, int32_t windowWidth,
                         int32_t windowHeight, dwContextHandle_t dwSdk);


    // Render frame
    void renderCameraTexture(dwImageStreamerHandle_t streamer, dwRendererHandle_t renderer);

    // Run inference using Lane Detector
    void runDetector(dwImageCUDA* frame, float32_t drawScaleLinesX, float32_t drawScaleLinesY,
                     dwLaneDetectorHandle_t laneDetector, dwRenderBufferHandle_t renderBuffer,
                     dwRendererHandle_t renderer);

    // Run whole detection pipeline
#ifdef VIBRANTE
    dwStatus runSingleCameraPipelineNvmedia(dwImageCUDA &frameCUDArgba,
                                            dwImageNvMedia &frameNVMrgba,
                                            float32_t drawScaleLinesX,
                                            float32_t drawScaleLinesY,
                                            dwLaneDetectorHandle_t laneDetector,
                                            dwSensorHandle_t cameraSensor,
                                            dwRenderBufferHandle_t renderBuffer,
                                            dwRendererHandle_t renderer,
                                            dwImageStreamerHandle_t nvm2cudaYUV,
                                            dwImageStreamerHandle_t nvm2glRGBA,
                                            dwImageFormatConverterHandle_t yuv2rgbaNvm,
                                            dwImageFormatConverterHandle_t yuv2rgbaCuda);
#else
    dwStatus runSingleCameraPipelineCuda(dwImageCUDA &frameCUDArgba, float32_t drawScaleLinesX,
                                         float32_t drawScaleLinesY,
                                         dwLaneDetectorHandle_t laneDetector, dwSensorHandle_t cameraSensor,
                                         dwRenderBufferHandle_t renderBuffer,
                                         dwRendererHandle_t renderer,
                                         dwImageStreamerHandle_t cuda2gl,
                                         dwImageFormatConverterHandle_t yuv2rgbaCUDA);
#endif
    bool initLaneNet(dwLaneDetectorHandle_t *gLaneDetector, dwImageProperties *gCameraImageProperties, 
                    cudaStream_t gCudaStream, dwContextHandle_t dwSdk, ProgramArguments &gArguments);
    void release(dwLaneDetectorHandle_t *gLaneDetector);
} // namespace laneDetection

#endif // SAMPLES_LANEDETECTIONCOMMON_HPP__
