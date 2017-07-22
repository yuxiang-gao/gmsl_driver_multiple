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

#include "DNNInference.hpp"
#include <iostream>
#include <algorithm>
#include <string.h>

/// Threshold for accepting bounding boxes as detections
#define COVG_THRESHOLD 0.6
/// Number of simultaneous frames (i.e. number of cameras)
#define INFERENCE_N_SIMFRAMES 1
/// Output layers of the network
#define COVERAGE_LAYER  "coverage"
#define BBOX_LAYER      "bboxes"

DNNInference::~DNNInference()
{
    if (m_dataConditionerHandle) {
        dwDataConditioner_release(&m_dataConditionerHandle);
    }

    if (m_networkHandle) {
        dwDNN_release(&m_networkHandle);
    }
    if (m_hostOutputData[0]) {
        delete[] m_hostOutputData[0];
    }
    if (m_hostOutputData[1]) {
        delete[] m_hostOutputData[1];
    }
    if (m_deviceOutputData[0]) {
        cudaFree(m_deviceOutputData[0]);
    }
    if (m_deviceOutputData[1]) {
        cudaFree(m_deviceOutputData[1]);
    }
    if (m_deviceInputData) {
        cudaFree(m_deviceInputData);
    }
}

void DNNInference::buildFromTensorRT(const char *const fileName)
{
    dwStatus status;

    // Load network
    status = dwDNN_initializeTensorRTFromFile(&m_networkHandle, m_context, fileName);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot initialize TensorRT Network: " << dwGetStatusName(status) << std::endl;
        m_success = false;
        return ;
    }

    initializeIO();
}

void DNNInference::buildFromCaffe(const char *const caffePrototxtFilename,
                                  const char *const caffeWeightsFilename)
{
    // Initialize DNN Context
    dwStatus status;

    // Load network
    status = dwDNN_initializeCaffeFromFile(&m_networkHandle, m_context, caffePrototxtFilename,
                                           caffeWeightsFilename, INFERENCE_N_SIMFRAMES);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot initialize Caffe Network: " << dwGetStatusName(status) << std::endl;
        m_success = false;
        return ;
    }

    initializeIO();
}

void DNNInference::reset()
{
    dwDNN_reset(m_networkHandle);
    m_bboxCount = 0;
}

bool DNNInference::inferSingleFrame(std::vector<dwBox2D> *bboxList,
                                    const dwImageCUDA *const frame, bool doClustering)
{    
    dwRect roi;

    roi.x = (frame->prop.width - m_networkInputDimensions.width) / 2;
    roi.y = (frame->prop.height - m_networkInputDimensions.height) / 2;
    roi.width = m_networkInputDimensions.width;
    roi.height = m_networkInputDimensions.height;    

    dwDataConditioner_prepareData(m_deviceInputData, &frame, 1, &roi, cudaAddressModeClamp, m_dataConditionerHandle);
   
    dwDNN_infer(&m_deviceOutputData[0], &m_deviceInputData, m_networkHandle);

    // Copy output data back
    cudaError_t cudaError =
        cudaMemcpy(m_hostOutputData[0], m_deviceOutputData[0],
                   sizeof(float32_t) * m_networkOutputSize[0],
                   cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        std::cerr << "Cannot copy back coverage memory: " << cudaGetErrorName(cudaError)
                  << std::endl;
        return false;
    }
    cudaError = cudaMemcpy(m_hostOutputData[1], m_deviceOutputData[1],
                           sizeof(float32_t) * m_networkOutputSize[1],
                           cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
        std::cerr << "Cannot copy back bounding box memory: " << cudaGetErrorName(cudaError)
                  << std::endl;
        return false;
    }

    float32_t *outConf = m_hostOutputData[m_cvgIdx];
    float32_t *outBBox = m_hostOutputData[m_bboxIdx];
    bool status = interpretOutput(bboxList, outConf, outBBox, &roi, doClustering);
    if (!status) {
        std::cerr << "Network output parsing failed." << std::endl;
        return false;
    }

    return true;
}

void DNNInference::initializeIO()
{
    dwStatus status;

    // Setup dimensions
    status = dwDNN_getInputSize(&m_networkInputDimensions, 0, m_networkHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot retrieve input size from network: " << dwGetStatusName(status)
                  << std::endl;
        m_success = false;
        return ;
    }
    status = dwDNN_getOutputSize(&m_networkOutputDimensions[0], 0, m_networkHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot retrieve output size from network: " << dwGetStatusName(status)
                  << std::endl;
        m_success = false;
        return ;
    }
    status = dwDNN_getOutputSize(&m_networkOutputDimensions[1], 1, m_networkHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot retrieve output size from network: " << dwGetStatusName(status)
                  << std::endl;
        m_success = false;
        return ;
    }
    m_networkInputSize = m_networkInputDimensions.channels * m_networkInputDimensions.height *
                         m_networkInputDimensions.width;
    m_networkOutputSize[0] = m_networkOutputDimensions[0].channels *
                             m_networkOutputDimensions[0].height *
                             m_networkOutputDimensions[0].width;
    m_networkOutputSize[1] = m_networkOutputDimensions[1].channels *
                             m_networkOutputDimensions[1].height *
                             m_networkOutputDimensions[1].width;

    status = dwDNN_getOutputIndex(&m_cvgIdx, COVERAGE_LAYER, m_networkHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot find output blob with name "
                  << COVERAGE_LAYER << std::endl;
        m_success = false;
        return ;
    }
    status = dwDNN_getOutputIndex(&m_bboxIdx, BBOX_LAYER, m_networkHandle);
    if (status != DW_SUCCESS) {
        std::cerr << "Cannot find output blob with name "
                  << BBOX_LAYER << std::endl;
        m_success = false;
        return ;
    }

    // Allocate GPU memory based on the dimensions
    cudaError_t cudaError =
        cudaMalloc((void **)&m_deviceInputData,
                   sizeof(float32_t) * INFERENCE_N_SIMFRAMES * m_networkInputSize);
    if (cudaError != cudaSuccess) {
        std::cerr << "Cannot allocate input memory for network: " << cudaGetErrorName(cudaError)
                  << std::endl;
        m_success = false;
        return ;
    }
    cudaError = cudaMalloc((void **)&m_deviceOutputData[0],
                           sizeof(float32_t) * INFERENCE_N_SIMFRAMES * m_networkOutputSize[0]);
    if (cudaError != cudaSuccess) {
        std::cerr << "Cannot allocate coverage memory for network: " << cudaGetErrorName(cudaError)
                  << std::endl;
        m_success = false;
        return ;
    }
    cudaError = cudaMalloc((void **)&m_deviceOutputData[1],
                           sizeof(float32_t) * INFERENCE_N_SIMFRAMES * m_networkOutputSize[1]);
    if (cudaError != cudaSuccess) {
        std::cerr << "Cannot allocate bounding box memory for network: "
                  << cudaGetErrorName(cudaError) << std::endl;
        m_success = false;
        return ;
    }

    m_hostOutputData[0] = new float32_t[INFERENCE_N_SIMFRAMES * m_networkOutputSize[0]];
    m_hostOutputData[1] = new float32_t[INFERENCE_N_SIMFRAMES * m_networkOutputSize[1]];

    dwDataConditionerParams dataConditionerParams;
    dwDataConditioner_initParams(&dataConditionerParams);

    dataConditionerParams.splitPlanes = true;

    // Initialize data conditioner
    status = dwDataConditioner_initialize(&m_dataConditionerHandle, &m_networkInputDimensions,
                                          &dataConditionerParams, 0, m_context);

    if (status != DW_SUCCESS) {
        std::cerr << "Cannot initialize DataConditioner: " << dwGetStatusName(status) << std::endl;
        m_success = false;
        return ;
    }

    m_success = true;
}

bool DNNInference::interpretOutput(std::vector<dwBox2D> *bboxes,
                                   const float32_t *outConf, const float32_t *outBBox,
				                   const dwRect *const roi,
                                   bool doClustering)
{
    uint16_t gridH    = m_networkOutputDimensions[0].height;
    uint16_t gridW    = m_networkOutputDimensions[0].width;
    uint16_t cellSize = m_networkInputDimensions.height / gridH;
    uint32_t gridSize = gridH * gridW;
    m_bboxCount = 0U;

    for (uint16_t gridY = 0U; gridY < gridH; ++gridY) {
        const float32_t *outConfRow = &outConf[gridY * gridW];
        for (uint16_t gridX = 0U; gridX < gridW; ++gridX) {
            float32_t conf = outConfRow[gridX];
            if ((conf > COVG_THRESHOLD) && (m_bboxCount < m_maxBBox)) {
                // This is a detection!
                float32_t imageX = (float32_t)gridX * (float32_t)cellSize;
                float32_t imageY = (float32_t)gridY * (float32_t)cellSize;
                uint32_t offset  = gridY * gridW + gridX;

                float32_t boxX1;
                float32_t boxY1;
                float32_t boxX2;
                float32_t boxY2;

                dwDataConditioner_outputPositionToInput(&boxX1, &boxY1, outBBox[offset] + imageX,
                                                        outBBox[gridSize + offset] + imageY, roi,
                                                        m_dataConditionerHandle);
                dwDataConditioner_outputPositionToInput(&boxX2, &boxY2,
                                                        outBBox[gridSize * 2 + offset] + imageX,
                                                        outBBox[gridSize * 3 + offset] + imageY, roi,
                                                        m_dataConditionerHandle);
                dwBox2D bbox;
                bbox.width  = static_cast<int32_t>(std::round(boxX2 - boxX1));
                bbox.height = static_cast<int32_t>(std::round(boxY2 - boxY1));
                bbox.x = static_cast<int32_t>(std::round(boxX1));
                bbox.y = static_cast<int32_t>(std::round(boxY1));

                m_bboxConfList[m_bboxCount++] = std::make_pair(bbox, conf);
            }
        }
    }

    // Merge overlapping bounding boxes by non-maximum suppression
    if (doClustering) {
        nonMaxSuppression(bboxes, 0.5f);
    } else {
        for (uint32_t bboxIdx = 0U; bboxIdx < m_bboxCount; ++bboxIdx) {
            bboxes->push_back(m_bboxConfList[bboxIdx].first);
        }
    }

    return true;
}

void DNNInference::nonMaxSuppression(std::vector<dwBox2D> *bboxes, float32_t overlapThresh)
{
    std::vector<BBoxConf>::iterator bboxConfEnd = m_bboxConfList.begin() + m_bboxCount;
    std::sort(m_bboxConfList.begin(), bboxConfEnd,
              [](std::pair<dwBox2D, float32_t> elem1,
                 std::pair<dwBox2D, float32_t> elem2) -> bool {
        return elem1.second < elem2.second;
    });

    for (auto objItr = m_bboxConfList.begin(); objItr != bboxConfEnd; ++objItr) {
        const auto& objA = *objItr;
        bool keepObj = true;
        for (auto next = objItr + 1; next != bboxConfEnd; ++next) {
            const auto& objB = *next;
            const dwBox2D &objABox = objA.first;
            const dwBox2D &objBBox = objB.first;

            int32_t objARight = objABox.x + objABox.width;
            int32_t objABottom = objABox.y + objABox.height;
            int32_t objBRight = objBBox.x + objBBox.width;
            int32_t objBBottom = objBBox.y + objBBox.height;

            float32_t ovl =  overlap(objABox, objBBox)
                / std::min( area(objABox), area(objBBox) );

            bool is_new_box_inside_old_box = (objBBox.x > objABox.x) &&
                                             (objBRight < objARight) &&
                                             (objBBox.y > objABox.y) &&
                                             (objBBottom < objABottom);

            if (ovl > overlapThresh || is_new_box_inside_old_box)
                keepObj = false;
        }
        if (keepObj)
            bboxes->push_back(objA.first);
    }
}

float32_t DNNInference::overlap(const dwBox2D &boxA,
                                const dwBox2D &boxB) const
{

    int32_t overlapWidth = std::min(boxA.x + boxA.width,
                                    boxB.x + boxB.width) -
                           std::max(boxA.x, boxB.x);
    int32_t overlapHeight = std::min(boxA.y + boxA.height,
                                     boxB.y + boxB.height) -
                            std::max(boxA.y, boxB.y);

    return (overlapWidth < 0 || overlapHeight < 0) ? 0.0f : (float32_t)(overlapWidth * overlapHeight);
}

