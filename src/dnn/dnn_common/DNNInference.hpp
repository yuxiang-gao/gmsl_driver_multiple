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

#ifndef SAMPLES_DNNCOMMON_DNNINFERENCE_HPP__
#define SAMPLES_DNNCOMMON_DNNINFERENCE_HPP__

#include <dw/dnn/DNN.h>
#include <dw/dataconditioner/DataConditioner.h>
#include <dw/renderer/Renderer.h>

#include <cuda_runtime.h>
#include <vector>

class DNNInference
{
  public:
    typedef std::pair<dwBox2D,float32_t> BBoxConf;

    DNNInference(dwContextHandle_t context)
        : m_networkHandle(NULL)
        , m_dataConditionerHandle(NULL)
        , m_deviceInputData(NULL)
    {
        m_context             = context;
        m_hostOutputData[0]   = NULL;
        m_hostOutputData[1]   = NULL;
        m_deviceOutputData[0] = NULL;
        m_deviceOutputData[1] = NULL;
        m_bboxConfList.resize(m_maxBBox);
    }

    ~DNNInference();

    inline bool isLoaded()
    {
        return m_success;
    }

    void buildFromTensorRT(const char *const fileName);
    void buildFromCaffe(const char *const caffePrototxtFilename, const char *const caffeWeightsFilename);

    void reset();

    bool inferSingleFrame(std::vector<dwBox2D> *bboxList,
                          const dwImageCUDA *const frame, bool doClustering);

  protected:
    void initializeIO();

    bool interpretOutput(std::vector<dwBox2D> *bbox, const float32_t *outConf,
                         const float32_t *outBBox, const dwRect *const roi, bool doClustering);

    void nonMaxSuppression(std::vector<dwBox2D> *bboxes, float32_t overlapThresh);

    float32_t overlap(const dwBox2D &boxA, const dwBox2D &boxB) const;

    inline int32_t area(const dwBox2D &box) const {
        return box.width * box.height;
    }

  private:
    dwContextHandle_t m_context;
    dwDNNHandle_t m_networkHandle;
    dwDataConditionerHandle_t m_dataConditionerHandle;

    dwBlobSize m_networkInputDimensions;
    dwBlobSize m_networkOutputDimensions[2];

    uint32_t m_networkInputSize;
    uint32_t m_networkOutputSize[2];

    float32_t *m_deviceInputData;
    float32_t *m_deviceOutputData[2];
    float32_t *m_hostOutputData[2];

    uint16_t m_cvgIdx;
    uint16_t m_bboxIdx;

    std::vector<BBoxConf> m_bboxConfList;
    uint32_t m_bboxCount;
    static const uint32_t m_maxBBox = 100U;

    bool m_success;
};

#endif // SAMPLES_DNNCOMMON_DNNINFERENCE_HPP__
