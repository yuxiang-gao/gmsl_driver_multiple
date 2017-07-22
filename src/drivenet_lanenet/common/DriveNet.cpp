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

#include <stdexcept>
#include <iostream>
#include <cmath>

#include "DriveNet.hpp"

//------------------------------------------------------------------------------
DriveNet::DriveNet(dwContextHandle_t ctx)
    : sdk(ctx)
{
}

//------------------------------------------------------------------------------
DriveNet::~DriveNet()
{
    for (uint32_t classIdx = 0U; classIdx < objectClustering.size(); ++classIdx) {
        if (objectClustering[classIdx]) {
            dwObjectClustering_release(&objectClustering[classIdx]);
        }
    }

    releaseTracker();
    releaseDetector();
}

//------------------------------------------------------------------------------
void DriveNet::resetTracker()
{
    dwObjectTracker_reset(objectTracker);
}

//------------------------------------------------------------------------------
void DriveNet::resetDetector()
{
    for (uint32_t classIdx = 0U; classIdx < classLabels.size(); ++classIdx) {
        numClusters[classIdx] = 0U;
        numMergedObjects[classIdx] = 0U;
        numProposals[classIdx] = 0U;
        numTrackedObjects[classIdx] = 0U;
    }
}

//------------------------------------------------------------------------------
bool DriveNet::initTracker(const dwImageProperties& rcbProperties, cudaStream_t stream)
{
    // get all classes which DriveNet provide
    uint32_t numClasses = DW_DRIVENET_NUM_CLASSES;

    // initialize ObjectTracker - it will be required to track detected instances over multiple frames
    // for better understanding how ObjectTracker works see sample_object_tracker

    dwObjectFeatureTrackerParams featureTrackingParams;
    dwObjectTrackerParams objectTrackingParams[DW_OBJECT_MAX_CLASSES];
    dwObjectTracker_initDefaultParams(&featureTrackingParams, objectTrackingParams, numClasses);
    featureTrackingParams.maxFeatureCount = 8000;
    featureTrackingParams.detectorScoreThreshold = 0.0001f;
    featureTrackingParams.iterationsLK = 10;
    featureTrackingParams.windowSizeLK = 8;

    for (uint32_t classIdx = 0U; classIdx < numClasses; ++classIdx) {
        objectTrackingParams[classIdx].confRateTrackMax = 0.05f;
        objectTrackingParams[classIdx].confRateTrackMin = 0.01f;
        objectTrackingParams[classIdx].confRateDetect = 0.5f;
        objectTrackingParams[classIdx].confThreshDiscard = 0.0f;
        objectTrackingParams[classIdx].maxFeatureCountPerBox = 200;
    }

    dwStatus result = dwObjectTracker_initialize(&objectTracker, sdk, &rcbProperties, &featureTrackingParams,
                                                  objectTrackingParams, numClasses);
    if (result != DW_SUCCESS) {
        std::cerr << "Failed to create object tracker" << std::endl;
        return false;
    }

    dwObjectTracker_setCUDAStream(stream, objectTracker);

    return true;
}

//------------------------------------------------------------------------------
bool DriveNet::initDetector(const dwImageProperties& rcbProperties, cudaStream_t cudaStream)
{
    // select a valuable parameters for best DriveNet results
    dwDriveNet_initDefaultParams(&drivenetParams);
    drivenetParams.maxNumImages         = 2U;       // we will provide 2 images, i.e. full resolution and selected ROI in the center
    drivenetParams.enableFuseObjects    = DW_TRUE;  // we want result from two both image to be combined
    drivenetParams.imageHeight          = rcbProperties.height;
    drivenetParams.imageWidth           = rcbProperties.width;
    drivenetParams.maxProposalsPerClass = maxProposalsPerClass;

    // if we have compute capability >= 6.2 (iGPU), then use FP16 model, otherwise FP32
    // note that the quality of FP16 based detection might be worse then of FP32
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        if (prop.major > 6 || (prop.major == 6 && prop.minor >= 2)) {
            drivenetParams.networkType          = DW_DRIVENET_TYPE_FP16;
        }else{
            drivenetParams.networkType          = DW_DRIVENET_TYPE_FP32;
        }
    }

    // Initialize DriveNet with parameters
    dwStatus result = DW_FAILURE;
    result = dwDriveNet_initialize(&driveNet, sdk, &drivenetParams);
    if (result != DW_SUCCESS) {
        std::cerr << "Failed to create DriveNet" << std::endl;
        return false;
    }
    dwDriveNet_setCUDAStream(cudaStream, driveNet);

    // since our input images might have a different aspect ratio as the input to drivenet
    // we setup the ROI such that the crop happens from the top of the image
    float32_t aspectRatio = 1.0f;
    {
        dwBlobSize inputBlob;
        dwDriveNet_getInputBlobsize(&inputBlob, driveNet);

        aspectRatio = static_cast<float32_t>(inputBlob.height) / static_cast<float32_t>(inputBlob.width);
    }

    // 1st image is a full resolution image as it comes out from the RawPipeline (cropped to DriveNet aspect ratio)
    dwRect fullROI;
    {
        fullROI = {0, 0, static_cast<int32_t>(rcbProperties.width),
                         static_cast<int32_t>(rcbProperties.width * aspectRatio)};
        dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
                                              0.0f, 1.0f, 0.0f,
                                              0.0f, 0.0f, 1.0f}};

        dwDriveNet_setROI(0, &fullROI, &transformation, driveNet);
    }

    // 2nd image is a cropped out region within the 1/4-3/4 of the original image in the center
    {
        dwRect ROI = {fullROI.width/4, fullROI.height/4,
                      fullROI.width/2, fullROI.height/2};
        dwTransformation2D transformation = {{1.0f, 0.0f, 0.0f,
                                              0.0f, 1.0f, 0.0f,
                                              0.0f, 0.0f, 1.0f}};

        dwDriveNet_setROI(1, &ROI, &transformation, driveNet);
    }

    // fill out member structure according to the ROIs
    dwDriveNet_getROI(&drivenetParams.ROIs[0], &drivenetParams.transformations[0], 0, driveNet);
    dwDriveNet_getROI(&drivenetParams.ROIs[1], &drivenetParams.transformations[1], 1, driveNet);

    // get all classes which DriveNet provide
    uint32_t numClasses = DW_DRIVENET_NUM_CLASSES;

    // Get which label name for each class id
    classLabels.resize(numClasses);
    for (uint32_t classIdx = 0U; classIdx < numClasses; ++classIdx) {
        const char *classLabel;
        dwDriveNet_getClassLabel(&classLabel, static_cast<dwDriveNetClass>(classIdx), driveNet);
        classLabels[classIdx] = classLabel;
    }

    // Initialize arrays for the pipeline
    postClusteringThresholds.resize(numClasses, 0);
    objectClustering.resize(numClasses, DW_NULL_HANDLE);
    objectClusters.resize(numClasses, std::vector<dwObject>(maxProposalsPerClass));
    objectClustersFiltered.resize(numClasses, std::vector<dwObject>(maxProposalsPerClass));
    objectProposals.resize(numClasses, std::vector<dwObject>(maxProposalsPerClass));
    objectsTracked.resize(numClasses, std::vector<dwObject>(maxProposalsPerClass));
    objectsMerged.resize(numClasses, std::vector<dwObject>(maxProposalsPerClass));

    numTrackedObjects.resize(numClasses, 0);
    numMergedObjects.resize(numClasses, 0);
    numClusters.resize(numClasses, 0);
    numProposals.resize(numClasses, 0);

    dnnBoxList.resize(numClasses);

    // initialize object clustering modules for each of the detected object class
    // the selected set of parameters are chosen to achieve best results with the current implementation
    dwObjectClusteringParams defaultClusteringParams{};
    {
        defaultClusteringParams.algorithm = DW_CLUSTERING_NVDRIVENET;
        defaultClusteringParams.enableATHRFilter = false;
        defaultClusteringParams.maxClusters = maxClustersPerClass;
        defaultClusteringParams.maxProposals = maxProposalsPerClass;
        defaultClusteringParams.minBoxes = 1;
    }

    // Car
    {
        dwObjectClusteringParams clusteringParams = defaultClusteringParams;
        clusteringParams.epsilon = 0.31304391937154946;
        clusteringParams.minSumOfConfidences = 0.0024699802059650056;
        result = dwObjectClustering_initialize(&objectClustering[DW_DRIVENET_CLASS_CAR], sdk, &clusteringParams);
        if (result != DW_SUCCESS) {
            std::cerr << "Failed to create object cluster" << std::endl;
            return false;
        }
        postClusteringThresholds[DW_DRIVENET_CLASS_CAR] = 0.95;
    }

    // Traffic sign
    {
        dwObjectClusteringParams clusteringParams = defaultClusteringParams;
        clusteringParams.epsilon = 0.1609227498777174;
        clusteringParams.minSumOfConfidences = 0.00778513476090107;
        result = dwObjectClustering_initialize(&objectClustering[DW_DRIVENET_CLASS_TRAFFIC_SIGN], sdk, &clusteringParams);
        if (result != DW_SUCCESS) {
            std::cerr << "Failed to create object cluster" << std::endl;
            return false;
        }
        postClusteringThresholds[DW_DRIVENET_CLASS_TRAFFIC_SIGN] = 0.85f;
    }

    // Bicycle
    {
        dwObjectClusteringParams clusteringParams = defaultClusteringParams;
        clusteringParams.epsilon = 0.2576750586181174;
        clusteringParams.minSumOfConfidences = 0.007776450489658193;
        result = dwObjectClustering_initialize(&objectClustering[DW_DRIVENET_CLASS_BICYCLE], sdk, &clusteringParams);
        if (result != DW_SUCCESS) {
            std::cerr << "Failed to create object cluster" << std::endl;
            return false;
        }
        postClusteringThresholds[DW_DRIVENET_CLASS_BICYCLE] = 0.975f;
    }

    // Truck
    {
        dwObjectClusteringParams clusteringParams = defaultClusteringParams;
        clusteringParams.epsilon = 0.31304391937154946;
        clusteringParams.minSumOfConfidences = 0.0024699802059650056;
        result = dwObjectClustering_initialize(&objectClustering[DW_DRIVENET_CLASS_TRUCK], sdk, &clusteringParams);
        if (result != DW_SUCCESS) {
            std::cerr << "Failed to create object cluster" << std::endl;
            return false;
        }
        postClusteringThresholds[DW_DRIVENET_CLASS_TRUCK] = 0.975f;
    }

    // Person
    {
        dwObjectClusteringParams clusteringParams = defaultClusteringParams;
        clusteringParams.epsilon = 0.2576750586181174;
        clusteringParams.minSumOfConfidences = 0.007776450489658193;
        result = dwObjectClustering_initialize(&objectClustering[DW_DRIVENET_CLASS_PEDESTRIAN], sdk, &clusteringParams);
        if (result != DW_SUCCESS) {
            std::cerr << "Failed to create object cluster" << std::endl;
            return false;
        }
        postClusteringThresholds[DW_DRIVENET_CLASS_PEDESTRIAN] = 0.975f;
    }

    return true;
}

//------------------------------------------------------------------------------
void DriveNet::releaseDetector()
{
    if (driveNet) {
        dwDriveNet_release(&driveNet);
        driveNet = DW_NULL_HANDLE;
    }
}

//------------------------------------------------------------------------------
void DriveNet::releaseTracker()
{
    if (objectTracker) {
        dwObjectTracker_release(&objectTracker);
        objectTracker = DW_NULL_HANDLE;
    }
}

//------------------------------------------------------------------------------
void DriveNet::filterClusters(dwObject out[], size_t *nOut, const dwObject in[], size_t nIn, float32_t threshold)
{
    uint32_t outIdx = 0U;
    for (uint32_t objIdx = 0U; objIdx < nIn; ++objIdx) {
        if (in[objIdx].totalConfidence >= (threshold)) {
            out[outIdx++] = in[objIdx];
        }
    }

    *nOut = outIdx;
}

//------------------------------------------------------------------------------
const std::vector<std::pair<dwBox2D,std::string>>& DriveNet::getResult(uint32_t classIdx)
{
    return dnnBoxList[classIdx];
}

//------------------------------------------------------------------------------
dwStatus DriveNet::inferDetectorAsync(const dwImageCUDA* rcbImage)
{
    // we feed two images to the DriveNet module, the first one will have full ROI
    // the second one, is the same image, however with an ROI cropped in the center
    const dwImageCUDA* rcbImagePtr[2] = {rcbImage, rcbImage};
    return dwDriveNet_inferDeviceAsync(rcbImagePtr, 2U, driveNet);
}

//------------------------------------------------------------------------------
dwStatus DriveNet::inferTrackerAsync(const dwImageCUDA* rcbImage)
{
    // track feature points on the rcb image
    return dwObjectTracker_featureTrackDeviceAsync(rcbImage, objectTracker);
}

//------------------------------------------------------------------------------
void DriveNet::processResults()
{
    dwDriveNet_interpretHost(2U, driveNet);

    // for each detection class, we do
    for (uint32_t classIdx = 0U; classIdx < classLabels.size(); ++classIdx) {

        // track detection from last frame given new feature tracker responses
        dwObjectTracker_boxTrackHost(objectsTracked[classIdx].data(), &numTrackedObjects[classIdx],
                                     objectsMerged[classIdx].data(), numMergedObjects[classIdx],
                                     classIdx, objectTracker);

        // extract new detections from DriveNet
        dwDriveNet_getDetectedObjects(objectProposals[classIdx].data(), &numProposals[classIdx],
                                      0U, classIdx, driveNet);

        // cluster the new detections to find actual clustered response
        dwObjectClustering_cluster(objectClusters[classIdx].data(), &numClusters[classIdx],
                                   objectProposals[classIdx].data(), numProposals[classIdx],
                                   objectClustering[classIdx]);

        // filter all clusters which response is lower than a predefined threshold
        size_t numFilteredClusters = 0;
        filterClusters(objectClustersFiltered[classIdx].data(), &numFilteredClusters,
                       objectClusters[classIdx].data(), numClusters[classIdx],
                       postClusteringThresholds[classIdx]);

        // the new response should be at new location as detected by DriveNet
        // in addition we have previously tracked response from last time
        // we hence now merge both detections to find the actual response for the current frame
        const dwObject *toBeMerged[2] = {objectsTracked[classIdx].data(),
                                         objectClustersFiltered[classIdx].data()};
        const size_t sizes[2] = {numTrackedObjects[classIdx], numFilteredClusters};
        dwObject_merge(objectsMerged[classIdx].data(), &numMergedObjects[classIdx],
                       maxClustersPerClass, toBeMerged, sizes, 2U, 0.1f, 0.1f, sdk);

        // extract now the actual bounding box of merged response in pixel coordinates to render on screen
        dnnBoxList[classIdx].resize(numMergedObjects[classIdx]);

        for (uint32_t objIdx = 0U; objIdx < numMergedObjects[classIdx]; ++objIdx) {
            const dwObject &obj = objectsMerged[classIdx][objIdx];
            dwBox2D &box = dnnBoxList[classIdx][objIdx].first;
            box.x = static_cast<int32_t>(std::round(obj.box.x));
            box.y = static_cast<int32_t>(std::round(obj.box.y));
            box.width = static_cast<int32_t>(std::round(obj.box.width));
            box.height = static_cast<int32_t>(std::round(obj.box.height));

            dnnBoxList[classIdx][objIdx].second = classLabels[classIdx];
        }
    }
}
