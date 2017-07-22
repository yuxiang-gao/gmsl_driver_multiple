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

#ifndef DRIVENET_COMNMON_H_
#define DRIVENET_COMNMON_H_

#include <vector>
#include <string>


// DriveNet
#include <dw/object/DriveNet.h>
#include <dw/object/Tracker.h>
#include <dw/object/Clustering.h>


class DriveNet
{
public:

    DriveNet(dwContextHandle_t ctx);
    ~DriveNet();

    bool initTracker(const dwImageProperties& rcbProperties, cudaStream_t stream);
    bool initDetector(const dwImageProperties& rcbProperties, cudaStream_t stream);
    void releaseDetector();
    void releaseTracker();

    void resetDetector();
    void resetTracker();

    dwStatus inferDetectorAsync(const dwImageCUDA* rcbImage);
    dwStatus inferTrackerAsync(const dwImageCUDA* rcbImage);
    void processResults();

    size_t getNumClasses() { return classLabels.size(); }

    const std::vector<std::pair<dwBox2D,std::string>>& getResult(uint32_t classIdx);

    // DriveNet
    dwDriveNetHandle_t driveNet = DW_NULL_HANDLE;
    dwDriveNetParams drivenetParams{};

    // Tracker/Clustering
    dwObjectTrackerHandle_t objectTracker = DW_NULL_HANDLE;
    std::vector<dwObjectClusteringHandle_t> objectClustering;

    // -------- parameters -------------
    // Maximum number of proposals per class object class
    static const uint32_t maxProposalsPerClass = 400U;
    // Maximum number of objects (clustered proposals) per object class
    static const uint32_t maxClustersPerClass = 100U;

private:
    void filterClusters(dwObject out[], size_t *nOut, const dwObject in[], size_t nIn, float32_t threshold);

    // -------- parameters -------------
    dwContextHandle_t sdk;

    // Thresholds for filtering out detections after clustering
    std::vector<float32_t> postClusteringThresholds;


    // -------- intermediate results ------

    // Number of detected proposals per object class
    std::vector<size_t> numProposals;
    // List of detected object proposals per object class
    std::vector<std::vector<dwObject>> objectProposals;

    // Number of clusters per object class
    std::vector<size_t> numClusters;
    // List of clusters per object class
    std::vector<std::vector<dwObject>> objectClusters;
    // List of filtered clusters per object classgNumClusters[classIdx]
    std::vector<std::vector<dwObject>> objectClustersFiltered;

    // Number of merged objects per object class
    std::vector<size_t> numMergedObjects;
    // List of merged objects per object class
    std::vector<std::vector<dwObject>> objectsMerged;

    // Number of tracked objects per object class
    std::vector<size_t> numTrackedObjects;
    // List of tracked objects per object class
    std::vector<std::vector<dwObject>> objectsTracked;

    // Labels of each class
    std::vector<std::string> classLabels;

    // Vector of pairs of boxes and class label ids
    std::vector<std::vector<std::pair<dwBox2D,std::string>>> dnnBoxList;
};

#endif
