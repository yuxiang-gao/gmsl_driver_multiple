# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

#project(dw_samples_drivenet_common C CXX)
project(dw_samples_drivenet_lanenet_common C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
   DriveNet.cpp
   DriveNet.hpp

   common.hpp
   common.cpp
   LaneDetectionCommon.hpp
   LaneDetectionCommon.cpp
)

set(LIBRARIES
    dw_samples_common
)

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
cuda_add_library(${PROJECT_NAME} ${SOURCES} STATIC)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
#set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")
