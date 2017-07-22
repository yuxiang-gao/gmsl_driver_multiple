# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

#-------------------------------------------------------------------------------
# Dependencies
#-------------------------------------------------------------------------------
set(glfw3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/glfw-3.1.1" CACHE PATH '' FORCE)
set(glew_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/glew-1.13.0" CACHE PATH '' FORCE)
set(lodepng_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/lodepng-20150912" CACHE PATH '' FORCE)
find_package(glfw3 REQUIRED CONFIG)
find_package(lodepng CONFIG REQUIRED)

if(WINDOWS OR LINUX)
    find_package(glew REQUIRED CONFIG)
endif()

if(VIBRANTE)
    set(vibrante_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/${SDK_ARCH_DIR}/vibrante" CACHE PATH '' FORCE)
    find_package(vibrante REQUIRED CONFIG)
    find_package(EGL REQUIRED)
    add_definitions(-DDW_USE_NVMEDIA)
    add_definitions(-DDW_USE_EGL)
    set(DW_USE_NVMEDIA ON)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
endif()

# Hide settings in default cmake view
mark_as_advanced(glfw3_DIR glew_DIR lodepng_DIR vibrante_DIR)
