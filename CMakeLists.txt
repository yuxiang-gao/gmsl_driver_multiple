cmake_minimum_required(VERSION 2.8.3)
project(gmsl_driver_multiple)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  camera_info_manager
  nodelet
)

set(Driveworks_INCLUDE_DIR /usr/local/driveworks/include)
set(Driveworks_LIBRARY /usr/local/driveworks/lib)
link_directories(${Driveworks_LIBRARY})
set(glfw_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/linux-aarch64/glfw-3.1.1/include)
set(glfw_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/linux-aarch64/glfw-3.1.1/lib)
link_directories(${glfw_LIBRARY})

set(VIBRANTE TRUE)
add_definitions(-DVIBRANTE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)
find_package(GLEW REQUIRED)
find_package(Driveworks REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES drivenet_lanenet_ros
#  CATKIN_DEPENDS cv_bridge iamge_transport libglew-dev roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Driveworks_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${GLEW_INCLUDE_DIR}
  ${glfw_INCLUDE_DIR}
  ${BOOST_INCLUDE_DIR}
)


include_directories(/usr/share/visionworks/sources/3rdparty/nvmedia/
/usr/share/visionworks/VisionWorks-1.6-Samples/3rdparty/glfw3/include/
/usr/local/driveworks/include)
## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/drivenet_lanenet_ros.cpp
# )
set(COMMON_SOURCES
    #src/common/SampleFramework.cpp
    src/common/ProgramArguments.cpp
    src/common/Grid.cpp
    #src/common/Shader.cpp
    #src/common/VBO.cpp
    #src/common/GPUProgram.cpp
    src/common/WindowEGL.cpp
    #src/common/MathUtils.cpp
    #src/common/MouseView3D.cpp
    src/common/ConsoleColor.cpp
    #src/common/ProfilerCUDA.cpp
    #src/common/WindowGLFW.cpp
    src/common/SampleFramework.cpp

)

set(NODE_NAME ${PROJECT_NAME}_node)
cuda_add_executable(${NODE_NAME} src/gmsl_driver_multiple/main.cpp src/gmsl_driver_multiple/cv_connection.cpp ${COMMON_SOURCES})
add_dependencies(${NODE_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${NODE_NAME} PRIVATE
    ${catkin_LIBRARIES}
    nvmedia  
    ${GLEW_LIBRARY}
    driveworks
    GLESv2
    EGL
    drm
    ${BOOST_LIBRARY}
    ${CUDA_LIBRARY}
    ${OpenCV_LIBS} 
)
install(TARGETS ${NODE_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/drivenet_lanenet_ros_node.cpp)

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_drivenet_lanenet_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
