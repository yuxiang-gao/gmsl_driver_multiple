# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

project(dw_samples_common C CXX)

#-------------------------------------------------------------------------------
# Project files
#-------------------------------------------------------------------------------
set(SOURCES
    Checks.hpp
    SampleFramework.hpp
    SampleFramework.cpp

    ProgramArguments.hpp
    ProgramArguments.cpp

    Window.hpp

    Grid.hpp
    Grid.cpp

    Shader.cpp
    Shader.hpp
    VBO.hpp
    VBO.cpp
    GPUProgram.cpp
    GPUProgram.hpp
    WindowGLFW.hpp
    WindowGLFW.cpp

    MathUtils.hpp
    MathUtils.cpp
    MouseView3D.cpp
    MouseView3D.hpp

    Mat4.hpp

    ConsoleColor.hpp
    ConsoleColor.cpp

    StatsCounter.hpp
    CudaTimer.hpp

    ProfilerCUDA.hpp
    ProfilerCUDA.cpp

    ${SDK_BINARY_DIR}/configured/samples/common/DataPath.hpp
)

set(LIBRARIES ${Driveworks_LIBRARIES})

if(NOT VIBRANTE)
    list(APPEND LIBRARIES
        GLEW::glew_s
    )
endif()

if(VIBRANTE)
    list(APPEND SOURCES
        WindowEGL.hpp
        WindowEGL.cpp
    )
    list(APPEND LIBRARIES
        drm
    )
endif()

list(APPEND LIBRARIES glfw)

if(LINUX)
    list(APPEND LIBRARIES
        ${X11_LIBRARIES} ${X11_Xrandr_LIB}
        ${X11_Xcursor_LIB} ${X11_Xxf86vm_LIB}
        ${X11_Xinerama_LIB} ${X11_Xi_LIB}
        ${CMAKE_THREAD_LIBS_INIT}
    )
endif()

#-------------------------------------------------------------------------------
# Final target
#-------------------------------------------------------------------------------
add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/.. ${lodepng_DIR}/include)
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "Samples")
