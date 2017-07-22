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

#include "Shader.hpp"
#include <fstream>
#include <iostream>
#include <string>

Shader::Shader(GLenum type, const char *src, SourceType srcType)
    : shader(0)
{
    std::string source;
    shader = glCreateShader(type);
    if (shader == 0) {
        std::cout << "Failed to create shader!" << std::endl;
        return;
    }
    if (srcType == File) {
        std::ifstream read(src);
        char c;
        while (read.get(c))
            source += c;
        read.close();
    } else {
        source += src;
    }

    char const *sources[] = {
        source.c_str()};
    glShaderSource(shader, 1, sources, NULL);
    glCompileShader(shader);

    GLint compiled;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        if (srcType == File)
            std::cout << "Failed to compile shader in file " << src << std::endl;
        else
            std::cout << "Failed to compile shader" << std::endl;
        std::cout << "Source: " << std::endl
                  << source << std::endl;

        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char *log = new char[infoLen];
            glGetShaderInfoLog(shader, infoLen, nullptr, log);
            std::cout << "Error compiling shader: " << std::endl
                      << log << std::endl;
            delete[] log;
        }
        glDeleteShader(shader);
        shader = 0;
    }
}

Shader::~Shader()
{
    if (shader != 0)
        glDeleteShader(shader);
}

GLuint Shader::getGLId()
{
    if (shader == 0) {
        std::cout << "Invalid shader selected!" << std::endl;
    }
    return shader;
}
