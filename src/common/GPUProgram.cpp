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

#include "GPUProgram.hpp"
#include <dw/core/EGL.h>
#include <iostream>

#ifndef glPatchParameteri
typedef void (*PFNGLPATCHPARAMETERIEXTPROC)(GLenum pname, GLint value);
static PFNGLPATCHPARAMETERIEXTPROC glPatchParameteri = nullptr;
#endif

#ifndef GL_EXT_tessellation_shader
#define GL_PATCH_VERTICES_EXT 0x8E72
#endif

GPUProgram::GPUProgram()
    : patchVertices(0)
{
    program = glCreateProgram();
    if (program == 0) {
        std::cout << "Failed to create GPU program!" << std::endl;
    }
}

GPUProgram::GPUProgram(std::shared_ptr<Shader> vertexShader, std::shared_ptr<Shader> fragmentShader)
    : patchVertices(0)
    , fShader(fragmentShader)
    , vShader(vertexShader)

{
    program = glCreateProgram();
    if (program == 0) {
        std::cout << "Failed to create GPU program!" << std::endl;
    }

    attachShader(*vertexShader);
    attachShader(*fragmentShader);
    linkProgram();
}

GPUProgram::~GPUProgram()
{
    glDeleteProgram(program);
}

void GPUProgram::attachShader(Shader &shader)
{
    if (program == 0) {
        std::cout << "Invalid GPU program!" << std::endl;
        return;
    }

    GLuint id = shader.getGLId();
    if (id == 0) {
        std::cout << "Invalid shader!" << std::endl;
        return;
    }

    glAttachShader(program, id);
}

void GPUProgram::linkProgram()
{
    glLinkProgram(program);
    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        std::cout << "Failed to link program!" << std::endl;
        GLint infoLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char *log = new char[infoLen];
            glGetProgramInfoLog(program, infoLen, nullptr, log);
            std::cout << "Error linking shader: " << std::endl
                      << log << std::endl;
            delete[] log;
        }
        glDeleteProgram(program);
    }
}

GLuint GPUProgram::getAttribLocation(std::string &name)
{
    std::map<std::string, GLint>::const_iterator elem = attribMap.find(name);
    if (elem != attribMap.end()) {
        return elem->second;
    } else {
        GLint location = glGetAttribLocation(program, name.c_str());
        if (location <= -1) {
            std::cout << "Received bad location " << location << " for attribute " << name << std::endl;
        }
        uniformMap[name] = location;
        return location;
    }
}

GLuint GPUProgram::getAttribLocation(const char *name)
{
    std::string temp(name);
    return getAttribLocation(temp);
}

GLint GPUProgram::getUniformLocation(const char *name)
{
    std::string temp(name);
    return getUniformLocation(temp);
}

GLint GPUProgram::getUniformLocation(std::string &name)
{
    std::map<std::string, GLint>::const_iterator elem = uniformMap.find(name);
    if (elem != uniformMap.end()) {
        return elem->second;
    } else {
        GLint location = glGetUniformLocation(program, name.c_str());
        if (location <= -1) {
            std::cout << "Received bad location " << location << " for uniform " << name << std::endl;
        }
        uniformMap[name] = location;
        return location;
    }
}

void GPUProgram::bind()
{
    glUseProgram(program);
    if (patchVertices > 0 && glPatchParameteri) {
        glPatchParameteri(GL_PATCH_VERTICES_EXT, patchVertices);
    }
}

void GPUProgram::unbind()
{
    glUseProgram(0);
    if (patchVertices > 0 && glPatchParameteri) {
        glPatchParameteri(GL_PATCH_VERTICES_EXT, 0);
    }
}

void GPUProgram::setPatchVertices(GLint patchVertices)
{
    this->patchVertices = patchVertices;
    if (glPatchParameteri == nullptr) {
#ifdef GL_EXT_tessellation_shader
        glPatchParameteri = (PFNGLPATCHPARAMETERIEXTPROC)eglGetProcAddress("glPatchParameteriEXT");
        if (glPatchParameteri == nullptr) {
            std::cout << "Failed to load glPatchParameteriEXT" << std::endl;
        }
#endif
    }
}
