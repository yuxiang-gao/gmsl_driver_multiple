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

#ifndef SAMPLES_COMMON_GPUPROGRAM_HPP__
#define SAMPLES_COMMON_GPUPROGRAM_HPP__

#include "Shader.hpp"
#include <string>
#include <map>
#include <memory>

class GPUProgram
{
  public:
    // create an empty program (not linked)
    GPUProgram();
    // create and link a program with a vertex and a fragment shader
    GPUProgram(std::shared_ptr<Shader> vertexShader, std::shared_ptr<Shader> fragmentShader);
    // release program
    ~GPUProgram();
    // attach a shader to program (only possible of not linked)
    void attachShader(Shader &shader);
    // link program
    void linkProgram();
    // get an attribute location
    GLuint getAttribLocation(std::string &name);
    GLuint getAttribLocation(const char *name);
    // get a uniform location
    GLint getUniformLocation(std::string &name);
    GLint getUniformLocation(const char *name);
    // bind program
    void bind();
    // remove binding of (any) program
    void unbind();
    // set number of patch vertices (tesselation only)
    void setPatchVertices(GLint patchVertices);

  private:
    std::map<std::string, GLint> attribMap;
    std::map<std::string, GLint> uniformMap;
    GLuint program;
    GLint patchVertices;
    std::shared_ptr<Shader> fShader;
    std::shared_ptr<Shader> vShader;
};

#endif // SAMPLES_COMMON_GPUPROGRAM_HPP__
