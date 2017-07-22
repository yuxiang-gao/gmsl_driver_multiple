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

#include "VBO.hpp"
#include <iostream>

#if !defined(GL_ES_VERSION_3_0) && !defined(GL_VERSION_3_1)
typedef void (*PFNGLDRAWARRAYSINSTANCEDPROC)(GLenum mode, GLint first, GLsizei count, GLsizei primcount);
static PFNGLDRAWARRAYSINSTANCEDPROC glDrawArraysInstanced = nullptr;
#endif

VBO::VBO(float *data, int vertexCount, int *subelements, int subelementscount, GLenum usage)
    : vertices(vertexCount)
    , subElementsCount(subelementscount)
    , strideSize(0)
{
    if (subElementsCount == 0 || vertexCount == 0) {
        std::cout << "Invalid VBO data sizes!" << std::endl;
        return;
    }

    subElements = new int[subElementsCount];
    for (int i = 0; i < subElementsCount; ++i) {
        strideSize += subelements[i];
        subElements[i] = subelements[i];
    }
    glGenBuffers(1, &buffer);
    bind();
    glBufferData(GL_ARRAY_BUFFER, strideSize * vertexCount * sizeof(float), data, usage);
    unbind();
}

VBO::~VBO()
{
    glDeleteBuffers(1, &buffer);
    delete[] subElements;
}

void VBO::bind(void)
{
    glBindBuffer(GL_ARRAY_BUFFER, buffer);
}

void VBO::unbind(void)
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VBO::setPointerAndBind(GLuint *locations, int count)
{
    bind();
    if (count != subElementsCount && count != 1) {
        std::cout << "Locations count and Vertex count does not match!" << std::endl;
        return;
    }
    if (count == 1) {
        glEnableVertexAttribArray(locations[0]);
        glVertexAttribPointer(locations[0], strideSize, GL_FLOAT, GL_FALSE, strideSize * sizeof(float), 0);
    } else {
        size_t offset = 0;
        for (int i = 0; i < count; ++i) {
            glEnableVertexAttribArray(locations[i]);
            glVertexAttribPointer(locations[i], subElements[i],
                                  GL_FLOAT, GL_FALSE, strideSize * sizeof(float), (void *)offset);
            offset += subElements[i] * sizeof(float);
        }
    }
}

void VBO::unsetPointerAndUnbind(GLuint *locations, int count)
{
    if (count != subElementsCount && count != 1) {
        std::cout << "Locations count and Vertex count does not match!" << std::endl;
        return;
    }
    for (int i = 0; i < count; ++i) {
        glDisableVertexAttribArray(locations[i]);
    }
    unbind();
}

void VBO::draw(GLuint *locations, int count, GLenum mode, int instances)
{
    setPointerAndBind(locations, count);
    if (instances == 1)
        glDrawArrays(mode, 0, vertices);
    else
        glDrawArraysInstanced(mode, 0, mode, instances);

    unsetPointerAndUnbind(locations, count);
}
