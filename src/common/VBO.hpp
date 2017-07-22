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

#ifndef SAMPLES_COMMON_VBO_HPP__
#define SAMPLES_COMMON_VBO_HPP__

#include <dw/gl/GL.h>

class VBO
{
  public:
    // create VBO and upload data to it
    //   data: data to upload (float data)
    //   vertexCount: number of vertices
    //   subelements: list of subelementscount values. Defines number of attributes per vertex and index
    //   subelementscount: number of elements in subelements
    //   usage: hint to GL of main usage of VBO (use GL values)
    VBO(float *data, int vertexCount, int *subelements, int subelementscount, GLenum usage);
    // release VBO
    ~VBO();
    // bind VBO
    void bind(void);
    // remove binding of (any) VBO
    void unbind(void);
    // bind buffer and set attribute pointers
    //   locations: attribute location of each subelement specified in constructor
    //   count: length of list in locations
    void setPointerAndBind(GLuint *locations, int count);
    // remove binding of any buffer and remove attribute pointers
    //   locations: attribute location of each subelement specified in constructor
    //   count: length of list in locations
    void unsetPointerAndUnbind(GLuint *locations, int count);
    // draw VBO content (includes setPointerAndBind() and unsetPointerAndUnbind() steps)
    //   locations: attribute location of each subelement specified in constructor
    //   count: length of list in locations
    //   mode: set drawing mode (use GL values)
    //   instances: draw instances times. Values larger 1 use instancing
    void draw(GLuint *locations, int count, GLenum mode, int instances = 1);

  private:
    GLuint buffer;
    int vertices;
    int subElementsCount;
    int *subElements;
    int strideSize;
};

#endif // SAMPLES_COMMON_VBO_HPP__
