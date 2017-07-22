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

#ifndef SAMPLES_COMMON_MATHUTILS_HPP__
#define SAMPLES_COMMON_MATHUTILS_HPP__

#include <math.h>

#define DEG2RAD(x) (static_cast<float>(x) * 0.01745329251994329575f)
#define RAD2DEG(x) (static_cast<float>(x) * 57.29577951308232087721f)

//Note all 4x4 matrices here are in column-major ordering

////////////////////////////////////////////////////////////
void cross(float dst[3], const float x[3], const float y[3]);

////////////////////////////////////////////////////////////
void normalize(float dst[3]);
void lookAt(float M[16], const float eye[3], const float center[3], const float up[3]);
void frustum(float M[16], const float l, const float r,
                          const float b, const float t,
                          const float n, const float f);

void perspective(float M[16], float fovy, float aspect, float n, float f);
void ortho(float M[16], float fovy, float aspect, float n, float f);

void quaternionToRotationMatrix(float rotation[16], const float quaternion[4]);
void positionToTranslateMatrix(float translate[16], const float position[3]);

#endif // SAMPLES_COMMON_MATHUTILS_HPP__
