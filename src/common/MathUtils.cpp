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

#include "MathUtils.hpp"

//Note all 4x4 matrices here are in column-major ordering

////////////////////////////////////////////////////////////
void cross(float dst[3], const float x[3], const float y[3])
{
    dst[0] = x[1] * y[2] - x[2] * y[1];
    dst[1] = -(x[0] * y[2] - x[2] * y[0]);
    dst[2] = x[0] * y[1] - x[1] * y[0];
}

////////////////////////////////////////////////////////////
void normalize(float dst[3])
{
    float normSqr = dst[0] * dst[0] + dst[1] * dst[1] + dst[2] * dst[2];

    if (normSqr > 0.0) {
        float invNorm = 1.0f / sqrt(normSqr);

        dst[0] = dst[0] * invNorm;
        dst[1] = dst[1] * invNorm;
        dst[2] = dst[2] * invNorm;
    }
}

void lookAt(float M[16], const float eye[3], const float center[3], const float up[3])
{
    float x[3], y[3], z[3];

    // make rotation matrix

    // Z vector
    z[0] = eye[0] - center[0];
    z[1] = eye[1] - center[1];
    z[2] = eye[2] - center[2];
    normalize(z);

    // Y vector
    y[0] = up[0];
    y[1] = up[1];
    y[2] = up[2];

    // X vector = Y cross Z
    cross(x, y, z);

    // Recompute Y = Z cross X
    cross(y, z, x);

    // cross product gives area of parallelogram, which is < 1.0 for
    // non-perpendicular unit-length vectors; so normalize x, y here
    normalize(x);
    normalize(y);

    M[0] = x[0];
    M[1] = y[0];
    M[2] = z[0];
    M[3] = 0.0;

    M[4] = x[1];
    M[5] = y[1];
    M[6] = z[1];
    M[7] = 0.0;

    M[8]  = x[2];
    M[9]  = y[2];
    M[10] = z[2];
    M[11] = 0.0;

    M[12] = -x[0] * eye[0] - x[1] * eye[1] - x[2] * eye[2];
    M[13] = -y[0] * eye[0] - y[1] * eye[1] - y[2] * eye[2];
    M[14] = -z[0] * eye[0] - z[1] * eye[1] - z[2] * eye[2];
    M[15] = 1.0;
}

void frustum(float M[16], const float l, const float r, const float b, const float t, const float n, const float f)
{
    M[0] = ((float)(2.0)) * n / (r - l);
    M[1] = 0.0;
    M[2] = 0.0;
    M[3] = 0.0;

    M[4] = 0.0;
    M[5] = ((float)(2.0)) * n / (t - b);
    M[6] = 0.0;
    M[7] = 0.0;

    M[8]  = (r + l) / (r - l);
    M[9]  = (t + b) / (t - b);
    M[10] = -(f + n) / (f - n);
    M[11] = -1.0;

    M[12] = 0.0;
    M[13] = 0.0;
    M[14] = -(((float)(2.0)) * f * n) / (f - n);
    M[15] = 0.0;
}

void perspective(float M[16], float fovy, float aspect, float n, float f)
{
    float xmin, xmax, ymin, ymax;

    ymax = n * (float)tan(fovy * 0.5f);
    ymin = -ymax;

    xmin = ymin * aspect;
    xmax = ymax * aspect;

    frustum(M, xmin, xmax, ymin, ymax, n, f);
}

void ortho(float M[16], const float l, const float r, const float b, const float t, const float n, const float f)
{
    // https://msdn.microsoft.com/ru-ru/library/windows/desktop/dd373965.aspx

    M[0] = (2.0f) / (r - l);
    M[1] = 0.0f;
    M[2] = 0.0f;
    M[3] = 0.0f;

    M[4] = 0.0f;
    M[5] = (2.0f) / (t - b);
    M[6] = 0.0f;
    M[7] = 0.0f;

    M[8]  = 0.0f;
    M[9]  = 0.0f;
    M[10] = (-2.0f) / (f - n);
    M[11] = 0.0;

    M[12] = -(r + l) / (r - l);
    M[13] = -(t + b) / (t - b);
    M[14] = -(f + n) / (f - n);
    M[15] = 1.0;
}

void ortho(float M[16], float fovy, float aspect, float n, float f)
{
    float xmin, xmax, ymin, ymax;

    ymax = n * (float)tan(fovy * 0.5f);
    ymin = -ymax;

    xmin = ymin * aspect;
    xmax = ymax * aspect;

    ortho(M, xmin, xmax, ymin, ymax, n, f);
}


void quaternionToRotationMatrix(float rotation[16], const float quaternion[4])
{
    float q[4] = { quaternion[0], quaternion[1], quaternion[2], quaternion[3] };
    {
        float dist2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
        if (fabs(dist2 - 1) > 1e-6) {
            float dist = sqrt(dist2);
            q[0] /= dist;
            q[1] /= dist;
            q[2] /= dist;
            q[3] /= dist;
        }
    }

    rotation[0] = 1 - 2 * q[1] * q[1] - 2 * q[2] * q[2];
    rotation[1] = 2 * (q[0] * q[1] + q[2] * q[3]);
    rotation[2] = 2 * (q[0] * q[2] + q[1] * q[3]);

    rotation[4] = 2 * (q[0] * q[1] - q[2] * q[3]);
    rotation[5] = 1 - 2 * q[0] * q[0] - 2 * q[2] * q[2];
    rotation[6] = 2 * (q[1] * q[2] + q[0] * q[3]);

    rotation[8] = 2 * (q[0] * q[2] + q[1] * q[3]);
    rotation[9] = 2 * (q[1] * q[2] - q[0] * q[3]);
    rotation[10] = 1 - 2 * q[0] * q[0] - 2 * q[1] * q[1];
}

void positionToTranslateMatrix(float translate[16], const float position[3])
{
    translate[3 * 4] = position[0];
    translate[3 * 4 + 1] = position[1];
    translate[3 * 4 + 2] = position[2];
}
