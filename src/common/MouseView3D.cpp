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

#include "MouseView3D.hpp"
#include <math.h>
#include <common/MathUtils.hpp>
#include <common/Mat4.hpp>

MouseView3D::MouseView3D()
    : m_windowAspect(1.0f)
    , m_fovRads(DEG2RAD(60.0f))
    , m_radius(8)
    , m_mouseLeft(false)
    , m_mouseRight(false)
{
    m_center[0] = 0;
    m_center[1] = 0;
    m_center[2] = 0;

    m_up[0] = 0;
    m_up[1] = 0;
    m_up[2] = 1;

    m_angles[0] = DEG2RAD(180.0f);
    m_angles[1] = DEG2RAD(30.0f);

    m_currentPos[0] = -1.0f;
    m_currentPos[1] = -1.0f;

    updateEye();
    updateMatrices();
}

void MouseView3D::updateEye()
{
    m_eye[0] = m_radius * cos(m_angles[1]) * cos(m_angles[0]) + m_center[0];
    m_eye[1] = m_radius * cos(m_angles[1]) * sin(m_angles[0]) + m_center[1];
    m_eye[2] = m_radius * sin(m_angles[1]) + m_center[2];
}

void MouseView3D::mouseDown(int button, float x, float y)
{
    m_currentPos[0] = x;
    m_currentPos[1] = y;

    m_startAngles[0] = m_angles[0];
    m_startAngles[1] = m_angles[1];

    m_startCenter[0] = m_center[0];
    m_startCenter[1] = m_center[1];
    m_startCenter[2] = m_center[2];

    m_mouseLeft  = (button == 0);
    m_mouseRight = (button == 1);
}

void MouseView3D::mouseUp(int button, float x, float y)
{
    (void)button;
    (void)x;
    (void)y;
    m_mouseLeft  = false;
    m_mouseRight = false;
}

void MouseView3D::mouseMove(float x, float y)
{
    float pos[] = {x, y};

    if (m_mouseLeft) {
        // update deltaAngle
        m_angles[0] = m_startAngles[0] - 0.01f * (pos[0] - m_currentPos[0]);
        m_angles[1] = m_startAngles[1] + 0.01f * (pos[1] - m_currentPos[1]);

        // Limit the vertical angle (5 to 85 degrees)
        if (m_angles[1] > DEG2RAD(85))
            m_angles[1] = DEG2RAD(85);

        if (m_angles[0] < DEG2RAD(5))
            m_angles[0] = DEG2RAD(5);

        updateEye();
        updateMatrices();
    } else if (m_mouseRight) {
        //Translation
        float t[3];
        t[0] = 0.1f * (pos[0] - m_currentPos[0]);
        t[1] = 0;
        t[2] = 0.1f * (pos[1] - m_currentPos[1]);

        float mt[3];
        Mat4_Rtxp(mt, m_modelView, t);

        m_center[0] = m_startCenter[0] + mt[0];
        m_center[1] = m_startCenter[1] + mt[1];
        m_center[2] = m_startCenter[2] + mt[2];

        updateEye();
        updateMatrices();
    }
}

void MouseView3D::mouseWheel(float dx, float dy)
{
    (void)dx;

    float tmpRadius = m_radius - dy * 1.5f;

    if (tmpRadius > 0.0f) {
        m_radius = tmpRadius;
        updateEye();
    }
}

void MouseView3D::setWindowAspect(float aspect)
{
    m_windowAspect = aspect;
    updateMatrices();
}

void MouseView3D::updateMatrices()
{
    lookAt(m_modelView, m_eye, m_center, m_up);
    perspective(m_projection, m_fovRads, 1.0f * m_windowAspect, 0.1f, 1000.0f);
}
