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

#ifndef SAMPLES_COMMON_MOUSEVIEW3D_HPP__
#define SAMPLES_COMMON_MOUSEVIEW3D_HPP__

class MouseView3D
{
  public:
    MouseView3D();

    //4x4 matrix in col-major format
    const float *getModelView() const
    {
        return m_modelView;
    }

    //4x4 matrix in col-major format
    const float *getProjection() const
    {
        return m_projection;
    }

    void setWindowAspect(float aspect);

    void mouseDown(int button, float x, float y);
    void mouseUp(int button, float x, float y);
    void mouseMove(float x, float y);
    void mouseWheel(float dx, float dy);

  private:
    float m_modelView[16];
    float m_projection[16];

    float m_windowAspect; // width/height
    float m_fovRads;
    float m_center[3];
    float m_up[3];
    float m_eye[3];

    // MOUSE NAVIGATION VARIABLES
    float m_startAngles[2];
    float m_startCenter[3];

    float m_radius;
    float m_angles[2];

    bool m_mouseLeft;
    bool m_mouseRight;
    float m_currentPos[2];

    void updateEye();
    void updateMatrices();
};

#endif // SAMPLES_COMMON_MOUSEVIEW3D_HPP__
