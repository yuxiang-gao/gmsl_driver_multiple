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

#include "Grid.hpp"

//------------------------------------------------------------------------------
void configureGrid(GridData_t *grid,
                   uint32_t windowWidth,
                   uint32_t windowHeight,
                   uint32_t imageWidth,
                   uint32_t imageHeight,
                   uint32_t cellCount)
{
    uint32_t rows = 1;
    uint32_t cols = 1;
    bool increaseRows = false;
    while( rows * cols < cellCount ) {
      if(increaseRows)
          rows++;
      else
          cols++;
      increaseRows = !increaseRows;
    }

    float camera_aspect_ratio = static_cast<float>(imageWidth) /
     static_cast<float>(imageHeight);

    uint32_t render_width = windowWidth / cols;
    uint32_t render_height = static_cast<uint32_t>(render_width / camera_aspect_ratio);

    if( render_height * rows > windowHeight ) {
        render_height = windowHeight / rows;
        render_width = static_cast<uint32_t>(render_height * camera_aspect_ratio);
    }

    grid->rows = rows;
    grid->cols = cols;
    grid->offsetX = 0;
    grid->offsetY = windowHeight - render_height;
    grid->cellWidth = render_width;
    grid->cellHeight = render_height;
}

//------------------------------------------------------------------------------
void gridCellRect(dwRect *rect,
              const GridData_t &grid,
              uint32_t cellIdx)
{
    //Set area
    int row = cellIdx / grid.cols;
    int col = cellIdx % grid.cols;

    rect->width   = grid.cellWidth;
    rect->height  = grid.cellHeight;
    rect->x = grid.offsetX + grid.cellWidth*col;
    rect->y = grid.offsetY - grid.cellHeight*row;
}
