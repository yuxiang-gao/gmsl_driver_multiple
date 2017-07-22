/*
 * Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef _DEV_ERROR_H_
#define _DEV_ERROR_H_

typedef enum {
    EXT_IMG_DEV_NO_ERROR = 0,
    EXT_IMG_DEV_NO_DATA_ACTIVITY,
    EXT_IMG_DEV_VIDEO_LINK_ERROR,
    EXT_IMG_DEV_VSYNC_DETECT_FAILURE,
    EXT_IMG_DEV_NUM_FAILURE_TYPES
} ExtImgDevFailureType;

#endif /* _DEV_ERROR_H_ */
