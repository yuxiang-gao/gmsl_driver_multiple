/* Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved. All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */
#ifndef _IMG_DEV_H_
#define _IMG_DEV_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "nvcommon.h"
#include "nvmedia.h"
#include "nvmedia_image.h"
#include "nvmedia_icp.h"
#include "nvmedia_isc.h"
#include "dev_error.h"

#define MAX_AGGREGATE_IMAGES 4

#define CAM_ENABLE_DEFAULT 0x0001  // only enable cam link 0
#define CAM_MASK_DEFAULT   0x0000  // do not mask any link
#define CSI_OUT_DEFAULT    0x3210  // cam link i -> csiout i

// count enabled camera links
#define EXTIMGDEV_MAP_COUNT_ENABLED_LINKS(enable) \
             ((enable & 0x1) + ((enable >> 4) & 0x1) + \
             ((enable >> 8) & 0x1) + ((enable >> 12) & 0x1))
// convert aggegate number to cam_enable
#define EXTIMGDEV_MAP_N_TO_ENABLE(n) \
             ((((1 << n) - 1) & 0x0001) + ((((1 << n) - 1) << 3) &0x0010) + \
              ((((1 << n) - 1) << 6) & 0x0100) + ((((1 << n) - 1) << 9) & 0x1000))

typedef void ExtImgDevDriver;

typedef struct {
    unsigned int                enable; // camera[3..0] enable, value:0/1. eg 0x1111
    unsigned int                mask;   // camera[3..0] mask,   value:0/1. eg 0x0001
    unsigned int                csiOut; // camera[3..0] csi outmap, value:0/1/2/3. eg. 0x3210
} ExtImgDevMapInfo;

typedef struct {
    char                       *moduleName;
    char                       *resolution;
    char                       *inputFormat;
    char                       *interface;
    NvU32                       i2cDevice;
    NvU32                       desAddr;
    NvU32                       brdcstSerAddr;
    NvU32                       serAddr[MAX_AGGREGATE_IMAGES];
    NvU32                       brdcstSensorAddr;
    NvU32                       sensorAddr[MAX_AGGREGATE_IMAGES];
    NvU32                       sensorsNum;
    NvMediaBool                 enableEmbLines; /* TBD : this flag will be optional for
                                                 * on chip ISP in the sensor
                                                 * such as OV10635,
                                                 * if not, this flag is mandatory */
    char                       *board; /* Optional */
    NvMediaBool                 initialized; /* Optional:
                                              * Already initialized doesn't need to */
    NvMediaBool                 slave;  /* Optional :
                                         * Doesn't need to control sensor/serializer
                                         * through aggregator */
    NvMediaBool                 enableSimulator; /* Optional
                                                  * This flag is not to open actual
                                                  * isc-dev, it is for running isc
                                                  * without actual device. */
    ExtImgDevMapInfo           *camMap;
} ExtImgDevParam;

typedef struct {
    unsigned short              width;
    unsigned short              height;
    unsigned int                embLinesTop;
    unsigned int                embLinesBottom;
    NvMediaICPInputFormatType   inputFormatType;
    NvMediaBitsPerPixel         bitsPerPixel;
    NvMediaRawPixelOrder        pixelOrder;
    NvMediaICPInterfaceType     interface;
    NvMediaBool                 doubledPixel; /* for raw11x2 or raw16 */
} ExtImgDevProperty;

typedef struct {
    ExtImgDevDriver            *driver;
    ExtImgDevProperty           property;
    // ISC
    NvMediaISCRootDevice       *iscRoot;
    NvMediaISCDevice           *iscDeserializer;
    NvMediaISCDevice           *iscSerializer[MAX_AGGREGATE_IMAGES];
    NvMediaISCDevice           *iscSensor[MAX_AGGREGATE_IMAGES];
    NvMediaISCDevice           *iscBroadcastSerializer;
    NvMediaISCDevice           *iscBroadcastSensor;
    NvU32                       sensorsNum;
} ExtImgDevice;

ExtImgDevice *
ExtImgDevInit(ExtImgDevParam *configParam);

void
ExtImgDevDeinit(ExtImgDevice *device);

NvMediaStatus
ExtImgDevGetError(
    ExtImgDevice *device,
    NvU32 *link,
    ExtImgDevFailureType *errorType);

NvMediaStatus
ExtImgDevRegisterCallback(
    ExtImgDevice *device,
    NvU32 sigNum,
    void (*cb)(void *),
    void *context);

#ifdef __cplusplus
};      /* extern "C" */
#endif

#endif /* _IMG_DEV_H_ */
