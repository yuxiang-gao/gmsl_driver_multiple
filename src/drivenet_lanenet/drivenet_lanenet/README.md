# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

@page dwx_object_tracker_drivenet_sample DriveNet Sample

The DriveNet sample is a sophisticated, multi-class, higher
resolution example that uses the NVIDIA proprietary deep neural
network (DNN) to perform object detection. For a simpler, basic,
single-class example, see [Object Detector using dwDetector](@ref dwx_object_dwdetector).

The DriveNet sample shows a more complete and sophisticated implementation of
object detection built around the NVIDIA proprietary network architecture. It provides
significantly enhanced detection quality when compared to @ref dwx_object_dwdetector.
For more information on DriveNet and how to customize it for your applications,
consult your NVIDIA sales or business representative.

![Multiclass object detector on an RCCB stream using DriveNet](sample_drivenet.png)

The table below shows the differences between @ref dwx_object_dwdetector and DriveNet:

| | Basic Object Detector (dwDetector) | DriveNet |
|-------------|------------|------------|
| Input Video Format | H.264 RGB | RAW video (RCCB fp16) or RCCB live camera |
| Detection Quality | Basic | High quality |
| Object Classes | Car/truck only | Car/truck, person, bicycle/motorcycle, road sign |
| Network | GoogLeNet-derived, network structure available as prototxt | NVIDIA proprietary DriveNet architecture |
| Conditions Supported | Daytime, clear weather | Daytime, clear weather |
| Post-processing for False Positive Reduction | Basic | More advanced, confidence model-based |
| Object tracking over time | Enabled | Enabled |
| Camera position supported | Front | Front, rear, or side |

DriveNet detects objects by performing inference on each frame of a RAW video/camera stream.
It clusters and tracks the objects with parameters defined in the sample.

Depending on the run platform, this sample supports Camera and Video modes:
- On NVIDIA<sup>&reg;</sup> Vibrante<sup>&trade;</sup>: Camera and Video
- On Linux: only Video

The DriveNet sample expects RAW video or live camera input data from an AR0231 (revision >= 4) sensor with an RCCB color filter
and a resolution of 1920 x 1208, which is then cropped and scaled down by half to 960 x 540 in typical
usage.

The DriveNet sample uses foveal detection mode. It sends two images to the DriveNet network:
- One image is the full resolution original RCB_fp16 camera frame. In the above image,
  a blue bounding box identifies this region.
- The other image is the cropped center-region of the original image. In the above image,
  a yellow bounding box identifies this region. DriveNet retains the region's
  aspect ratio by internally scaling the cropped region to fit the network input
  dimensions.

A follow-up algorithm clusters detections from both images to compute a more
stable response.

## Limitations ##

@note The version of DriveNet included in this release is optimized for daytime, clear-weather data. It
does not perform well in dark or rainy conditions.


The DriveNet network is trained to support any of the following six camera configurations:
* Front camera location with a 60&deg; field of view
* Rear camera location with a 60&deg; field of view
* Front-left camera location with a 120&deg; field of view
* Front-right camera location with a 120&deg; field of view
* Rear-left camera location with a 120&deg; field of view
* Rear-right camera location with a 120&deg; field of view

The DriveNet network works on any of the above camera positions, without additional configuration changes.

The DriveNet network is trained primarily on data collected in the United States. It may have reduced
accuracy in other locales, particularly for road sign shapes that do not exist in the U.S.

## Running the Sample

### To run the sample on a video on Vibrante

    ./sample_drivenet --input-type=video --video=<video file.raw>

### To run the sample on a camera on Vibrante

    ./sample_drivenet --input-type=camera --camera-type=<rccb camera type> --csi-port=<csi port> --camera-index=<camera idx on csi port>

where `<rccb camera type>` is one of the following: `ar0231-rccb`, `ar0231-rccb-ssc`, `ar0231-rccb-bae`, `ar0231-rccb-ss3322`, `ar0231-rccb-ss3323`

### To run the sample on Linux

    ./sample_drivenet --video=<video file.raw>

## Output

The sample creates a window, displays a video, and overlays bounding boxes for detected objects.
The color of the bounding boxes represent the classes that it detects:

    Red: Cars
    Green: Traffic Signs
    Blue: Bicycles
    Cyan: Trucks
    Yellow: Pedestrians
