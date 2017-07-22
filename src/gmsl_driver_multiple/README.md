# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

@page dwx_mul_camera_gmsl_sample Multiple GMSL Camera Capture Sample

![4 x OV10640-b1 + 4 x OV10635 cameras capturing in parallel](sample_camera_multiple_gmsl.png)

The Multiple GMSL Camera Capture sample uses the GMSL camera interface, if
available, on DRIVE PX 2. The sample opens an X window showing the input from all
selected cameras on the specified CSI ports. The sample supports offscreen mode
in case a headless DRIVE PX 2 is used. However, in headless mode, no output can be
observed besides the logging output on the console.

The sample is capable of capturing the input from up to 12 cameras
simultaneously.

## Setting Up Cameras

For information about the physical location of the ports on the DRIVE
board, see "Camera Setup under Configuration and Setup" in <em>Vibrante Linux
SDK/PDK 4.1 for DRIVE PX 2 Development Guide</em>.

@note It is not possible to intermix cameras of different types on the same CSI
port. Doing so causes corruption in the camera preview.

## Testing

One or multiple cameras must be connected to the DRIVE PX 2 CSI ports. The sample
requires a specific type of camera to be connected to the specific port. For
more information, see `sample_sensor_info` for supported camera types.

Specify the camera types by using:

    --type-ab (for AB port)
    --type-cd (for CD port)
    --type-ef (for EF port)

For example,

    ./sample_camera_multiple_gmsl --type-ab=ar0231 --type-cd=c-ov10640-b1

The argument `--selector-mask` enables you to select specific cameras on any
port to used for capturing. The format of the mask is a string with 0s and 1s
indicating disabling and enabling of the camera, respectively. The mask is
parsed from left to right and applied on the CSI ports in the increasing order
ab, cd, ef (i.e., counting camera connectors from left to right on DRIVE PX 2).

For example, to select the 2nd camera from ab port and 3rd and 4th from cd port,
use:

    --selector-mask=01000011


If desired, the sample can also be executed in headless mode on Vibrante
platforms by adding the argument:
    --offscreen=1

Screenshots from all cameras can be captured by pressing `s` while the sample is running.

@note

 - Some camera types will fail to load in this sample if they do
   not have YUV output for rendering (e.g., `ar0231-rccb-ssc` will fail
   because it only supports RAW output). Please see `sample_camera_gmsl_raw`
   for rendering RAW output from a camera.
 - Continental cameras model OV10640 can be launched under different names: the
   generic name is `c-ov10640-b1` and is kept for backward compatibility. 
   The two other modes `ov10640-svc210` and `ov10640-svc212` will activate lense specific
   configurations for an improved image quality. 

#### Cross-CSI Synchronization
It is possible to activate this feature by specifying

    --cross-csi-sync=1

Cameras supporting synchronization accross CSI ports will be triggered synchronously.
More precisely, when this feature is disabled (by default it is 0), difference in time points of captured frames from
cameras at different csi-ports might be large, on average > 60ms. When this mode is active the frames are synchronized
over the span of all csi-ports and a latency of 1ms on average (worst case observed was 5 ms) can be observed.

@note
 - Not all cameras support cross-csi synchronization. At he moment only OV10635 and AR0231 are supported
 - cross csi syncrhonization is only supported for all cameras of the same type


## Run on Tegra B

#### Master Mode Prerequisites
Before running camera applications only on Tegra B, you must disable FRSYNC and
the forward/reverse control channel of Tegra A aggregator.
Please look at *Camera Setup (P2379)* in *Vibrante Linux SDK/PDK for DRIVE PX 2 Development Guide*.
This guide will show show you how to:
* Turn MAX9286 aggregator on.
* Disable FRSYNC and the forward/reverse control channel on MAX9286 aggregator to avoid any interference with MAX96799.

#### Master Mode
After you have addressed the prerequisites (above) and rebooted Tegra B,
you can run camera applications on Tegra B.
Be aware that running camera applications on Tegra A re-enables the forward/reverse control channel and FRSYNC from MAX9286
and the procecedure of activating camera for Tegra B need to be repeated.

#### Slave Mode
Cameras can be captured on Tegra B in slave mode, i.e. when they are already captured
by an application on Tegra A. In such case, it is possible to specify the "slave" flag, which can
be 0 or 1. If slave is 1, then Tegra B will not be able to run cameras
autonomously, but it requires the camera to be run at the same time from Tegra A.
If slave is false, then Tegra B can control any camera that is not currently
being used by Tegra A.

