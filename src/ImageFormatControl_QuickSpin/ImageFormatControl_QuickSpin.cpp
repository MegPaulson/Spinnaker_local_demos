//=============================================================================
// Copyright (c) 2001-2023 FLIR Systems, Inc. All Rights Reserved.
//
// This software is the confidential and proprietary information of FLIR
// Integrated Imaging Solutions, Inc. ("Confidential Information"). You
// shall not disclose such Confidential Information and shall use it only in
// accordance with the terms of the license agreement you entered into
// with FLIR Integrated Imaging Solutions, Inc. (FLIR).
//
// FLIR MAKES NO REPRESENTATIONS OR WARRANTIES ABOUT THE SUITABILITY OF THE
// SOFTWARE, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, OR NON-INFRINGEMENT. FLIR SHALL NOT BE LIABLE FOR ANY DAMAGES
// SUFFERED BY LICENSEE AS A RESULT OF USING, MODIFYING OR DISTRIBUTING
// THIS SOFTWARE OR ITS DERIVATIVES.
//=============================================================================

/**
 *  @example ImageFormatControl_QuickSpin.cpp
 *
 *  @brief ImageFormatControl_QuickSpin.cpp shows how to apply custom image
 *	settings to the camera using the QuickSpin API. QuickSpin is a subset of
 *	the Spinnaker library that allows for simpler node access and control.
 *
 *	This example demonstrates customizing offsets X and Y, width and height,
 *	and the pixel format. Ensuring custom values fall within an acceptable
 *	range is also touched on. Retrieving and setting node values using
 *	QuickSpin is the only portion of the example that differs from
 *	ImageFormatControl.
 *
 *  A much wider range of topics is covered in the full Spinnaker examples than
 *  in the QuickSpin ones. There are only enough QuickSpin examples to
 *  demonstrate node access and to get started with the API; please see full
 *  Spinnaker examples for further or specific knowledge on a topic.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// This function configures a number of settings on the camera including
// offsets X and Y, width, height, and pixel format. These settings must be
// applied before spinCameraBeginAcquisition() is called; otherwise, those
// nodes would be read only. Also, it is important to note that settings are
// applied immediately. This means if you plan to reduce the width and move
// the x offset accordingly, you need to apply such changes in the appropriate
// order.
int ConfigureCustomImageSettings(CameraPtr pCam)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING CUSTOM IMAGE SETTINGS ***" << endl << endl;

    try
    {
        //
        // Apply mono 8 pixel format
        //
        // *** NOTES ***
        // In QuickSpin, enumeration nodes are as easy to set as other node
        // types. This is because enum values representing each entry node
        // are added to the API.
        //
        if (IsReadable(pCam->PixelFormat) && IsWritable(pCam->PixelFormat))
        {
            pCam->PixelFormat.SetValue(PixelFormat_Mono8);

            cout << "Pixel format set to " << pCam->PixelFormat.GetCurrentEntry()->GetSymbolic() << "..." << endl;
        }
        else
        {
            cout << "Pixel format not available..." << endl;
            result = -1;
        }

        //
        // Apply minimum to offset X
        //
        // *** NOTES ***
        // Numeric nodes have both a minimum and maximum. A minimum is retrieved
        // with the method GetMin(). Sometimes it can be important to check
        // minimums to ensure that your desired value is within range.
        //
        if (IsReadable(pCam->OffsetX) && IsWritable(pCam->OffsetX))
        {
            pCam->OffsetX.SetValue(pCam->OffsetX.GetMin());

            cout << "Offset X set to " << pCam->OffsetX.GetValue() << "..." << endl;
        }
        else
        {
            cout << "Offset X not available..." << endl;
            result = -1;
        }

        //
        // Apply minimum to offset Y
        //
        // *** NOTES ***
        // It is often desirable to check the increment as well. The increment
        // is a number of which a desired value must be a multiple. Certain
        // nodes, such as those corresponding to offsets X and Y, have an
        // increment of 1, which basically means that any value within range
        // is appropriate. The increment is retrieved with the method GetInc().
        //
        if (IsReadable(pCam->OffsetY) && IsWritable(pCam->OffsetY))
        {
            pCam->OffsetY.SetValue(pCam->OffsetY.GetMin());

            cout << "Offset Y set to " << pCam->OffsetY.GetValue() << "..." << endl;
        }
        else
        {
            cout << "Offset Y not available..." << endl;
            result = -1;
        }

        //
        // Set maximum width
        //
        // *** NOTES ***
        // Other nodes, such as those corresponding to image width and height,
        // might have an increment other than 1. In these cases, it can be
        // important to check that the desired value is a multiple of the
        // increment.
        //
        // This is often the case for width and height nodes. However, because
        // these nodes are being set to their maximums, there is no real reason
        // to check against the increment.
        //
        if (IsReadable(pCam->Width) && IsWritable(pCam->Width) && pCam->Width.GetInc() != 0 &&
            pCam->Width.GetMax() != 0)
        {
            pCam->Width.SetValue(pCam->Width.GetMax());

            cout << "Width set to " << pCam->Width.GetValue() << "..." << endl;
        }
        else
        {
            cout << "Width not available..." << endl;
            result = -1;
        }

        //
        // Set maximum height
        //
        // *** NOTES ***
        // A maximum is retrieved with the method GetMax(). A node's minimum and
        // maximum should always be a multiple of its increment.
        //
        if (IsReadable(pCam->Height) && IsWritable(pCam->Height) && pCam->Height.GetInc() != 0 &&
            pCam->Height.GetMax() != 0)
        {
            pCam->Height.SetValue(pCam->Height.GetMax());

            cout << "Height set to " << pCam->Height.GetValue() << "..." << endl;
        }
        else
        {
            cout << "Height not available..." << endl;
            result = -1;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int PrintDeviceInfo(CameraPtr pCam)
{
    int result = 0;

    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try
    {
        INodeMap& nodeMap = pCam->GetTLDeviceNodeMap();

        FeatureList_t features;
        CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsReadable(category))
        {
            category->GetFeatures(features);

            FeatureList_t::const_iterator it;
            for (it = features.begin(); it != features.end(); ++it)
            {
                CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = (CValuePtr)pfeatureNode;
                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                cout << endl;
            }
        }
        else
        {
            cout << "Device control information not readable." << endl;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function acquires and saves 10 images from a device; please see
// Acquisition example for more in-depth comments on the acquisition of images.
int AcquireImages(CameraPtr pCam)
{
    int result = 0;

    cout << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try
    {
        // Set acquisition mode to continuous
        if (!IsWritable(pCam->AcquisitionMode))
        {
            cout << "Unable to set acquisition mode to continuous. Aborting..." << endl << endl;
            return -1;
        }

        pCam->AcquisitionMode.SetValue(AcquisitionMode_Continuous);

        cout << "Acquisition mode set to continuous..." << endl;

        // Begin acquiring images
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        // Get device serial number for filename
        gcstring deviceSerialNumber("");

        if (IsReadable(pCam->DeviceSerialNumber))
        {
            deviceSerialNumber = pCam->DeviceSerialNumber.GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        // Retrieve, convert, and save images
        const int k_numImages = 10;

        //
        // Create ImageProcessor instance for post processing images
        //
        ImageProcessor processor;

        //
        // Set default image processor color processing method
        //
        // *** NOTES ***
        // By default, if no specific color processing algorithm is set, the image
        // processor will default to NEAREST_NEIGHBOR method.
        //
        processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
        {
            try
            {
                // Retrieve next received image and ensure image completion
                ImagePtr pResultImage = pCam->GetNextImage(1000);

                if (pResultImage->IsIncomplete())
                {
                    cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl
                         << endl;
                }
                else
                {
                    // Print image information
                    cout << "Grabbed image " << imageCnt << ", width = " << pResultImage->GetWidth()
                         << ", height = " << pResultImage->GetHeight() << endl;

                    // Convert image to mono 8
                    ImagePtr convertedImage = processor.Convert(pResultImage, PixelFormat_Mono8);

                    // Create a unique filename
                    ostringstream filename;

                    filename << "ImageFormatControlQS-";
                    if (deviceSerialNumber != "")
                    {
                        filename << deviceSerialNumber.c_str() << "-";
                    }
                    filename << imageCnt << ".jpg";

                    // Save image
                    convertedImage->Save(filename.str().c_str());

                    cout << "Image saved at " << filename.str() << endl;
                }

                // Release image
                pResultImage->Release();

                cout << endl;
            }
            catch (Spinnaker::Exception& e)
            {
                cout << "Error: " << e.what() << endl;
                result = -1;
            }
        }

        // End acquisition
        pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function acts as the body of the example; please see
// NodeMapInfo_QuickSpin example for more in-depth comments on setting
// up cameras.
int RunSingleCamera(CameraPtr pCam)
{
    int result = 0;

    try
    {
        // Initialize camera
        pCam->Init();

        // Print device info
        result = PrintDeviceInfo(pCam);

        // Configure custome image settings
        result = result | ConfigureCustomImageSettings(pCam);

        // Acquire images
        result = result | AcquireImages(pCam);

        // Deinitialize camera
        pCam->DeInit();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// Example entry point; please see Enumeration_QuickSpin example for more
// in-depth comments on preparing and cleaning up the system.
int main(int /*argc*/, char** /*argv*/)
{
    int result = 0;

    // Print application build information
    cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
    cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
         << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
         << endl;

    // Retrieve list of cameras from the system
    CameraList camList = system->GetCameras();

    unsigned int numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl << endl;

    // Finish if there are no cameras
    if (numCameras == 0)
    {
        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();

        cout << "Not enough cameras!" << endl;
        cout << "Done! Press Enter to exit..." << endl;
        getchar();

        return -1;
    }

    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        cout << endl << "Running example for camera " << i << "..." << endl;

        result = result | RunSingleCamera(camList.GetByIndex(i));

        cout << "Camera " << i << " example complete..." << endl << endl;
    }

    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return result;
}
