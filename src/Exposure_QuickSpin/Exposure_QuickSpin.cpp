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
 *	@example Exposure_QuickSpin.cpp
 *
 *	@brief Exposure_QuickSpin.cpp shows how to customize image exposure time
 *	using the QuickSpin API. QuickSpin is a subset of the Spinnaker library
 *	that allows for simpler node access and control.
 *
 *  This example prepares the camera, sets a new exposure time, and restores
 *  the camera to its default state. Ensure custom values fall within an
 *  acceptable range is also touched on. Retrieving and setting information
 *  is the only portion of hte example that differs from Exposure.
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

// This function configures a custom exposure time. Automatic exposure is turned
// off in order to allow for the customization, and then the custom setting is
// applied.
int ConfigureExposure(CameraPtr pCam)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING EXPOSURE ***" << endl << endl;

    try
    {
        //
        // Turn off automatic exposure mode
        //
        // *** NOTES ***
        // Automatic exposure prevents the manual configuration of exposure
        // times and needs to be turned off for this example. Enumerations
        // representing entry nodes have been added to QuickSpin. This allows
        // for the much easier setting of enumeration nodes to new values.
        //
        // In C++, the naming convention of QuickSpin enums is the name of the
        // enumeration node followed by an underscore and the symbolic of
        // the entry node. Selecting "Off" on the "ExposureAuto" node is
        // thus named "ExposureAuto_Off".
        //
        // *** LATER ***
        // Exposure time can be set automatically or manually as needed. This
        // example turns automatic exposure off to set it manually and back
        // on to return the camera to its default state.
        //
        if (!IsWritable(pCam->ExposureAuto))
        {
            cout << "Unable to disable automatic exposure. Aborting..." << endl << endl;
            return -1;
        }

        pCam->ExposureAuto.SetValue(ExposureAuto_Off);

        cout << "Automatic exposure disabled..." << endl;

        //
        // Set exposure time manually; exposure time recorded in microseconds
        //
        // *** NOTES ***
        // Notice that the node is checked for availability and writability
        // prior to the setting of the node. In QuickSpin, availability is
        // ensured by checking for null while writability is ensured by checking
        // the access mode.
        //
        // Further, it is ensured that the desired exposure time does not exceed
        // the maximum. Exposure time is counted in microseconds - this can be
        // found out either by retrieving the unit with the GetUnit() method or
        // by checking SpinView.
        //
        if (!IsReadable(pCam->ExposureTime) || !IsWritable(pCam->ExposureTime))
        {
            cout << "Unable to set exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // Ensure desired exposure time does not exceed the maximum
        const double exposureTimeMax = pCam->ExposureTime.GetMax();
        double exposureTimeToSet = 2000000.0;

        if (exposureTimeToSet > exposureTimeMax)
        {
            exposureTimeToSet = exposureTimeMax;
        }

        pCam->ExposureTime.SetValue(exposureTimeToSet);

        cout << std::fixed << "Shutter time set to " << exposureTimeToSet << " us..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function returns the camera to a normal state by re-enabling automatic
// exposure.
int ResetExposure(CameraPtr pCam)
{
    int result = 0;

    try
    {
        //
        // Turn automatic exposure back on
        //
        // *** NOTES ***
        // Automatic exposure is turned on in order to return the camera to its
        // default state.
        //
        if (!IsWritable(pCam->ExposureAuto))
        {
            cout << "Unable to enable automatic exposure (node retrieval). Non-fatal error..." << endl << endl;
            return -1;
        }

        pCam->ExposureAuto.SetValue(ExposureAuto_Continuous);

        cout << "Automatic exposure enabled..." << endl << endl;
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
            cout << "Device control information not available." << endl;
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

        if (IsReadable(pCam->TLDevice.DeviceSerialNumber))
        {
            deviceSerialNumber = pCam->TLDevice.DeviceSerialNumber.GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        // Get the value of exposure time to set an appropriate timeout for GetNextImage
        if (!IsReadable(pCam->ExposureTime))
        {
            cout << "Unable to read exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit
        // being used in GetNextImage
        uint64_t timeout = static_cast<uint64_t>(pCam->ExposureTime.GetValue() / 1000 + 1000);

        // Retrieve, convert, and save images
        const int k_numImages = 5;

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
                // By default, GetNextImage will block indefinitely until an image arrives.
                // In this example, the timeout value is set to [exposure time + 1000]ms to ensure that an image has
                // enough time to arrive under normal conditions
                ImagePtr pResultImage = pCam->GetNextImage(timeout);

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

                    filename << "ExposureQS-";
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
    int err = 0;

    try
    {
        // Initialize camera
        pCam->Init();

        // Print device info
        result = PrintDeviceInfo(pCam);

        // Configure exposure
        err = ConfigureExposure(pCam);
        if (err < 0)
        {
            return err;
        }

        // Acquire images
        result = result | AcquireImages(pCam);

        // Reset exposure
        result = result | ResetExposure(pCam);

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
