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
 *    @example Exposure.cpp
 *
 *    @brief Exposure.cpp shows how to set a custom exposure time on a device. It
 *    relies on information provided in the Enumeration, Acquisition, and
 *    NodeMapInfo examples.
 *
 *    This example shows the processes of preparing the camera, setting a custom
 *    exposure time, and restoring the camera to its default state (without power
 *    cycling). Ensuring custom values do not fall out of range is also touched
 *    on.
 *
 *    Following this, we suggest familiarizing yourself with the ImageFormatControl
 *    example if you haven't already. ImageFormatControl is another example on
 *    camera customization that is shorter and simpler than many of the others.
 *    Once comfortable with Exposure and ImageFormatControl, we suggest checking
 *    out any of the longer, more complicated examples related to camera
 *    configuration: ChunkData, LookupTable, Sequencer, or Trigger.
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
int ConfigureExposure(INodeMap& nodeMap)
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
        // time and needs to be turned off. Some models have auto-exposure
        // turned off by default
        //
        // *** LATER ***
        // Exposure time can be set automatically or manually as needed. This
        // example turns automatic exposure off to set it manually and back
        // on in order to return the camera to its default state.
        //
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (IsReadable(ptrExposureAuto) &&
            IsWritable(ptrExposureAuto))
        {
            CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
            if (IsReadable(ptrExposureAutoOff))
            {
                ptrExposureAuto->SetIntValue(ptrExposureAutoOff->GetValue());
                cout << "Automatic exposure disabled..." << endl;
            }
        }
        else 
        {
            CEnumerationPtr ptrAutoBright = nodeMap.GetNode("autoBrightnessMode");
            if (!IsReadable(ptrAutoBright) ||
                !IsWritable(ptrAutoBright))
            {
                cout << "Unable to get or set exposure time. Aborting..." << endl << endl;
                return -1;
            }
            cout << "Unable to disable automatic exposure. Expected for some models... " << endl;
            cout << "Proceeding..." << endl;
            result = 1;
        }

        //
        // Set exposure time manually; exposure time recorded in microseconds
        //
        // *** NOTES ***
        // The node is checked for availability and writability prior to the
        // setting of the node. Further, it is ensured that the desired exposure
        // time does not exceed the maximum. Exposure time is counted in
        // microseconds. This information can be found out either by
        // retrieving the unit with the GetUnit() method or by checking SpinView.
        //
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime) ||
            !IsWritable(ptrExposureTime))
        {
            cout << "Unable to get or set exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // Ensure desired exposure time does not exceed the maximum
        const double exposureTimeMax = ptrExposureTime->GetMax();
        double exposureTimeToSet = 500000.0;

        if (exposureTimeToSet > exposureTimeMax)
        {
            exposureTimeToSet = exposureTimeMax;
        }

        ptrExposureTime->SetValue(exposureTimeToSet);

        cout << std::fixed << "Exposure time set to " << exposureTimeToSet << " us..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function returns the camera to its default state by re-enabling automatic
// exposure.
int ResetExposure(INodeMap& nodeMap)
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
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (!IsReadable(ptrExposureAuto) ||
            !IsWritable(ptrExposureAuto))
        {
            cout << "Unable to enable automatic exposure (node retrieval). Non-fatal error..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrExposureAutoContinuous = ptrExposureAuto->GetEntryByName("Continuous");
        if (!IsReadable(ptrExposureAutoContinuous))
        {
            cout << "Unable to enable automatic exposure (enum entry retrieval). Non-fatal error..." << endl << endl;
            return -1;
        }

        ptrExposureAuto->SetIntValue(ptrExposureAutoContinuous->GetValue());

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
int PrintDeviceInfo(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try
    {
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
// Acquisition example for more in-depth comments on acquiring images.
int AcquireImages(CameraPtr pCam, INodeMap& nodeMap, INodeMap& nodeMapTLDevice)
{
    int result = 0;

    cout << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try
    {
        // Set acquisition mode to continuous
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsReadable(ptrAcquisitionMode) ||
            !IsWritable(ptrAcquisitionMode))
        {
            cout << "Unable to get or set acquisition mode to continuous (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to get acquisition mode to continuous (entry 'continuous' retrieval). Aborting..." << endl
                 << endl;
            return -1;
        }

        int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        cout << "Acquisition mode set to continuous..." << endl;

        // Begin acquiring images
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        // Retrieve device serial number for filename
        gcstring deviceSerialNumber("");

        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrStringSerial))
        {
            deviceSerialNumber = ptrStringSerial->GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        // Get the value of exposure time to set an appropriate timeout for GetNextImage
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime))
        {
            cout << "Unable to read exposure time. Aborting..." << endl << endl;
            return -1;
        }
        // The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit
        // being used in GetNextImage
        uint64_t timeout = static_cast<uint64_t>(ptrExposureTime->GetValue() / 1000 + 1000);

        // Retrieve, convert, and save images
        const unsigned int k_numImages = 5;

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

                    filename << "Exposure-";
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

// This function acts as the body of the example; please see NodeMapInfo example
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
    int result = 0;
    int err = 0;

    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure exposure
        err = ConfigureExposure(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Acquire images
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);

        // Reset exposure
        if (err == 0) 
        {
            result = result | ResetExposure(nodeMap);
        }
        else 
        {
            cout << "Skipping exposure reset" << endl << endl;
        }
        

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

// Example entry point; please see Enumeration example for additional
// comments on the steps in this function.
int main(int /*argc*/, char** /*argv*/)
{
    // Since this application saves images in the current folder
    // we must ensure that we have permission to write to this folder.
    // If we do not have permission, fail right away.
    FILE* tempFile = fopen("test.txt", "w+");
    if (tempFile == nullptr)
    {
        cout << "Failed to create file in current folder.  Please check "
                "permissions."
             << endl;
        cout << "Press Enter to exit..." << endl;
        getchar();
        return -1;
    }
    fclose(tempFile);
    remove("test.txt");

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