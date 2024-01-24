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
 *   @example CounterAndTimer.cpp
 *
 *   @brief CounterAndTimer.cpp shows how to setup a Pulse Width Modulation (PWM)
 *   signal using counters and timers. The camera will output the PWM signal via
 *   strobe, and capture images at a rate defined by the PWM signal as well.
 *   Users should take care to use a PWM signal within the camera's max
 *   framerate (by default, the PWM signal is set to 50 Hz).
 *
 *   Counter and Timer functionality is only available for BFS and Oryx Cameras.
 *   Some cameras lack all the functionality required to output PWM. We have
 *   included a demonstration to acquire an image every 2 seconds via the counter
 *   in this example. 
 *   For details on the hardware setup, see our kb article, "Using Counter and
 *   Timer Control";
 * https://www.flir.com/support-center/iis/machine-vision/application-note/using-counter-and-timer-control
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

inline int AssignEntryToEnum(INodeMap& nodeMap, gcstring enumName, gcstring entryName)
{
    CEnumerationPtr enumPtr = nodeMap.GetNode(enumName);
    if (!IsReadable(enumPtr) ||
        !IsWritable(enumPtr))
    {
        cout << "Unable to get or set " << enumName << "(enum retrieval). Aborting..." << endl;
        return -1;
    }

    CEnumEntryPtr entryPtr = enumPtr->GetEntryByName(entryName);
    if (!IsReadable(entryPtr))
    {
        cout << "Unable to get " << entryName << "(entry retrieval). Aborting..." << endl;
        return -1;
    }

    enumPtr->SetIntValue(entryPtr->GetValue());
    return 0;
}

int SetupCounterAndTimer(INodeMap& nodeMap)
{
    int result = 0;
    cout << endl << "Configuring Pulse Width Modulation signal" << endl;
    try
    {
        if (AssignEntryToEnum(nodeMap, "CounterSelector", "Counter0") == -1)
        {
            return -1;
        }
        if(AssignEntryToEnum(nodeMap, "CounterEventSource", "MHzTick") == -1)
        {
            return -1;
        }
        

        CIntegerPtr ptrCounterDuration = nodeMap.GetNode("CounterDuration");
        if (!IsReadable(ptrCounterDuration) ||
            !IsWritable(ptrCounterDuration))
        {
            cout << "Unable to get or set Counter Duration (integer retrieval). Aborting..." << endl << endl;
            return -1;
        }
        ptrCounterDuration->SetValue(14000);
        // Set Counter Delay to 6000
        CIntegerPtr ptrCounterDelay = nodeMap.GetNode("CounterDelay");
        if (!IsReadable(ptrCounterDelay) ||
            !IsWritable(ptrCounterDelay))
        {
            cout << "Unable to get or set Counter Delay (integer retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrCounterDelay->SetValue(6000);

        // Determine Duty Cycle of PWM signal
        int64_t dutyCycle = (int64_t)(
            (float)ptrCounterDuration->GetValue() /
            ((float)ptrCounterDuration->GetValue() + (float)ptrCounterDelay->GetValue()) * 100);

        cout << endl << "The duty cycle has been set to " << dutyCycle << "%" << endl;

        // Determine pulse rate of PWM signal
        int64_t pulseRate =
            (int64_t)(1000000 / ((float)ptrCounterDuration->GetValue() + (float)ptrCounterDelay->GetValue()));

        cout << endl << "The pulse rate has been set to " << pulseRate << "Hz" << endl;

        if (AssignEntryToEnum(nodeMap, "CounterTriggerSource", "FrameTriggerWait") == -1)
        {
            return -1;
        }
        if (AssignEntryToEnum(nodeMap, "CounterTriggerActivation", "LevelHigh") == -1)
        {
            return -1;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl << endl;
        result = -1;
    }

    return result;
}

int SetupCounterAndTimerReduced(INodeMap& nodeMap)
{
    int result = 0;
    cout << endl << "Configuring Reduced CounterAndTimer Demo" << endl << endl;
    try
    {
        // Counter mode Active seems to block counter configuration and exposure configuration.
        if (AssignEntryToEnum(nodeMap, "counterMode", "Off") == -1)
        {
            return -1;
        }
        
        cout << "Turning off counter mode to enable counter configuration changes" << endl;

        if (AssignEntryToEnum(nodeMap, "counterSelector", "Counter1") == -1)
        {
            return -1;
        }
        
        if (AssignEntryToEnum(nodeMap, "counterIncrementalSource", "InternalClock") == -1)
        {
            return -1;
        }

        CIntegerPtr ptrCounterDuration = nodeMap.GetNode("counterDuration");
        if (!IsReadable(ptrCounterDuration) ||
            !IsWritable(ptrCounterDuration))
        {
            cout << "Unable to get or set Counter Duration (integer retrieval). Aborting..." << endl;
            return -1;
        }
        ptrCounterDuration->SetValue(2000000);
        cout << "Set Counter Duration to 2 seconds" << endl;

        if (AssignEntryToEnum(nodeMap, "counterStartSource", "ExposureStart") == -1)
        {
            return -1;
        }
        if (AssignEntryToEnum(nodeMap, "counterResetSource", "Counter1End") == -1)
        {
            return -1;
        }
        if (AssignEntryToEnum(nodeMap, "TriggerMode", "On") == -1)
        {
            return -1;
        }
        
        if (AssignEntryToEnum(nodeMap, "TriggerSource", "Counter1End") == -1)
        {
            return -1;
        }

        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrCounterDuration) ||
            !IsWritable(ptrCounterDuration))
        {
            cout << "Unable to get or set Exposure Time (integer retrieval). Aborting..." << endl;
            return -1;
        }
        ptrExposureTime->SetValue(15000.0);
        cout << "Set Exposure Time to 15,000us" << endl;

        cout << "Re-activating counter mode" << endl;
        if (AssignEntryToEnum(nodeMap, "counterMode", "Active") == -1)
        {
            return -1;
        }

        cout << endl << "Configuration finished" << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl << endl;
        result = -1;
    }

    return result;
}
// Configure GPIO to output the PWM signal
int ConfigureDigitalIO(INodeMap& nodeMap)
{
    int result = 0;
    const gcstring cameraFamilyBFS = "BFS";
    const gcstring cameraFamilyOryx = "ORX";

    cout << endl << "Configuring GPIO strobe output" << endl;

    try
    {
        // Determine camera family
        CStringPtr ptrDeviceModelName = nodeMap.GetNode("DeviceModelName");
        if (!IsReadable(ptrDeviceModelName))
        {
            cout << "Unable to determine camera family. Aborting..." << endl << endl;
            return -1;
        }

        gcstring cameraModel = ptrDeviceModelName->GetValue();

        if (cameraModel.find(cameraFamilyBFS) != std::string::npos)
        {
            if(AssignEntryToEnum(nodeMap, "LineSelector", "Line1") == -1)
            {
                return -1;
            }
        }
        else if (cameraModel.find(cameraFamilyOryx) != std::string::npos)
        {
            if (AssignEntryToEnum(nodeMap, "LineSelector", "Line2") == -1)
            {
                return -1;
            }

            // Set Line Mode to output
            if (AssignEntryToEnum(nodeMap, "LineMode", "Output") == -1)
            {
                return -1;
            }
        }

        // Set Line Source for Selected Line to Counter 0 Active
        if (AssignEntryToEnum(nodeMap, "LineSource", "Counter0Active") == -1)
        {
            return -1;
        }

        if (cameraModel.find(cameraFamilyBFS) != std::string::npos)
        {
            // Change Line Selector to Line 2 and Enable 3.3 Voltage Rail
            if (AssignEntryToEnum(nodeMap, "LineSelector", "Line2") == -1)
            {
                return -1;
            }

            CBooleanPtr ptrVoltageEnable = nodeMap.GetNode("V3_3Enable");
            if (!IsWritable(ptrVoltageEnable))
            {
                cout << "Unable to set Voltage Enable (boolean retrieval). Aborting..." << endl << endl;
                return -1;
            }

            ptrVoltageEnable->SetValue(true);
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl << endl;
        result = -1;
    }

    return result;
}


// This function configures the camera to set a manual exposure value and enables
// camera to be triggered by the PWM signal.
int ConfigureExposureandTrigger(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << "Configuring Exposure and Trigger" << endl;

    try
    {
        
        // Turn off auto exposure
        if (AssignEntryToEnum(nodeMap, "ExposureAuto", "Off") == -1)
        {
            return -1;
        }
        // Set Exposure Time to less than 1/50th of a second (5000 us is used as an example)
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsWritable(ptrExposureTime))
        {
            cout << "Unable to set Exposure Time (float retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrExposureTime->SetValue(5000);

        // Ensure trigger mode is off
        //
        // *** NOTES ***
        // The trigger must be disabled in order to configure
        //
        if (AssignEntryToEnum(nodeMap, "TriggerMode", "Off") == -1)
        {
            return -1;
        }
        
        // Set Trigger Source to Counter 0 Start
        if (AssignEntryToEnum(nodeMap, "TriggerSource", "Counter0Start") == -1)
        {
            return -1;
        }

        // Set Trigger Overlap to Readout
        if (AssignEntryToEnum(nodeMap, "TriggerOverlap", "ReadOut") == -1)
        {
            return -1;
        }

        // Turn trigger mode on
        if (AssignEntryToEnum(nodeMap, "TriggerMode", "On") == -1)
        {
            return -1;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl << endl;
        result = -1;
    }

    return result;
}

// This function acquires and saves 10 images from a device; please see
// Acquisition example for more in-depth comments on acquiring images.
int AcquireImages(CameraPtr pCam, INodeMap& nodeMap, INodeMap& nodeMapTLDevice, bool reduced)
{
    int result = 0;

    cout << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try
    {
        // Set acquisition mode to continuous
        if (AssignEntryToEnum(nodeMap, "AcquisitionMode", "Continuous") == -1)
        {
            return -1;
        }
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

        // Retrieve, convert, and save images
        const unsigned int k_numImages = 10;

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

        if (reduced) 
        {
            CCommandPtr ptrSoftwareTrigger = nodeMap.GetNode("TriggerSoftware");
            if (!IsWritable(ptrSoftwareTrigger))
            {
                cout << "Failed to execute software trigger... Aborting" << endl;
            }
            ptrSoftwareTrigger->Execute();
        }

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
        {
            try
            {
                // Retrieve next received image and ensure image completion
                ImagePtr pResultImage = pCam->GetNextImage(3000);

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

                    filename << "CounterAndTimer-";
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
                cout << "Error: " << e.what() << endl << endl;
                result = -1;
            }
        }

        // End acquisition
        pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl << endl;
        result = -1;
    }

    return result;
}

// This function returns the camera to a normal state by turning off trigger
// mode.
//
// *** NOTES ***
// This function turns off trigger mode, but does not change the trigger
// source.
//
int ResetTrigger(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        // Turn trigger mode back off
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode) ||
            !IsWritable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Non-fatal error..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Non-fatal error..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function acts as the body of the example; please see the NodeMapInfo example
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

        // Configure Counter and Timer setup
        err = SetupCounterAndTimer(nodeMap);
        if (err < 0)
        {
            cout << "Regular setup failed... Trying reduced functionality demo" << endl;
            // Try reduced functionality setup
            err = SetupCounterAndTimerReduced(nodeMap);
            if (err < 0)
            {
                return err;
            }
            // Set err to different value if reduced functionality setup succeeds
            err = 1;
        }

        if (err == 0)
        {
            // Configure DigitalIO (GPIO output)
            err = ConfigureDigitalIO(nodeMap);
            if (err < 0)
            {
                return err;
            }

            // Configure Exposure and Trigger
            err = ConfigureExposureandTrigger(nodeMap);
            if (err < 0)
            {
                return err;
            }
        }

        // Acquire images. 
        // err == 0, acquire normally
        // err == 1, start acquisition using sofware trigger for reduced functionality
        if (err == 0)
        {
            result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice, false);
        }
        else if (err == 1)
        {
            result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice, true);
        }

        // Reset trigger
        result = result | ResetTrigger(nodeMap);

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

// Example entry point; please see Enumeration example for more in-depth
// comments on preparing and cleaning up the system.
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

    //
    // Create shared pointer to camera
    //
    // *** NOTES ***
    // The CameraPtr object is a shared pointer, and will generally clean itself
    // up upon exiting its scope. However, if a shared pointer is created in the
    // same scope that a system object is explicitly released (i.e. this scope),
    // the reference to the shared point must be broken manually.
    //
    // *** LATER ***
    // Shared pointers can be terminated manually by assigning them to nullptr.
    // This prevents an exception from being thrown when releasing the system.
    //
    CameraPtr pCam = nullptr;

    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera
        pCam = camList.GetByIndex(i);

        cout << endl << "Running example for camera " << i << "..." << endl;

        // Run example
        result = result | RunSingleCamera(pCam);

        cout << endl << "Camera " << i << " example complete..." << endl << endl;
    }

    //
    // Release reference to the camera
    //
    // *** NOTES ***
    // Had the CameraPtr object been created within the for-loop, it would not
    // be necessary to manually break the reference because the shared pointer
    // would have automatically cleaned itself up upon exiting the loop.
    //
    pCam = nullptr;

    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return result;
}
