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
 *  @example LogicBlock.cpp
 *
 *  @brief LogicBlock.cpp shows how to use logic blocks to detect missing
 *  triggers and refire.  It relies on information provided in the
 *  Acquisition and Trigger examples.
 *
 *  A logic block is a collection of combinatorial logic and latches that allows
 *  users to create new, custom signals inside the camera. These custom signals
 *  can be used by the camera (for example to trigger exposure) or sent out to
 *  integrate with external systems.
 *
 *   Logic Block functionality is only available for BFS and Oryx Cameras.
 *   For details on logic blocks and how this example works, see our kb article,
 *   "Using Logic Blocks with Blackfly S and Oryx";
 * https://www.flir.com/support-center/iis/machine-vision/application-note/using-logic-blocks-with-blackfly-s-and-oryx
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

// This function configures the camera to use a trigger. First, trigger mode is
// set to off in order to select the trigger source. Once the trigger source
// has been selected, trigger mode is then enabled.
int ConfigureTrigger(INodeMap& nodeMap)
{
    cout << endl << endl << "*** CONFIGURING TRIGGER ***" << endl << endl;
    int result = 0;

    try
    {
        //
        // Ensure trigger mode off
        //
        // *** NOTES ***
        // The trigger must be disabled in order to configure the source
        //
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode) ||
            !IsWritable(ptrTriggerMode))
        {
            cout << "Unable to disable trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerModeOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsReadable(ptrTriggerModeOff))
        {
            cout << "Unable to disable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOff->GetValue());

        cout << "Trigger mode disabled..." << endl;

        // Set trigger source to Logic Block 0
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (!IsReadable(ptrTriggerSource) ||
            !IsWritable(ptrTriggerSource))
        {
            cout << "Unable to set trigger mode (node retrieval). Aborting..." << endl;
            return -1;
        }

        // Set trigger mode to software
        CEnumEntryPtr ptrTriggerSourceLogicBlock0 = ptrTriggerSource->GetEntryByName("LogicBlock0");
        if (!IsReadable(ptrTriggerSourceLogicBlock0))
        {
            cout << "Unable to set trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerSource->SetIntValue(ptrTriggerSourceLogicBlock0->GetValue());

        cout << "Trigger source set to software..." << endl;

        // Set trigger activation to level high
        CEnumerationPtr ptrTriggerActivation = nodeMap.GetNode("TriggerActivation");
        if (!IsReadable(ptrTriggerActivation) ||
            !IsWritable(ptrTriggerActivation))
        {
            cout << "Unable to get or set trigger activation (enum retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerActivationLH = ptrTriggerActivation->GetEntryByName("LevelHigh");
        if (!IsReadable(ptrTriggerActivationLH))
        {
            cout << "Unable to get trigger mode ( entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerActivation->SetIntValue(ptrTriggerActivationLH->GetValue());

        cout << "Trigger activation set to level high..." << endl;

        // Turn trigger mode on
        CEnumEntryPtr ptrTriggerModeOn = ptrTriggerMode->GetEntryByName("On");
        if (!IsReadable(ptrTriggerModeOn))
        {
            cout << "Unable to enable trigger mode (enum entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrTriggerMode->SetIntValue(ptrTriggerModeOn->GetValue());

        cout << "Trigger mode turned back on..." << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function does the logic block configuration.
int ConfigureLogicBlock(INodeMap& nodeMap)
{
    cout << endl << endl << "*** CONFIGURING LOGIC BLOCKS ***" << endl << endl;

    // Select Logic Block 0
    CEnumerationPtr ptrLogicBlockSelector = nodeMap.GetNode("LogicBlockSelector");
    if (!IsReadable(ptrLogicBlockSelector) ||
        !IsWritable(ptrLogicBlockSelector))
    {
        cout << "Unable to get or set logic block selector to Logic Block 0 (node retrieval). Non-fatal error..." << endl;
        return -1;
    }

    CEnumEntryPtr ptrLogicBlock0 = ptrLogicBlockSelector->GetEntryByName("LogicBlock0");
    if (!IsReadable(ptrLogicBlock0))
    {
        cout << "Unable to get logic block selector to Logic Block 0 (enum entry retrieval). Non-fatal error..."
             << endl;
        return -1;
    }

    ptrLogicBlockSelector->SetIntValue(ptrLogicBlock0->GetValue());

    cout << "Logic Block 0 selected...." << endl;

    // Set Logic Block Lut to Enable
    CEnumerationPtr ptrLBLUTSelector = nodeMap.GetNode("LogicBlockLUTSelector");
    if (!IsReadable(ptrLBLUTSelector) ||
        !IsWritable(ptrLBLUTSelector))
    {
        cout << "Unable to get or set LUT logic block selector to Enable (node retrieval). Non-fatal error..." << endl;
        return -1;
    }

    CEnumEntryPtr ptrLBLUTEnable = ptrLBLUTSelector->GetEntryByName("Enable");
    if (!IsReadable(ptrLBLUTEnable))
    {
        cout << "Unable to get LUT logic block selector to Enable (enum entry retrieval). Non-fatal error..." << endl;
        return -1;
    }

    ptrLBLUTSelector->SetIntValue(ptrLBLUTEnable->GetValue());

    cout << "Logic Block LUT set to to Enable..." << endl;

    // Set Logic Block LUT Output Value All to 0xFC
    CIntegerPtr ptrLBLUTOutputValueAll = nodeMap.GetNode("LogicBlockLUTOutputValueAll");
    if (!IsWritable(ptrLBLUTOutputValueAll))
    {
        cout << "Unable to set value to LUT logic block output value all (integer retrieval). Non-fatal error..."
             << endl;
        return -1;
    }

    ptrLBLUTOutputValueAll->SetValue(252);

    cout << "Logic Block LUT Output Value All set to to OxFC..." << endl;

    // Set Logic Block LUT Input Selector to Input 0
    CEnumerationPtr ptrLBLUTInputSelector = nodeMap.GetNode("LogicBlockLUTInputSelector");
    if (!IsReadable(ptrLBLUTInputSelector) ||
        !IsWritable(ptrLBLUTInputSelector))
    {
        cout << "Unable to get or set LUT logic block input selector to Input 0 (node retrieval). Non-fatal error..." << endl;
        return -1;
    }

    CEnumEntryPtr ptrLBLUTInput0 = ptrLBLUTInputSelector->GetEntryByName("Input0");
    if (!IsReadable(ptrLBLUTInput0))
    {
        cout << "Unable to get LUT logic block selector to Input 0 (enum entry retrieval). Non-fatal error..." << endl;
        return -1;
    }

    ptrLBLUTInputSelector->SetIntValue(ptrLBLUTInput0->GetValue());

    cout << "Logic Block LUT Input Selector set to to Input 0 ..." << endl;

    // Set Logic Block LUT Input Source to FrameTriggerWait
    CEnumerationPtr ptrLBLUTSource = nodeMap.GetNode("LogicBlockLUTInputSource");
    if (!IsReadable(ptrLBLUTSource) ||
        !IsWritable(ptrLBLUTSource))
    {
        cout << "Unable to get or set LUT logic block input source to Frame Trigger Wait (node retrieval). Non-fatal error..."
             << endl;
        return -1;
    }

    CEnumEntryPtr ptrLBLUTSourceFTW = ptrLBLUTSource->GetEntryByName("FrameTriggerWait");
    if (!IsReadable(ptrLBLUTSourceFTW))
    {
        cout << "Unable to get LUT logic block input source to Frame Trigger Wait (enum entry retrieval). Non-fatal "
                "error..."
             << endl;
        return -1;
    }

    ptrLBLUTSource->SetIntValue(ptrLBLUTSourceFTW->GetValue());

    cout << "Logic Block LUT Input Source set to to Frame Trigger Wait ..." << endl;

    // Set Logic Block LUT Activation Type to Level High
    CEnumerationPtr ptrLBLUTActivation = nodeMap.GetNode("LogicBlockLUTInputActivation");
    if (!IsReadable(ptrLBLUTActivation) ||
        !IsWritable(ptrLBLUTActivation))
    {
        cout << "Unable to get or set LUT logic block input activation to level high (node retrieval). Non-fatal error..."
             << endl;
        return -1;
    }

    CEnumEntryPtr ptrLBLUTActivationLevelHigh = ptrLBLUTActivation->GetEntryByName("LevelHigh");
    if (!IsReadable(ptrLBLUTActivationLevelHigh))
    {
        cout
            << "Unable to get LUT logic block input activation to level high (enum entry retrieval). Non-fatal error..."
            << endl;
        return -1;
    }

    ptrLBLUTActivation->SetIntValue(ptrLBLUTActivationLevelHigh->GetValue());

    cout << "Logic Block LUT Input Activation set to level high..." << endl;

    // Set Logic Block LUT Input Selector to Input 1
    CEnumEntryPtr ptrLBLUTInput1 = ptrLBLUTInputSelector->GetEntryByName("Input1");
    if (!IsReadable(ptrLBLUTInput1))
    {
        cout << "Unable to get LUT logic block Input selector to input 1 (enum entry retrieval). Non-fatal error..."
             << endl;
        return -1;
    }

    ptrLBLUTInputSelector->SetIntValue(ptrLBLUTInput1->GetValue());

    cout << "Logic Block LUT Input Selector set to to Input 1 ..." << endl;

    // Set Logic Block LUT Source to User Output 0
    CEnumEntryPtr ptrLBLUTSourceUO0 = ptrLBLUTSource->GetEntryByName("UserOutput0");
    if (!IsReadable(ptrLBLUTSourceUO0))
    {
        cout << "Unable to get LUT logic block input source to User Output 0 (enum entry retrieval). Non-fatal error..."
             << endl;
        return -1;
    }

    ptrLBLUTSource->SetIntValue(ptrLBLUTSourceUO0->GetValue());

    cout << "Logic Block LUT Input Source set to to User Output 0 ..." << endl;

    // Set Logic Block LUT Activation Type to Rising Edge
    CEnumEntryPtr ptrLBLUTActivationRisingEdge = ptrLBLUTActivation->GetEntryByName("RisingEdge");
    if (!IsReadable(ptrLBLUTActivationRisingEdge))
    {
        cout << "Unable to get LUT logic block input activation to Rising Edge (enum entry retrieval). Non-fatal "
                "error..."
             << endl;
        return -1;
    }

    ptrLBLUTActivation->SetIntValue(ptrLBLUTActivationRisingEdge->GetValue());

    cout << "Logic Block LUT Input Activation set to Rising Edge..." << endl;

    // Set Logic Block LUT Input Selector to Input 2
    CEnumEntryPtr ptrLBLUTInput2 = ptrLBLUTInputSelector->GetEntryByName("Input2");
    if (!IsReadable(ptrLBLUTInput2))
    {
        cout << "Unable to get LUT logic block selector to input 2 (enum entry retrieval). Non-fatal error..." << endl;
        return -1;
    }

    ptrLBLUTInputSelector->SetIntValue(ptrLBLUTInput2->GetValue());

    cout << "Logic Block LUT Input Selector set to to Input 2 ..." << endl;

    // Set Logic Block LUT Source to Exposure Start
    CEnumEntryPtr ptrLBLUTSourceExposureStart = ptrLBLUTSource->GetEntryByName("ExposureStart");
    if (!IsReadable(ptrLBLUTSourceExposureStart))
    {
        cout
            << "Unable to get LUT logic block input source to Exposure Start (enum entry retrieval). Non-fatal error..."
            << endl;
        return -1;
    }

    ptrLBLUTSource->SetIntValue(ptrLBLUTSourceExposureStart->GetValue());

    cout << "Logic Block LUT Input Source set to to Exposure Start ..." << endl;

    // Set Logic Block LUT Activation Type to Rising Edge
    ptrLBLUTActivation->SetIntValue(ptrLBLUTActivationRisingEdge->GetValue());

    cout << "Logic Block LUT Input Activation set to Rising Edge..." << endl;

    // Set Logic Block Lut Selector to Value
    CEnumEntryPtr ptrLBLUTValue = ptrLBLUTSelector->GetEntryByName("Value");
    if (!IsReadable(ptrLBLUTValue))
    {
        cout << "Unable to get LUT logic block selector to Value (enum entry retrieval). Non-fatal error..." << endl;
        return -1;
    }

    ptrLBLUTSelector->SetIntValue(ptrLBLUTValue->GetValue());

    cout << "Logic Block LUT set to to Value..." << endl;

    // Set Logic Block LUT output Value All to 0x4C
    ptrLBLUTOutputValueAll->SetValue(76);

    cout << "Logic Block LUT Output Value All set to to Ox4C..." << endl << endl;

    return 0;
}

// This function retrieves two images, using user output 0 as a makeshift trigger
int GrabTwoImages(INodeMap& nodeMap, CameraPtr pCam)
{
    int result = 0;

    try
    {
        // Select User Output 0
        CEnumerationPtr ptrUserOutputSelector = nodeMap.GetNode("UserOutputSelector");
        if (!IsReadable(ptrUserOutputSelector) ||
            !IsWritable(ptrUserOutputSelector))
        {
            cout << "Unable to get or set User Output Selector (enum retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrUserOutput0 = ptrUserOutputSelector->GetEntryByName("UserOutput0");
        if (!IsReadable(ptrUserOutput0))
        {
            cout << "Unable to get User Output Selector (entry retrieval). Aborting..." << endl;
            return -1;
        }

        ptrUserOutputSelector->SetIntValue(ptrUserOutput0->GetValue());

        // Set user output value to true to capture an image
        CBooleanPtr ptrUserOutputValue = nodeMap.GetNode("UserOutputValue");
        if (!IsWritable(ptrUserOutputValue))
        {
            cout << "Unable to set User Output value (boolean retrieval). Aborting..." << endl;
            return -1;
        }

        // Trigger the camera twice with two consecutive triggers
        //
        // *** NOTES ***
        // User Output 0's value is changed from false to true twice,
        // causing the camera to be triggered to capture an image for each
        // transition (rising edge). One trigger captures an image,
        // and the second trigger will be detected and cause the
        // camera to capture an image once the first triggered image
        // is complete.
        //
        ptrUserOutputValue->SetValue(false);
        ptrUserOutputValue->SetValue(true);
        ptrUserOutputValue->SetValue(false);
        ptrUserOutputValue->SetValue(true);
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function returns the camera to a normal state by turning off trigger
// mode.
int ResetTrigger(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn trigger mode back off
        //
        // *** NOTES ***
        // Once all images have been captured, turn trigger mode back off to
        // restore the camera to a clean state.
        //
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

        cout << "Trigger mode disabled..." << endl << endl;
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

    cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

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

        // Turn off auto exposure functionality
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (!IsReadable(ptrExposureAuto) ||
            !IsWritable(ptrExposureAuto))
        {
            cout << "Unable to get or set Exposure Auto (enum retrieval). Aborting...." << endl;
            return -1;
        }

        CEnumEntryPtr ptrExposureAutoEntry = ptrExposureAuto->GetEntryByName("Off");
        if (!IsReadable(ptrExposureAutoEntry))
        {
            cout << "Unable to get Exposure Auto (enum entry Retrieval). Aborting..." << endl;
        }

        ptrExposureAuto->SetIntValue(ptrExposureAutoEntry->GetValue());

        // Set exposure time to 0.5 seconds
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsWritable(ptrExposureTime))
        {
            cout << "Unable to set Exposure Time (float retrieaval).  Aborting...";
        }

        ptrExposureTime->SetValue(500000);

        // Begin acquiring images
        //
        // *** NOTES ***
        // Due to the logic block configuration we are using, after
        // first initializing the logic block setup, the camera will automatically
        // capture an image after calling beginacquisition(); calling endacquisition
        // then beginacquisition() will clear this image and restart the capture
        //
        pCam->BeginAcquisition();
        pCam->EndAcquisition();
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

        unsigned int imageCnt = 0;

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

        while (imageCnt < k_numImages)
        {
            try
            {
                // Retrieve the next two images from the trigger
                result = result | GrabTwoImages(nodeMap, pCam);

                for (unsigned int imageCntb = 0; imageCntb < 2; imageCntb++)
                {
                    // Retrieve the next received image
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

                        filename << "LogicBlock-";
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
                    imageCnt++;
                }
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

        // Configure Logic Block
        err = ConfigureLogicBlock(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Configure trigger
        err = ConfigureTrigger(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Acquire images
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);

        // Reset trigger
        result = result | ResetTrigger(nodeMap);

        // Reset Exposure
        result = result | ResetExposure(nodeMap);

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
    if (tempFile == NULL)
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
