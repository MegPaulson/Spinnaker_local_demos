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
 *	@example LookupTable.cpp
 *
 *	@brief LookupTable.cpp shows how to configure lookup tables on the camera.
 *	It relies on information provided in the Enumeration, Acquisition, and
 *	NodeMapInfo examples.
 *
 *	It can also be helpful to familiarize yourself with the ImageFormatControl
 *	and Exposure examples. As they are somewhat shorter and simpler, either
 *	provides a strong introduction to camera customization.
 *
 *	Lookup tables allow for the customization and control of individual pixels.
 *	This can be a very powerful and deeply useful tool; however, because use
 *	cases are context dependent, this example only explores lookup table
 *	configuration.
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

// This function handles the error prints when a node or entry is 
// not readable on the connected camera
void PrintRetrieveNodeFailure(string node, string name)
{
    cout << "Unable to get " << node << " (" << name << " " << node << " retrieval failed)." << endl << endl;
    cout << "The " << node << " may not be readable on all camera models..." << endl;
    cout << "Please try a Blackfly S camera." << endl << endl;
}

// This function configures lookup tables linearly. This involves selecting the
// type of lookup table, finding the appropriate increment calculated from the
// maximum value, and enabling lookup tables on the camera.
int ConfigureLookupTables(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING LOOKUP TABLES ***" << endl << endl;

    try
    {
        //
        // Select lookup table type
        //
        // *** NOTES ***
        // Setting the lookup table selector. It is important to note that this
        // does not enable lookup tables.
        //
        CEnumerationPtr ptrLUTSelector = nodeMap.GetNode("LUTSelector");
        if (!IsReadable(ptrLUTSelector) ||
            !IsWritable(ptrLUTSelector))
        {
            // Some cameras have an LUT type that must be "User Defined"
            CEnumerationPtr ptrLUTType = nodeMap.GetNode("lutType");
            CEnumEntryPtr ptrLUTTypeType = nodeMap.GetNode("UserDefined");
            if (!IsReadable(ptrLUTTypeType) ||
                !IsWritable(ptrLUTTypeType)) 
            {
                PrintRetrieveNodeFailure("node", "LUTSelector");
                return -1;
            }
            ptrLUTType->SetIntValue(static_cast<int64_t>(ptrLUTTypeType->GetValue()));
        }

        CEnumEntryPtr ptrLUTSelectorLUT = ptrLUTSelector->GetEntryByName("LUT1");
        // Try out alternate node name
        if (!IsReadable(ptrLUTSelectorLUT))
        {
            ptrLUTSelectorLUT = ptrLUTSelector->GetEntryByName("UserDefined1");
            if (!IsReadable(ptrLUTSelectorLUT))
            {
                PrintRetrieveNodeFailure("entry", "LUTSelector LUT1 or User Defined 1");
                return -1;
            }
            cout << "Lookup table selector set to User Defined 1..." << endl;
        }
        else
        {
            cout << "Lookup table selector set to LUT 1..." << endl;
        }
        ptrLUTSelector->SetIntValue(static_cast<int64_t>(ptrLUTSelectorLUT->GetValue()));

        //
        // Determine pixel increment and set indexes and values as desired
        //
        // *** NOTES ***
        // To get the pixel increment, the maximum range of the value node must
        // first be retrieved. The value node represents an index, so its value
        // should be one less than a power of 2 (e.g. 511, 1023, etc.). Add 1 to
        // this index to get the maximum range. Divide the maximum range by 512
        // to calculate the pixel increment.
        //
        // Finally, all values (in the value node) and their corresponding
        // indexes (in the index node) need to be set. The goal of this example
        // is to set the lookup table linearly. As such, the slope of the values
        // should be set according to the increment, but the slope of the
        // indexes is inconsequential.
        //
        // Retrieve value node
        CIntegerPtr ptrLUTValue = nodeMap.GetNode("LUTValue");
        if (!IsReadable(ptrLUTValue) ||
            !IsWritable(ptrLUTValue))
        {
            PrintRetrieveNodeFailure("node", "LUTValue");
            return -1;
        }
        // Retrieve index node
        CIntegerPtr ptrLUTIndex = nodeMap.GetNode("LUTIndex");
        if (!IsWritable(ptrLUTIndex))
        {
            PrintRetrieveNodeFailure("node", "LUTIndex");
            return -1;
        }

        // Retrieve maximum value
        int maxVal = (int)ptrLUTValue->GetMax();
        cout << "\tMaximum Value: " << maxVal << endl;
        
        // Retrieve maximum index
        int maxIndex = (int)ptrLUTIndex->GetMax();
        cout << "\tMaximum Index: " << maxIndex << endl;

        // Calculate increment
        int increment = maxVal / maxIndex;

        // Set LUT Values
        if (increment > 0) 
        {
            cout << "\tIncrement: " << increment << endl;
            for(int i = 0; i < maxIndex; i++)
            {
                ptrLUTIndex->SetValue(i);
                ptrLUTValue->SetValue(i * increment);
            }
            
        }
        else 
        {
            int denominator = maxIndex / maxVal;
            cout << "\tIncrement: 1/" << denominator << endl;
            for(int i = 0; i < maxIndex; i++)
            {
                ptrLUTIndex->SetValue(i);
                ptrLUTValue->SetValue(i / denominator);
            }
        }

        cout << "All lookup table values set..." << endl;

        //
        // Enable lookup tables
        //
        // *** NOTES ***
        // Once lookup tables have been configured, don't forget to enable them
        // with the appropriate node.
        //
        // *** LATER ***
        // Once the images with lookup tables have been collected, turn the
        // feature off with the same node.
        //
        CBooleanPtr ptrLUTEnable = nodeMap.GetNode("LUTEnable");
        if (!IsWritable(ptrLUTEnable))
        {
            // Try alternate name
            CEnumerationPtr ptrLUTMode = nodeMap.GetNode("lutMode");
            if (!IsWritable(ptrLUTMode)) 
            {
                PrintRetrieveNodeFailure("node", "LUTEnable");
                return -1;
            }
            CEnumEntryPtr ptrLUTModeActive = ptrLUTMode->GetEntryByName("Active");
            ptrLUTMode->SetIntValue(ptrLUTModeActive->GetValue());
        }
        else
        {
            ptrLUTEnable->SetValue(true);
        }

        

        cout << "Lookup tables enabled..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function resets the camera by disabling lookup tables.
int ResetLookupTables(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        //
        // Disable lookup tables
        //
        // *** NOTES ***
        // Turn look up tables off when they are not needed to reduce overhead.
        //
        CBooleanPtr ptrLUTEnable = nodeMap.GetNode("LUTEnable");
        if (!IsWritable(ptrLUTEnable))
        {
            // Try alternate name
            CEnumerationPtr ptrLUTMode = nodeMap.GetNode("lutMode");
            if (!IsWritable(ptrLUTMode)) 
            {
                cout << endl << "Unable to disable lookup tables. Non-fatal error..." << endl << endl;
                return -1;
            }
            else 
            {
                CEnumEntryPtr ptrLUTModeOff = ptrLUTMode->GetEntryByName("Off");
                ptrLUTMode->SetIntValue(ptrLUTModeOff->GetValue());
            }

        }
        else 
        {
            ptrLUTEnable->SetValue(false);
        }
        

        cout << "Lookup tables disabled..." << endl << endl;
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

                    filename << "LookupTable-";
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

        // Configure lookup tables
        err = ConfigureLookupTables(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Acquire images
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);

        // Reset lookup tables
        result = result | ResetLookupTables(nodeMap);

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
// commnets on the steps in this function.
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
