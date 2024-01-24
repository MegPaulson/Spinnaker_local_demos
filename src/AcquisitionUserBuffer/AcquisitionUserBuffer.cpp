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
 *  @example AcquisitionUserBuffer.cpp
 *
 *  @brief AcquisitionUserBuffer.cpp shows how to use User Buffers for image
 *  acquisition.  The acquisition engine uses a pool of memory buffers.  The
 *  memory of a buffer can be allocated by the library (default) or the user.
 *  User Buffers refer to the latter.  This example relies on information
 *  provided in the Acquisition example.
 *
 *  This example demonstrates setting up the user allocated memory just before
 *  the acquisition of images.  First, the size of each buffer is determined
 *  based on the data payload size.  Then, depending on the the number of
 *  buffers (numBuffers) specified, the corresponding amount of memory is
 *  allocated.  Finally, after setting the buffer ownership to be users,
 *  the image acquisition can commence.
 *
 *  It is important to note that if the user provides the memory for the
 *  buffers, the user is ultimately responsible for freeing up memory.
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

// Whether the user memory is contiguous or non-contiguous
const bool isContiguous = true;

// Disables or enables heartbeat on GEV cameras so debugging does not incur timeout errors
int ConfigureGVCPHeartbeat(CameraPtr pCam, bool enable)
{
    //
    // Write to boolean node controlling the camera's heartbeat
    //
    // *** NOTES ***
    // This applies only to GEV cameras.
    //
    // GEV cameras have a heartbeat built in, but when debugging applications the
    // camera may time out due to its heartbeat. Disabling the heartbeat prevents
    // this timeout from occurring, enabling us to continue with any necessary 
    // debugging.
    //
    // *** LATER ***
    // Make sure that the heartbeat is reset upon completion of the debugging.  
    // If the application is terminated unexpectedly, the camera may not locked
    // to Spinnaker indefinitely due to the the timeout being disabled.  When that 
    // happens, a camera power cycle will reset the heartbeat to its default setting.

    // Retrieve TL device nodemap
    INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

    // Retrieve GenICam nodemap
    INodeMap& nodeMap = pCam->GetNodeMap();

    CEnumerationPtr ptrDeviceType = nodeMapTLDevice.GetNode("DeviceType");
    if (!IsReadable(ptrDeviceType))
    {
        return -1;
    }

    if (ptrDeviceType->GetIntValue() != DeviceType_GigEVision)
    {
        return 0;
    }

    if (enable)
    {
        cout << endl << "Resetting heartbeat..." << endl << endl;
    }
    else
    {
        cout << endl << "Disabling heartbeat..." << endl << endl;
    }

    CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
    if (!IsWritable(ptrDeviceHeartbeat))
    {
        cout << "Unable to configure heartbeat. Continuing with execution as this may be non-fatal..."
            << endl
            << endl;
    }
    else
    {
        ptrDeviceHeartbeat->SetValue(enable);

        if (!enable)
        {
            cout << "WARNING: Heartbeat has been disabled for the rest of this example run." << endl;
            cout << "         Heartbeat will be reset upon the completion of this run.  If the " << endl;
            cout << "         example is aborted unexpectedly before the heartbeat is reset, the" << endl;
            cout << "         camera may need to be power cycled to reset the heartbeat." << endl << endl;
        }
        else
        {
            cout << "Heartbeat has been reset." << endl;
        }
    }

    return 0;
}

int ResetGVCPHeartbeat(CameraPtr pCam)
{
    return ConfigureGVCPHeartbeat(pCam, true);
}

int DisableGVCPHeartbeat(CameraPtr pCam)
{
    return ConfigureGVCPHeartbeat(pCam, false);
}

// This function acquires and saves 10 images from a device.
int AcquireImages(CameraPtr pCam, INodeMap& nodeMap, INodeMap& nodeMapTLDevice)
{
    int result = 0;

    cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try
    {
        // Set acquisition mode to continuous

        // Retrieve enumeration node from nodemap
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsReadable(ptrAcquisitionMode) ||
            !IsWritable(ptrAcquisitionMode))
        {
            cout << "Unable to get or set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve entry node from enumeration node
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        // Retrieve Stream Parameters device nodemap
        const INodeMap& sNodeMap = pCam->GetTLStreamNodeMap();

        // Set stream buffer Count Mode to manual
        CEnumerationPtr ptrStreamBufferCountMode = sNodeMap.GetNode("StreamBufferCountMode");
        if (!IsReadable(ptrStreamBufferCountMode) ||
            !IsWritable(ptrStreamBufferCountMode))
        {
            cout << "Unable to get or set Buffer Count Mode (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrStreamBufferCountModeManual = ptrStreamBufferCountMode->GetEntryByName("Manual");
        if (!IsReadable(ptrStreamBufferCountModeManual))
        {
            cout << "Unable to get Buffer Count Mode entry (Entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrStreamBufferCountMode->SetIntValue(ptrStreamBufferCountModeManual->GetValue());

        cout << "Stream Buffer Count Mode set to manual..." << endl;

        cout << "Acquisition mode set to continuous..." << endl;

        //
        // Allocate buffers
        //
        // *** NOTES ***
        //
        // When allocating memory for user buffers, keep in mind that implicitly you are specifying how many
        // buffers are used for the acquisition engine.  There are two ways to set user buffers for Spinnaker
        // to utilize.  You can either pass a pointer to a contiguous buffer, or pass a pointer to pointers to
        // non-contiguous buffers into the library.  In either case, you will be responsible for allocating and
        // de-allocating the memory buffers that the pointers point to.
        //
        // The acquisition engine will be utilizing a bufferCount equal to totalSize divided by bufferSize,
        // where totalSize is the total allocated memory in bytes, and bufferSize is the image payload size.
        //
        // This example here demonstrates how to determine how much memory needs to be allocated based on the
        // retrieved payload size from the node map for both cases.
        //
        CIntegerPtr ptrPayloadSize = nodeMap.GetNode("PayloadSize");
        if (!IsReadable(ptrPayloadSize))
        {
            cout << "Unable to determine the payload size from the nodemap. Aborting..." << endl << endl;
            return -1;
        }
        uint64_t bufferSize = ptrPayloadSize->GetValue();

        // Calculate the 1024 aligned image size to be used for USB cameras
        CEnumerationPtr ptrDeviceType = pCam->GetTLDeviceNodeMap().GetNode("DeviceType");
        if (ptrDeviceType != nullptr && ptrDeviceType->GetIntValue() == DeviceType_USB3Vision)
        {
            const uint64_t usbPacketSize = 1024;
            bufferSize = ((bufferSize + usbPacketSize - 1) / usbPacketSize) * usbPacketSize;
        }

        const unsigned int numBuffers = 10;

        // Contiguous memory buffer
        unique_ptr<unsigned char[]> pMemBuffersContiguous;
        // Non-contiguous memory buffer
        vector<unique_ptr<unsigned char[]>> ppMemBuffersNonContiguous;
        vector<void*> ppMemBuffersNonContiguousVoid;

        // Set buffer ownership to user.
        // This must be set before using user buffers when calling BeginAcquisition().
        // If not set, BeginAcquisition() will use the system's buffers.
        if (pCam->GetBufferOwnership() != SPINNAKER_BUFFER_OWNERSHIP_USER)
        {
            pCam->SetBufferOwnership(SPINNAKER_BUFFER_OWNERSHIP_USER);
        }

        // Contiguous memory buffer
        if (isContiguous)
        {
            try
            {
                // Smart pointers will clean themselves up when they go out of scope,
                // so there is no need to clean them up manually
                pMemBuffersContiguous =
                    unique_ptr<unsigned char[]>(new unsigned char[numBuffers * static_cast<unsigned int>(bufferSize)]);
            }
            catch (bad_alloc& /*e*/)
            {
                cout << "Unable to allocate the memory required. Aborting..." << endl << endl;
                return -1;
            }

            pCam->SetUserBuffers(pMemBuffersContiguous.get(), numBuffers * bufferSize);

            cout << "User-allocated memory 0x" << hex << static_cast<void*>(&pMemBuffersContiguous)
                 << " will be used for user buffers..." << endl;
        }
        // Non-contiguous memory buffer
        else
        {
            try
            {
                // Smart pointers will clean themselves up when they go out of scope,
                // so there is no need to clean them up manually
                for (unsigned int i = 0; i < numBuffers; i++)
                {
                    ppMemBuffersNonContiguous.emplace_back(
                        unique_ptr<unsigned char[]>(new unsigned char[static_cast<unsigned int>(bufferSize)]));
                }
            }
            catch (bad_alloc& /*e*/)
            {
                cout << "Unable to allocate the memory required. Aborting..." << endl << endl;
                return -1;
            }

            for (unsigned int i = 0; i < ppMemBuffersNonContiguous.size(); i++)
            {
                ppMemBuffersNonContiguousVoid.emplace_back(ppMemBuffersNonContiguous.at(i).get());
            }

            const uint64_t bufferCount = ppMemBuffersNonContiguousVoid.size();

            pCam->SetUserBuffers(ppMemBuffersNonContiguousVoid.data(), bufferCount, bufferSize);

            cout << "User-allocated memory:" << endl;
            for (size_t i = 0; i < ppMemBuffersNonContiguousVoid.size(); i++)
            {
                cout << "\t0x" << hex << &ppMemBuffersNonContiguousVoid.at(i) << endl;
            }
            cout << "will be used for user buffers..." << endl;
        }

        // Begin acquiring images
        pCam->BeginAcquisition();

        // Retrieve the resulting stream buffer count nBuffers
        // Note: the buffer count result is dependent on the Stream Buffer Count Mode (Auto/Manual).
        // For Manual mode, Spinnaker uses the allocated memory size and payload size to calculate the number
        // of buffers. For auto mode, a deprecated buffer count mode, Spinnaker used additional information
        // such as frame rate to determine the number of buffers.
        CIntegerPtr ptrStreamBufferCountResult = sNodeMap.GetNode("StreamBufferCountResult");
        if (!IsReadable(ptrStreamBufferCountResult))
        {
            cout << "Unable to retrieve Buffer Count result (node retrieval). Aborting..." << endl << endl;
            return -1;
        }
        const int64_t streamBufferCountResult = ptrStreamBufferCountResult->GetValue();

        cout << "Resulting stream buffer count: " << dec << streamBufferCountResult << "." << endl << endl;

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
                // Retrieve next received image
                ImagePtr pResultImage = pCam->GetNextImage(1000);

                // Ensure image completion
                if (pResultImage->IsIncomplete())
                {
                    // Retrieve and print the image status description
                    cout << "Image incomplete: " << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
                         << "..." << endl
                         << endl;
                }
                else
                {
                    // Print image information; height and width recorded in pixels

                    const size_t width = pResultImage->GetWidth();

                    const size_t height = pResultImage->GetHeight();

                    cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;

                    // Convert image to mono 8
                    ImagePtr convertedImage = processor.Convert(pResultImage, PixelFormat_Mono8);

                    // Create a unique filename
                    ostringstream filename;

                    filename << "AcquisitionUserBuffer-";
                    if (!deviceSerialNumber.empty())
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

    // The smart pointers are cleaned up so you have no more allocated memory.
    // Therefore, we reset the buffer ownership to the system.
    if (pCam->GetBufferOwnership() != SPINNAKER_BUFFER_OWNERSHIP_SYSTEM)
    {
        pCam->SetBufferOwnership(SPINNAKER_BUFFER_OWNERSHIP_SYSTEM);
    }

    return result;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int PrintDeviceInfo(INodeMap& nodeMap)
{
    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;
    int result = 0;

    try
    {
        FeatureList_t features;
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsReadable(category))
        {
            category->GetFeatures(features);

            for (auto it = features.begin(); it != features.end(); ++it)
            {
                const CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
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

// This function acts as the body of the example; please see NodeMapInfo example
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
    int result = 0;

    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure heartbeat for GEV camera
#ifdef _DEBUG
        result = result | DisableGVCPHeartbeat(pCam);
#else
        result = result | ResetGVCPHeartbeat(pCam);
#endif

        // Acquire images
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);

#ifdef _DEBUG
        // Reset heartbeat for GEV camera
        result = result | ResetGVCPHeartbeat(pCam);
#endif

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

    const unsigned int numCameras = camList.GetSize();

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

    // Create shared pointer to camera
    CameraPtr pCam = nullptr;

    int result = 0;

    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera
        pCam = camList.GetByIndex(i);

        cout << endl << "Running example for camera " << i << "..." << endl;

        // Run example
        result = result | RunSingleCamera(pCam);

        cout << "Camera " << i << " example complete..." << endl << endl;
    }

    // Release reference to the camera
    pCam = nullptr;

    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return result;
}
