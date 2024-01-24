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
 *  @example FileAccess_Quickspin.cpp
 *
 *  @brief FileAccess_Quickspin.cpp shows how to read and write images using
 *  camera File Access function.
 *  This example uploads an image to the camera File Access storage and also
 *  download the image from the camera File Access storage and saves it to
 *  the disk.
 *  It also provides debug message when debug mode is turned on giving more
 *  detail status of the progress and error messages to the users.
 *
 *  It relies on information provided in the
 *  Enumeration, Acquisition, and NodeMapInfo examples.
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

static bool _enableDebug = false;
static gcstring _fileSelector = "UserFile1";

// Print out operation result message
static void PrintResultMessage(bool result)
{
    if (result)
    {
        cout << "\n*** OPERATION COMPLETE ***\n";
    }
    else
    {
        cout << "\n*** OPERATION FAILED ***\n";
    }
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

// Intializes the system
bool InitializeSystem(SystemPtr& system, CameraList& camList, CameraPtr& pCam)
{
    // Retrieve singleton reference to system object
    system = System::GetInstance();

    // Retrieve list of cameras from the system
    camList = system->GetCameras();

    const unsigned int numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl << endl;

    // Stop if there are no cameras
    if (numCameras == 0)
    {
        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();

        cout << "Not enough cameras!" << endl;
        cout << "Done! Press Enter to exit..." << endl;
        getchar();

        return false;
    }

    //
    // It creates shared pointer to camera
    //
    // *** NOTES ***
    // The CameraPtr object is a shared pointer, and will generally clean itself
    // up upon exiting its scope. However, if a shared pointer is created in the
    // same scope that a system object is explicitly released (i.e. this scope),
    // the reference to the shared point must be broken manually.
    //
    // *** LATER ***
    // Shared pointers can be terminated manually by assigning them to nullptr.
    // This keeps releasing the system from throwing an exception.
    //
    pCam = nullptr;

    // Run example on 1st camera
    pCam = camList.GetByIndex(0);

    return true;
}

// Print out debug message
static void PrintDebugMessage(string msg)
{
#if DEBUG
    _enableDebug = true;
#endif
    if (_enableDebug)
    {
        cout << msg << endl;
    }
}

// Acquire 5 images from a device.
bool AcquireImages(CameraPtr pCam, INodeMap& nodeMap, INodeMap& nodeMapTLDevice, ImagePtr pReferenceImage)
{
    cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;
    bool result = true;

    try
    {
        //
        // Set acquisition mode to continuous
        //
        // *** NOTES ***
        // Because the example acquires and saves 5 images, setting acquisition
        // mode to continuous lets the example finish. If set to single frame
        // or multiframe (at a lower number of images), the example would just
        // hang. This would happen because the example has been written to
        // acquire 5 images while the camera would have been programmed to
        // retrieve less than that.
        //
        // Setting the value of an enumeration node is slightly more complicated
        // than other node types. Two nodes must be retrieved: first, the
        // enumeration node is retrieved from the nodemap; and second, the entry
        // node is retrieved from the enumeration node. The integer value of the
        // entry node is then set as the new value of the enumeration node.
        //
        // Notice that both the enumeration and the entry nodes are checked for
        // availability and readability/writability. Enumeration nodes are
        // generally readable and writable whereas their entry nodes are only
        // ever readable.
        //
        // Retrieve enumeration node from nodemap
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsReadable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
        {
            cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
            return false;
        }

        // Retrieve entry node from enumeration node
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            return false;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        PrintDebugMessage("Acquisition mode set to continuous...");

        // Apply Small Pixel Format
        CEnumEntryPtr ptrPixelFormat = pCam->PixelFormat.GetEntry(PixelFormat_Mono8);
        if (IsReadable(ptrPixelFormat))
        {
            pCam->PixelFormat.SetValue(PixelFormat_Mono8);
        }
        else
        {
            // Use Bayer8 if Mono8 is not available
            pCam->PixelFormat.SetValue(PixelFormat_BayerRG8);
        }

        //
        // Begin acquiring images
        //
        // *** NOTES ***
        // What happens when the camera begins acquiring images depends on the
        // acquisition mode. Single frame captures only a single image, multi
        // frame catures a set number of images, and continuous captures a
        // continuous stream of images. Because the example calls for the
        // retrieval of 10 images, continuous mode has been set.
        //
        // *** LATER ***
        // Image acquisition must be ended when no more images are needed.
        //
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        //
        // Retrieve device serial number for filename
        //
        // *** NOTES ***
        // The device serial number is retrieved in order to keep cameras from
        // overwriting one another. Grabbing image IDs could also accomplish
        // this.
        //
        string deviceSerialNumber("");
        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrStringSerial))
        {
            deviceSerialNumber = ptrStringSerial->GetValue();

            PrintDebugMessage("Device serial number retrieved as " + deviceSerialNumber + "...");
        }
        cout << endl;

        // Retrieve, convert, and save images
        const unsigned int k_numImages = 5;

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
        {

            try
            {
                //
                // Retrieve next received image
                //
                // *** NOTES ***
                // Capturing an image houses images on the camera buffer. Trying
                // to capture an image that does not exist will hang the camera.
                //
                // *** LATER ***
                // Once an image from the buffer is saved and/or no longer
                // needed, the image must be released in order to keep the
                // buffer from filling up.
                //
                ImagePtr pResultImage = pCam->GetNextImage(1000);

                //
                // Ensure image completion
                //
                // *** NOTES ***
                // Images can easily be checked for completion. This should be
                // done whenever a complete image is expected or required.
                // Further, check image status for a little more insight into
                // why an image is incomplete.
                //
                if (pResultImage->IsIncomplete())
                {
                    cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl
                         << endl;
                }
                else
                {
                    //
                    // Print image information; height and width recorded in pixels
                    //
                    // *** NOTES ***
                    // Images have quite a bit of available metadata including
                    // things such as CRC, image status, and offset values, to
                    // name a few.
                    //
                    PrintDebugMessage(
                        "Grabbed image " + to_string(static_cast<long long>(imageCnt)) +
                        ", width = " + to_string(static_cast<long long>(pResultImage->GetWidth())) +
                        ", height = " + to_string(static_cast<long long>(pResultImage->GetHeight())));

                    // It deep copies to ImagePtr to pass it on
                    pReferenceImage->DeepCopy(pResultImage);
                }

                //
                // Release image
                //
                // *** NOTES ***
                // Images retrieved directly from the camera (i.e. non-converted
                // images) need to be released in order to keep from filling the
                // buffer.
                //
                pResultImage->Release();
            }
            catch (Spinnaker::Exception& e)
            {
                cout << "Error: " << e.what() << endl;
                result = false;
            }
        }

        //
        // End acquisition
        //
        // *** NOTES ***
        // Ending acquisition appropriately helps ensure that devices clean up
        // properly and do not need to be power-cycled to maintain integrity.
        //
        pCam->EndAcquisition();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = false;
    }
    return result;
}

// Execute delete operation
bool ExecuteDeleteCommand(CameraPtr pCam)
{
    PrintDebugMessage("Deleting file...");
    bool result = true;

    try
    {
        pCam->FileOperationSelector.SetValue(FileOperationSelector_Delete);
        pCam->FileOperationExecute.Execute();

        if (pCam->FileOperationStatus.GetValue() != FileOperationStatus_Success)
        {
            cout << "Failed to delete file!" << endl;
            return false;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what() << endl;
        result = false;
    }
    return result;
}

// Open camera file for writing
bool OpenFileToWrite(CameraPtr pCam)
{
    bool result = true;

    PrintDebugMessage("Opening file for writing...");
    try
    {
        pCam->FileOperationSelector.SetValue(FileOperationSelector_Open);
        pCam->FileOpenMode.SetValue(FileOpenMode_Write);
        pCam->FileOperationExecute.Execute();

        if (pCam->FileOperationStatus.GetValue() != FileOperationStatus_Success)
        {
            cout << "Failed to open file for writing!" << endl;
            return false;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = false;
    }
    return result;
}

// Execute write operation
bool ExecuteWriteCommand(CameraPtr pCam)
{
    bool result = true;

    try
    {
        pCam->FileOperationSelector.SetValue(FileOperationSelector_Write);
        pCam->FileOperationExecute.Execute();

        if (pCam->FileOperationStatus.GetValue() != FileOperationStatus_Success)
        {
            cout << "Failed to write to file!" << endl;
            return false;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = false;
    }
    return result;
}

// Close the file
bool CloseFile(CameraPtr pCam)
{
    bool result = true;
    PrintDebugMessage("Closing file...");

    try
    {
        pCam->FileOperationSelector.SetValue(FileOperationSelector_Close);
        pCam->FileOperationExecute.Execute();

        if (pCam->FileOperationStatus.GetValue() != FileOperationStatus_Success)
        {
            cout << "Failed to close file!" << endl;
            return false;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = false;
    }
    return result;
}

// Upload the image to the camera file
bool UploadImage()
{
    SystemPtr system;
    CameraList camList;
    CameraPtr pCam;
    ImagePtr pReferenceImage;
    bool result = true;

    // Initialize System
    if (!InitializeSystem(system, camList, pCam))
    {
        PrintResultMessage(false);
        return false;
    }

    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
        PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        ImagePtr pReferenceImage = Image::Create();

        // Acquire images
        if (!AcquireImages(pCam, nodeMap, nodeMapTLDevice, pReferenceImage))
        {
            PrintResultMessage(false);
            return false;
        }

        // Save a raw image for debugging purpose
        if (_enableDebug)
        {
            try
            {
                cout << "\nSaving raw image to disk for debugging purpose..." << endl;

                // Form file path
                ostringstream filename;
                filename << "rawImage.png";

                pReferenceImage->Save(filename.str().c_str());
                cout << "Image saved at " << filename.str() << endl;
            }
            catch (Spinnaker::Exception& e)
            {
                cout << "Unable to save an image file : " << e.what();
            }
        }

        cout << endl << "*** UPLOADING IMAGE ***" << endl;

        PrintDebugMessage("Fetching \"UserFile1\" Entry from FileSelectorNode...");
        // Check file selector support
        if (pCam->FileSelector == NULL)
        {
            cout << "File selector not supported on device!";
            return false;
        }

        NodeList_t selectorList;
        pCam->FileSelector.GetEntries(selectorList);

        for (unsigned int i = 0; i < selectorList.size(); i++)
        {
            // Get current enum entry node
            CEnumEntryPtr node = selectorList.at(i);

            PrintDebugMessage("Setting value to FileSelectorNode...");

            // Check file selector entry support
            if (!node || !IsReadable(node))
            {
                // Go to next entry node
                cout << node->GetSymbolic() << " not supported!" << endl;
                continue;
            }

            if (node->GetSymbolic().compare(_fileSelector) == 0)
            {
                PrintDebugMessage("Setting value to FileSelectorNode...");
                // Set file selector
                pCam->FileSelector.SetIntValue((int64_t)node->GetNumericValue());

                // Delete file on camera before writing in case camera runs out of space
                if (pCam->FileSize.GetValue() > 0)
                {
                    if (ExecuteDeleteCommand(pCam) != true)
                    {
                        cout << "Failed to delete file!" << endl;
                        continue;
                    }
                }

                // Open file on camera for write
                if (OpenFileToWrite(pCam) != true)
                {
                    cout << "Failed to open file!" << endl;
                    // may be file was not closed properly last time.
                    // Close and re-open again
                    if (!CloseFile(pCam))
                    {
                        // It fails to close the file. Abort!
                        cout << "Problem opening file node." << endl;
                        return false;
                    }

                    // File was closed. Open again.
                    if (!OpenFileToWrite(pCam))
                    {
                        // Fails again. Abort!
                        cout << "Problem opening file node." << endl;
                        return false;
                    }
                }

                // Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
                if (pCam->FileAccessLength.GetValue() < pCam->FileAccessBuffer.GetLength())
                {
                    try
                    {
                        pCam->FileAccessLength.SetValue(pCam->FileAccessBuffer.GetLength());
                    }
                    catch (Spinnaker::Exception& e)
                    {
                        cout << "Unable to set FileAccessLength to FileAccessBuffer length : " << e.what();
                    }
                }

                // Set File Access Offset to zero if its not
                pCam->FileAccessOffset.SetValue(0);

                // Compute number of write operations required
                int64_t totalBytesToWrite = pReferenceImage->GetBufferSize();
                int64_t intermediateBufferSize = pCam->FileAccessLength.GetValue();
                int64_t writeIterations = (totalBytesToWrite / intermediateBufferSize) +
                                          (totalBytesToWrite % intermediateBufferSize == 0 ? 0 : 1);

                if (totalBytesToWrite == 0)
                {
                    cout << "Empty Image. No data will be written to camera." << endl;
                    return false;
                }

                PrintDebugMessage("Start saving image on camera...");

                PrintDebugMessage("Total Bytes to write : " + to_string(static_cast<long long>(totalBytesToWrite)));
                PrintDebugMessage("FileAccessLength : " + to_string(static_cast<long long>(intermediateBufferSize)));
                PrintDebugMessage("Write Iterations : " + to_string(static_cast<long long>(writeIterations)));

                int64_t index = 0;
                int64_t bytesLeftToWrite = totalBytesToWrite;
                int64_t totalBytesWritten = 0;
                bool paddingRequired = false;
                int numPaddings = 0;

                cout << "Writing data to device" << endl;

                unsigned char* pImageData = static_cast<unsigned char*>(pReferenceImage->GetData());

                for (unsigned int i = 0; i < writeIterations; i++)
                {
                    // Check whether padding is required
                    if (intermediateBufferSize > bytesLeftToWrite)
                    {
                        // Check for multiple of 4 bytes
                        unsigned int remainder = bytesLeftToWrite % 4;
                        if (remainder != 0)
                        {
                            paddingRequired = true;
                            numPaddings = 4 - remainder;
                        }
                    }

                    // Setup data to write
                    int64_t tmpBufferSize = intermediateBufferSize <= bytesLeftToWrite
                                                ? intermediateBufferSize
                                                : (bytesLeftToWrite + numPaddings);
                    std::unique_ptr<unsigned char> tmpBuffer(
                        new unsigned char[static_cast<unsigned int>(tmpBufferSize)]);
                    memcpy(
                        tmpBuffer.get(),
                        &pImageData[index],
                        static_cast<unsigned int>(
                            (intermediateBufferSize <= bytesLeftToWrite) ? intermediateBufferSize : bytesLeftToWrite));

                    if (paddingRequired)
                    {
                        // Fill padded bytes
                        for (int j = 0; j < numPaddings; j++)
                        {
                            unsigned char* pTmpBuffer = tmpBuffer.get();
                            pTmpBuffer[bytesLeftToWrite + j] = 255;
                        }
                    }

                    // Update index for next write iteration
                    index = index +
                            (intermediateBufferSize <= bytesLeftToWrite ? intermediateBufferSize : bytesLeftToWrite);

                    // Write to FileAccessBufferNode
                    pCam->FileAccessBuffer.Set(tmpBuffer.get(), tmpBufferSize);

                    if (intermediateBufferSize > bytesLeftToWrite)
                    {
                        // Update FileAccessLength node appropriately to prevent garbage data outside the range of
                        // the uploaded file to be written to the camera
                        pCam->FileAccessLength.SetValue(bytesLeftToWrite);
                    }

                    // Perform Write command
                    if (!ExecuteWriteCommand(pCam))
                    {
                        cout << "Writing to stream failed!" << endl;
                        break;
                    }

                    // Verify size of bytes written
                    int64_t sizeWritten = pCam->FileOperationResult.GetValue();

                    // Log current file access offset
                    PrintDebugMessage("File Access Offset: " + to_string(pCam->FileAccessOffset.GetValue()));
                    // Keep track of total bytes written
                    totalBytesWritten += sizeWritten;
                    PrintDebugMessage(
                        "Bytes written: " + to_string(static_cast<long long>(totalBytesWritten)) + " of " +
                        to_string(static_cast<long long>(totalBytesToWrite)));
                    // Keep track of bytes left to write
                    bytesLeftToWrite = totalBytesToWrite - totalBytesWritten;
                    PrintDebugMessage("Bytes left: " + to_string(static_cast<long long>(bytesLeftToWrite)));

                    cout << "Progress : " << (i * 100 / writeIterations) << " %" << endl;
                }

                cout << "Writing complete" << endl;

                if (!CloseFile(pCam))
                {
                    cout << "Failed to close file!" << endl;
                }
            }
        }

        //
        // Release reference to the camera
        //
        // *** NOTES ***
        // Had the CameraPtr object been created within the for-loop, it would not
        // be necessary to manually break the reference because the shared pointer
        // would have automatically cleaned itself up upon exiting the loop.
        //
        pCam->DeInit();
        pCam = nullptr;

        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();

        cout << endl << "Done! Press Enter to exit..." << endl;
        getchar();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = false;
    }
    return result;
}

// Open camera file to read
bool OpenFileToRead(CameraPtr pCamera)
{
    bool result = true;
    cout << "Opening file for reading..." << endl;
    try
    {
        pCamera->FileOperationSelector.SetValue(FileOperationSelector_Open);
        pCamera->FileOpenMode.SetValue(FileOpenMode_Read);
        pCamera->FileOperationExecute.Execute();

        if (pCamera->FileOperationStatus.GetValue() != FileOperationStatus_Success)
        {
            cout << "Failed to open file for reading!" << endl;
            return false;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what() << endl;
        result = false;
    }
    return result;
}

// Execute read operation
bool ExecuteReadCommand(CameraPtr pCamera)
{
    bool result = true;

    try
    {
        pCamera->FileOperationSelector.SetValue(FileOperationSelector_Read);
        pCamera->FileOperationExecute.Execute();

        if (pCamera->FileOperationStatus.GetValue() != FileOperationStatus_Success)
        {
            cout << "Failed to read file!" << endl;
            return false;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what() << endl;
        result = false;
    }
    return result;
}

// Download the image to the disk from camera file
bool DownloadImage()
{
    SystemPtr system;
    CameraList camList;
    CameraPtr pCam;
    bool result = true;

    // Initialize System
    if (!InitializeSystem(system, camList, pCam))
    {
        PrintResultMessage(false);
        return false;
    }

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
    cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
         << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
         << endl;

    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
        PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        cout << endl << "*** DOWNLOADING IMAGE ***" << endl;

        // Check file selector support
        PrintDebugMessage("Fetching \"UserFile1\" Entry from FileSelectorNode...");
        if (pCam->FileSelector == NULL)
        {
            cout << "File selector not supported on device!";
            return false;
        }

        NodeList_t selectorList;
        pCam->FileSelector.GetEntries(selectorList);

        for (unsigned int i = 0; i < selectorList.size(); i++)
        {
            // Get current enum entry node
            CEnumEntryPtr node = selectorList.at(i);

            PrintDebugMessage("Setting value to FileSelectorNode...");
            // Check file selector entry support
            if (!node || !IsReadable(node))
            {
                // Go to next entry node
                cout << node->GetSymbolic() << " not supported!" << endl;
                continue;
            }

            if (node->GetSymbolic().compare(_fileSelector) == 0)
            {
                // Set file selector
                pCam->FileSelector.SetIntValue((int64_t)node->GetNumericValue());

                int64_t bytesToRead = pCam->FileSize.GetValue();
                if (bytesToRead == 0)
                {
                    cout << "No data available to read!" << endl;
                    continue;
                }

                PrintDebugMessage("Total data to download : " + to_string(static_cast<long long>(bytesToRead)));

                // Open file on camera for reading
                if (OpenFileToRead(pCam) != true)
                {
                    cout << "Failed to open file!" << endl;
                    continue;
                }

                // Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
                if (pCam->FileAccessLength.GetValue() < pCam->FileAccessBuffer.GetLength())
                {
                    try
                    {
                        pCam->FileAccessLength.SetValue(pCam->FileAccessBuffer.GetLength());
                    }
                    catch (Spinnaker::Exception& e)
                    {
                        cout << "Unable to set FileAccessLength to FileAccessBuffer length : " << e.what() << endl;
                    }
                }

                // Set File Access Offset to zero if its not
                pCam->FileAccessOffset.SetValue(0);

                // Compute number of read operations required
                int64_t intermediateBufferSize = pCam->FileAccessLength.GetValue();
                int64_t iterations =
                    (bytesToRead / intermediateBufferSize) + (bytesToRead % intermediateBufferSize == 0 ? 0 : 1);

                PrintDebugMessage("Fetching image from camera.");

                int64_t index = 0;
                int64_t totalSizeRead = 0;
                std::unique_ptr<unsigned char> dataBuffer(new unsigned char[static_cast<unsigned int>(bytesToRead)]);
                uint8_t* pData = dataBuffer.get();

                for (unsigned int i = 0; i < iterations; i++)
                {
                    if (!ExecuteReadCommand(pCam))
                    {
                        cout << "Reading stream failed!" << endl;
                        break;
                    }

                    // Verify size of bytes read
                    int64_t sizeRead = pCam->FileOperationResult.GetValue();

                    // Read from buffer Node
                    pCam->FileAccessBuffer.Get(&pData[index], sizeRead);

                    // Update index for next read iteration
                    index = index + sizeRead;

                    // Keep track of total bytes read
                    totalSizeRead += sizeRead;
                    PrintDebugMessage(
                        "Bytes read: " + to_string(static_cast<long long>(totalSizeRead)) + " of " +
                        to_string(static_cast<long long>(bytesToRead)));
                    cout << "Progress : " << (i * 100 / iterations) << " %" << endl;
                }

                PrintDebugMessage("Reading complete");

                if (!CloseFile(pCam))
                {
                    cout << "Failed to close file!" << endl;
                }

                cout << endl;

                // Form file path
                ostringstream filename;
                filename << "DeviceStreamRead-";

                if (pCam->DeviceSerialNumber != NULL)
                {
                    filename << pCam->DeviceSerialNumber.GetValue().c_str() << "-";
                }

                filename << ".bmp";

                // Image should be captured with Mono8 or Bayer8, it sets camera to correct pixel format
                // in order to grab image ROI
                CEnumEntryPtr ptrPixelFormat = pCam->PixelFormat.GetEntry(PixelFormat_Mono8);
                if (IsReadable(ptrPixelFormat) || IsWritable(ptrPixelFormat))
                {
                    pCam->PixelFormat.SetValue(PixelFormat_Mono8);
                }
                else
                {
                    // Use Bayer8 if Mono8 is not available
                    pCam->PixelFormat.SetValue(PixelFormat_BayerGB8);
                }

                const int64_t width = pCam->Width.GetValue();
                const int64_t height = pCam->Height.GetValue();
                const int64_t offSetX = pCam->OffsetX.GetValue();
                const int64_t offSetY = pCam->OffsetX.GetValue();
                PixelFormatEnums pixelFormat = pCam->PixelFormat.GetValue();

                // Form image and save image
                cout << "Width : " << width << endl;
                cout << "Height : " << height << endl;
                cout << "OffSetX : " << offSetX << endl;
                cout << "OffSetY : " << offSetY << endl;

                ImagePtr pImage = Image::Create(
                    static_cast<size_t>(width),
                    static_cast<size_t>(height),
                    static_cast<size_t>(offSetX),
                    static_cast<size_t>(offSetY),
                    pixelFormat,
                    pData);

                if (_enableDebug)
                {
                    try
                    {
                        cout << "\nSaving raw image to disk for debugging purpose..." << endl;

                        // Form file path
                        ostringstream filename;
                        filename << "downloaded_output.bmp";

                        pImage->Save(filename.str().c_str());
                        cout << "Image saved at " << filename.str() << endl;
                    }
                    catch (Spinnaker::Exception& e)
                    {
                        cout << "Unable to write image data to the PC : " << e.what() << endl;
                    }
                }

                pImage->Save(filename.str().c_str());
                cout << "Image saved at " << filename.str() << endl;
                cout << "\n*** SAVING IMAGE ***\n";
            }
        }
        //
        // Release reference to the camera
        //
        // *** NOTES ***
        // Had the CameraPtr object been created within the for-loop, it would not
        // be necessary to manually break the reference because the shared pointer
        // would have automatically cleaned itself up upon exiting the loop.
        //
        pCam->DeInit();
        pCam = nullptr;

        // Clear camera list before releasing system
        camList.Clear();

        // Release system
        system->ReleaseInstance();

        cout << endl << "Done! Press Enter to exit..." << endl;
        getchar();
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what() << endl;
        result = false;
    }
    return result;
}

// Print out usage of the application
void PrintUsage()
{
    cout << "Usage: FileAcess </u | /d>" << endl;
    cout << "Options:" << endl;
    cout << "/u : Grab an image and store it on camera." << endl;
    cout << "/d : Download saved image from camera and store it on desktop." << endl;
    cout << "/v : Enable verbose output." << endl;
    cout << "/? : Print usage informaion." << endl;
    cout << endl << endl;
}

// Example entry point; please see Enumeration example for more in-depth
// comments on preparing and cleaning up the system.
int main(int argc, char* argv[])
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

    // Change arguments to string for easy read
    std::vector<std::string> args(argv, argv + argc);

    if (args.size() == 1)
    {
        // Print usage when no arguments specified
        PrintUsage();
        return result;
    }

    bool uploadImage = false;
    bool downloadImage = false;

    // Check verbose output flag first
    for (size_t i = 1; i < args.size(); ++i)
    {
        if (args[i] == "/v" || args[i] == "/V")
        {
            _enableDebug = true;
        }

        if (args[i] == "?")
        {
            PrintUsage();
            return result;
        }
        
        if (args[i] == "/u" || args[i] == "/U")
        {
            uploadImage = true;
        }
        
        if (args[i] == "/d" || args[i] == "/D")
        {
            downloadImage = true;
        }
    }

    if (uploadImage)
    {
        if (!UploadImage())
        {
            PrintResultMessage(false);
            result = -1;
        }

        return result;
    }

    if (downloadImage)
    {
        if (!DownloadImage())
        {
            PrintResultMessage(false);
            result = -1;
        }
        return result;
    }

    return result;
}
