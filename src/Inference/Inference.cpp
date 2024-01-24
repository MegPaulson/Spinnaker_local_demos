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
 *  @example Inference.cpp
 *
 *  @brief Inference.cpp shows how to perform the following:
 *  - Upload custom inference neural networks to the camera (DDR or Flash)
 *  - Inject sample test image
 *  - Enable/Configure chunk data
 *  - Enable/Configure trigger inference ready sync
 *  - Acquire images
 *  - Display inference data from acquired image chunk data
 *  - Disable previously configured camera configurations
 *
 *  Inference is only available for Firefly deep learning cameras.
 *  See the related content section on the Firefly DL product page for relevant
 *  documentation.
 *
 *  https://www.flir.com/products/firefly-dl/
 *
 *  It can also be helpful to familiarize yourself with the Acquisition,
 *  ChunkData and FileAccess_QuickSpin examples.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include "ChunkDataInference.h"
#include <iostream>
#include <fstream>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// Use the following enum and global constant to select whether inference network
// type is Detection or Classification.
enum InferenceNetworkType
{
    /**
     * This network determines the  most likely class given a set of predetermined,
     * trained options. Object detection can also provide a location within the
     * image (in the form of a "bounding box" surrounding the class), and can
     * detect multiple objects.
     */
    DETECTION,
    /**
     * This network determines the best option from a list of predetermined options;
     * the camera gives a percentage that determines the likelihood of the currently
     * perceived image being one of the classes it has been trained to recognize.
     */
    CLASSIFICATION
};

const InferenceNetworkType chosenInferenceNetworkType = CLASSIFICATION;

// Use the following enum and global constant to select whether uploaded inference
// network and injected image should be written to camera flash or DDR
enum FileUploadPersistence
{
    FLASH, // Slower upload but data persists after power cycling the camera
    DDR    // Faster upload but data clears after power cycling the camera
};

const FileUploadPersistence chosenFileUploadPersistence = DDR;

// The example provides two existing custom networks that can be uploaded
// on to the camera to demonstrate classification and detection capabilities.
// "Network_Classification" file is created with Tensorflow using a mobilenet
// neural network for classifying flowers.
// "Network_Detection" file is created with Caffe using mobilenet SSD network
// for people object detection.
// Note: Make sure these files exist on the system and are accessible by the example
const std::string networkFilePath =
    (chosenInferenceNetworkType == CLASSIFICATION ? "Network_Classification" : "Network_Detection");

// The example provides two raw images that can be injected into the camera
// to demonstrate camera inference classification and detection capabilities. Jpeg
// representation of the raw images can be found packaged with the example with
// the names "Injected_Image_Classification_Daisy.jpg" and "Injected_Image_Detection_Aeroplane.jpg".
// Note: Make sure these files exist on the system and are accessible by the example
const std::string injectedImageFilePath =
    (chosenInferenceNetworkType == CLASSIFICATION ? "Injected_Image_Classification.raw"
                                                  : "Injected_Image_Detection.raw");

// The injected images have different ROI sizes so the camera needs to be
// configured to the appropriate width and height to match the injected image
const unsigned int injectedImageWidth = (chosenInferenceNetworkType == CLASSIFICATION ? 640 : 720);
const unsigned int injectedImageHeight = (chosenInferenceNetworkType == CLASSIFICATION ? 400 : 540);

// The sample classification inference network file was trained with the following
// data set labels
// Note: This list should match the list of labels used during the training
//       stage of the network file
const char* arrayLabelClassification[] = {"daisy", "dandelion", "roses", "sunflowers", "tulips"};
const std::vector<std::string> labelClassification(arrayLabelClassification, end(arrayLabelClassification));

// The sample detection inference network file was trained with the following
// data set labels
// Note: This list should match the list of labels used during the training
//       stage of the network file
const char* arrayLabelDetection[] = {"background", "aeroplane", "bicycle",   "bird",   "boat",        "bottle",
                                     "bus",        "car",       "cat",       "chair",  "cow",         "diningtable",
                                     "dog",        "horse",     "motorbike", "person", "pottedplant", "sheep",
                                     "sofa",       "train",     "monitor"};
const std::vector<std::string> labelDetection(arrayLabelDetection, end(arrayLabelDetection));

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
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsReadable(category))
        {
            category->GetFeatures(features);

            for (FeatureList_t::const_iterator it = features.begin(); it != features.end(); ++it)
            {
                CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
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

// This function executes a file delete operation on the camera.
bool CameraDeleteFile(INodeMap& nodeMap)
{
    bool result = true;

    CIntegerPtr ptrFileSize = nodeMap.GetNode("FileSize");
    if (!IsReadable(ptrFileSize))
    {
        cout << "Unable to query FileSize. Aborting..." << endl;
        return false;
    }

    if (ptrFileSize->GetValue() == 0)
    {
        // No file uploaded yet, skip delete
        cout << "No files found, skipping file deletion." << endl;
        return true;
    }

    cout << "Deleting file..." << endl;
    try
    {
        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsReadable(ptrFileOperationSelector) ||
            !IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to configure FileOperationSelector. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationDelete = ptrFileOperationSelector->GetEntryByName("Delete");
        if (!IsReadable(ptrFileOperationDelete))
        {
            cout << "Unable to configure FileOperationSelector Delete. Aborting..." << endl;
            return false;
        }

        ptrFileOperationSelector->SetIntValue(static_cast<int64_t>(ptrFileOperationDelete->GetNumericValue()));

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to configure FileOperationExecute. Aborting..." << endl;
            return false;
        }

        ptrFileOperationExecute->Execute();

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to query FileOperationStatus. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (!IsReadable(ptrFileOperationStatusSuccess))
        {
            cout << "Unable to query FileOperationStatus Success. Aborting..." << endl;
            return false;
        }

        if (ptrFileOperationStatus->GetCurrentEntry() != ptrFileOperationStatusSuccess)
        {
            cout << "Failed to delete file! File Operation Status : "
                 << ptrFileOperationStatus->GetCurrentEntry()->GetSymbolic() << endl;
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

// This function executes file open/write on the camera, sets the uploaded file persistence
// and attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write.
bool CameraOpenFile(INodeMap& nodeMap)
{
    bool result = true;

    cout << "Opening file for writing..." << endl;
    try
    {
        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsReadable(ptrFileOperationSelector) ||
            !IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to configure FileOperationSelector. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationOpen = ptrFileOperationSelector->GetEntryByName("Open");
        if (!IsReadable(ptrFileOperationOpen))
        {
            cout << "Unable to configure FileOperationSelector Open. Aborting..." << endl;
            return false;
        }

        ptrFileOperationSelector->SetIntValue(static_cast<int64_t>(ptrFileOperationOpen->GetNumericValue()));

        CEnumerationPtr ptrFileOpenMode = nodeMap.GetNode("FileOpenMode");
        if (!IsReadable(ptrFileOpenMode) ||
            !IsWritable(ptrFileOpenMode))
        {
            cout << "Unable to configure ptrFileOpenMode. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOpenModeWrite = ptrFileOpenMode->GetEntryByName("Write");
        if (!IsReadable(ptrFileOpenModeWrite))
        {
            cout << "Unable to configure FileOperationSelector Write. Aborting..." << endl;
            return false;
        }

        ptrFileOpenMode->SetIntValue(static_cast<int64_t>(ptrFileOpenModeWrite->GetNumericValue()));

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to configure FileOperationExecute. Aborting..." << endl;
            return false;
        }

        ptrFileOperationExecute->Execute();

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to query FileOperationStatus. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (!IsReadable(ptrFileOperationStatusSuccess))
        {
            cout << "Unable to query FileOperationStatus Success. Aborting..." << endl;
            return false;
        }

        if (ptrFileOperationStatus->GetCurrentEntry() != ptrFileOperationStatusSuccess)
        {
            cout << "Failed to open file for writing!" << endl;
            return false;
        }

        // Set file upload persistence settings
        CBooleanPtr ptrFileWriteToFlash = nodeMap.GetNode("FileWriteToFlash");
        if (IsWritable(ptrFileWriteToFlash))
        {
            if (chosenFileUploadPersistence == FLASH)
            {
                ptrFileWriteToFlash->SetValue(true);
                cout << "FileWriteToFlash is set to true" << endl;
            }
            else
            {
                ptrFileWriteToFlash->SetValue(false);
                cout << "FileWriteToFlash is set to false" << endl;
            }
        }

        // Attempt to set FileAccessLength to FileAccessBufferNode length to speed up the write
        CIntegerPtr ptrFileAccessLength = nodeMap.GetNode("FileAccessLength");
        if (!IsReadable(ptrFileAccessLength) || !IsWritable(ptrFileAccessLength))
        {
            cout << "Unable to query/configure FileAccessLength. Aborting..." << endl;
            return false;
        }

        CRegisterPtr ptrFileAccessBuffer = nodeMap.GetNode("FileAccessBuffer");
        if (!IsReadable(ptrFileAccessBuffer))
        {
            cout << "Unable to query FileAccessBuffer. Aborting..." << endl;
            return false;
        }

        if (ptrFileAccessLength->GetValue() < ptrFileAccessBuffer->GetLength())
        {
            try
            {
                ptrFileAccessLength->SetValue(ptrFileAccessBuffer->GetLength());
            }
            catch (Spinnaker::Exception& e)
            {
                cout << "Unable to set FileAccessLength to FileAccessBuffer length : " << e.what();
            }
        }

        // Set File Access Offset to zero
        CIntegerPtr ptrFileAccessOffset = nodeMap.GetNode("FileAccessOffset");
        if (!IsWritable(ptrFileAccessOffset))
        {
            cout << "Unable to query/configure ptrFileAccessOffset. Aborting..." << endl;
            return false;
        }

        ptrFileAccessOffset->SetValue(0);
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = false;
    }

    return result;
}

// This function executes a file write operation on the camera.
bool CameraWriteToFile(INodeMap& nodeMap)
{
    bool result = true;

    try
    {
        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsReadable(ptrFileOperationSelector) ||
            !IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to configure FileOperationSelector. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationWrite = ptrFileOperationSelector->GetEntryByName("Write");
        if (!IsReadable(ptrFileOperationWrite))
        {
            cout << "Unable to configure FileOperationSelector Write. Aborting..." << endl;
            return false;
        }

        ptrFileOperationSelector->SetIntValue(static_cast<int64_t>(ptrFileOperationWrite->GetNumericValue()));

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to configure FileOperationExecute. Aborting..." << endl;
            return false;
        }

        ptrFileOperationExecute->Execute();

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to query FileOperationStatus. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (!IsReadable(ptrFileOperationStatusSuccess))
        {
            cout << "Unable to query FileOperationStatus Success. Aborting..." << endl;
            return false;
        }

        if (ptrFileOperationStatus->GetCurrentEntry() != ptrFileOperationStatusSuccess)
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

// This function executes a file close operation on the camera.
bool CameraCloseFile(INodeMap& nodeMap)
{
    bool result = true;

    cout << "Closing file..." << endl;
    try
    {
        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsReadable(ptrFileOperationSelector) ||
            !IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to configure FileOperationSelector. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationClose = ptrFileOperationSelector->GetEntryByName("Close");
        if (!IsReadable(ptrFileOperationClose))
        {
            cout << "Unable to configure FileOperationSelector Close. Aborting..." << endl;
            return false;
        }

        ptrFileOperationSelector->SetIntValue((int64_t)ptrFileOperationClose->GetNumericValue());

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to configure FileOperationExecute. Aborting..." << endl;
            return false;
        }

        ptrFileOperationExecute->Execute();

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to query FileOperationStatus. Aborting..." << endl;
            return false;
        }

        CEnumEntryPtr ptrFileOperationStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (!IsReadable(ptrFileOperationStatusSuccess))
        {
            cout << "Unable to query FileOperationStatus Success. Aborting..." << endl;
            return false;
        }

        if (ptrFileOperationStatus->GetCurrentEntry() != ptrFileOperationStatusSuccess)
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

// This function loads a file given a file path on the system into memory.
std::vector<char> LoadFileIntoMemory(const string& filename)
{
    ifstream ifs(filename, ios::binary | ios::ate | ios::in);
    if (ifs.fail())
    {
        cout << "Failed to open " << filename << endl;
        std::vector<char> data;
        return data;
    }

    const ifstream::pos_type pos = ifs.tellg();
    std::vector<char> data(static_cast<unsigned int>(pos));
    ifs.seekg(0, ios::beg);
    ifs.read(&data[0], pos);

    return data;
}

// This function uploads a file on the system to the camera given the selected
// file selector entry.
int UploadFileToCamera(INodeMap& nodeMap, const std::string& fileSelectorEntryName, const std::string& filePath)
{
    cout << endl << endl << "*** CONFIGURING FILE SELECTOR ***" << endl << endl;

    CEnumerationPtr ptrFileSelector = nodeMap.GetNode("FileSelector");
    if (!IsReadable(ptrFileSelector) ||
        !IsWritable(ptrFileSelector))
    {
        cout << "Unable to configure FileSelector. Aborting..." << endl;
        return -1;
    }

    CEnumEntryPtr ptrInferenceSelectorEntry = ptrFileSelector->GetEntryByName(fileSelectorEntryName.c_str());
    if (!IsReadable(ptrInferenceSelectorEntry))
    {
        cout << "Unable to query FileSelector entry " << fileSelectorEntryName << ". Aborting..." << endl;
        return -1;
    }

    // Set file selector to entry
    cout << "Setting FileSelector to " << ptrInferenceSelectorEntry->GetSymbolic() << "..." << endl;
    ptrFileSelector->SetIntValue(static_cast<int64_t>(ptrInferenceSelectorEntry->GetNumericValue()));

    // Delete file on camera before writing in case camera runs out of space
    if (CameraDeleteFile(nodeMap) != true)
    {
        cout << "Failed to delete existing file for selector entry " << ptrInferenceSelectorEntry->GetSymbolic()
             << ". Aborting..." << endl;
        return -1;
    }

    // Open file on camera for write
    if (CameraOpenFile(nodeMap) != true)
    {
        // File may not be closed properly last time
        // Close and re-open again
        if (!CameraCloseFile(nodeMap))
        {
            // It fails to close the file. Abort!
            cout << "Problem opening file node. Aborting..." << endl;
            return -1;
        }

        // File was closed. Open again.
        if (!CameraOpenFile(nodeMap))
        {
            // Fails again. Abort!
            cout << "Problem opening file node. Aborting..." << endl;
            return -1;
        }
    }

    CIntegerPtr ptrFileAccessLength = nodeMap.GetNode("FileAccessLength");
    if (!IsReadable(ptrFileAccessLength) || !IsWritable(ptrFileAccessLength))
    {
        cout << "Unable to query FileAccessLength. Aborting..." << endl;
        return -1;
    }

    CRegisterPtr ptrFileAccessBuffer = nodeMap.GetNode("FileAccessBuffer");
    if (!IsWritable(ptrFileAccessBuffer))
    {
        cout << "Unable to query FileAccessBuffer. Aborting..." << endl;
        return -1;
    }

    CIntegerPtr ptrFileOperationResult = nodeMap.GetNode("FileOperationResult");
    if (!IsReadable(ptrFileOperationResult))
    {
        cout << "Unable to query FileOperationResult. Aborting..." << endl;
        return -1;
    }

    // Load network file from path depending on network type
    std::vector<char> fileBytes = LoadFileIntoMemory(filePath);
    if (fileBytes.size() == 0)
    {
        cout << "Failed to load file path : " << filePath << ". Aborting..." << endl;
        return -1;
    }

    // Compute number of write operations required
    const int64_t totalBytesToWrite = fileBytes.size();
    int64_t intermediateBufferSize = ptrFileAccessLength->GetValue();
    const int64_t writeIterations =
        (totalBytesToWrite / intermediateBufferSize) + (totalBytesToWrite % intermediateBufferSize == 0 ? 0 : 1);

    if (totalBytesToWrite == 0)
    {
        cout << "Empty Image. No data will be written to camera. Aborting..." << endl;
        return -1;
    }

    cout << "Start uploading \"" << filePath << "\" to device..." << endl;

    cout << "Total Bytes to write : " << to_string(static_cast<long long>(totalBytesToWrite)) << endl;
    cout << "FileAccessLength : " << to_string(static_cast<long long>(intermediateBufferSize)) << endl;
    cout << "Write Iterations : " << to_string(static_cast<long long>(writeIterations)) << endl;

    int64_t index = 0;
    int64_t bytesLeftToWrite = totalBytesToWrite;
    int64_t totalBytesWritten = 0;
    bool paddingRequired = false;
    int numPaddings = 0;

    cout << "Writing data to device..." << endl;

    char* pFileData = fileBytes.data();

    for (unsigned int i = 0; i < writeIterations; i++)
    {
        // Check whether padding is required
        if (intermediateBufferSize > bytesLeftToWrite)
        {
            // Check for multiple of 4 bytes
            const unsigned int remainder = bytesLeftToWrite % 4;
            if (remainder != 0)
            {
                paddingRequired = true;
                numPaddings = 4 - remainder;
            }
        }

        // Setup data to write
        const int64_t tmpBufferSize =
            intermediateBufferSize <= bytesLeftToWrite ? intermediateBufferSize : (bytesLeftToWrite + numPaddings);
        std::unique_ptr<unsigned char> tmpBuffer(new unsigned char[static_cast<unsigned int>(tmpBufferSize)]);
        memcpy(
            tmpBuffer.get(),
            &pFileData[index],
            static_cast<size_t>(
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
        index = index + (intermediateBufferSize <= bytesLeftToWrite ? intermediateBufferSize : bytesLeftToWrite);

        // Write to FileAccessBufferNode
        ptrFileAccessBuffer->Set(tmpBuffer.get(), tmpBufferSize);

        if (intermediateBufferSize > bytesLeftToWrite)
        {
            // Update FileAccessLength node appropriately to prevent garbage data outside the range of
            // the uploaded file to be written to the camera
            ptrFileAccessLength->SetValue(bytesLeftToWrite);
        }

        // Perform Write command
        if (!CameraWriteToFile(nodeMap))
        {
            cout << "Writing to stream failed! Aborting..." << endl;
            return -1;
        }

        // Verify size of bytes written
        const int64_t sizeWritten = ptrFileOperationResult->GetValue();

        // Keep track of total bytes written
        totalBytesWritten += sizeWritten;

        // Keep track of bytes left to write
        bytesLeftToWrite = totalBytesToWrite - totalBytesWritten;

        cout << "Progress : " << (i * 100 / writeIterations) << " %" << '\r' << flush;
    }

    cout << "Writing complete" << endl;

    if (!CameraCloseFile(nodeMap))
    {
        cout << "Failed to close file!" << endl;
    }

    return 0;
}

// This function deletes the file uploaded to the camera given the selected
// file selector entry.
int DeleteFileOnCamera(INodeMap& nodeMap, const std::string& fileSelectorEntryName)
{
    cout << endl << endl << "*** CLEANING UP FILE SELECTOR ***" << endl << endl;

    CEnumerationPtr ptrFileSelector = nodeMap.GetNode("FileSelector");
    if (!IsReadable(ptrFileSelector) ||
        !IsWritable(ptrFileSelector))
    {
        cout << "Unable to configure FileSelector. Aborting..." << endl;
        return -1;
    }

    CEnumEntryPtr ptrInferenceSelectorEntry = ptrFileSelector->GetEntryByName(fileSelectorEntryName.c_str());
    if (!IsReadable(ptrInferenceSelectorEntry))
    {
        cout << "Unable to query FileSelector entry " << fileSelectorEntryName << ". Aborting..." << endl;
        return -1;
    }

    // Set file selector to entry
    cout << "Setting FileSelector to " << ptrInferenceSelectorEntry->GetSymbolic() << endl;
    ptrFileSelector->SetIntValue(static_cast<int64_t>(ptrInferenceSelectorEntry->GetNumericValue()));

    // Delete file on camera before writing in case camera runs out of space
    if (CameraDeleteFile(nodeMap) != true)
    {
        cout << "Failed to delete existing file for selector entry " << ptrInferenceSelectorEntry->GetSymbolic()
             << ". Aborting..." << endl;
        return -1;
    }

    return 0;
}

// This function enables or disables the given chunk data type based on
// the specified entry name.
int SetChunkEnable(INodeMap& nodeMap, const gcstring& entryName, const bool enable)
{
    int result = 0;

    // Retrieve the chunk data selector node
    const CEnumerationPtr ptrChunkSelector = nodeMap.GetNode("ChunkSelector");
    if (!IsReadable(ptrChunkSelector))
    {
        cout << "Unable to retrieve chunk selector (enum retrieval). Aborting..." << endl;
        return -1;
    }

    const CEnumEntryPtr ptrEntry = ptrChunkSelector->GetEntryByName(entryName);
    if (!IsReadable(ptrEntry))
    {
        return -1;
    }

    if (!IsWritable(ptrChunkSelector))
    {
        return -1;
    }

    ptrChunkSelector->SetIntValue(ptrEntry->GetValue());

    // Enable the boolean, thus enabling the corresponding chunk data
    cout << entryName << " ";
    CBooleanPtr ptrChunkEnable = nodeMap.GetNode("ChunkEnable");
    if (!IsReadable(ptrChunkEnable))
    {
        cout << "not readable" << endl;
        return -1;
    }
    if (enable)
    {
        if (ptrChunkEnable->GetValue())
        {
            cout << "enabled" << endl;
        }
        else if (IsWritable(ptrChunkEnable))
        {
            ptrChunkEnable->SetValue(true);
            cout << "enabled" << endl;
        }
        else
        {
            cout << "not writable" << endl;
            result = -1;
        }
    }
    else
    {
        if (!ptrChunkEnable->GetValue())
        {
            cout << "disabled" << endl;
        }
        else if (IsWritable(ptrChunkEnable))
        {
            ptrChunkEnable->SetValue(false);
            cout << "disabled" << endl;
        }
        else
        {
            cout << "not writable" << endl;
            result = -1;
        }
    }

    return result;
}

// This function configures the camera to add inference chunk data to each image.
// When chunk data is turned on, the data is made available in both the nodemap
// and each image.
int ConfigureChunkData(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING CHUNK DATA ***" << endl << endl;

    try
    {
        //
        // Activate chunk mode
        //
        // *** NOTES ***
        // Once enabled, chunk data will be available at the end of the payload
        // of every image captured until it is disabled. Chunk data can also be
        // retrieved from the nodemap.
        //
        CBooleanPtr ptrChunkModeActive = nodeMap.GetNode("ChunkModeActive");
        if (!IsWritable(ptrChunkModeActive))
        {
            cout << "Unable to activate chunk mode. Aborting..." << endl;
            return -1;
        }
        ptrChunkModeActive->SetValue(true);
        cout << "Chunk mode activated..." << endl;

        // Enable inference related chunks in chunk data
        // Enable chunk data inference Frame Id
        result = SetChunkEnable(nodeMap, "InferenceFrameId", true);
        if (result == -1)
        {
            cout << "Unable to enable Inference Frame Id chunk data. Aborting..." << endl;
            return result;
        }

        if (chosenInferenceNetworkType == DETECTION)
        {
            // Detection network type

            // Enable chunk data inference bounding box
            result = SetChunkEnable(nodeMap, "InferenceBoundingBoxResult", true);
            if (result == -1)
            {
                cout << "Unable to enable Inference Bounding Box chunk data. Aborting..." << endl;
                return result;
            }
        }
        else
        {
            // Classification network type

            // Enable chunk data inference result
            result = SetChunkEnable(nodeMap, "InferenceResult", true);
            if (result == -1)
            {
                cout << "Unable to enable Inference Result chunk data. Aborting..." << endl;
                return result;
            }

            // Enable chunk data inference confidence
            result = SetChunkEnable(nodeMap, "InferenceConfidence", true);
            if (result == -1)
            {
                cout << "Unable to enable Inference Confidence chunk data. Aborting..." << endl;
                return result;
            }
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function disables each type of chunk data before disabling chunk data mode.
int DisableChunkData(INodeMap& nodeMap)
{
    cout << endl << endl << "*** DISABLING CHUNK DATA ***" << endl << endl;

    int result = 0;
    try
    {
        // Retrieve the selector node
        const CEnumerationPtr ptrChunkSelector = nodeMap.GetNode("ChunkSelector");

        if (!IsReadable(ptrChunkSelector))
        {
            cout << "Unable to retrieve chunk selector. Aborting..." << endl;
            return -1;
        }

        result = SetChunkEnable(nodeMap, "InferenceFrameId", false);
        if (result == -1)
        {
            cout << "Unable to disable Inference Frame Id chunk data. Aborting..." << endl;
            return result;
        }

        if (chosenInferenceNetworkType == DETECTION)
        {
            // Detection network type

            // Disable chunk data inference bounding box
            result = SetChunkEnable(nodeMap, "InferenceBoundingBoxResult", false);
            if (result == -1)
            {
                cout << "Unable to disable Inference Bounding Box chunk data. Aborting..." << endl;
                return result;
            }
        }
        else
        {
            // Classification network type

            // Disable chunk data inference result
            result = SetChunkEnable(nodeMap, "InferenceResult", false);
            if (result == -1)
            {
                cout << "Unable to disable Inference Result chunk data. Aborting..." << endl;
                return result;
            }

            // Disable chunk data inference confidence
            result = SetChunkEnable(nodeMap, "InferenceConfidence", false);
            if (result == -1)
            {
                cout << "Unable to disable Inference Confidence chunk data. Aborting..." << endl;
                return result;
            }
        }

        // Deactivate ChunkMode
        CBooleanPtr ptrChunkModeActive = nodeMap.GetNode("ChunkModeActive");
        if (!IsWritable(ptrChunkModeActive))
        {
            cout << "Unable to deactivate chunk mode. Aborting..." << endl;
            return -1;
        }
        ptrChunkModeActive->SetValue(false);
        cout << "Chunk mode deactivated..." << endl;

        // Disable Inference
        CBooleanPtr ptrInferenceEnable = nodeMap.GetNode("InferenceEnable");
        if (!IsWritable(ptrInferenceEnable))
        {
            cout << "Unable to disable inference. Aborting..." << endl;
            return -1;
        }
        ptrInferenceEnable->SetValue(false);
        cout << "Inference disabled..." << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function displays the inference-related chunk data from the image.
int DisplayChunkData(const ImagePtr pImage)
{
    int result = 0;

    cout << "Printing chunk data from image..." << endl;

    try
    {
        //
        // Retrieve chunk data from image
        //
        // *** NOTES ***
        // When retrieving chunk data from an image, the data is stored in a
        // a ChunkData object and accessed with getter functions.
        //
        const ChunkData chunkData = pImage->GetChunkData();

        const int64_t inferenceFrameID = chunkData.GetInferenceFrameId();
        cout << "\tInference Frame ID: " << inferenceFrameID << endl;

        if (chosenInferenceNetworkType == DETECTION)
        {
            InferenceBoundingBoxResult boxResult = chunkData.GetInferenceBoundingBoxResult();
            const int16_t boxCount = boxResult.GetBoxCount();

            cout << "\tInference Bounding Box Result:" << endl;

            if (boxCount == 0)
            {
                cout << "\tNo bounding box" << endl;
            }

            for (int16_t i = 0; i < boxCount; ++i)
            {
                const InferenceBoundingBox box = boxResult.GetBoxAt(i);
                switch (box.boxType)
                {
                case INFERENCE_BOX_TYPE_RECTANGLE:
                    cout << "\tBox[" << i + 1 << "]: "
                         << "Class " << box.classId << " ("
                         << (static_cast<uint64_t>(box.classId) < labelDetection.size() ? labelDetection[box.classId]
                                                                                        : "N/A")
                         << ") - " << box.confidence * 100 << "% - "
                         << "Rectangle ("
                         << "X=" << box.rect.topLeftXCoord << ", "
                         << "Y=" << box.rect.topLeftYCoord << ", "
                         << "W=" << box.rect.bottomRightXCoord - box.rect.topLeftXCoord << ", "
                         << "H=" << box.rect.bottomRightYCoord - box.rect.topLeftYCoord << ")" << endl;
                    break;
                case INFERENCE_BOX_TYPE_CIRCLE:
                    cout << "\tBox[" << i + 1 << "]: "
                         << "Class " << box.classId << " ("
                         << (static_cast<uint64_t>(box.classId) < labelDetection.size() ? labelDetection[box.classId]
                                                                                        : "N/A")
                         << ") - " << box.confidence * 100 << "% - "
                         << "Circle ("
                         << "X=" << box.circle.centerXCoord << ", "
                         << "Y=" << box.circle.centerYCoord << ", "
                         << "R=" << box.circle.radius << ")" << endl;
                    break;
                case INFERENCE_BOX_TYPE_ROTATED_RECTANGLE:
                    cout << "\tBox[" << i + 1 << "]: "
                         << "Class " << box.classId << " ("
                         << (static_cast<uint64_t>(box.classId) < labelDetection.size() ? labelDetection[box.classId]
                                                                                        : "N/A")
                         << ") - " << box.confidence * 100 << "% - "
                         << "Rotated Rectangle ("
                         << "X1=" << box.rotatedRect.topLeftXCoord << ", "
                         << "Y1=" << box.rotatedRect.topLeftYCoord << ", "
                         << "X2=" << box.rotatedRect.bottomRightXCoord << ", "
                         << "Y2=" << box.rotatedRect.bottomRightYCoord << ", "
                         << "angle=" << box.rotatedRect.rotationAngle << ")" << endl;
                    break;
                default:
                    cout << "\tBox[" << i + 1 << "]: "
                         << "Class " << box.classId << " ("
                         << (static_cast<uint64_t>(box.classId) < labelDetection.size() ? labelDetection[box.classId]
                                                                                        : "N/A")
                         << ") - " << box.confidence * 100 << "% - "
                         << "Unknown bounding box type (not supported)" << endl;
                    break;
                }
            }
        }
        else
        {
            uint64_t inferenceResult = chunkData.GetInferenceResult();
            cout << "\tInference Result: " << inferenceResult << " ("
                 << (inferenceResult < labelClassification.size()
                         ? labelClassification[static_cast<unsigned int>(inferenceResult)]
                         : "N/A")
                 << ")" << endl;

            double inferenceConfidence = chunkData.GetInferenceConfidence();
            cout << "\tInference Confidence: " << inferenceConfidence << endl;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function disables trigger mode on the camera.
int DisableTrigger(INodeMap& nodeMap)
{
    cout << endl << endl << "*** DISABLING TRIGGER ***" << endl << endl;
    int result = 0;
    try
    {
        // Configure TriggerMode
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode) ||
            !IsWritable(ptrTriggerMode))
        {
            cout << "Unable to configure TriggerMode. Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerOff = ptrTriggerMode->GetEntryByName("Off");
        if (!IsReadable(ptrTriggerOff))
        {
            cout << "Unable to query TriggerMode Off. Aborting..." << endl;
            return -1;
        }

        cout << "Configure TriggerMode to " << ptrTriggerOff->GetSymbolic() << endl;
        ptrTriggerMode->SetIntValue(static_cast<int64_t>(ptrTriggerOff->GetNumericValue()));
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = -1;
    }

    return result;
}

// This function configures camera to run in "inference sync" trigger mode.
int ConfigureTrigger(INodeMap& nodeMap)
{
    cout << endl << endl << "*** CONFIGURING TRIGGER ***" << endl << endl;
    int result = 0;

    try
    {
        // Configure TriggerSelector
        CEnumerationPtr ptrTriggerSelector = nodeMap.GetNode("TriggerSelector");
        if (!IsReadable(ptrTriggerSelector) ||
            !IsWritable(ptrTriggerSelector))
        {
            cout << "Unable to configure TriggerSelector. Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrFrameStart = ptrTriggerSelector->GetEntryByName("FrameStart");
        if (!IsReadable(ptrFrameStart))
        {
            cout << "Unable to query TriggerSelector FrameStart. Aborting..." << endl;
            return -1;
        }

        cout << "Configure TriggerSelector to " << ptrFrameStart->GetSymbolic() << endl;
        ptrTriggerSelector->SetIntValue(static_cast<int64_t>(ptrFrameStart->GetNumericValue()));

        // Configure TriggerSource
        CEnumerationPtr ptrTriggerSource = nodeMap.GetNode("TriggerSource");
        if (!IsReadable(ptrTriggerSource) ||
            !IsWritable(ptrTriggerSource))
        {
            cout << "Unable to configure TriggerSource. Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrInferenceReady = ptrTriggerSource->GetEntryByName("InferenceReady");
        if (!IsReadable(ptrInferenceReady))
        {
            cout << "Unable to query TriggerSource InferenceReady. Aborting..." << endl;
            return -1;
        }

        cout << "Configure TriggerSource to " << ptrInferenceReady->GetSymbolic() << endl;
        ptrTriggerSource->SetIntValue(static_cast<int64_t>(ptrInferenceReady->GetNumericValue()));

        // Configure TriggerMode
        CEnumerationPtr ptrTriggerMode = nodeMap.GetNode("TriggerMode");
        if (!IsReadable(ptrTriggerMode) ||
            !IsWritable(ptrTriggerMode))
        {
            cout << "Unable to configure TriggerMode. Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrTriggerOn = ptrTriggerMode->GetEntryByName("On");
        if (!IsReadable(ptrTriggerOn))
        {
            cout << "Unable to query TriggerMode On. Aborting..." << endl;
            return -1;
        }

        cout << "Configure TriggerMode to " << ptrTriggerOn->GetSymbolic() << endl;
        ptrTriggerMode->SetIntValue(static_cast<int64_t>(ptrTriggerOn->GetNumericValue()));
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what();
        result = -1;
    }

    return result;
}

// This function enables/disables inference on the camera and configures the inference network type
int ConfigureInference(INodeMap& nodeMap, bool isEnabled)
{
    int result = 0;

    if (isEnabled)
    {
        cout << endl
             << endl
             << "*** CONFIGURING INFERENCE ("
             << ((chosenInferenceNetworkType == DETECTION) ? "DETECTION" : "CLASSIFICATION") << ") ***" << endl
             << endl;
    }
    else
    {
        cout << endl << endl << "*** DISABLING INFERENCE ***" << endl << endl;
    }

    try
    {
        if (isEnabled)
        {
            // Set Network Type to Detection
            CEnumerationPtr ptrInferenceNetworkTypeSelector = nodeMap.GetNode("InferenceNetworkTypeSelector");
            if (!IsReadable(ptrInferenceNetworkTypeSelector) ||
                !IsWritable(ptrInferenceNetworkTypeSelector))
            {
                cout << "Unable to query InferenceNetworkTypeSelector. Aborting..." << endl;
                return -1;
            }

            const gcstring networkTypeString =
                (chosenInferenceNetworkType == DETECTION) ? "Detection" : "Classification";

            // Retrieve entry node from enumeration node
            CEnumEntryPtr ptrInferenceNetworkType = ptrInferenceNetworkTypeSelector->GetEntryByName(networkTypeString);
            if (!IsReadable(ptrInferenceNetworkType))
            {
                cout << "Unable to set inference network type to " << networkTypeString
                     << " (entry retrieval). Aborting..." << endl
                     << endl;
                return -1;
            }

            ptrInferenceNetworkTypeSelector->SetIntValue(
                static_cast<int64_t>(ptrInferenceNetworkType->GetNumericValue()));

            cout << "Inference network type set to " << networkTypeString << "..." << endl;
        }

        // Enable/Disable inference
        cout << (isEnabled ? "Enabling" : "Disabling") << " inference..." << endl;
        CBooleanPtr ptrInferenceEnable = nodeMap.GetNode("InferenceEnable");
        if (!IsWritable(ptrInferenceEnable))
        {
            cout << "Unable to enable inference. Aborting..." << endl;
            return -1;
        }

        ptrInferenceEnable->SetValue(isEnabled);
        cout << "Inference " << (isEnabled ? "enabled..." : "disabled...") << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function configures camera test pattern to make use of the injected test image for inference
int ConfigureTestPattern(INodeMap& nodeMap, bool isEnabled)
{
    int result = 0;

    if (isEnabled)
    {
        cout << endl << endl << "*** CONFIGURING TEST PATTERN ***" << endl << endl;
    }
    else
    {
        cout << endl << endl << "*** DISABLING TEST PATTERN ***" << endl << endl;
    }

    try
    {
        // Set TestPatternGeneratorSelector to PipelineStart
        CEnumerationPtr ptrTestPatternGeneratorSelector = nodeMap.GetNode("TestPatternGeneratorSelector");
        if (!IsReadable(ptrTestPatternGeneratorSelector) ||
            !IsWritable(ptrTestPatternGeneratorSelector))
        {
            cout << "Unable to query TestPatternGeneratorSelector. Aborting..." << endl;
            return -1;
        }

        if (isEnabled)
        {
            CEnumEntryPtr ptrTestPatternGeneratorPipelineStart =
                ptrTestPatternGeneratorSelector->GetEntryByName("PipelineStart");
            if (!IsReadable(ptrTestPatternGeneratorPipelineStart))
            {
                cout << "Unable to query TestPatternGeneratorSelector PipelineStart. Aborting..." << endl;
                return -1;
            }

            ptrTestPatternGeneratorSelector->SetIntValue(
                static_cast<int64_t>(ptrTestPatternGeneratorPipelineStart->GetNumericValue()));
            cout << "TestPatternGeneratorSelector set to " << ptrTestPatternGeneratorPipelineStart->GetSymbolic()
                 << "..." << endl;
        }
        else
        {
            CEnumEntryPtr ptrTestPatternGeneratorSensor = ptrTestPatternGeneratorSelector->GetEntryByName("Sensor");
            if (!IsReadable(ptrTestPatternGeneratorSensor))
            {
                cout << "Unable to query TestPatternGeneratorSelector Sensor. Aborting..." << endl;
                return -1;
            }

            ptrTestPatternGeneratorSelector->SetIntValue(
                static_cast<int64_t>(ptrTestPatternGeneratorSensor->GetNumericValue()));
            cout << "TestPatternGeneratorSelector set to " << ptrTestPatternGeneratorSensor->GetSymbolic() << "..."
                 << endl;
        }

        // Set TestPattern to InjectedImage
        CEnumerationPtr ptrTestPattern = nodeMap.GetNode("TestPattern");
        if (!IsReadable(ptrTestPattern) || !IsWritable(ptrTestPattern))
        {
            cout << "Unable to query TestPattern. Aborting..." << endl;
            return -1;
        }

        if (isEnabled)
        {
            CEnumEntryPtr ptrInjectedImage = ptrTestPattern->GetEntryByName("InjectedImage");
            if (!IsReadable(ptrInjectedImage))
            {
                cout << "Unable to query TestPattern InjectedImage. Aborting..." << endl;
                return -1;
            }

            ptrTestPattern->SetIntValue(static_cast<int64_t>(ptrInjectedImage->GetNumericValue()));
            cout << "TestPattern set to " << ptrInjectedImage->GetSymbolic() << "..." << endl;
        }
        else
        {
            CEnumEntryPtr ptrTestPatternOff = ptrTestPattern->GetEntryByName("Off");
            if (!IsReadable(ptrTestPatternOff))
            {
                cout << "Unable to query TestPattern Off. Aborting..." << endl;
                return -1;
            }

            ptrTestPattern->SetIntValue(static_cast<int64_t>(ptrTestPatternOff->GetNumericValue()));
            cout << "TestPattern set to " << ptrTestPatternOff->GetSymbolic() << "..." << endl;
        }

        if (isEnabled)
        {
            // The inject images have different ROI sizes so camera needs to be configured to the appropriate
            // injected width and height
            CIntegerPtr ptrInjectedWidth = nodeMap.GetNode("InjectedWidth");
            if (!IsReadable(ptrInjectedWidth) ||
                !IsWritable(ptrInjectedWidth))
            {
                cout << "Unable to query InjectedWidth. Aborting..." << endl;
                return -1;
            }

            ptrInjectedWidth->SetValue(isEnabled ? injectedImageWidth : ptrInjectedWidth->GetMax());

            CIntegerPtr ptrInjectedHeight = nodeMap.GetNode("InjectedHeight");
            if (!IsReadable(ptrInjectedHeight) ||
                !IsWritable(ptrInjectedHeight))
            {
                cout << "Unable to query InjectedHeight. Aborting..." << endl;
                return -1;
            }

            ptrInjectedHeight->SetValue(isEnabled ? injectedImageHeight : ptrInjectedHeight->GetMax());
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected exception : " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function acquires and saves 10 images from a device; please see
// Acquisition example for more in-depth comments on acquiring images.
int AcquireImages(const CameraPtr& pCam, INodeMap& nodeMap, INodeMap& nodeMapTLDevice)
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
            cout << "Unable to set acquisition mode to continuous (node retrieval). Aborting..." << endl;
            return -1;
        }

        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting..." << endl
                 << endl;
            return -1;
        }

        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

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
                         << ", height = " << pResultImage->GetHeight() << "." << endl;

                    // Display chunk data
                    result = DisplayChunkData(pResultImage);

                    pResultImage->Save("image.jpg");
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
int RunSingleCamera(const CameraPtr& pCam)
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

        // Check to make sure camera supports inference
        cout << endl << "Checking camera inference support..." << endl;
        CBooleanPtr ptrInferenceEnable = nodeMap.GetNode("InferenceEnable");
        if (!IsWritable(ptrInferenceEnable))
        {
            cout << "Inference is not supported on this camera. Aborting..." << endl;
            return -1;
        }

        // Upload custom inference network onto the camera
        // The inference network file is in a movidius specific neural network format.
        // Uploading the network to the camera allows for "inference on the edge" where
        // camera can apply deep learning on a live stream. Refer to "Getting Started
        // with Firefly-DL" for information on how to create your own custom inference
        // network files using pre-existing neural network.
        err = UploadFileToCamera(nodeMap, "InferenceNetwork", networkFilePath);
        if (err < 0)
        {
            return err;
        }

        // Upload injected test image
        // Instead of applying deep learning on a live stream, the camera can be
        // tested with an injected test image.
        err = UploadFileToCamera(nodeMap, "InjectedImage", injectedImageFilePath);
        if (err < 0)
        {
            return err;
        }

        // Configure inference
        err = ConfigureInference(nodeMap, true);
        if (err < 0)
        {
            return err;
        }

        // Configure test pattern to make use of the injected image
        err = ConfigureTestPattern(nodeMap, true);
        if (err < 0)
        {
            return err;
        }

        // Configure trigger
        // When enabling inference results via chunk data, the results that accompany a frame
        // will likely not be the frame that inference was run on. In order to guarantee that
        // the chunk inference results always correspond to the frame that they are sent with,
        // the camera needs to be put into the "inference sync" trigger mode.
        // Note: Enabling this setting will limit frame rate so that every frame contains new
        //       inference dataset. To not limit the frame rate, you can enable InferenceFrameID
        //       chunk data to help determine which frame is associated with a particular
        //       inference data.
        err = ConfigureTrigger(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Configure chunk data
        err = ConfigureChunkData(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Acquire images and display chunk data
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice);

        // Disable chunk data
        err = DisableChunkData(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Disable trigger
        err = DisableTrigger(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Disable test pattern
        err = ConfigureTestPattern(nodeMap, false);
        if (err < 0)
        {
            return err;
        }

        // Disable inference
        err = ConfigureInference(nodeMap, false);
        if (err < 0)
        {
            return err;
        }

        // Clear injected test image
        err = DeleteFileOnCamera(nodeMap, "InjectedImage");
        if (err < 0)
        {
            return err;
        }

        // Clear uploaded inference network
        err = DeleteFileOnCamera(nodeMap, "InferenceNetwork");
        if (err < 0)
        {
            return err;
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