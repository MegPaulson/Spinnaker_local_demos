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
 *  @example AcquisitionMultipleCamerasWriteToFiles.cpp
 *
 *  @brief AcquisitionMultipleCamerasWriteToFiles.cpp shows how to acquire
 *  images from one or more cameras and write the images to a binary file. Thereafter,
 *  the acquired images can be retrieved from the file and saved using a desired file
 *  format supported by the current Spinnaker SDK.
 *
 *  This example covers most of the basics for getting started with the
 *  Spinnaker API including acquiring system objects, camera list acquisition
 *  and initialization of cameras, acquisition of images and writing to a file,
 *  image retrieval from a file, conversion and saving with desired file format is
 *  covered as well.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <assert.h>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// Directory to save data to.
// Modify this to save images to a different directory
const string kDestinationDirectory = "";

// Number of images to grab
const unsigned int k_numImages = 30;

// This struct defines image information for each unique device
// An ImageInfo struct is created for each device detected
struct ImageInfo
{
    size_t imageWidth;
    size_t imageHeight;
    PixelFormatEnums pixelFormat;
    string imageFileName;
    std::shared_ptr<fstream> imageFile;

    ImageInfo(string filename)
        : imageWidth(0), imageHeight(0), pixelFormat(UNKNOWN_PIXELFORMAT), imageFileName(filename)
    {
    }
};

// This vector stores all the ImageInfo for all cameras
vector<ImageInfo> imageInfos;

// Create files to save each of the camera images
bool CreateFiles(unsigned int numCameras)
{
    bool result = true;
    for (unsigned int cameraCnt = 0; cameraCnt < numCameras; cameraCnt++)
    {
        stringstream sstream;
        string tmpFilename;

        sstream << kDestinationDirectory << "camera" << cameraCnt << ".tmp";
        sstream >> tmpFilename;

        imageInfos.push_back(ImageInfo(tmpFilename));

        cout << "Creating " << tmpFilename << "..." << endl;

        // Create temporary files
        imageInfos.at(cameraCnt).imageFile = std::make_shared<fstream>(
            tmpFilename.c_str(), fstream::trunc | fstream::in | fstream::out | fstream::binary);

        if (!imageInfos.at(cameraCnt).imageFile)
        {
            assert(false);
            result = false;
        }
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

// This function configure each of the cameras including the acquisition mode
bool ConfigureCameras(CameraList& camList, unsigned int numCameras)
{
    bool result = true;

    cout << endl << endl << "*** CONFIGURING CAMERAS... ***" << endl << endl;

    try
    {
        for (unsigned int cameraCnt = 0; cameraCnt < numCameras; cameraCnt++)
        {
            // Get the camera node Map
            INodeMap& nodeMap = camList.GetByIndex(cameraCnt)->GetNodeMap();

            // Set acquisition mode to continuous
            CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
            if (!IsReadable(ptrAcquisitionMode))
            {
                cout << "Unable to get acquisition mode to continuous (node retrieval). Aborting..." << endl << endl;
                return false;
            }

            CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
            if (!IsReadable(ptrAcquisitionModeContinuous))
            {
                cout << "Unable to set acquisition mode to continuous (entry 'continuous' retrieval). Aborting..."
                     << endl
                     << endl;
                return false;
            }

            int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

            if (!IsWritable(ptrAcquisitionMode))
            {
                cout << "Unable to set acquisition mode to continuous (node retrieval). Aborting..." << endl << endl;
                return false;
            }

            ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

            cout << "Camera[" << cameraCnt << "]: Acquisition mode set to continuous..." << endl;

            CEnumerationPtr ptrPixelFormat = nodeMap.GetNode("PixelFormat");

            if (!IsWritable(ptrPixelFormat))
            {
                cout << "Unable to set Pixel Format mode (node retrieval). Aborting..." << endl << endl;
                return false;
            }

            CEnumEntryPtr ptrPixelFormatBayerRG8 = ptrPixelFormat->GetEntryByName("BayerRG8");
            CEnumEntryPtr ptrMono8 = ptrPixelFormat->GetEntryByName("Mono8");

            if (IsReadable(ptrPixelFormatBayerRG8))
            {
                ptrPixelFormat->SetIntValue(ptrPixelFormatBayerRG8->GetValue());

                // Keep the track of the pixel Format
                imageInfos.at(cameraCnt).pixelFormat = PixelFormatEnums::PixelFormat_BayerRG8;
                cout << "Camera[" << cameraCnt
                     << "]: Pixel format set to: " << ptrPixelFormat->GetCurrentEntry()->GetName() << endl;
            }
            else if (IsReadable(ptrMono8))
            {
                ptrPixelFormat->SetIntValue(ptrMono8->GetValue());

                // Keep the track of the pixel Format
                imageInfos.at(cameraCnt).pixelFormat = PixelFormatEnums::PixelFormat_Mono8;

                cout << "Camera[" << cameraCnt << "]: Pixel format set to "
                     << ptrPixelFormat->GetCurrentEntry()->GetName() << endl;
            }
            else
            {
                cout << "Unable to set pixel format (enum entry retrieval). Aborting..." << endl << endl;
                return false;
            }
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = false;
    }

    return result;
}

// This function acquires and saves k_numImages images from a device; please see
// Acquisition example for more in-depth comments on acquiring images.
bool AcquireImagesAndSaveToFile(CameraList& camList, unsigned int numCameras)
{
    bool result = true;

    uint64_t missedImageCnts = 0;

    cout << endl << endl << "*** ACQUIRING AND SAVING IMAGES TO A FILE ***" << endl << endl;

    try
    {

        // Begin acquiring images in all cameras
        for (unsigned int cameraCnt = 0; cameraCnt < numCameras; cameraCnt++)
        {
            camList.GetByIndex(cameraCnt)->BeginAcquisition();
            cout << "Camera[" << cameraCnt << "]: Started acquiring images" << endl;
        }

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
        {
            // Loop through each of the cameras
            for (unsigned int cameraCnt = 0; cameraCnt < numCameras; cameraCnt++)
            {
                try
                {
                    // Retrieve the next received image
                    ImagePtr pResultImage = camList.GetByIndex(cameraCnt)->GetNextImage(1000);

                    if (pResultImage->IsIncomplete())
                    {
                        cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl
                             << endl;
                    }
                    else
                    {
                        char* imageData = static_cast<char*>(pResultImage->GetData());

                        // Do the writing
                        imageInfos.at(cameraCnt).imageFile->write(imageData, pResultImage->GetImageSize());

                        // Check if the writing is successful
                        if (!imageInfos.at(cameraCnt).imageFile->good())
                        {
                            cout << "Error writing to file for camera " << cameraCnt << " !" << endl;
                            return false;
                        }

                        if (imageCnt == 0)
                        {
                            // This is the first image, save image size
                            imageInfos.at(cameraCnt).imageHeight = pResultImage->GetHeight();
                            imageInfos.at(cameraCnt).imageWidth = pResultImage->GetWidth();
                        }
                    }

                    // Release image
                    pResultImage->Release();
                }
                catch (Spinnaker::Exception& e)
                {
                    cout << "Error: " << e.what() << endl;
                    result = false;
                }
            }
        }
        // End acquisition for all cameras
        for (unsigned int cameraCnt = 0; cameraCnt < numCameras; cameraCnt++)
        {
            // Check for any dropped images
            CIntegerPtr pDroppedImages =
                camList.GetByIndex(cameraCnt)->GetTLStreamNodeMap().GetNode("StreamDroppedFrameCount");
            if (IsReadable(pDroppedImages))
            {
                if (pDroppedImages->GetValue() > 0)
                {
                    missedImageCnts += pDroppedImages->GetValue();
                    cout << pDroppedImages->GetValue() << " images "
                         << " missed at camera " << cameraCnt << endl;
                }
            }
            else
            {
                cout << "Unable to determine the dropped frame count from the nodemap at camera " << cameraCnt << endl
                     << endl;
            }
            camList.GetByIndex(cameraCnt)->EndAcquisition();
            cout << "Camera[" << cameraCnt << "]: Stop acquiring images " << endl;
        }
        cout << endl;

        cout << "We missed a total of " << missedImageCnts << " images!" << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = false;
    }

    return result;
}

bool RetrieveImagesFromFiles(unsigned int numCameras, string fileFormat = "bmp")
{
    bool result = true;

    try
    {
        // Loop through the saved files for each camera and retrieve the images
        for (unsigned int cameraCnt = 0; cameraCnt < numCameras; cameraCnt++)
        {
            string tempFilename = imageInfos.at(cameraCnt).imageFileName;

            size_t imageSize = imageInfos.at(cameraCnt).imageHeight * imageInfos.at(cameraCnt).imageWidth;

            cout << "Opening " << tempFilename.c_str() << "..." << endl;

            std::shared_ptr<fstream> rawFile = imageInfos.at(cameraCnt).imageFile;

            if (!rawFile->good())
            {
                cout << "Error opening file: " << imageInfos.at(cameraCnt).imageFileName.c_str() << " Aborting..."
                     << endl;

                return false;
            }

            cout << "Splitting images" << endl;
            rawFile->seekg(0);

            // Read image into buffer
            for (int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
            {
                string readImageFilename;

                stringstream sstream;

                std::shared_ptr<char> pBuffer(new char[imageSize], std::default_delete<char[]>());

                rawFile->read(pBuffer.get(), imageSize);

                // Import image into Image structure
                ImagePtr pImage = Image::Create(
                    imageInfos.at(cameraCnt).imageWidth,
                    imageInfos.at(cameraCnt).imageHeight,
                    0,
                    0,
                    imageInfos.at(cameraCnt).pixelFormat,
                    pBuffer.get());

                // Create file location and file name
                sstream << kDestinationDirectory << "camera" << cameraCnt << "_" << imageCnt << "." << fileFormat;

                sstream >> readImageFilename;

                //  Save image to disk
                pImage->Save(readImageFilename.c_str());

                // Check if reading is successful
                if (!rawFile->good())
                {
                    cout << "Error reading from image " << imageCnt << " for camera " << cameraCnt << ". Aborting..."
                         << endl;

                    return false;
                }

                cout << "Camera[" << cameraCnt << "]: Retrieved image " << imageCnt << endl;
            }

            cout << endl << endl;
        }
    }

    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        return false;
    }
    return result;
}
// This function acts as the body of the example; please see NodeMapInfo example
// for more in-depth comments on setting up cameras.
int RunCameras(CameraList& camList, unsigned int numCameras)
{
    int result = 0;

    try
    {
        // Retrieve TL device nodemap, print device information, and initialize camera
        for (unsigned int i = 0; i < numCameras; i++)
        {
            cout << endl << "Printing device info for camera " << i << "..." << endl;

            INodeMap& nodeMapTLDevice = camList.GetByIndex(i)->GetTLDeviceNodeMap();

            result = PrintDeviceInfo(nodeMapTLDevice);

            cout << endl << "Initializing camera " << i << "..." << endl;

            camList.GetByIndex(i)->Init();
        }

        // Create files to write
        if (!CreateFiles(numCameras))
        {
            cout << "There was an error creating the files. Aborting..." << endl;
            return -1;
        }

        // Configure each of the cameras before starting the acquisition
        if (!ConfigureCameras(camList, numCameras))
        {
            return -1;
        }

        // Acquire Imageas from each camera and save to a file
        if (!AcquireImagesAndSaveToFile(camList, numCameras))
        {
            return -1;
        }

        // Retrieve all the images from each file and
        // save the image using a particular file format
        if (!RetrieveImagesFromFiles(numCameras, "bmp"))
        {
            return -1;
        }

        // Deinitialize camera
        for (unsigned int i = 0; i < numCameras; i++)
        {
            cout << endl << "Deinitializing camera " << i << "..." << endl;

            camList.GetByIndex(i)->DeInit();
        }
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
    string testFile = kDestinationDirectory + "test.txt";
    FILE* tempFile = fopen(testFile.c_str(), "w+");
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
    remove(testFile.c_str());

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

    // Run example on all cameras
    result = result | RunCameras(camList, numCameras);

    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return result;
}