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
 *  @example Compression.cpp
 *
 *  @brief Compression.cpp shows how to utilize the compression features on a
 *  supported camera and in the Spinnaker SDK. It relies on information
 *  provided in the Acquisition, Enumeration, ChunkData, and NodeMapInfo
 *  examples.
 *
 *  This example covers all of the following: the preparation of a camera to
 *  acquire compressed images (or compressed chunk images), image retrieval,
 *  image saving, loading compressed images from disk, reconstructing compressed
 *  images, and converting compressed images.
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

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// Use the following global constant to select whether to stream with compressed
// chunk images along with other compressed related data
const bool enableChunkData = false;

struct CompressedImageInfo
{
    string fileName;
    size_t compressedImageSize;
    size_t imageWidth;
    size_t imageHeight;
    size_t imageXOffset;
    size_t imageYOffset;
    PixelFormatEnums imagePixelFormat;

    CompressedImageInfo(
        const string& fileName,
        size_t imageSize,
        size_t width,
        size_t height,
        size_t xOffset,
        size_t yOffset,
        PixelFormatEnums pixelFormat)
        : fileName(fileName), compressedImageSize(imageSize), imageWidth(width), imageHeight(height),
          imageXOffset(xOffset), imageYOffset(yOffset), imagePixelFormat(pixelFormat)
    {
    }
};

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
// Please see Acquisition example for more in-depth comments on acquiring,
// converting and saving images from a device.
int AcquireImages(
    CameraPtr pCam,
    INodeMap& nodeMap,
    INodeMap& nodeMapTLDevice,
    vector<CompressedImageInfo>& compressedImageInfos)
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
            cout << "Unable to get acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
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
                    //
                    // Print image information
                    //
                    // *** NOTES ***
                    // IsCompressed() can be used to verify if the grabbed images indeed have compression enabled.
                    //
                    const size_t width = pResultImage->GetWidth();
                    const size_t height = pResultImage->GetHeight();
                    const bool isCompressed = pResultImage->IsCompressed();

                    cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height
                         << ", IsCompressed = " << (isCompressed ? "true" : "false");

                    //
                    // Here we utilize the chunk data feature to retrieve the compression ratio of
                    // each compressed image. For more in-depth comments about using chunk data,
                    // please see the ChunkData example.
                    //
                    if (enableChunkData)
                    {
                        ChunkData chunkData = pResultImage->GetChunkData();
                        const double compressionRatio = static_cast<double>(chunkData.GetCompressionRatio());
                        const int64_t chunkImageCRC = chunkData.GetCRC();
                        cout << ", compression ratio = " << compressionRatio << ", CRC = " << chunkImageCRC;
                    }
                    cout << endl;

                    //
                    // If chunk data is enabled, chunk image CRC will be available by default and we could
                    // check if the library computed checksum of the image payload matches with chunk
                    // data provided image checksum. Note that mismatching CRC could lead to decompression
                    // errors and image integrity issues.
                    //
                    if (pResultImage->HasCRC())
                    {
                        if (!pResultImage->CheckCRC())
                        {
                            cout << "WARNING: CRC mismatch could lead to image decompression failures" << endl;
                        }
                    }

                    // Create a unique filename
                    ostringstream filename;

                    filename << "Compression-";
                    if (!deviceSerialNumber.empty())
                    {
                        filename << deviceSerialNumber.c_str() << "-";
                    }
                    filename << imageCnt;

                    // Save image
                    //
                    // *** NOTES ***
                    // In this example, we are demonstrating how to save compressed images to disk
                    // and later, how to load the saved images for post processing.
                    //
                    // In order to save the images in their compressed form, we must save with the .raw
                    // format because saving in other file formats automatically performs decompression
                    // and color processing. Regardless of whether chunk data is enabled or not, only the
                    // image data will be saved in the raw file and none of the other chunk data enabled
                    // will be preserved when the raw file is loaded back in memory.
                    //
                    // Note that in the CompressedImageInfo, we are saving all the information required
                    // to reconstruct this image when we load it back from disk later.
                    // Also note that GetImageSize() here returns the size of the image data in its
                    // compressed form.
                    //
                    pResultImage->Save(filename.str().c_str(), SPINNAKER_IMAGE_FILE_FORMAT_RAW);
                    cout << "Image saved at " << filename.str() << ".raw" << endl;

                    const CompressedImageInfo imageInfo(
                        filename.str(),
                        pResultImage->GetImageSize(),
                        width,
                        height,
                        pResultImage->GetXOffset(),
                        pResultImage->GetYOffset(),
                        pResultImage->GetPixelFormat());
                    compressedImageInfos.push_back(imageInfo);
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
        return -1;
    }

    return result;
}

// This function prints the device information of the camera from the transport
// layer. Please see NodeMapInfo example for more in-depth comments on printing
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

// This function configures the necessary features to enable image chunk data on the camera.
bool EnableImageChunkData(INodeMap& nodeMap)
{
    try
    {
        cout << endl << "Configuring camera settings to enable image chunk data..." << endl;

        //
        // Enable chunk mode and image chunk data for compression ratio
        //
        // *** NOTES ***
        // Enabling chunk data for compression ratio allows us to inspect the compression ratio of
        // each individual image. This is not a requirement for using the image compression features.
        //
        // For more in-depth comments about using chunk data, see the ChunkData example
        //

        bool chunkEnabled = false;

        // Activate chunk mode
        CBooleanPtr ptrChunkModeActive = nodeMap.GetNode("ChunkModeActive");
        if (IsWritable(ptrChunkModeActive))
        {
            ptrChunkModeActive->SetValue(true);

            // Retrieve the selector node
            CEnumerationPtr ptrChunkSelector = nodeMap.GetNode("ChunkSelector");
            if (IsReadable(ptrChunkSelector) && IsWritable(ptrChunkSelector))
            {
                CEnumEntryPtr ptrCompressionRatioEntry = ptrChunkSelector->GetEntryByName("CompressionRatio");
                if (IsReadable(ptrCompressionRatioEntry))
                {
                    ptrChunkSelector->SetIntValue(ptrCompressionRatioEntry->GetValue());

                    // Retrieve corresponding boolean
                    CBooleanPtr ptrChunkEnable = nodeMap.GetNode("ChunkEnable");

                    // Enable the boolean, thus enabling the chunk data
                    if (IsWritable(ptrChunkEnable))
                    {
                        ptrChunkEnable->SetValue(true);
                        chunkEnabled = true;
                    }
                }
            }
        }

        cout << "CompressionRatio chunk data " << (chunkEnabled ? "enabled" : "not enabled") << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected error while configuring image chunk data: " << e.what() << endl;
        return false;
    }

    return true;
}

// This function shows how to disable image chunk data
bool DisableImageChunkData(INodeMap& nodeMap)
{
    try
    {
        cout << endl << "Disabling image chunk data..." << endl;

        // Disable image compression on the camera
        CEnumerationPtr ptrChunkSelector = nodeMap.GetNode("ChunkSelector");
        if (IsReadable(ptrChunkSelector) && IsWritable(ptrChunkSelector))
        {
            CEnumEntryPtr ptrCompressionRatioEntry = ptrChunkSelector->GetEntryByName("CompressionRatio");
            if (IsReadable(ptrCompressionRatioEntry))
            {
                ptrChunkSelector->SetIntValue(ptrCompressionRatioEntry->GetValue());

                // Retrieve corresponding boolean
                CBooleanPtr ptrChunkEnable = nodeMap.GetNode("ChunkEnable");

                // Disable the boolean, thus disabling the CompressionRatio chunk data
                if (IsWritable(ptrChunkEnable))
                {
                    ptrChunkEnable->SetValue(false);
                }
            }
        }

        // De-activate chunk mode
        CBooleanPtr ptrChunkModeActive = nodeMap.GetNode("ChunkModeActive");
        if (IsWritable(ptrChunkModeActive))
        {
            ptrChunkModeActive->SetValue(false);
        }

        cout << "Disabled image chunk data..." << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected error while disabling image chunk data: " << e.what() << endl;
        return false;
    }

    return true;
}

// This function configures the necessary features to enable image compression on the camera.
bool EnableImageCompression(INodeMap& nodeMap)
{
    try
    {
        cout << endl << "Configuring camera settings to enable image compression..." << endl;

        // Select supported pixel format
        //
        // *** NOTES ***
        // Currently, only the BayerRG8 and Mono8 pixel formats support image compression.
        // User must first select one of these pixel formats for the image compression features to be available.
        //
        CEnumerationPtr ptrPixelFormat = nodeMap.GetNode("PixelFormat");
        if (!IsReadable(ptrPixelFormat) ||
            !IsWritable(ptrPixelFormat))
        {
            cout << "Unable to get or set pixel format. Aborting..." << endl << endl;
            return false;
        }

        // Disable ISP
        //
        // *** NOTES ***
        // The image signal processor (ISP) has to be disabled before any compression features can be configured.
        //
        // Note that we recommend setting the pixel format before disabling ISP because sometimes selecting
        // a pixel format may result in the ISP being enabled again.
        //
        CBooleanPtr ptrIspEnable = nodeMap.GetNode("IspEnable");
        if (IsWritable(ptrIspEnable))
        {
            ptrIspEnable->SetValue(false);
            cout << "IspEnable set to false..." << endl;
        }

        CEnumEntryPtr ptrPixelFormatBayerRG8 = ptrPixelFormat->GetEntryByName("BayerRG8");
        if (!IsReadable(ptrPixelFormatBayerRG8))
        {
            CEnumEntryPtr ptrPixelFormatMono8 = ptrPixelFormat->GetEntryByName("Mono8");
            if (!IsReadable(ptrPixelFormatMono8))
            {
                cout << "Unable to get pixel format to BayerRG8 or Mono8. Aborting..." << endl << endl;
                return false;
            }

            ptrPixelFormat->SetIntValue(ptrPixelFormatMono8->GetValue());
            cout << "Pixel format set to " << ptrPixelFormatMono8->GetSymbolic() << "..." << endl;
        }
        else
        {
            ptrPixelFormat->SetIntValue(ptrPixelFormatBayerRG8->GetValue());
            cout << "Pixel format set to " << ptrPixelFormatBayerRG8->GetSymbolic() << "..." << endl;
        }

        // Select the lossless compression feature
        //
        // *** NOTES ***
        // The lossless image compression allows the camera to achieve higher acquisition frame rate by reducing
        // the image size. However, the compression ratio can vary significantly depending on the image complexity.
        //
        CEnumerationPtr ptrCompressionMode = nodeMap.GetNode("ImageCompressionMode");
        if (!IsWritable(ptrCompressionMode))
        {
            cout << "Unable to set image compression mode to Lossless (enum retrieval). Aborting..." << endl << endl;
            return false;
        }

        CEnumEntryPtr ptrCompressionModeLossless = ptrCompressionMode->GetEntryByName("Lossless");
        if (!IsReadable(ptrCompressionModeLossless))
        {
            cout << "Unable to get image compression mode to Lossless (entry retrieval). Aborting..." << endl << endl;
            return false;
        }

        ptrCompressionMode->SetIntValue(ptrCompressionModeLossless->GetValue());
        cout << "Compression mode set to " << ptrCompressionModeLossless->GetSymbolic() << "..." << endl;

        // Print the current compression configurations
        cout << endl << "*** COMPRESSION SETTINGS ***" << endl << endl;
        cout << "Compression Mode: "
             << (IsReadable(ptrCompressionMode) ? ptrCompressionMode->ToString() : "Node not readable") << endl;

        // Number of bytes in each block of compressed data that can be decoded independently of other blocks
        CIntegerPtr ptrCompressionBlockSize = nodeMap.GetNode("LosslessCompressionBlockSize");
        cout << "Compression Block Size: "
             << (IsReadable(ptrCompressionBlockSize) ? ptrCompressionBlockSize->ToString() : "Node not readable")
             << endl;

        // The ratio between the uncompressed image size and compressed image size
        CFloatPtr ptrCompressionRatio = nodeMap.GetNode("CompressionRatio");
        cout << "Compression Ratio: "
             << (IsReadable(ptrCompressionRatio) ? ptrCompressionRatio->ToString() : "Node not readable") << endl;

        // Determines how the camera handles situations where the datarate of the compressed frames exceeds the datarate
        // of the interface.
        string compressionSaturationPriority = "Node not readable";
        CEnumerationPtr ptrCompressionSaturationPriority = nodeMap.GetNode("CompressionSaturationPriority");
        if (IsReadable(ptrCompressionSaturationPriority))
        {
            CEnumEntryPtr ptrCompressionSaturationPriorityEntry = ptrCompressionSaturationPriority->GetCurrentEntry();
            if (IsReadable(ptrCompressionSaturationPriorityEntry))
            {
                compressionSaturationPriority = ptrCompressionSaturationPriorityEntry->GetSymbolic();
            }
        }
        cout << "Compression Saturation Priority: " << compressionSaturationPriority << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected error while configuring image compression: " << e.what() << endl;
        return false;
    }

    return true;
}

// This function shows how to disable image compression
bool DisableImageCompression(INodeMap& nodeMap)
{
    try
    {
        cout << endl << "Disabling image compression..." << endl;

        // Disable image compression on the camera
        //
        // *** NOTES ***
        // Disabling image compression on the camera is as easy as setting the ImageCompressionMode to "Off"
        //
        CEnumerationPtr ptrCompressionMode = nodeMap.GetNode("ImageCompressionMode");
        if (!IsReadable(ptrCompressionMode) ||
            !IsWritable(ptrCompressionMode))
        {
            cout << "Unable to set image compression mode to Off (enum retrieval). Aborting..." << endl << endl;
            return false;
        }

        CEnumEntryPtr ptrCompressionModeOff = ptrCompressionMode->GetEntryByName("Off");
        if (!IsReadable(ptrCompressionModeOff))
        {
            cout << "Unable to get image compression mode to Off (entry retrieval). Aborting..." << endl << endl;
            return false;
        }

        ptrCompressionMode->SetIntValue(ptrCompressionModeOff->GetValue());
        cout << "Compression mode set to " << ptrCompressionModeOff->GetSymbolic() << "..." << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Unexpected error while disabling image compression: " << e.what() << endl;
        return false;
    }

    return true;
}

// This function loads compressed images from file, performs post-processing on them and
// saves the resulting images
int ProcessCompressedImagesFromFile(const vector<CompressedImageInfo>& compressedImageInfos)
{
    int result = 0;

    // Set the number of decompression threads to use
    //
    // *** NOTES ***
    // The number of threads used by Spinnaker when performing image decompression is
    // defaulted to be equal to one less than the number of concurrent threads
    // supported by the system. This should already result in the fastest
    // decompression speed.
    //
    // However, this can be configured through the SetNumDecompressionThreads() function,
    // and this setting will persist for all future calls to Convert().
    //
    // A higher thread count will result in faster decompression but higher CPU usage, whereas
    // a lower thread count will result in slower decompression but lower CPU usage.
    //
    const unsigned int kNumDecompressionThreads = 4;

    //
    // Create ImageProcessor instance for post processing images
    //
    ImageProcessor processor;

    //
    // Set default image processor color processing method
    //
    // *** NOTES ***
    // By default, if no specific color processing algorithm Is set, the image
    // processor will default to NEAREST_NEIGHBOR method.
    //
    processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

    try
    {
        processor.SetNumDecompressionThreads(kNumDecompressionThreads);
        cout << "Number of decompression threads set to " << kNumDecompressionThreads << endl << endl;
    }
    catch (Spinnaker::Exception& se)
    {
        cout << "Unexpected error when setting the number of decompression threads to " << kNumDecompressionThreads
             << endl;
        cout << "Error: " << se.what() << endl;
        result = -1;
    }

    for (auto imageInfo = compressedImageInfos.begin(); imageInfo != compressedImageInfos.end(); ++imageInfo)
    {
        // Load previously saved compressed images from disk
        ifstream file(imageInfo->fileName + ".raw", ios::binary | ios::in);
        if (!file)
        {
            cout << "Failed to load image " << imageInfo->fileName << endl;
            result = -1;
            continue;
        }

        cout << "Loading compressed image from '" << imageInfo->fileName << ".raw'" << endl;

        std::shared_ptr<char> imageBuffer(new char[imageInfo->compressedImageSize], std::default_delete<char[]>());
        file.read(imageBuffer.get(), imageInfo->compressedImageSize);
        file.close();

        try
        {
            ImagePtr loadedCompressedImage = Image::Create(
                imageInfo->imageWidth,
                imageInfo->imageHeight,
                imageInfo->imageXOffset,
                imageInfo->imageYOffset,
                imageInfo->imagePixelFormat,
                imageBuffer.get(),
                SPINNAKER_TLPAYLOAD_TYPE_LOSSLESS_COMPRESSED, // Note: Chunk data is not preserved when saved to disk
                                                  //       therefore the non-chunk payload type is specified
                imageInfo->compressedImageSize);

            //
            // Perform decompression and image processing
            //
            // *** NOTES ***
            // When converting a compressed image, Spinnaker will first decompress the image
            // before performing other conversion steps such as debayering. As noted above,
            // the decompression portion of the operation may be multi-threaded depending on
            // the number of threads set by the most recent call to SetNumDecompressionThreads().
            //
            // Note that if we only wanted Spinnaker to perform decompression without further image
            // processing, we could call Convert with the same pixel format as the compressed image.
            //
            ImagePtr convertedImage = processor.Convert(loadedCompressedImage, PixelFormat_RGB8);

            // Save converted image
            convertedImage->Save(imageInfo->fileName.c_str(), SPINNAKER_IMAGE_FILE_FORMAT_JPEG);

            cout << "Image saved at " << imageInfo->fileName << ".jpg" << endl;
        }
        catch (Spinnaker::Exception& se)
        {
            cout << "Unexpected error when processing " << imageInfo->fileName << endl;
            cout << "Error: " << se.what() << endl;
            result = -1;
            continue;
        }
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

        // Enable chunk data features
        if (enableChunkData && !EnableImageChunkData(nodeMap))
        {
            cout << "Failed to enable image chunk data. Please check if image chunk data is supported on this camera"
                 << endl;
            return -1;
        }

        // Enable compression features
        if (!EnableImageCompression(nodeMap))
        {
            cout << "Failed to enable image compression. Please check if image compression is supported on this camera"
                 << endl;
            return -1;
        }

        // Acquire images
        vector<CompressedImageInfo> compressedImageInfos;
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice, compressedImageInfos);

        // Load compressed images from file and perform post-processing
        result = result | ProcessCompressedImagesFromFile(compressedImageInfos);

        // Disable image compression
        if (!DisableImageCompression(nodeMap))
        {
            cout << "Failed to disable image compression." << endl;
            result = -1;
        }

        // Disable chunk data
        if (enableChunkData && !DisableImageChunkData(nodeMap))
        {
            cout << "Failed to disable image chunk data." << endl;
            result = -1;
        }

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