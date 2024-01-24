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
 *  @example Planar.cpp
 *
 *  @brief Planar.cpp shows how to acquire and manage images from each stream
 *  of a camera that supports planar and JPEG12 transmission.
 *
 *  Each stream returns a single plane of the planar image and this example
 *  demonstrates how to:
 * - synchronize the stream grabs
 * - serialize and de-serialize planar images to and from disk, and
 * - post process the de-serialized planar image.
 *
 *  This example also touches on the preparation and cleanup of a camera just
 *  before and just after the acquisition of images. Image retrieval and
 *  conversion, grabbing image data, and saving images are all covered as well.
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

// Retrieve, convert, and save images
const unsigned int k_numImages = 10;

// Demonstrates acquiring synchronized planar images with image list events
// instead of polling images with GetNextImageSync()
const bool acquireWithImageListEvents = true;

// Planar camera settings
const int width = 2048;
const int height = 2048;
const int offsetX = 0;
const int offsetY = 0;
const double frameRate = 10.0;
const double compressionRatio = 6.0;

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

// This class defines the properties, parameters, and the event handler itself. Take a
// moment to notice what parts of the class are mandatory, and what have been
// added for demonstration purposes. First, any class used to define image list event
// handlers must inherit from ImageEventHandler. Second, the method signature of
// OnImageListEvent() must also be consistent. Everything else - including the constructor,
// deconstructor, properties, body of OnImageListEvent(), and other functions -
// is particular to the example.
class ImageListEventHandlerImpl : public ImageListEventHandler
{
  public:
    // The constructor retrieves the serial number and initializes the image
    // counter to 1.
    ImageListEventHandlerImpl(CameraPtr pCam)
    {
        // Retrieve device serial number
        INodeMap& nodeMap = pCam->GetTLDeviceNodeMap();

        m_deviceSerialNumber = "";
        CStringPtr ptrDeviceSerialNumber = nodeMap.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrDeviceSerialNumber))
        {
            m_deviceSerialNumber = ptrDeviceSerialNumber->GetValue();
        }

        // Initialize image counter to 1
        m_imageCnt = 1;

        // Release reference to camera
        pCam = nullptr;
    }
    ~ImageListEventHandlerImpl()
    {
    }

    // This method defines an image event. In it, the image that triggered the
    // event is serialized to disk for processing at a later time.
    void OnImageListEvent(ImageList planarImage)
    {
        // *** NOTES ***
        // An OnImageListEvent signals and returns a synchronized image set in an ImageList
        // object based on the frame ID. The ImageList object is simply a generic container
        // for one or more ImagePtr objects.
        //
        // For a planar image, the ImageList object represents the planar image containing
        // the four (R, GR, GB, B) color planes that made up a color image. The four color
        // planes that are returned in the ImageList are guaranteed to be synchronized
        // in timestamp and Frame ID.
        //
        // *** LATER ***
        // Once the image list is saved and/or no longer needed, the image list must be
        // released in order to keep the image buffer from filling up.
        cout << "Received planar image: " << m_imageCnt << endl;

        // Save a maximum of 10 planar images
        if (m_imageCnt <= k_numImages)
        {
            for (unsigned int i = 0; i < planarImage.GetSize(); i++)
            {
                cout << "  Plane:" << i << " FrameID:" << planarImage.GetByIndex(i)->GetFrameID() << endl;
                if (planarImage.GetByIndex(i)->IsIncomplete())
                {
                    cout << "  Plane:" << i << " FrameID:" << planarImage.GetByIndex(i)->GetFrameID()
                         << "Incomplete Image" << endl;
                }
            }

            //
            // Serialization
            //
            // The planar image is saved to file with an extension "sil" for post-processing later.
            ostringstream objFilename;

            objFilename << "Planar-";
            if (!m_deviceSerialNumber.empty())
            {
                objFilename << m_deviceSerialNumber.c_str() << "-";
            }
            objFilename << m_imageCnt << ".sil";

            cout << "  Saving Planar Image to: " << objFilename.str() << "...";
            planarImage.Save(objFilename.str().c_str());
            cout << " Done." << endl << endl;

            //
            // Release image list
            //
            // *** NOTES ***
            // ImageList retrieved from the camera need to be released
            // in order to keep from filling the buffer. ImageList will be
            // cleaned up if the object goes out of scope as well.
            planarImage.Release();
        }

        // Increment image counter
        m_imageCnt++;
    }

    // Getter for image counter
    unsigned int GetImageCount() const
    {
        return m_imageCnt;
    }

  private:
    unsigned int m_imageCnt;
    string m_deviceSerialNumber;
};

bool IsDevicePlanar(CameraPtr pCam)
{
    // Planar camera should have 4 data streams.
    return (pCam->GetNumDataStreams() == 4);
}

gcstring GetSerialNumber(CameraPtr pCam)
{
    INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

    gcstring deviceSerialNumber("");
    CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
    if (IsReadable(ptrStringSerial))
    {
        deviceSerialNumber = ptrStringSerial->GetValue();
    }

    return deviceSerialNumber;
}

// This helper function allows the example to sleep in both Windows and Linux
// systems. Note that Windows sleep takes milliseconds as a parameter while
// Linux systems take microseconds as a parameter.
void SleepyWrapper(int milliseconds)
{
#if defined WIN32 || defined _WIN32 || defined WIN64 || defined _WIN64
    Sleep(milliseconds);
#else
    usleep(1000 * milliseconds);
#endif
}

int SetAcquisitionContinuous(INodeMap& nodeMap)
{
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
    return 0;
}

// This helper function allows the example to set the buffer handling mode and
// buffer count for the specified stream nodemap.  For Planar, it is highly
// recommended that all four streams use the same buffer handling mode.
int ConfigureHostStream(CameraPtr pCam)
{
    // Configure host buffer handling mode and buffer count for each stream
    const uint64_t numStreams = pCam->GetNumDataStreams();

    for (uint64_t streamIndex = 0; streamIndex < numStreams; streamIndex++)
    {
        // Retrieve TL stream nodemap for each stream stream index
        INodeMap& streamNodemap = pCam->GetTLStreamNodeMap(streamIndex);

        // Retrieve and modify Stream Buffer Handling Mode
        CEnumerationPtr ptrHandlingMode = streamNodemap.GetNode("StreamBufferHandlingMode");
        if (!IsReadable(ptrHandlingMode) ||
            !IsWritable(ptrHandlingMode))
        {
            cout << "Unable to get or set Buffer Handling mode (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrHandlingModeEntry = ptrHandlingMode->GetCurrentEntry();
        if (!IsReadable(ptrHandlingModeEntry))
        {
            cout << "Unable to get Buffer Handling mode (Entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrHandlingModeEntry = ptrHandlingMode->GetEntryByName("OldestFirst");
        ptrHandlingMode->SetIntValue(ptrHandlingModeEntry->GetValue());
        cout << endl << "StreamBufferHandlingMode set to: " << ptrHandlingModeEntry->GetDisplayName() << endl;

        // Retrieve and modify Stream Buffer Count
        CIntegerPtr ptrBufferCount = streamNodemap.GetNode("StreamBufferCountManual");
        if (!IsReadable(ptrBufferCount) ||
            !IsWritable(ptrBufferCount))
        {
            cout << "Unable to get or set Buffer Count (Integer node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        cout << "StreamBufferCountManual default : " << ptrBufferCount->GetValue() << endl;
        cout << "StreamBufferCountManual maximum : " << ptrBufferCount->GetMax() << endl;

        ptrBufferCount->SetValue(ptrBufferCount->GetMax());
        cout << "StreamBufferCountManual set to : " << ptrBufferCount->GetValue() << endl;
    }

    return 0;
}

int ConfigureDeviceStream(INodeMap& nodeMap)
{
    // Note the following information for configuring the ideal packet delay (delay between each transmitted packet):
    //
    // Camera firmware will report the current bandwith required by the camera in the 'DeviceLinkCurrentThroughput'
    // node. When 'DeviceLinkThroughputLimit' node is set by the user, firmware will automatically increase packet delay
    // to fulfill the requested throughput. Therefore, once camera is configured for the desired framerate/image
    // size/etc, set 'DeviceLinkCurrentThroughput' node value into 'DeviceLinkThroughputLimit' node, and packet size
    // will be automatically set by the camera.

    CIntegerPtr ptrCurrentThroughput = nodeMap.GetNode("DeviceLinkCurrentThroughput");
    CIntegerPtr ptrThroughputLimit = nodeMap.GetNode("DeviceLinkThroughputLimit");

    if (!IsReadable(ptrCurrentThroughput))
    {
        cout << "Unable to read ptrCurrentThroughput" << endl;
        return -1;
    }

    if (!IsReadable(ptrThroughputLimit) ||
        !IsWritable(ptrThroughputLimit))
    {
        cout << "Unable to read or write ptrThroughputLimit (node retrieval; camera channel selector" << endl;
        return -1;
    }

    cout << "Current camera throughput: " << ptrCurrentThroughput->GetValue() << endl;

    // If the 'DeviceLinkCurrentThroughput' value is lower than the minimum, set the lowest possible value allowed by
    // the 'DeviceLinkCurrentThroughput' node
    if (ptrThroughputLimit->GetMin() > ptrCurrentThroughput->GetValue())
    {
        cout << "DeviceLinkCurrentThroughput node minimum of: " << ptrThroughputLimit->GetMin()
             << " is higher than current throughput we desire to set (" << ptrCurrentThroughput->GetValue() << ")"
             << endl;
        ptrThroughputLimit->SetValue(ptrThroughputLimit->GetMin());
    }
    else
    {
        // Set 'DeviceLinkCurrentThroughput' value into 'DeviceLinkThroughputLimit' node
        ptrThroughputLimit->SetValue(ptrCurrentThroughput->GetValue());
    }

    cout << "DeviceLinkThroughputLimit set to: " << ptrThroughputLimit->GetValue() << endl << endl;

    // Planar cameras expects 4 stream channels
    // Stream channel specific nodes are configured by first selecting the channel through GevStreamChannelSelector
    // node, and then configuring the channel's related nodes.
    CIntegerPtr ptrStreamChannelCount = nodeMap.GetNode("DeviceStreamChannelCount");
    if (!IsAvailable(ptrStreamChannelCount))
    {
        cout << "Unable to get stream channel count. Aborting..." << endl << endl;
        return -1;
    }

    const int64_t streamChannelCount = ptrStreamChannelCount->GetValue();
    for (int64_t channelIndex = 0; channelIndex < streamChannelCount; channelIndex++)
    {
        // Set channel selector
        CIntegerPtr ptrChannelSelector = nodeMap.GetNode("GevStreamChannelSelector");
        if (!IsWritable(ptrChannelSelector))
        {
            cout << "Unable to set stream channel selector. Aborting..." << endl << endl;
            return -1;
        }
        ptrChannelSelector->SetValue(channelIndex);

        // Set Stream Channel ('SC') specific values. Camera may only use one value across all channels.
        CIntegerPtr ptrPacketSize = nodeMap.GetNode("GevSCPSPacketSize");
        if (!IsReadable(ptrPacketSize) ||
            !IsWritable(ptrPacketSize))
        {
            cout << "Unable to read or write packet size for stream channel " << channelIndex << endl;
            return -1;
        }

        ptrPacketSize->SetValue(9000);
        cout << "Stream channel " << channelIndex << " PacketSize set to: " << ptrPacketSize->GetValue() << endl;

        // Read Packet Delay as a check
        CIntegerPtr ptrPacketDelay = nodeMap.GetNode("GevSCPD");
        if (!IsReadable(ptrPacketDelay))
        {
            cout << "Unable to Read packet delay for channel " << channelIndex << endl;
            return -1;
        }

        cout << "Stream channel " << channelIndex << " Packet Delay: " << ptrPacketDelay->GetValue() << endl;
    }

    return 0;
}

int SetAcquisitionFrameRate(INodeMap& nodeMap)
{
    CBooleanPtr ptrAcquisitionFrameRateEnable = nodeMap.GetNode("AcquisitionFrameRateEnable");
    if (!IsWritable(ptrAcquisitionFrameRateEnable))
    {
        cout << "Unable to set Acquisition Frame Rate Enable to true (enum retrieval). Aborting..." << endl << endl;
        return -1;
    }
    ptrAcquisitionFrameRateEnable->SetValue(true);

    CFloatPtr ptrAcquisitionFrameRate = nodeMap.GetNode("AcquisitionFrameRate");
    if (!IsReadable(ptrAcquisitionFrameRate) ||
        !IsWritable(ptrAcquisitionFrameRate))
    {
        cout << "Unable to set Acquisition Frame Rate" << endl << endl;
        return -1;
    }
    ptrAcquisitionFrameRate->SetValue(frameRate);

    cout << "Set Acquisition Frame Rate to  " << ptrAcquisitionFrameRate->GetValue() << endl;

    return 0;
}

int ConfigureCamera(INodeMap& nodeMap)
{
    // Sets up camera in a preset configuration. Adjust accordingly for desired streaming scenario.
    SetAcquisitionContinuous(nodeMap);

    // Set nodes with integer values
    std::map<std::string, int> intNodeEntries;

    // The MaxDatarateThreshold node specifies the desired max data rate in Bytes/second for datarate sum of all JPEG12
    // stream channels combined.  MaxDatarateThreshold can be calculated based on the ROI, the camera frame rate
    // and the desired compression rate.
    //
    //      MaxDatarateThreshold = (Width * Height * 12 * FrameRate) / (CompressionRatio * 8) (Bps)

    const int maxDataRateThreshold = static_cast<int>(width * height * 12 * frameRate / compressionRatio / 8);

    // The step increment sizes for the data rate nodes are in units of 500Bps so the final value set needs to be rounded to the nearest 500 Bps.
    const int nodeStepSize = 500;
    const int maxDataRateThresholdRounded = (maxDataRateThreshold % nodeStepSize == 0) ? maxDataRateThreshold : (maxDataRateThreshold + nodeStepSize - (maxDataRateThreshold % nodeStepSize));

    // MaxDatarateThreshold is automatically split evenly into JpegDatarateStreamN nodes. 
    // Alternatively, JpegDatarateStream0, JpegDatarateStream1, JpegDatarateStream2, JpegDatarateStream3 can be set individually
    intNodeEntries[std::string("MaxDatarateThreshold")] = maxDataRateThresholdRounded;

    // Configure the camera's region of interest (ROI)
    intNodeEntries[std::string("OffsetX")] = offsetX;
    intNodeEntries[std::string("OffsetY")] = offsetY;
    intNodeEntries[std::string("Width")] = width;
    intNodeEntries[std::string("Height")] = height;

    for (map<std::string, int>::iterator it = intNodeEntries.begin(); it != intNodeEntries.end(); ++it)
    {
        std::string nodeName = it->first;
        const int nodeVal = it->second;

        cout << "Setting " << nodeName << " to " << nodeVal << '.' << endl;
        CIntegerPtr ptrNode = nodeMap.GetNode(nodeName.c_str());
        if (!IsReadable(ptrNode) ||
            !IsWritable(ptrNode))
        {
            cout << "Unable to set node: " << nodeName << ". Aborting..." << endl << endl;
            return -1;
        }

        ptrNode->SetValue(nodeVal);
        cout << " New Value: " << ptrNode->GetValue() << endl;
    }

    cout << endl;

    ConfigureDeviceStream(nodeMap);

    SetAcquisitionFrameRate(nodeMap);

    return 0;
}

// This function configures the camera to add chunk data to each image. It does
// this by enabling each type of chunk data after enabling chunk data mode.
// When chunk data mode is turned on, the data is made available in both the nodemap
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
            cout << "Unable to activate chunk mode. Aborting..." << endl << endl;
            return -1;
        }

        ptrChunkModeActive->SetValue(true);

        cout << "Chunk mode activated..." << endl;

        //
        // Enable all types of chunk data
        //
        // *** NOTES ***
        // Enabling chunk data requires working with nodes: "ChunkSelector"
        // is an enumeration selector node and "ChunkEnable" is a boolean. It
        // requires retrieving the selector node (which is of enumeration node
        // type), selecting the entry of the chunk data to be enabled, retrieving
        // the corresponding boolean, and setting it to true.
        //
        // In this example, all chunk data is enabled, so these steps are
        // performed in a loop. Once this is complete, chunk mode still needs to
        // be activated.
        //
        NodeList_t entries;

        // Retrieve the selector node
        CEnumerationPtr ptrChunkSelector = nodeMap.GetNode("ChunkSelector");

        if (!IsReadable(ptrChunkSelector) ||
            !IsWritable(ptrChunkSelector))
        {
            cout << "Unable to retrieve chunk selector. Aborting..." << endl << endl;
            return -1;
        }

        // Retrieve entries
        ptrChunkSelector->GetEntries(entries);

        cout << "Enabling entries..." << endl;

        for (size_t i = 0; i < entries.size(); i++)
        {
            // Select entry to be enabled
            CEnumEntryPtr ptrChunkSelectorEntry = entries.at(i);

            // Go to next node if problem occurs
            if (!IsReadable(ptrChunkSelectorEntry))
            {
                continue;
            }

            ptrChunkSelector->SetIntValue(ptrChunkSelectorEntry->GetValue());

            cout << "\t" << ptrChunkSelectorEntry->GetSymbolic() << ": ";

            // Retrieve corresponding boolean
            CBooleanPtr ptrChunkEnable = nodeMap.GetNode("ChunkEnable");

            // Enable the boolean, thus enabling the corresponding chunk data
            if (!IsAvailable(ptrChunkEnable))
            {
                cout << "not available" << endl;
                result = -1;
            }
            else if (ptrChunkEnable->GetValue())
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
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function displays a select amount of chunk data from the image. Unlike
// accessing chunk data via the nodemap, there is no way to loop through all
// available data.
int DisplayChunkData(ImagePtr pImage)
{
    int result = 0;

    cout << " Printing chunk data from image..." << endl;

    try
    {
        //
        // Retrieve chunk data from image
        //
        // *** NOTES ***
        // When retrieving chunk data from an image, the data is stored in a
        // a ChunkData object and accessed with getter functions.
        //
        ChunkData chunkData = pImage->GetChunkData();

        //
        // Retrieve exposure time; exposure time recorded in microseconds
        //
        // *** NOTES ***
        // Floating point numbers are returned as a float64_t. This can safely
        // and easily be statically cast to a double.
        //
        double exposureTime = static_cast<double>(chunkData.GetExposureTime());
        std::cout << "\tExposure time: " << exposureTime << endl;

        //
        // Retrieve frame ID
        //
        // *** NOTES ***
        // Integers are returned as an int64_t. As this is the typical integer
        // data type used in the Spinnaker SDK, there is no need to cast it.
        //
        int64_t frameID = chunkData.GetFrameID();
        cout << "\tFrame ID: " << frameID << endl;

        // Retrieve gain; gain recorded in decibels
        double gain = chunkData.GetGain();
        cout << "\tGain: " << gain << endl;

        // Retrieve height; height recorded in pixels
        int64_t height = chunkData.GetHeight();
        cout << "\tHeight: " << height << endl;

        // Retrieve offset X; offset X recorded in pixels
        int64_t offsetX = chunkData.GetOffsetX();
        cout << "\tOffset X: " << offsetX << endl;

        // Retrieve offset Y; offset Y recorded in pixels
        int64_t offsetY = chunkData.GetOffsetY();
        cout << "\tOffset Y: " << offsetY << endl;

        // Retrieve sequencer set active
        int64_t sequencerSetActive = chunkData.GetSequencerSetActive();
        cout << "\tSequencer set active: " << sequencerSetActive << endl;

        // Retrieve timestamp
        uint64_t timestamp = chunkData.GetTimestamp();
        cout << "\tTimestamp: " << timestamp << endl;

        // Retrieve width; width recorded in pixels
        int64_t width = chunkData.GetWidth();
        cout << "\tWidth: " << width << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int AcquireImages(CameraPtr pCam)
{
    cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try
    {
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl << endl;

        const gcstring deviceSerialNumber = GetSerialNumber(pCam);

        for (unsigned int imageCnt = 1; imageCnt <= k_numImages; imageCnt++)
        {
            //
            // Retrieve next received planar image
            //
            // *** NOTES ***
            // GetNextImageSync() captures an image from each stream and returns a synchronized
            // image set in an ImageList object based on the frame ID. The ImageList object is
            // simply a generic container for one or more ImagePtr objects.
            //
            // For a planar image, the ImageList object represents the planar image containing
            // the four (R, GR, GB, B) color planes that made up a color image. The four color
            // planes that are returned in the ImageList are guaranteed to be synchronized
            // in timestamp and Frame ID.
            //
            // *** LATER ***
            // Once the image list is saved and/or no longer needed, the image list must be
            // released in order to keep the image buffer from filling up.
            ImageList planarImage = pCam->GetNextImageSync(1000);
            cout << "Acquired planar image: " << imageCnt << endl;
            for (unsigned int i = 0; i < planarImage.GetSize(); i++)
            {
                cout << "  Plane:" << i << " FrameID:" << planarImage.GetByIndex(i)->GetFrameID() << endl;
                if (planarImage.GetByIndex(i)->IsIncomplete())
                {
                    cout << "  Plane:" << i << " FrameID:" << planarImage.GetByIndex(i)->GetFrameID()
                         << "Incomplete Image" << endl;
                }
            }

            //
            // Serialization
            //
            // The planar image is saved to file with an extension "sil" for post-processing later.
            ostringstream objFilename;

            objFilename << "Planar-";
            if (!deviceSerialNumber.empty())
            {
                objFilename << deviceSerialNumber.c_str() << "-";
            }
            objFilename << imageCnt << ".sil";

            cout << "  Saving Planar Image to: " << objFilename.str() << "...";
            planarImage.Save(objFilename.str().c_str());
            cout << " Done." << endl << endl;

            //
            // Release image list
            //
            // *** NOTES ***
            // ImageList retrieved from the camera need to be released
            // in order to keep from filling the buffer. ImageList will be
            // cleaned up if the object goes out of scope as well.
            planarImage.Release();
        }

        //
        // End acquisition
        //
        // *** NOTES ***
        // Ending acquisition appropriately helps ensure that devices clean up
        // properly and do not need to be power-cycled to maintain integrity.
        pCam->EndAcquisition();
        std::cout << "End acquisition" << std::endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Acquisition Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}

int AcquireImagesWithEvents(CameraPtr pCam)
{
    cout << endl << endl << "*** IMAGE ACQUISITION WITH EVENTS ***" << endl << endl;

    try
    {
        //
        // Create image list event handler
        //
        // *** NOTES ***
        // The class has been constructed to accept a camera pointer in order
        // to allow the saving of images with the device serial number.
        //
        ImageListEventHandlerImpl* imageEventHandler = new ImageListEventHandlerImpl(pCam);

        //
        // Register image event handler
        //
        // *** NOTES ***
        // Image list event handlers are registered to cameras. If there are multiple
        // cameras, each camera must have the image list event handlers registered to
        // it separately. Also, multiple image event handlers may be registered to a
        // single camera.
        //
        // *** LATER ***
        // Image list event handlers must be unregistered manually. This must be done
        // prior to releasing the system and while the image event handlers are still
        // in scope.
        //
        pCam->RegisterEventHandler(*imageEventHandler);

        //
        // Begin acquiring images
        //
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl << endl;

        //
        // Wait for images
        //
        // *** NOTES ***
        // In order to passively capture images using image events and
        // automatic polling, the main thread sleeps in increments of 200 ms
        // until 10 images have been acquired and saved.
        //
        const int sleepDuration = 200; // in milliseconds

        while (imageEventHandler->GetImageCount() <= k_numImages)
        {
            cout << "\t//" << endl;
            cout << "\t// Sleeping for " << sleepDuration << " ms. Grabbing images..." << endl;
            cout << "\t//" << endl;

            SleepyWrapper(sleepDuration);
        }

        //
        // End acquisition
        //
        // *** NOTES ***
        // Ending acquisition appropriately helps ensure that devices clean up
        // properly and do not need to be power-cycled to maintain integrity.
        pCam->EndAcquisition();
        std::cout << "End acquisition" << std::endl;

        //
        // Unregister image event handler
        //
        // *** NOTES ***
        // It is important to unregister all image event handlers from all cameras
        // they are registered to.
        //
        pCam->UnregisterEventHandler(*imageEventHandler);

        // Delete image event handler (because it is a pointer)
        delete imageEventHandler;

        cout << "Image events unregistered..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}

int ProcessImages(const char* deviceSerialNumber)
{
    cout << endl << endl << "Processing images..." << endl << endl;

    for (unsigned int imageCnt = 1; imageCnt <= k_numImages; imageCnt++)
    {
        ostringstream objFilename;
        objFilename << "Planar-";
        if (deviceSerialNumber)
        {
            objFilename << deviceSerialNumber << "-";
        }
        objFilename << imageCnt << ".sil";

        //
        // De-serialization
        //
        // The planar image is loaded from the "sil" file that was saved in AcquireImages() previously.
        //
        const ImageList planarImage = ImageList::Load(objFilename.str().c_str());

        //
        // Image Reconstruction
        //
        // The planar image is reconstructed by the ImageProcessor.
        // Here, the ImageList object is converted to a single color ImagePtr object.
        //

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
        processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_NEAREST_NEIGHBOR);

        // Saving the reconstructed image for inspection
        ostringstream destFilename;
        destFilename << "Planar-";
        if (deviceSerialNumber)
        {
            destFilename << deviceSerialNumber << "-";
        }
        destFilename << imageCnt << ".png";

        const ImagePtr pReconstructedImage = processor.Convert(planarImage, PixelFormat_BGR8);

        cout << " Saving to " << destFilename.str() << "..." << endl;
        pReconstructedImage->Save(destFilename.str().c_str());

        // Display chunk data
        DisplayChunkData(pReconstructedImage);

        cout << " Done." << endl << endl;
    }

    cout << endl;

    return 0;
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
        cout << "Print Device Info Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function acts as the body of the example; please see NodeMapInfo example
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
    int result;
    gcstring deviceSerialNumber = GetSerialNumber(pCam);

    if (!IsDevicePlanar(pCam))
    {
        cout << "Device serial number " << deviceSerialNumber << " is not a valid planar camera. Skipping..." << endl;
        return 0;
    }

    //
    // Real time image acquisition
    //
    // This try-catch block demonstrates the real time acquistion of the planar
    // images.  The acquired images are then saved to disk for post-processing
    // in the next try-catch block.
    //
    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapTLDevice);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        // Note: Must be done after camera Init()
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure device stream settings
        ConfigureCamera(nodeMap);

        // Configure chunk data settings
        ConfigureChunkData(nodeMap);

        // Configure host stream settings
        ConfigureHostStream(pCam);

        // Configure heartbeat for GEV camera
#ifdef _DEBUG
        result = result | DisableGVCPHeartbeat(pCam);
#else
        result = result | ResetGVCPHeartbeat(pCam);
#endif

        // Acquire images
        if (!acquireWithImageListEvents)
        {
            result = result | AcquireImages(pCam);
        }
        else
        {
            result = result | AcquireImagesWithEvents(pCam);
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
        std::cout << "Run Single Camera Acquisition Error: " << e.what() << std::endl;
        return -1;
    }

    //
    // Post-processing
    //
    // This try-catch block demonstrates the post-processing of the saved image
    // data.  This involves first loading the "*.sil" files that were saved
    // previously, then performing the image reconstruction.  Finally, the
    // reconstructed images are exported to an easily viewable format.
    //
    // Notice that the post-processing does not require the camera handle.
    //
    try
    {
        result = result | ProcessImages(deviceSerialNumber.c_str());
    }
    catch (Spinnaker::Exception& e)
    {
        std::cout << "Post Processing Error: " << e.what() << std::endl;
        return -1;
    }

    return result;
}

// Example entry point; please see Enumeration example for more in-depth
// comments on preparing and cleaning up the system.
int main(int argc, char** argv)
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
    // This keeps releasing the system from throwing an exception.
    //
    CameraPtr pCam = nullptr;

    int result = 0;

    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera
        pCam = camList.GetByIndex(i);

        cout << endl << "Running example for camera index " << i << "..." << endl << endl;

        // Run example
        result = result | RunSingleCamera(pCam);

        cout << "Camera " << i << " example complete..." << endl << endl;
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