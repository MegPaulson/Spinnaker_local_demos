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
 *  @example AcquisitionMultipleCameraRecovery.cpp
 *
 *  @brief AcquisitionMultipleCameraRecovery.cpp shows how to continously acquire
 *  images from multiple cameras using image events. It demonstrates the use of
 *  User Set Control to save persistent camera configurations, allowing for smooth
 *  camera recovery through interface events. This example relies on information
 *  provided in the ImageEvents, EnumerationEvents, ImageFormatControl, and
 *  Acquisition examples.
 *
 *  This example uses a global map to retain image information, including the number
 *  of images grabbed, the number of incomplete images and the number of removals
 *  for each camera over the duration of the example. Cameras may be added or
 *  removed after the example has started.
 *
 *  The example assumes each camera has a unique serial number and is capable of
 *  configuring User Set 1. Note that if a camera was configured and is disconnected
 *  before the example ends, it will not be reconfigured to use the default User Set.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

#include <iostream>
#include <iomanip>
#include <map>

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// This class defines the properties and methods for image event handling; please see
// ImageEvents example for more in-depth comments on setting up image event handlers.
class ImageEventHandlerImpl : public ImageEventHandler
{
  public:
    ImageEventHandlerImpl(string deviceSerial)
    {
        // The device serial number is stored so the grab information can be updated
        // on each image event.
        m_deviceSerialNumber = deviceSerial;
    }

    ~ImageEventHandlerImpl()
    {
    }

    //
    // *** NOTES ***
    // The implementation for OnImageEvent is defined later as it requires the implementation
    // of the GrabInfo struct below. GrabInfo relies on the implementation of the
    // ImageEventHandlerImpl constructor so it is defined first.
    //
    void OnImageEvent(ImagePtr image);

  private:
    string m_deviceSerialNumber;
};

// This struct defines grab information for each unique device
// A Grab Info object is created each time a new device is configured and is updated on device
// arrival, device removal, and image events.
struct GrabInfo
{
    unsigned int numImagesGrabbed;
    unsigned int numIncompleteImages;
    unsigned int numRemovals;
    std::shared_ptr<ImageEventHandlerImpl> imageEventHandler;

    GrabInfo(const string& deviceSerial) : numImagesGrabbed(0), numIncompleteImages(0), numRemovals(0)
    {
        imageEventHandler = make_shared<ImageEventHandlerImpl>(deviceSerial);
    }
};

// This map stores the Grab Info for all cameras.
std::map<std::string, GrabInfo> cameraGrabInfoMap;

//
// This method belongs to the ImageEventHandlerImpl class and defines an image event.
//
// *** NOTES ***
// Since the example has the potential of grabbing a large number of images, printouts are
// limited to every 10th image grabbed. The final grab information is printed on example
// completion. Images are not saved for this exmaple.
//
void ImageEventHandlerImpl::OnImageEvent(ImagePtr image)
{
    auto cameraGrabInfo = cameraGrabInfoMap.find(m_deviceSerialNumber);
    if (cameraGrabInfo == cameraGrabInfoMap.end())
    {
        cout << "Error OnImageEvent: camera " << m_deviceSerialNumber << " not found in grab info map" << endl;
        return;
    }

    // Check image retrieval status
    if (image->IsIncomplete())
    {
        // Increment the number of incomplete images
        cameraGrabInfo->second.numIncompleteImages++;
    }
    else
    {
        // Retrieve and increment the total number of images grabbed
        unsigned int& numImagesGrabbed = cameraGrabInfo->second.numImagesGrabbed;
        numImagesGrabbed++;

        if (numImagesGrabbed % 10 == 0)
        {
            cout << numImagesGrabbed << " Images grabbed for " << m_deviceSerialNumber << endl;
        }
    }
}

// A global camera list is updated on camera arrival/removals to ensure old cameras can be
// cleaned up when appropriate.
CameraList globalCamList;

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

// This helper function wraps GetCameras in a try catch. This ensures cleanup or setup
// continues even if the camera list failed to update.
void RefreshCameraList(SystemPtr system)
{
    try
    {
        globalCamList = system->GetCameras();
    }
    catch (Exception& e)
    {
        cout << "Error refreshing camera list: " << e.what() << endl;
    }
}

// Provide the function signature for RunSingleCamera so it can be triggered on a
// device arrival.
bool ConfigureCamera(CameraPtr pCam);

// This class defines the properties and functions for device arrivals and removals
// on an interface; please see EnumerationEvents example for more in-depth comments
// on setting up interface event handlers.
class InterfaceEventHandlerImpl : public InterfaceEventHandler
{
  public:
    InterfaceEventHandlerImpl(SystemPtr system) : m_system(system){};

    ~InterfaceEventHandlerImpl(){};

    // This function defines the arrival event on an interface. It refreshes the
    // global camera list and starts a RunSingleCamera thread for the device.
    void OnDeviceArrival(CameraPtr pCamera)
    {
        const std::string serialNum = std::string(pCamera->TLDevice.DeviceSerialNumber.ToString());

        // Refresh the global camera list
        RefreshCameraList(m_system);

        // Configure the camera and begin acquisition
        CameraPtr cam = globalCamList.GetBySerial(serialNum);
        if (cam.IsValid())
        {
            if (ConfigureCamera(cam))
            {
                try
                {
                    cam->BeginAcquisition();
                }
                catch (Exception& e)
                {
                    cout << "Error beginning acquisition: " << e.what() << endl;
                }
            }
            else
            {
                cout << "Error OnDeviceArrival: configuration for device " << serialNum << " failed.";
            }
        }
    }

    // This function defines removal events on an interface.
    void OnDeviceRemoval(CameraPtr pCamera)
    {
        // Log the device removal
        const string deviceSerial = std::string(pCamera->TLDevice.DeviceSerialNumber.ToString());
        cout << endl << deviceSerial << " Device Removal Detected" << endl << endl;

        // Refresh the global camera list
        RefreshCameraList(m_system);

        // Increment the number of removals for disconnected cameras
        auto cameraGrabInfo = cameraGrabInfoMap.find(deviceSerial);
        if (cameraGrabInfo != cameraGrabInfoMap.end())
        {
            cameraGrabInfo->second.numRemovals++;
        }
        else
        {
            cout << "Error OnDeviceRemoval: camera " << deviceSerial << " not found in grab info map" << endl;
        }
    }

  private:
    SystemPtr m_system;
};

// This function configures newly discovered cameras with desired settings, saves
// them to User Set 1 and sets this as the default. This way cameras will not need
// to be reconfigured if they are disconnected during the example. Note that this may
// overwrite current settings for User Set 1; please see ImageFormatControl example
// for more in-depth comments on camera configuration.
bool ConfigureUserSet1(CameraPtr pCam)
{
    bool result = true;

    try
    {
        // Get the camera node map
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Get User Set 1 from the User Set Selector
        CEnumerationPtr ptrUserSetSelector = nodeMap.GetNode("UserSetSelector");
        if (!IsReadable(ptrUserSetSelector) ||
            !IsWritable(ptrUserSetSelector))
        {
            cout << "Unable to set User Set Selector to User Set 1 (node retrieval). Aborting..." << endl << endl;
            return false;
        }

        CEnumEntryPtr ptrUserSet1 = ptrUserSetSelector->GetEntryByName("UserSet1");
        if (!IsReadable(ptrUserSet1))
        {
            cout << "Unable to get User Set Selector to User Set 1 (enum entry retrieval). Aborting..." << endl << endl;
            return false;
        }

        const int64_t userSet1 = ptrUserSet1->GetValue();

        // Set User Set Selector to User Set 1
        ptrUserSetSelector->SetIntValue(userSet1);

        // Set User Set Default to User Set 1
        // This ensures the camera will re-enumerate using User Set 1, instead of the default user set.
        CEnumerationPtr ptrUserSetDefault = nodeMap.GetNode("UserSetDefault");
        if (!IsWritable(ptrUserSetDefault))
        {
            cout << "Unable to set User Set Default to User Set 1 (node retrieval). Aborting..." << endl << endl;
            return false;
        }

        ptrUserSetDefault->SetIntValue(userSet1);

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
            cout << "Unable to set acquisition mode to continuous (enum entry retrieval). Aborting..." << endl << endl;
            return false;
        }

        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        if (!IsWritable(ptrAcquisitionMode))
        {
            cout << "Unable to set acquisition mode to continuous (node retrieval). Aborting..." << endl << endl;
            return false;
        }

        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        //
        // *** NOTES ***
        // Any additional settings should be changed before executing the user set save.
        //

        // Execute User Set Save to save User Set 1
        CCommandPtr ptrUserSetSave = nodeMap.GetNode("UserSetSave");
        if (!ptrUserSetSave.IsValid())
        {
            cout << "Unable to save Settings to User Set 1. Aborting..." << endl << endl;
            return false;
        }

        ptrUserSetSave->Execute();
    }
    catch (Exception& e)
    {
        cout << "Error configuring user set 1: " << e.what() << endl;
        result = false;
    }

    return result;
}

// This function resets the cameras default User Set to the Default User Set.
// Note that User Set 1 will retain the settings set during ConfigureUserSet1.
void ResetCameraUserSetToDefault(CameraPtr pCam)
{
    try
    {
        // Get the camera node map
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Get User Set Default from User Set Selector
        CEnumerationPtr ptrUserSetSelector = nodeMap.GetNode("UserSetSelector");
        if (!IsReadable(ptrUserSetSelector) ||
            !IsWritable(ptrUserSetSelector))
        {
            cout << "Unable to set User Set Selector to Default (node retrieval). Aborting..." << endl << endl;
            return;
        }

        CEnumEntryPtr ptrUserSetDefaultEntry = ptrUserSetSelector->GetEntryByName("Default");
        if (!IsReadable(ptrUserSetDefaultEntry))
        {
            cout << "Unable to get User Set Selector to Default (enum entry retrieval). Aborting..." << endl << endl;
            return;
        }

        const int64_t userSetDefault = ptrUserSetDefaultEntry->GetValue();

        // Set User Set Selector back to User Set Default
        ptrUserSetSelector->SetIntValue(userSetDefault);

        // Set User Set Default to User Set Default
        CEnumerationPtr ptrUserSetDefault = nodeMap.GetNode("UserSetDefault");
        if (!IsWritable(ptrUserSetDefault))
        {
            cout << "Unable to set User Set Default to User Set 1 (node retrieval). Aborting..." << endl << endl;
            return;
        }

        ptrUserSetDefault->SetIntValue(userSetDefault);

        // Execute User Set Load to load User Set Default
        CCommandPtr ptrUserSetLoad = nodeMap.GetNode("UserSetLoad");
        if (!ptrUserSetLoad.IsValid())
        {
            cout << "Unable to load Settings from User Set Default. Aborting..." << endl << endl;
            return;
        }

        ptrUserSetLoad->Execute();
    }
    catch (Exception& e)
    {
        cout << "Error resetting camera to use default user set: " << e.what() << endl;
    }
}

// This helper function retrieves the device serial number from the cameras nodemap.
string GetDeviceSerial(CameraPtr pCam)
{
    INodeMap& nodeMap = pCam->GetTLDeviceNodeMap();
    CStringPtr ptrDeviceSerialNumber = nodeMap.GetNode("DeviceSerialNumber");
    if (IsReadable(ptrDeviceSerialNumber))
    {
        return string(ptrDeviceSerialNumber->GetValue());
    }
    return "";
}

// This function configures new cameras to use camera settings from User Set 1 and
// registers an image event handler on new and rediscovered cameras.
bool ConfigureCamera(CameraPtr pCam)
{
    bool result = true;

    try
    {
        // Initialize camera
        pCam->Init();

        // Get the device serial number
        const string deviceSerialNumber = GetDeviceSerial(pCam);

        // Check if the camera has been connected before
        auto camera = cameraGrabInfoMap.find(deviceSerialNumber);
        const bool cameraFound = camera != cameraGrabInfoMap.end();

        cout << endl
             << endl
             << "*** " << (cameraFound ? "RESUM" : "START") << "ING RECOVERY EXAMPLE FOR " << deviceSerialNumber
             << " ***" << endl
             << endl;

        // Configure newly discovered cameras to use User Set 1
        if (!cameraFound)
        {
            // Add a new entry into the camera grab info map using the default GrabInfo constructor
            cameraGrabInfoMap.insert(std::make_pair(deviceSerialNumber, GrabInfo(deviceSerialNumber)));

            cout << "Configuring device " << deviceSerialNumber << endl;

            // Configure the new camera to use User Set 1
            // Return if it is unsuccessful
            if (!ConfigureUserSet1(pCam))
            {
                return false;
            }
        }

        //
        // Register the imageEvent to the camera
        //
        // *** NOTES ***
        // An image event handler is created in the GrabInfo default constructor for every
        // new camera. The event handler can be reused for cameras that have been reconnected.
        //
        pCam->RegisterEventHandler(*(cameraGrabInfoMap.find(deviceSerialNumber)->second.imageEventHandler));
    }
    catch (Exception& e)
    {
        cout << "Error Running single camera: " << e.what() << endl;
        result = false;
    }

    return result;
}

// This helper function prints the complete grab info results for each camera.
void PrintExampleStatistics()
{
    cout << endl << "Printing Final Statistics " << endl << endl;
    const unsigned int printWidth = 20;

    cout << setw(printWidth) << left << "Serial Number:" << setw(printWidth) << left
         << "Images Grabbed:" << setw(printWidth) << left << "Incomplete Images:" << setw(printWidth) << left
         << "Camera Removals:" << endl;

    for (auto camInfo = cameraGrabInfoMap.begin(); camInfo != cameraGrabInfoMap.end(); camInfo++)
    {
        cout << setw(printWidth) << left << camInfo->first << setw(printWidth) << left
             << camInfo->second.numImagesGrabbed << setw(printWidth) << left << camInfo->second.numIncompleteImages
             << setw(printWidth) << left << camInfo->second.numRemovals << endl;
    }
}

// Example entry point; please see EnumerationEvents example for more in-depth
// comments on preparing and cleaning up the system.
int main(int /*argc*/, char** /*argv*/)
{
    // Since this application saves images in the current folder
    // we must ensure that we have permission to write to this folder.
    // If we do not have permission, fail right away.
    FILE* tempFile = fopen("test.txt", "w+");
    if (tempFile == nullptr)
    {
        cout << "Failed to create file in current folder. Please check permissions." << endl;
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
         << endl
         << endl;

    // Retrieve list of cameras from the system
    globalCamList = system->GetCameras();
    cout << "Number of cameras detected: " << globalCamList.GetSize() << endl;

    // Configure each connected camera to use User Set 1 and register image events
    for (unsigned int camIndex = 0; camIndex < globalCamList.GetSize(); camIndex++)
    {
        CameraPtr cam = globalCamList[camIndex];
        if (!ConfigureCamera(cam))
        {
            cout << "Camera configuration for device " << GetDeviceSerial(cam) << " unsuccessful, aborting...";
            return -1;
        }
    }

    // Create an interface event handler to handle all interface events on the system.
    // Please see EnumerationEvents example for more in-depth comments on setting up system and interface event
    // handlers.
    InterfaceEventHandlerImpl interfaceEventHandler(system);
    system->RegisterEventHandler(interfaceEventHandler);

    cout << endl
         << "Cameras may be unplugged/plugged in after the example has started." << endl
         << "After the example begins, please press any key to end the example..." << endl;

    cout << endl << "Press any key to begin..." << endl << endl;
    getchar();

    // Begin Acquisition on all cameras
    for (unsigned int camIndex = 0; camIndex < globalCamList.GetSize(); camIndex++)
    {
        try
        {
            globalCamList[camIndex]->BeginAcquisition();
        }
        catch (Exception& e)
        {
            cout << "Error starting acquistion on camera at index " << camIndex << ": " << e.what() << endl;
        }
    }

    // Wait for user to stop the example by pressing any key
    getchar();

    // End Acquisition on all cameras
    for (unsigned int camIndex = 0; camIndex < globalCamList.GetSize(); camIndex++)
    {
        try
        {
            globalCamList[camIndex]->EndAcquisition();
        }
        catch (Exception& e)
        {
            cout << "Error ending acquistion on camera at index " << camIndex << ": " << e.what() << endl;
        }
    }

    // Unregister the interface events from the system
    system->UnregisterEventHandler(interfaceEventHandler);

    // Reset camera user sets and deinitialize all cameras
    for (unsigned int camIndex = 0; camIndex < globalCamList.GetSize(); ++camIndex)
    {
        CameraPtr cam = globalCamList[camIndex];
        if (cam.IsValid())
        {
            cout << "Resetting configuration for device " << GetDeviceSerial(cam) << endl;
            ResetCameraUserSetToDefault(cam);
            cam->DeInit();
        }
    }

    cout << endl;

    PrintExampleStatistics();

    cout << endl;

    // Clear camera list before releasing system
    globalCamList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press any key to exit..." << endl;
    getchar();

    return 0;
}