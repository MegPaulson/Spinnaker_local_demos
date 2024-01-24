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
 *	@example ImageEvents.cpp
 *
 *	@brief ImageEvents.cpp shows how to acquire images using the image event
 *	handler. It relies on information provided in the Enumeration, Acquisition,
 *	and NodeMapInfo examples.
 *
 *	It can also be helpful to familiarize yourself with the NodeMapCallback
 *	example, as nodemap callbacks follow the same general procedure as
 *	events, but with a few less steps.
 *
 *	This example creates a user-defined class, ImageEventHandlerImpl, that inherits
 *	from the Spinnaker class, ImageEventHandler. ImageEventHandlerImpl allows the user to
 *	define any properties, parameters, and the event itself while ImageEventHandler
 *	allows the child class to appropriately interface with Spinnaker.
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

// This class defines the properties, parameters, and the event handler itself. Take a
// moment to notice what parts of the class are mandatory, and what have been
// added for demonstration purposes. First, any class used to define image event handlers
// must inherit from ImageEventHandler. Second, the method signature of OnImageEvent()
// must also be consistent. Everything else - including the constructor,
// deconstructor, properties, body of OnImageEvent(), and other functions -
// is particular to the example.
class ImageEventHandlerImpl : public ImageEventHandler
{
  public:
    // The constructor retrieves the serial number and initializes the image
    // counter to 0.
    ImageEventHandlerImpl(CameraPtr pCam)
    {
        // Retrieve device serial number
        INodeMap& nodeMap = pCam->GetTLDeviceNodeMap();

        m_deviceSerialNumber = "";
        CStringPtr ptrDeviceSerialNumber = nodeMap.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrDeviceSerialNumber))
        {
            m_deviceSerialNumber = ptrDeviceSerialNumber->GetValue();
        }

        // Initialize image counter to 0
        m_imageCnt = 0;

        // Release reference to camera
        pCam = nullptr;

        //
        // Set default image processor color processing method
        //
        // *** NOTES ***
        // By default, if no specific color processing algorithm is set, the image
        // processor will default to NEAREST_NEIGHBOR method.
        //
        m_processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);
    }

    ~ImageEventHandlerImpl()
    {
    }

    // This method defines an image event. In it, the image that triggered the
    // event is converted and saved before incrementing the count. Please see
    // Acquisition_CSharp example for more in-depth comments on the acquisition
    // of images.
    void OnImageEvent(ImagePtr image)
    {
        // Save a maximum of 10 images
        if (m_imageCnt < mk_numImages)
        {
            cout << "Image event occurred..." << endl;

            // Check image retrieval status
            if (image->IsIncomplete())
            {
                cout << "Image incomplete with image status " << image->GetImageStatus() << "..." << endl << endl;
            }
            else
            {
                // Print image information
                cout << "Grabbed image " << m_imageCnt << ", width = " << image->GetWidth()
                     << ", height = " << image->GetHeight() << endl;

                // Convert image to mono 8
                ImagePtr convertedImage = m_processor.Convert(image, PixelFormat_Mono8);

                // Create a unique filename and save image
                ostringstream filename;

                filename << "ImageEvents-";
                if (m_deviceSerialNumber != "")
                {
                    filename << m_deviceSerialNumber.c_str() << "-";
                }
                filename << m_imageCnt << ".jpg";

                convertedImage->Save(filename.str().c_str());

                cout << "Image saved at " << filename.str() << endl << endl;

                // Increment image counter
                m_imageCnt++;
            }
        }
    }

    // Getter for image counter
    int getImageCount()
    {
        return m_imageCnt;
    }

    // Getter for maximum images
    int getMaxImages()
    {
        return mk_numImages;
    }

  private:
    static const unsigned int mk_numImages = 10;
    unsigned int m_imageCnt;
    string m_deviceSerialNumber;
    ImageProcessor m_processor;
};

// This function configures the example to execute image events by preparing and
// registering an image event.
int ConfigureImageEvents(CameraPtr pCam, ImageEventHandlerImpl*& imageEventHandler)
{
    int result = 0;

    try
    {
        //
        // Create image event handler
        //
        // *** NOTES ***
        // The class has been constructed to accept a camera pointer in order
        // to allow the saving of images with the device serial number.
        //
        imageEventHandler = new ImageEventHandlerImpl(pCam);

        //
        // Register image event handler
        //
        // *** NOTES ***
        // Image event handlers are registered to cameras. If there are multiple
        // cameras, each camera must have the image event handlers registered to it
        // separately. Also, multiple image event handlers may be registered to a
        // single camera.
        //
        // *** LATER ***
        // Image event handlers must be unregistered manually. This must be done prior
        // to releasing the system and while the image event handlers are still in
        // scope.
        //
        pCam->RegisterEventHandler(*imageEventHandler);
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function waits for the appropriate amount of images.  Notice that
// whereas most examples actively retrieve images, the acquisition of images is
// handled passively in this example.
int WaitForImages(ImageEventHandlerImpl*& imageEventHandler)
{
    int result = 0;

    try
    {
        //
        // Wait for images
        //
        // *** NOTES ***
        // In order to passively capture images using image events and
        // automatic polling, the main thread sleeps in increments of 200 ms
        // until 10 images have been acquired and saved.
        //
        const int sleepDuration = 200; // in milliseconds

        while (imageEventHandler->getImageCount() < imageEventHandler->getMaxImages())
        {
            cout << "\t//" << endl;
            cout << "\t// Sleeping for " << sleepDuration << " ms. Grabbing images..." << endl;
            cout << "\t//" << endl;

            SleepyWrapper(sleepDuration);
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This functions resets the example by unregistering the image event handler.
int ResetImageEvents(CameraPtr pCam, ImageEventHandlerImpl*& imageEventHandler)
{
    int result = 0;

    try
    {
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

// This function passively waits for images by calling WaitForImages(). Notice that
// this function is much shorter than the AcquireImages() function of other examples.
// This is because most of the code has been moved to the image event's OnImageEvent()
// method.
int AcquireImages(
    CameraPtr pCam,
    INodeMap& nodeMap,
    INodeMap& nodeMapTLDevice,
    ImageEventHandlerImpl*& imageEventHandler)
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
            cout << "Unable to get acquisition mode to continuous (enum entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        cout << "Acquisition mode set to continuous..." << endl;

        // Begin acquiring images
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        // Retrieve images using image event handler
        WaitForImages(imageEventHandler);

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

        // Configure image events
        ImageEventHandlerImpl* imageEventHandler;

        err = ConfigureImageEvents(pCam, imageEventHandler);
        if (err < 0)
        {
            return err;
        }

        // Acquire images using the image event handler
        result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice, imageEventHandler);

        // Reset image events
        result = result | ResetImageEvents(pCam, imageEventHandler);

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
// comments on the steps in this function.
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