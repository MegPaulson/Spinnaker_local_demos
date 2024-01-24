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
 *  @example NodeMapInfo_QuickSpin.cpp
 *
 *  @brief NodeMapInfo_QuickSpin.cpp shows how to interact with nodes
 *  using the QuickSpin API. QuickSpin is a subset of the Spinnaker library
 *  that allows for simpler node access and control.
 *
 *  This example demonstrates the retrieval of information from both the
 *  transport layer and the camera. Because the focus of this example is node
 *  access, which is where QuickSpin and regular Spinnaker differ, this
 *  example differs from NodeMapInfo quite a bit.
 *
 *  A much wider range of topics is covered in the full Spinnaker examples than
 *  in the QuickSpin ones. There are only enough QuickSpin examples to
 *  demonstrate node access and to get started with the API; please see full
 *  Spinnaker examples for further or specific knowledge on a topic.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// This function prints node information if applicable.
void PrintNodeInfo(CValuePtr pNode)
{
    //
    // Notice that each node is checked for availability and readability
    // prior to value retrieval. Checking for availability and readability
    // (or writability when applicable) whenever a node is accessed is
    // important in terms of error handling. If a node retrieval error
    // occurs but remains unhandled, an exception is thrown.
    //
    if (pNode != NULL && IsReadable(pNode))
    {
        cout << pNode->ToString() << endl;
        return;
    }

    cout << "unavailable" << endl;
}

// This function prints device information from the transport layer.
int PrintTransportLayerDeviceInfo(CameraPtr pCamera)
{
    int result = 0;

    try
    {
        //
        // Print device information from the transport layer
        //
        // *** NOTES ***
        // In QuickSpin, accessing device information on the transport layer is
        // accomplished via a camera's TLDevice property. The TLDevice property
        // houses nodes related to general device information such as the three
        // demonstrated below, device access status, XML and GUI paths and
        // locations, and GEV information to name a few. The TLDevice property
        // allows access to nodes that would generally be retrieved through the
        // TL device nodemap in full Spinnaker.
        //
        // Notice that each node is checked for availability and readability
        // prior to value retrieval. Checking for availability and readability
        // (or writability when applicable) whenever a node is accessed is
        // important in terms of error handling. If a node retrieval error
        // occurs but remains unhandled, an exception is thrown.
        //

        // Print device serial number
        cout << "Device serial number: ";
        PrintNodeInfo(&pCamera->TLDevice.DeviceSerialNumber);

        // Print device vendor name
        cout << "Device vendor name: ";
        PrintNodeInfo(&pCamera->TLDevice.DeviceVendorName);

        // Print device display name
        cout << "Device display name: ";
        PrintNodeInfo(&pCamera->TLDevice.DeviceDisplayName);

        cout << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function prints stream information from the transport layer.
int PrintTransportLayerStreamInfo(CameraPtr pCamera)
{
    int result = 0;

    try
    {
        //
        // Print stream information from the transport layer
        //
        // *** NOTES ***
        // In QuickSpin, accessing stream information on the transport layer is
        // accomplished via a camera's TLStream property. The TLStream property
        // houses nodes related to streaming such as the two demonstrated below,
        // buffer information, and GEV packet information to name a few. The
        // TLStream property allows access to nodes that would generally be
        // retrieved through the TL stream nodemap in full Spinnaker.
        //

        // Print stream ID
        cout << "Stream ID: ";
        PrintNodeInfo(&pCamera->TLStream.StreamID);

        // Print stream type
        cout << "Stream type: ";
        PrintNodeInfo(&pCamera->TLStream.StreamType);

        cout << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function prints information about the interface.
int PrintTransportLayerInterfaceInfo(const InterfacePtr& pInterface)
{
    int result = 0;

    try
    {
        //
        // Print interface information from the transport layer
        //
        // *** NOTES ***
        // In QuickSpin, accessing interface information is accomplished via an
        // interface's TLInterface property. The TLInterface property houses
        // nodes that hold information about the interface such as the three
        // demonstrated below, other general interface information, and
        // GEV addressing information. The TLInterface property allows access to
        // nodes that would generally be retrieved through the interface nodemap
        // in full Spinnaker.
        //
        // Interface nodes should also always be checked for availability and
        // readability (or writability when applicable). If a node retrieval
        // error occurs but remains unhandled, an exception is thrown.
        //

        // Print interface display name
        cout << "Interface display name: ";
		PrintNodeInfo(&pInterface->TLInterface.InterfaceDisplayName);

        // Print interface ID
        cout << "Interface ID: ";
		PrintNodeInfo(&pInterface->TLInterface.InterfaceID);

        // Print interface type
        cout << "Interface type: ";
		PrintNodeInfo(&pInterface->TLInterface.InterfaceType);

		//
		// Print information specific to the interface's host adapter
		// from the transport layer.
		//
		// *** NOTES ***
		// This information can help in determining which interface
		// to use for better performance as some host adapters may have more
		// significant physical limitations.
		//
		// Interface nodes should also always be checked for availability and
		// readability (or writability when applicable). If a node retrieval
		// error occurs but remains unhandled, an exception is thrown.
		//

		// Print host adapter name
		cout << "Host adapter name: ";
		PrintNodeInfo(&pInterface->TLInterface.HostAdapterName);

		// Print host adapter vendor name
		cout << "Host adapter vendor: ";
		PrintNodeInfo(&pInterface->TLInterface.HostAdapterVendor);

		// Print host adapter driver version
		cout << "Host adapter driver version: ";
		PrintNodeInfo(&pInterface->TLInterface.HostAdapterDriverVersion);

		cout << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function prints device information from the GenICam nodemap.
int PrintGenICamDeviceInfo(CameraPtr pCamera)
{
    int result = 0;

    try
    {
        //
        // Print device information from the camera
        //
        // *** NOTES ***
        // Most camera interaction happens through GenICam nodes. The
        // advantages of these nodes is that there is a lot more of them, they
        // allow for a much deeper level of interaction with a camera, and no
        // intermediate property (i.e. TLDevice or TLStream) is required. The
        // disadvantage is that they require initialization.
        //

        // Print exposure time
        cout << "Exposure time: ";
        PrintNodeInfo(&pCamera->ExposureTime);

        // Print black level
        cout << "Black level: ";
        PrintNodeInfo(&pCamera->BlackLevel);

        // Print height
        cout << "Height: ";
        PrintNodeInfo(&pCamera->Height);

        cout << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// Example entry point; this function prints transport layer information from
// each interface and transport and GenICam information from each
// camera.
int main(int /*argc*/, char** /*argv*/)
{
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

    // Retrieve list of interfaces from the system
    InterfaceList interfaceList = system->GetInterfaces();

    unsigned int numInterfaces = interfaceList.GetSize();

    cout << "Number of interfaces detected: " << numInterfaces << endl << endl;

    //
    // Print information on each interface
    //
    // *** NOTES ***
    // All USB 3 Vision and GigE Vision interfaces should enumerate for
    // Spinnaker.
    //
    cout << endl << "*** PRINTING INTERFACE INFORMATION ***" << endl << endl;

    for (unsigned int i = 0; i < numInterfaces; i++)
    {
        result = result | PrintTransportLayerInterfaceInfo(interfaceList.GetByIndex(i));
    }

    //
    // Print general device information on each camera from transport layer
    //
    // *** NOTES ***
    // Transport layer nodes do not require initialization in order to interact
    // with them.
    //
    cout << endl << "*** PRINTING TRANSPORT LAYER DEVICE INFORMATION ***" << endl << endl;

    for (unsigned int i = 0; i < numCameras; i++)
    {
        result = result | PrintTransportLayerDeviceInfo(camList.GetByIndex(i));
    }

    //
    // Print streaming information on each camera from transport layer
    //
    // *** NOTES ***
    // Again, initialization is not required to print information from the
    // transport layer; this is equally true of streaming information.
    //
    cout << endl << "*** PRINTING TRANSPORT LAYER STREAMING INFORMATION ***" << endl << endl;

    for (unsigned int i = 0; i < numCameras; i++)
    {
        result = result | PrintTransportLayerStreamInfo(camList.GetByIndex(i));
    }

    //
    // Print device information on each camera from GenICam nodemap
    //
    // *** NOTES ***
    // GenICam nodes require initialization in order to interact with
    // them; as such, this loop initializes the camera, prints some information
    // from the GenICam nodemap, and then deinitializes it. If the camera were
    // not initialized, node availability would fail.
    //
    cout << endl << "*** PRINTING GENICAM INFORMATION ***" << endl << endl;

    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Initialize camera
        camList.GetByIndex(i)->Init();

        // Print information
        result = result | PrintGenICamDeviceInfo(camList.GetByIndex(i));

        // Deinitialize camera
        camList.GetByIndex(i)->DeInit();
    }

    // Clear camera list before releasing system
    camList.Clear();

    // Clear interface list before releasing system
    interfaceList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return result;
}
