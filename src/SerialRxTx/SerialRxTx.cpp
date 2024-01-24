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
 *	@example SerialRxTx.cpp
 *
 *	@brief SerialRxTx.cpp shows how to communicate using Serial ports.
 *	It sets serial port settings in Spinnaker, opens and operates on FileAccess nodes and creates COM Port handle.
 *	After the setup, it transmits and receives simple data.
 *	It verifies the communication by transmitting data and reading data to/from the COM Port.
 *  Basic understanding of serial communication and knowledge of the installation process for a serial port is required.
 *
 *	THIS EXAMPLE ONLY WORKS IN WINDOWS OS
 *
 *  Machine setup steps to run the example:
 *  1. Your PC needs a serial port. If one does not already exist, purchase a USB to Serial port cable or adapter.
 *     After installing this, your PC will see a serial port in device manager.
 *     Note the COM port assigned to this serial port.
 *  2. From your PC's serial port, you need a RS232 to TTL converter
 *  3. To wire up from the RS232-TTL converter to the camera, use a FLIR supplied GPIO cable or make your own from the
 *     part numbers supplied in our technical reference manual. Wire up the TX/RX/GND pins
 *     between the adapter and the camera.
 *  4. Run the example. It should find the COM port, set up the serial settings, read anything
 *     sent to the camera, and then transmit ABCD over the serial port
 *
 *  The final setup of the machine to run the example should be:
 *  camera -> GPIO -> TTL-RS232 -> RS232 port
 *
 *  More information on how to configure and test serial port can be found here:
 *  https://www.flir.com/support-center/iis/machine-vision/application-note/configuring-and-testing-the-rs-232-serial-port/
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

#define COM_PORT_COUNT_MAX                            256
#define TWO_SECOND_DELAY                              2000
#define SERIAL_PORT_COMMUNICATION_TIMEOUT_MILLISECOND 1000
#define SERIAL_PORT_BAUD_RATE                         19200
#define SERIAL_PORT_STOP_BITS                         0
#define SERIAL_PORT_PARITY_BITS                       0
#define SERIAL_PORT_DELAY                             1500
#define DATA_BITS                                     8
#define MILLISECOND                                   1000

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

int ConfigureDevice(CameraPtr pCam, HANDLE& hFileHandle)
{
    int result = 0;

    cout << endl << endl << "*** SET SERIAL PORT, OPEN COM PORT, FILE ACCESS SETTINGS ***" << endl << endl;

    try
    {
        // It retrieves GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        cout << endl << "Setup Serial Port Settings " << endl;

        // This is for Serial port Receive settings
        CEnumerationPtr ptrSerialPortSelector = nodeMap.GetNode("SerialPortSelector");
        if (!IsWritable(ptrSerialPortSelector))
        {
            cout << "Unable to set Serial Port Selector. Aborting..." << endl << endl;
            return -1;
        }
        ptrSerialPortSelector->SetIntValue(0); // Serial port 0 is chosen

        CEnumerationPtr ptrSerialPortSource = nodeMap.GetNode("SerialPortSource");
        if (!IsWritable(ptrSerialPortSource))
        {
            cout << "Unable to set Serial Port Source. Aborting..." << endl << endl;
            return -1;
        }
        ptrSerialPortSource->SetIntValue(0); // line 0 chosen

        CEnumerationPtr ptrSerialPortBaudRate = nodeMap.GetNode("SerialPortBaudRate");
        if (!IsWritable(ptrSerialPortBaudRate))
        {
            cout << "Unable to set Serial Port BaudRate. Aborting..." << endl << endl;
            return -1;
        }
        ptrSerialPortBaudRate->SetIntValue(SERIAL_PORT_BAUD_RATE);

        CIntegerPtr ptrSerialPortDataBits = nodeMap.GetNode("SerialPortDataBits");
        if (!IsWritable(ptrSerialPortDataBits))
        {
            cout << "Unable to set Serial Port Data Bits. Aborting..." << endl << endl;
            return -1;
        }
        ptrSerialPortDataBits->SetValue(DATA_BITS);

        CEnumerationPtr ptrSerialPortStopBits = nodeMap.GetNode("SerialPortStopBits");
        if (!IsWritable(ptrSerialPortStopBits))
        {
            cout << "Unable to set Serial Port Stop Bits. Aborting..." << endl << endl;
            return -1;
        }
        ptrSerialPortStopBits->SetIntValue(SERIAL_PORT_STOP_BITS);

        CEnumerationPtr ptrSerialPortParity = nodeMap.GetNode("SerialPortParity");
        if (!IsWritable(ptrSerialPortParity))
        {
            cout << "Unable to set Serial Port Parity. Aborting..." << endl << endl;
            return -1;
        }
        ptrSerialPortParity->SetIntValue(SERIAL_PORT_PARITY_BITS);

        // This is for Serial port Transmit settings
        CEnumerationPtr ptrLineSelector = nodeMap.GetNode("LineSelector");
        if (!IsWritable(ptrLineSelector))
        {
            cout << "Unable to set Line Selector. Aborting..." << endl << endl;
            return -1;
        }
        ptrLineSelector->SetIntValue(2); // line 2 is selected

        CEnumerationPtr ptrLineMode = nodeMap.GetNode("LineMode");
        if (!IsWritable(ptrLineMode))
        {
            cout << "Unable to set Line Mode. Aborting..." << endl << endl;
            return -1;
        }
        ptrLineMode->SetIntValue(1); // output is selected

        CEnumerationPtr ptrLineSource = nodeMap.GetNode("LineSource");
        if (!IsWritable(ptrLineSource))
        {
            cout << "Unable to set Line Source. Aborting..." << endl << endl;
            return -1;
        }
        ptrLineSource->SetIntValue(30); // Serial port 0 is selected

        cout << endl << "Setup File Access Settings " << endl;

        CEnumerationPtr ptrFileSelector = nodeMap.GetNode("FileSelector");
        if (!IsWritable(ptrFileSelector))
        {
            cout << "Unable to set File Selector. Aborting..." << endl << endl;
            return -1;
        }
        ptrFileSelector->SetIntValue(9); // Serial Port is chosen

        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to set File Operation Selector. Aborting..." << endl << endl;
            return -1;
        }
        ptrFileOperationSelector->SetIntValue(0); // Open operation is chosen

        CEnumerationPtr ptrFileOpenMode = nodeMap.GetNode("FileOpenMode");
        if (!IsWritable(ptrFileOpenMode))
        {
            cout << "Unable to set File Open Mode. Aborting..." << endl << endl;
            return -1;
        }
        ptrFileOpenMode->SetIntValue(2); // readwrite mode is chosen

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to execute File Operation. Aborting..." << endl << endl;
            return -1;
        }

        cout << endl << "Execute file access open" << endl;

        ptrFileOperationExecute->Execute();

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to get File Operation Status. Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrFileStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (ptrFileOperationStatus->GetIntValue() != ptrFileStatusSuccess->GetValue())
        {
            cout << "Failed to open the file in the File Access Control." << endl;
            return 1;
        }

        Sleep(TWO_SECOND_DELAY); // It sleeps two seconds to avoid error from opening and closing com ports

        cout << endl << "Open COM Port Handle" << endl;
        string comPort;

        // It loops through COM ports to find which one the device is connected to
        for (unsigned int comPortIndex = 0; comPortIndex < COM_PORT_COUNT_MAX; comPortIndex++)
        {
            // This gets the COM port
            comPort = "\\\\.\\COM" + to_string(static_cast<long long>(comPortIndex));

            hFileHandle = CreateFileA(
                comPort.c_str(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                nullptr,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                nullptr);
            if (hFileHandle == INVALID_HANDLE_VALUE)
            {
                comPort = "";
                continue;
            }

            // This sets the read and write timeouts for the serial
            COMMTIMEOUTS comTimeout;
            if (!GetCommTimeouts(hFileHandle, &comTimeout))
            {
                cout << "Failed to get the timeout settings for COM" + to_string(static_cast<long long>(comPortIndex)) +
                            ". Windows Error Code: " + to_string(static_cast<long long>(GetLastError()))
                     << endl;
                return -1;
            }
            comTimeout.ReadTotalTimeoutConstant = SERIAL_PORT_COMMUNICATION_TIMEOUT_MILLISECOND;
            comTimeout.WriteTotalTimeoutConstant = SERIAL_PORT_COMMUNICATION_TIMEOUT_MILLISECOND;
            if (!SetCommTimeouts(hFileHandle, &comTimeout))
            {
                cout << "Failed to get the timeout settings for COM" + to_string(static_cast<long long>(comPortIndex)) +
                            ". Windows Error Code: " + to_string(static_cast<long long>(GetLastError()))
                     << endl;
                return -1;
            }
            // This clears the transmit buffer of the COM port
            PurgeComm(hFileHandle, PURGE_TXCLEAR);

            cout << endl
                 << "COM" + to_string(static_cast<long long>(comPortIndex)) << " port is connected to the Device"
                 << endl;
            // This sets up the COM port

            DCB comSettings = {0};
            comSettings.DCBlength = sizeof(comSettings);

            if (!GetCommState(hFileHandle, &comSettings))
            {
                cout << "Failed to get the communication settings for COM " +
                            to_string(static_cast<long long>(comPortIndex)) +
                            ". Windows Error Code: " + to_string(static_cast<long long>(GetLastError()))
                     << endl;
                return -1;
            }
            comSettings.ByteSize = DATA_BITS;
            comSettings.Parity = SERIAL_PORT_PARITY_BITS;
            comSettings.BaudRate = SERIAL_PORT_BAUD_RATE;
            comSettings.StopBits = SERIAL_PORT_STOP_BITS;
            if (!SetCommState(hFileHandle, &comSettings))
            {
                cout << "Failed to set the communication settings for COM " +
                            to_string(static_cast<long long>(comPortIndex)) + ". Windows Error Code: "
                     << endl;
                return -1;
            }
            break;
        }
        if (comPort == "")
        {
            cout << "The device was not found to be connected to a COM port between COM0 and COM" +
                        to_string(static_cast<long long>(COM_PORT_COUNT_MAX)) + ".";
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function receives data from the PC over serial port
int SerialRx(CameraPtr pCam, INodeMap& nodeMap, HANDLE& hFileHandle)
{
    int result = 0;
    try
    {
        // It writes test data to the serial port
        DWORD bytesWritten = 0;
        char* data = "ABCD";
        if (!WriteFile(hFileHandle, data, 4, &bytesWritten, nullptr))
        {
            cout << "Failed to write the test data to COM port Windows Error Code: " +
                        to_string(static_cast<long long>(GetLastError()))
                 << endl;
            return -1;
        }

        // This is to ensure that the data is transferred
        Sleep(
            (unsigned int)((double)MILLISECOND * bytesWritten / (SERIAL_PORT_BAUD_RATE / DATA_BITS)) +
            SERIAL_PORT_DELAY);

        CIntegerPtr ptrFileOperationResult = nodeMap.GetNode("FileOperationResult");
        if (!IsReadable(ptrFileOperationResult))
        {
            cout << "Unable to get File Operation Result. Aborting..." << endl << endl;
            return -1;
        }

        CIntegerPtr ptrFileSize = nodeMap.GetNode("FileSize");
        if (!IsReadable(ptrFileSize))
        {
            cout << "Unable to get File Size. Aborting..." << endl << endl;
            return -1;
        }

        CIntegerPtr ptrFileAccessLength = nodeMap.GetNode("FileAccessLength");
        if (!IsWritable(ptrFileAccessLength))
        {
            cout << "Unable to set File Access Length. Aborting..." << endl << endl;
            return -1;
        }

        CRegisterPtr ptrFileAccessBuffer = nodeMap.GetNode("FileAccessBuffer");
        if (!IsWritable(ptrFileAccessBuffer))
        {
            cout << "Unable to set File Access Buffer. Aborting..." << endl << endl;
            return -1;
        }

        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to set File Operation Selector. Aborting..." << endl << endl;
            return -1;
        }

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to execute File Operation. Aborting..." << endl << endl;
            return -1;
        }

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to get File Operation Status. Aborting..." << endl << endl;
            return -1;
        }

        cout << endl << "Set File Access to read operation " << endl;

        ptrFileOperationSelector->SetIntValue(2); // read operation is selected

        unsigned int totalBytesRead = 0;
        totalBytesRead = (unsigned int)ptrFileOperationResult->GetValue();
        char* serialDataRx = "";
        serialDataRx = new char[(unsigned int)ptrFileAccessBuffer->GetLength()];

        string dataRead = "";
        while (ptrFileSize->GetValue() > 0)
        {
            ptrFileOperationExecute->Execute();

            CEnumEntryPtr ptrFileStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
            if (ptrFileOperationStatus->GetIntValue() != ptrFileStatusSuccess->GetValue())
            {
                cout << "Failed to read the file in the File Access Control." << endl;
            }

            memset(serialDataRx, 0, (unsigned int)ptrFileAccessBuffer->GetLength());

            ptrFileAccessBuffer->Get((uint8_t*)serialDataRx, ptrFileAccessBuffer->GetLength());

            dataRead.append(serialDataRx, (unsigned int)ptrFileOperationResult->GetValue());
        }
        cout << endl << "Data received is : " << dataRead << endl;

        // It clears the data
        CCommandPtr serialReceiveQueueClear = nodeMap.GetNode("SerialReceiveQueueClear");
        if (!IsAvailable(serialReceiveQueueClear))
        {
            cout << "Unable to execute Serial Receive Queue Clear. Aborting..." << endl << endl;
            return -1;
        }

        serialReceiveQueueClear->Execute();

        delete[] serialDataRx;
        serialDataRx = nullptr;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function sends data to the PC over serial port
int SerialTx(CameraPtr pCam, INodeMap& nodeMap, HANDLE& hFileHandle)
{
    int result = 0;
    try
    {
        // This tests transmit function
        cout << endl << "Set File Access to write mode " << endl;

        CIntegerPtr ptrFileAccessLength = nodeMap.GetNode("FileAccessLength");
        if (!IsWritable(ptrFileAccessLength))
        {
            cout << "Unable to set File Access Length. Aborting..." << endl << endl;
            return -1;
        }

        CRegisterPtr ptrFileAccessBuffer = nodeMap.GetNode("FileAccessBuffer");
        if (!IsWritable(ptrFileAccessBuffer))
        {
            cout << "Unable to set File Access Buffer. Aborting..." << endl << endl;
            return -1;
        }

        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to set File Operation Selector. Aborting..." << endl << endl;
            return -1;
        }

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to exectue File Operation. Aborting..." << endl << endl;
            return -1;
        }

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to get File Operation Status. Aborting..." << endl << endl;
            return -1;
        }

        ptrFileOperationSelector->SetIntValue(3); // write mode is chosen

        ptrFileAccessLength->SetValue(4);
        char serialDataTx[] = "ABCD";
        ptrFileAccessBuffer->Set((uint8_t*)serialDataTx, 4);
        ptrFileOperationExecute->Execute();

        CEnumEntryPtr ptrFileStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (ptrFileOperationStatus->GetIntValue() != ptrFileStatusSuccess->GetValue())
        {
            cout << "Failed to write the file in the File Access Control." << endl;
            return -1;
        }

        // It ensures that the data is transferred
        Sleep((unsigned int)((double)MILLISECOND * 4 / (SERIAL_PORT_BAUD_RATE / DATA_BITS)) + SERIAL_PORT_DELAY);

        // This checks if the device received the test data
        DWORD bytesRead = 0;
        unsigned char* tempBytesRead[4] = {0};
        if (!ReadFile(hFileHandle, tempBytesRead, 4, &bytesRead, nullptr))
        {
            cout << "Failed to read the test data from COM port Windows Error Code: " +
                        to_string(static_cast<long long>(GetLastError()))
                 << endl;
            return -1;
        }

        string dataTransmitted(reinterpret_cast<const char*>(tempBytesRead), 4);

        cout << endl << "Data transmitted was " << dataTransmitted << endl;
    }

    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function closes file access and com port connection
int CleanUp(INodeMap& nodeMap, HANDLE& hFileHandle)
{
    int result = 0;
    try
    {
        CEnumerationPtr ptrFileOperationSelector = nodeMap.GetNode("FileOperationSelector");
        if (!IsWritable(ptrFileOperationSelector))
        {
            cout << "Unable to set File Operation Selector. Aborting..." << endl << endl;
            return -1;
        }

        CCommandPtr ptrFileOperationExecute = nodeMap.GetNode("FileOperationExecute");
        if (!IsWritable(ptrFileOperationExecute))
        {
            cout << "Unable to exectue File Operation. Aborting..." << endl << endl;
            return -1;
        }

        CEnumerationPtr ptrFileOperationStatus = nodeMap.GetNode("FileOperationStatus");
        if (!IsReadable(ptrFileOperationStatus))
        {
            cout << "Unable to get File Operation Status. Aborting..." << endl << endl;
            return -1;
        }

        // This sets to file access close mode
        ptrFileOperationSelector->SetIntValue(1);

        cout << endl << "Execute file access close" << endl;

        ptrFileOperationExecute->Execute();

        CEnumEntryPtr ptrFileStatusSuccess = ptrFileOperationStatus->GetEntryByName("Success");
        if (ptrFileOperationStatus->GetIntValue() != ptrFileStatusSuccess->GetValue())
        {
            cout << "Failed to close the file in the File Access Control." << endl;
        }

        // It clears the incoming and outgoing buffer of the communication port
        PurgeComm(hFileHandle, PURGE_RXCLEAR);
        PurgeComm(hFileHandle, PURGE_TXCLEAR);

        cout << endl << "Close Com Port handle" << endl;
        CloseHandle(hFileHandle);
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
        // It retrieves TL device nodemap and print device information
        INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapTLDevice);

        // It initializes camera
        pCam->Init();

        // It retrieves GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        HANDLE hFileHandle;

        err = ConfigureDevice(pCam, hFileHandle);
        if (err < 0)
        {
            return err;
        }

        // It receives data over serial port from PC
        result = result | SerialRx(pCam, nodeMap, hFileHandle);

        // It transmits data over serial port to PC
        result = result | SerialTx(pCam, nodeMap, hFileHandle);

        // It closes the handle and file access
        result = result | CleanUp(nodeMap, hFileHandle);

        // This is to deinitialize camera
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
    int result = 0;

    // Print application build information
    cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

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
    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera
        pCam = camList.GetByIndex(i);

        cout << endl << "Running example for camera " << i << "..." << endl;

        // Run example
        result = result | RunSingleCamera(camList.GetByIndex(i));

        cout << endl << "Camera " << i << " example complete..." << endl << endl;
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
