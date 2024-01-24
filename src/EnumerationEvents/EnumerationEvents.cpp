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
 *	@example EnumerationEvents.cpp
 *
 *	@brief EnumerationEvents.cpp explores arrival and removal events on
 *	interfaces and the system. It relies on information provided in the
 *	Enumeration, Acquisition, and NodeMapInfo examples.
 *
 *	It can also be helpful to familiarize yourself with the NodeMapCallback
 *	example, as nodemap callbacks follow the same general procedure as
 *	events, but with a few less steps.
 *
 *	This example creates two user-defined classes: InterfaceEventHandlerImpl and
 *	SystemEventHandlerImpl. These child classes allow the user to define properties,
 *	parameters, and the event itself while the parent classes - DeviceArrivalEventHandler,
 *	DeviceRemovalEventHandler, and InterfaceEventHandler - allow the child classes to interface
 *	with Spinnaker.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <sstream>
#if _MSC_VER != 1600
#include <mutex>
#endif

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

// This class defines the properties and functions for device arrivals and removals
// on an interface. Take special note of the signatures of the OnDeviceArrival()
// and OnDeviceRemoval() methods. Also, enumeration event handlers must inherit at least
// one of the three event types - DeviceArrivalEventHandler, DeviceRemovalEventHandler, and InterfaceEventHandler
// if they are to be registered to interfaces, the system, or both.
class InterfaceEventHandlerImpl : public InterfaceEventHandler
{
  public:
    //
    // Set the constructor and destructor
    //
    // *** NOTES ***
    // When constructing a generic InterfaceEventHandler to be registered to the system,
    // the handler will not have knowledge of which interface triggered the event callbacks.
    // On the other hand, InterfaceEventHandlerImpl does not need knowledge about the system if we
    // are constructing it to be registered to a specific interface.
    //
    InterfaceEventHandlerImpl(SystemPtr system)
        : m_system(system), m_interface(nullptr), m_interfaceID(""), m_registerToSystem(true){};

    InterfaceEventHandlerImpl(InterfacePtr iface, std::string interfaceID)
        : m_system(nullptr), m_interface(iface), m_interfaceID(interfaceID), m_registerToSystem(false){};

    ~InterfaceEventHandlerImpl(){};

    // Helper function to print the number of devices on an interface event registered to the system.
    // Note: Using strings in event print statements prevents text from overlapping in the console output
    void PrintGenericHandlerMessage(const unsigned long long deviceCount)
    {
        cout << "Generic interface event handler:\n";
        const bool singular = deviceCount == 1;
        const string genericMsg = "\tThere " + string(singular ? "is " : "are ") + to_string(deviceCount) +
                                  string(singular ? " device " : " devices ") + "on the system.\n";
        cout << genericMsg << endl;
    }

    // This function defines the arrival event handler on an interface. It prints out
    // the device serial number of the camera arriving and the interface
    // number. The argument is the serial number of the camera that triggered
    // the arrival event.
    // If the event handler was constructed to be registered to the system as a generic
    // InterfaceEventHandler, then we just retrieve the number of cameras
    // currently connected on the system and print it out.
    void OnDeviceArrival(CameraPtr pCamera)
    {
        if (m_registerToSystem)
        {
            const auto deviceCount = static_cast<unsigned long long>(m_system->GetCameras().GetSize());
            PrintGenericHandlerMessage(deviceCount);
        }
        else
        {
            cout << "Interface event handler:\n";
            const std::string arrivalMsg = "\tDevice " + std::string(pCamera->TLDevice.DeviceSerialNumber.ToString()) +
                                           " has arrived on interface '" + m_interfaceID + "'.\n";
            cout << arrivalMsg << endl;
        }
    }

    // This function defines removal event handlers on an interface. It prints out the
    // device serial number of the camera being removed and the interface
    // number. The argument is the serial number of the camera that triggered
    // the removal event.
    // If the event handler was constructed to be registered to the system as a generic
    // InterfaceEventHandler, then we just retrieve the number of cameras
    // currently connected on the system and print it out.
    void OnDeviceRemoval(CameraPtr pCamera)
    {
        if (m_registerToSystem)
        {
            try
            {
                // At this point the interface that the camera is connected to
                // may have been removed as well, so we need to handle
                // potential exceptions when calling GetCameras().
                const auto deviceCount = static_cast<unsigned long long>(m_system->GetCameras().GetSize());
                PrintGenericHandlerMessage(deviceCount);
            }
            catch (Spinnaker::Exception& e)
            {
                cout << "Error updating cameras: " << e.what() << endl;
            }
        }
        else
        {
            cout << "Interface event handler:\n";
            const std::string removalMsg = "\tDevice " + std::string(pCamera->TLDevice.DeviceSerialNumber.ToString()) +
                                           " was removed from interface '" + m_interfaceID + "'.\n";
            cout << removalMsg << endl;
        }
    }

    // This function returns the interface ID that the interface event handler is bound to.
    std::string GetInterfaceId()
    {
        return m_interfaceID;
    }

  private:
    SystemPtr m_system;
    InterfacePtr m_interface;
    std::string m_interfaceID;

    bool m_registerToSystem;
};

// This class defines the properties and methods for interface arrivals and removals
// on the system. Take special note of the signatures of the OnInterfaceArrival()
// and OnInterfaceRemoval() methods. Interface enumeration event handlers must inherit from
// SystemEventHandler.
class SystemEventHandlerImpl : public SystemEventHandler
{
  public:
    SystemEventHandlerImpl(SystemPtr system) : m_system(system)
    {
#if _MSC_VER == 1600
        // Initialize the mutex for Visual Studio 2010
        eventHandlersMutex = CreateMutex(NULL, false, NULL);
#endif
    };

    ~SystemEventHandlerImpl(){};

    // Helper function to lock the eventhandler mutex as std::mutex is not available for vs 2010
    void LockEventHandlerMutex()
    {
#if _MSC_VER == 1600
        WaitForSingleObject(eventHandlersMutex, INFINITE);
#else
        eventHandlersMutex.lock();
#endif
    }

    // Helper function to unlock the eventhandler mutex as std::mutex is not available for vs 2010
    void UnlockEventHandlerMutex()
    {
#if _MSC_VER == 1600
        ReleaseMutex(eventHandlersMutex);
#else
        eventHandlersMutex.unlock();
#endif
    }

    // This method defines the interface arrival event on the system.
    // It first prints the ID of the arriving interface, then
    // registers the interface event on the newly arrived interface.
    //
    // *** NOTES ***
    // Only arrival events for GEV interfaces are currently supported.
    void OnInterfaceArrival(InterfacePtr pInterface)
    {
        cout << "System event handler:\n";
        cout << "\tInterface '" << pInterface->TLInterface.InterfaceID.ToString() << "' has arrived on the system." << endl;

        // UpdateInterfaceList() only updates newly arrived or newly removed interfaces.
        // In particular, after this call:
        //
        // - Any pre-existing interfaces will still be valid.
        // - Any pointers to newly removed interfaces will be invalid.
        //
        // *** NOTES ***
        // - Invalid pointers will be re-validated if the interface comes back (arrives) with the same interface ID.
        //   If the interface ID changes, you can use the pointer populated by this callback or you must get a new 
        //   pointer from the updated interface list.
        //
        // - Interface indices used to access an interface with GetInterfaces() may change after updating the interface
        //   list. The interface at a particular index cannot be expected to remain at that index after calling
        //   UpdateInterfaceList().
        m_system->UpdateInterfaceList();

        CameraList cameraList = pInterface->GetCameras();
        const unsigned int numCameras = cameraList.GetSize();
        for (unsigned int camIdx = 0; camIdx < numCameras; camIdx++)
        {
            CameraPtr pCam = cameraList.GetByIndex(camIdx);
            INodeMap& nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
            CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
            if (IsReadable(ptrStringSerial))
            {
                const gcstring deviceSerialNumber = ptrStringSerial->GetValue();
                cout << "\tDevice " << string(deviceSerialNumber) << " is connected to interface '" << pInterface->TLInterface.InterfaceID.ToString() << "'." << endl;
            }
        }

        // Create interface event
        {
            LockEventHandlerMutex();

            try
            {
                shared_ptr<InterfaceEventHandlerImpl> interfaceEventHandler =
                    shared_ptr<InterfaceEventHandlerImpl>(new InterfaceEventHandlerImpl(pInterface, pInterface->TLInterface.InterfaceID.ToString().c_str()));
                m_interfaceEventHandlers.push_back(interfaceEventHandler);

                // Register interface event
                pInterface->RegisterEventHandler(*interfaceEventHandler);
                cout << "Event handler registered to interface '" << pInterface->TLInterface.InterfaceID.ToString() << "'..." << endl;
            }
            catch (exception& e)
            {
                cout << "Error registering interface event handler to '" << pInterface->TLInterface.InterfaceID.ToString() << "' :" << e.what()
                        << endl;
            }

            UnlockEventHandlerMutex();
        }
    }

    // This method defines the interface removal event on the system.
    // It prints the ID of the interface removed.
    //
    // *** NOTES ***
    // Only removal events for GEV interfaces are currently supported.
    void OnInterfaceRemoval(InterfacePtr pInterface)
    {
        cout << "System event handler:" << endl;
        cout << "\tInterface '"<< pInterface->TLInterface.InterfaceID.ToString() << "' was removed from the system." << endl;

        // Interface indices used to access an interface with GetInterfaces() may change after updating the interface
        // list. The interface at a particular index cannot be expected to remain at that index after calling
        // UpdateInterfaceList().
        m_system->UpdateInterfaceList();

        // Find the event handler that was registered to the removed interface and delete it.
        // Interface events are automatically unregistered when the interface is removed so it is not necessary to
        // manually unregister them.
        {
            LockEventHandlerMutex();
            try
            {
                unsigned int handlerIdx = 0;
                for (handlerIdx = 0; handlerIdx < m_interfaceEventHandlers.size(); handlerIdx++)
                {
                    if (m_interfaceEventHandlers[handlerIdx]->GetInterfaceId() == pInterface->TLInterface.InterfaceID.ToString().c_str())
                    {
                        m_interfaceEventHandlers.erase(m_interfaceEventHandlers.begin() + handlerIdx);
                        break;
                    }
                }
            }
            catch (exception& e)
            {
                cout << "Error erasing event handler from interface '" << pInterface->TLInterface.InterfaceID.ToString() << "' :" << e.what() << endl;
            }

            UnlockEventHandlerMutex();
        }
    }

    void RegisterInterfaceEventToSystem()
    {
        if (!m_interfaceEventHandlerOnSystem)
        {
            //
            // Create interface event handler for the system
            //
            // *** NOTES ***
            // The InterfaceEventHandler has been constructed to accept a system object in
            // order to print the number of cameras on the system.
            //
            m_interfaceEventHandlerOnSystem =
                shared_ptr<InterfaceEventHandlerImpl>(new InterfaceEventHandlerImpl(m_system));
        }

        //
        // Register interface event handler for the system
        //
        // *** NOTES ***
        // Arrival, removal, and interface event handlers can all be registered to
        // interfaces or the system. Do not think that interface event handlers can only be
        // registered to an interface. An interface event is merely a combination
        // of an arrival and a removal event.
        // Only arrival and removal events for GEV interfaces are currently supported.
        //
        // *** LATER ***
        // Arrival, removal, and interface event handlers must all be unregistered manually.
        // This must be done prior to releasing the system and while they are still
        // in scope.
        //
        m_system->RegisterEventHandler(*m_interfaceEventHandlerOnSystem);
        cout << "Interface event handler registered on the system..." << endl;
    }

    void UnregisterInterfaceEventFromSystem()
    {
        //
        // Unregister interface event handler from system object
        //
        // *** NOTES ***
        // It is important to unregister all arrival, removal, and interface event handlers
        // registered to the system.
        //
        if (m_interfaceEventHandlerOnSystem)
        {
            m_system->UnregisterEventHandler(*m_interfaceEventHandlerOnSystem);

            cout << "Interface event handler unregistered from system..." << endl;

            m_interfaceEventHandlerOnSystem = shared_ptr<InterfaceEventHandlerImpl>();
        }
    }

    void RegisterAllInterfaceEvents()
    {
        {
            LockEventHandlerMutex();

            if (!m_interfaceEventHandlers.empty())
            {
                m_interfaceEventHandlers.clear();
            }

            UnlockEventHandlerMutex();
        }

        InterfaceList interfaceList = m_system->GetInterfaces();
        unsigned int numInterfaces = interfaceList.GetSize();

        //
        // Create and register interface event handler to each interface
        //
        // *** NOTES ***
        // The process of event handler creation and registration on interfaces is similar
        // to the process of event handler creation and registration on the system. The
        // class for interfaces has been constructed to accept an interface and an
        // interface number (this is just to separate the interfaces).
        //
        // *** LATER ***
        // Arrival, removal, and interface event handlers must all be unregistered manually.
        // This must be done prior to releasing the system and while they are still
        // in scope.
        //
        for (unsigned int i = 0; i < numInterfaces; i++)
        {
            // Select interface
            InterfacePtr pInterface = interfaceList.GetByIndex(i);
            INodeMap& nodeMap = pInterface->GetTLNodeMap();

            const CStringPtr interfaceIDNode = nodeMap.GetNode("InterfaceID");
            // Ensure the node is valid
            if (!IsReadable(interfaceIDNode))
            {
                continue;
            }

            std::string interfaceID = std::string(interfaceIDNode->GetValue().c_str());

            {
                LockEventHandlerMutex();
                try
                {
                    // Create interface event handler
                    shared_ptr<InterfaceEventHandlerImpl> interfaceEventHandler =
                        shared_ptr<InterfaceEventHandlerImpl>(new InterfaceEventHandlerImpl(pInterface, interfaceID));
                    m_interfaceEventHandlers.push_back(interfaceEventHandler);

                    // Register interface event handler
                    pInterface->RegisterEventHandler(*m_interfaceEventHandlers[i]);

                    cout << "Event handler registered to interface '" << interfaceID << "'..." << endl;
                }
                catch (exception& e)
                {
                    cout << "Error registering event handler to interface '" << interfaceID << "' :" << e.what() << endl;
                }

                UnlockEventHandlerMutex();
            }
        }
        cout << endl;
    }

    void UnregisterAllInterfaceEvents()
    {
        InterfaceList interfaceList = m_system->GetInterfaces(false);
        unsigned int numInterfaces = interfaceList.GetSize();

        //
        // Unregister interface event handler from each interface
        //
        // *** NOTES ***
        // It is important to unregister all arrival, removal, and interface event handlers
        // from all interfaces that they may be registered to.
        //
        for (unsigned int i = 0; i < numInterfaces; i++)
        {
            // Select interface
            InterfacePtr pInterface = interfaceList.GetByIndex(i);
            INodeMap& nodeMap = pInterface->GetTLNodeMap();

            const CStringPtr interfaceIDNode = nodeMap.GetNode("InterfaceID");
            // Ensure the node is valid
            if (!IsReadable(interfaceIDNode))
            {
                continue;
            }

            std::string interfaceID = std::string(interfaceIDNode->GetValue().c_str());

            {
                LockEventHandlerMutex();

                try
                {
                    // Ensure that that the event is unregistered from the matching interface
                    for (unsigned int j = 0; j < m_interfaceEventHandlers.size(); j++)
                    {
                        if (interfaceID == m_interfaceEventHandlers[j]->GetInterfaceId())
                        {
                            pInterface->UnregisterEventHandler(*m_interfaceEventHandlers[j]);
                        }
                    }
                }
                catch (exception& e)
                {
                    cout << "Error unreigstering event handler to interface '" << interfaceID << "' :" << e.what()
                         << endl;
                }

                UnlockEventHandlerMutex();
            }
        }

        {
            LockEventHandlerMutex();

            m_interfaceEventHandlers.clear();

            cout << "Event handler unregistered from interfaces..." << endl;

            UnlockEventHandlerMutex();
        }
    }

  private:
    SystemPtr m_system;
    shared_ptr<InterfaceEventHandlerImpl> m_interfaceEventHandlerOnSystem;
    vector<shared_ptr<InterfaceEventHandlerImpl>> m_interfaceEventHandlers;

#if _MSC_VER == 1600
    HANDLE eventHandlersMutex;
#else
    std::mutex eventHandlersMutex;
#endif
};

// This function checks if GEV enumeration is enabled on the system.
void CheckGevEnabled(SystemPtr& pSystem)
{
    // Retrieve the System TL Nodemap and EnumerateGEVInterfaces node
    INodeMap& nodeMap = pSystem->GetTLNodeMap();
    const CBooleanPtr enumerateGevInterfacesNode = nodeMap.GetNode("EnumerateGEVInterfaces");

    // Ensure the node is valid
    if (IsReadable(enumerateGevInterfacesNode))
    {
        const bool gevEnabled = enumerateGevInterfacesNode->GetValue();

        // Check if node is enabled
        if (!gevEnabled)
        {
            cout << endl
                 << "WARNING: GEV Enumeration is disabled." << endl
                 << "If you intend to use GigE cameras please run the EnableGEVInterfaces shortcut" << endl
                 << "or set EnumerateGEVInterfaces to true and relaunch your application." << endl
                 << endl;
        }
        else
        {
            cout << "EnumerateGEVInterfaces is enabled. Continuing.." << endl;
        }
    }
    else
    {
        cout << "EnumerateGEVInterfaces node is not readable" << endl;
    }
}

// Example entry point; this function sets up the example to act appropriately
// upon arrival and removal events; please see Enumeration example for more
// in-depth comments on preparing and cleaning up the system.
int main(int /*argc*/, char** /*argv*/)
{
    // Print application build information
    cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
    cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
         << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
         << endl;

    // Check if GEV enumeration is enabled.
    CheckGevEnabled(system);

    // Retrieve list of cameras from the system
    CameraList camList = system->GetCameras();

    unsigned int numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl << endl;

    //
    // Retrieve list of interfaces from the system
    //
    // *** NOTES ***
    // macOS interfaces are only registered if they are active.
    // This example will have varied outcomes depending on the order
    // in which we receive the interface and the device arrival events.
    // In most cases on macOS, since the interface and the device arrive
    // at the same time, we will see the interface arrival event and
    // the interface will already have one device connected to it.
    // This means we will not see another device arrival event.
    // Device removal events however, are guaranteed to be fired before
    // interface removal events.
    //
    InterfaceList interfaceList = system->GetInterfaces();

    unsigned int numInterfaces = interfaceList.GetSize();

    cout << "Number of interfaces detected: " << numInterfaces << endl << endl;

    cout << endl << "*** CONFIGURING ENUMERATION EVENTS ***" << endl << endl;

    //
    // Create system event handler
    //
    // *** NOTES ***
    // The SystemEventHandlerImpl has been constructed to accept a system object in
    // order to register/unregister events to/from the system object
    //
    SystemEventHandlerImpl systemEventHandler(system);

    //
    // Register system event to the system
    //
    // *** NOTES ***
    // A system event is merely a combination of an interface arrival and an
    // interface removal event.
    // This feature is currently only supported for GEV interface arrivals and removals.
    //
    // *** LATER ***
    // Interface arrival and removal events must all be unregistered manually.
    // This must be done prior to releasing the system and while they are still
    // in scope.
    //
    system->RegisterEventHandler(systemEventHandler);

    systemEventHandler.RegisterInterfaceEventToSystem();
    systemEventHandler.RegisterAllInterfaceEvents();

    // Wait for user to plug in and/or remove camera devices
    cout << endl << "Ready! Remove/Plug in cameras to test or press Enter to exit..." << endl << endl;
    getchar();

    systemEventHandler.UnregisterAllInterfaceEvents();
    systemEventHandler.UnregisterInterfaceEventFromSystem();

    //
    // Unregister system event handler from system object
    //
    // *** NOTES ***
    // It is important to unregister all interface arrival and removal event handlers
    // registered to the system.
    //
    system->UnregisterEventHandler(systemEventHandler);

    cout << "System event handler unregistered from system..." << endl;

    // Clear camera list before releasing system
    camList.Clear();

    // Clear interface list before releasing system
    interfaceList.Clear();

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return 0;
}
