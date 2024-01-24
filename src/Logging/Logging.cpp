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
 *	@example Logging.cpp
 *
 *	@brief Logging.cpp shows how to create a handler to access logging events.
 *	It relies on information provided in the Enumeration, Acquisition, and
 *	NodeMapInfo examples.
 *
 *	It can also be helpful to familiarize yourself with the NodeMapCallback
 *	example, as nodemap callbacks follow the same general procedure as
 *	events, but with a few less steps.
 *
 *	This example creates a user-defined class, LoggingEventHandlerImpl, that inherits
 *	from the Spinnaker class, LoggingEventHandler. The child class allows the user to
 *	define any properties, parameters, and the event itself while LoggingEventHandler
 *	allows the child class to appropriately interface with the Spinnaker SDK.
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

// Define callback priority threshold; please see documentation for additional
// information on logging level philosophy.
const SpinnakerLogLevel k_LoggingLevel = SPINNAKER_LOG_LEVEL_DEBUG;

// Although logging events are just as flexible and extensible as other events,
// they are generally only used for logging purposes, which is why a number of
// helpful functions that provide logging information have been added. Generally,
// if the purpose is not logging, one of the other event types is probably more
// appropriate.
class LoggingEventHandlerImpl : public LoggingEventHandler
{
    // This function displays readily available logging information.
    void OnLogEvent(LoggingEventDataPtr loggingEventDataPtr)
    {
        cout << "--------Log Event Received----------" << endl;
        cout << "Category: " << loggingEventDataPtr->GetCategoryName() << endl;
        cout << "Priority Value: " << loggingEventDataPtr->GetPriority() << endl;
        cout << "Priority Name: " << loggingEventDataPtr->GetPriorityName() << endl;
        cout << "Timestmap: " << loggingEventDataPtr->GetTimestamp() << endl;
        cout << "NDC: " << loggingEventDataPtr->GetNDC() << endl;
        cout << "Thread: " << loggingEventDataPtr->GetThreadName() << endl;
        cout << "Message: " << loggingEventDataPtr->GetLogMessage() << endl;
        cout << "------------------------------------" << endl << endl;
    }
};

// Example entry point; notice the volume of data that the logging event handler
// prints out on debug despite the fact that very little really happens in this
// example. Because of this, it may be better to have the logger set to lower
// level in order to provide a more concise, focused log.
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

    //
    // Create and register the logging event handler
    //
    // *** NOTES ***
    // Logging events are registered to the system. Take note that a logging
    // event handler is very verbose when the logging level is set to debug.
    //
    // *** LATER ***
    // Logging event handlers must be unregistered manually. This must be done prior to
    // releasing the system and while the logging event handlers are still in scope.
    //
    LoggingEventHandlerImpl loggingEventHandler;
    system->RegisterLoggingEventHandler(loggingEventHandler);

    //
    // Set callback priority level
    //
    // *** NOTES ***
    // Please see documentation for up-to-date information on the logging
    // philosophies of the Spinnaker SDK.
    //
    system->SetLoggingEventPriorityLevel(k_LoggingLevel);

    // Retrieve list of cameras from the system
    CameraList camList = system->GetCameras();

    unsigned int numCameras = camList.GetSize();

    cout << "Number of cameras detected: " << numCameras << endl << endl;

    // Clear camera list before releasing system
    camList.Clear();

    //
    // Unregister logging event handler
    //
    // *** NOTES ***
    // It is important to unregister all logging event handlers from the system.
    //
    system->UnregisterLoggingEventHandler(loggingEventHandler);

    // Release system
    system->ReleaseInstance();

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return 0;
}
