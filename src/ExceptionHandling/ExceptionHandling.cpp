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
 *	@example ExceptionHandling.cpp
 *
 *	@brief ExceptionHandling.cpp shows the catching of an exception in
 *	Spinnaker. Following this, check out the Acquisition or NodeMapInfo examples
 *	if you haven't already. Acquisition demonstrates image acquisition while
 *	NodeMapInfo explores retrieving information from various node types.
 *
 *	This example shows three typical paths of exception handling in Spinnaker:
 *	catching the exception as a Spinnaker exception, as a standard exception,
 *	or as a standard exception which is then cast to a Spinnaker exception.
 *
 *	Once comfortable with Acquisition, ExceptionHandling, and NodeMapInfo, we
 *	suggest checking out AcquisitionMultipleCamera, NodeMapCallback, or
 *	SaveToAvi. AcquisitionMultipleCamera demonstrates simultaneously acquiring
 *	images from a number of cameras, NodeMapCallback serves as a good
 *	introduction to programming with callbacks and events, and SaveToAvi
 *	exhibits video creation.
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
using namespace std;

// Use the following enum and global constant to select type of exception
// handling used for the example.
enum exceptionType
{
    SPINNAKER_EXCEPTION,
    STANDARD_EXCEPTION,
    STANDARD_CAST_TO_SPINNAKER
};

const exceptionType chosenException = SPINNAKER_EXCEPTION;

// This helper function causes a Spinnaker exception by still holding a camera
// reference while attempting to release the system.
void causeSpinnakerException()
{
    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
    cout << "Spinnaker library version: " << spinnakerLibraryVersion.major << "." << spinnakerLibraryVersion.minor
         << "." << spinnakerLibraryVersion.type << "." << spinnakerLibraryVersion.build << endl
         << endl;

    cout << "System retrieved..." << endl;

    //
    // Retrieve list of cameras from the system
    //
    // *** NOTES ***
    // Retrieving this camera list without clearing it is the cause of the
    // -1004 RESOURCES_IN_USE exception when the example is run.
    //
    CameraList cameraList = system->GetCameras();

    cout << "Camera list retrieved..." << endl;

    // The exception will only be thrown if no cameras are connected
    if (cameraList.GetSize() == 0)
    {
        cout << endl << "Not enough cameras!" << endl << endl;

        return;
    }

    //
    // Release system
    //
    // *** NOTES ***
    // One of the requirements of cleanly releasing a system object is that
    // all references to resources are released first. In this example, the
    // cameras referenced by the camera list are still present.
    //
    system->ReleaseInstance();

    cout << "System released; this part of the code should not be reached..." << endl << endl;
}

// This helper function causes a standard exception by attempting to access an
// member of a vector that is out of range.
void causeStandardException()
{
    const int k_maxNum = 10;
    vector<int> numbers;

    // The vector is initialized with 10 members, from indexes 0 to 9.
    for (int i = 0; i < k_maxNum; i++)
    {
        numbers.push_back(i);
    }

    cout << "Vector initialized..." << endl << endl;

    // The number attempting to be called here is index 10, or the 11th member,
    // which throws an out-of-range exception.
    cout << "The highest number in the vector is " << numbers.at(k_maxNum) << ".";
}

// Example entry point; this function demonstrates the handling of a variety
// of exception use-cases.
int main(int /*argc*/, char** /*argv*/)
{
    // Print application build information
    cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Cause and catch an exception.
    switch (chosenException)
    {
        //
        // Catch a Spinnaker exception
        //
        // *** NOTES ***
        // The Spinnaker library has a number of built-in exceptions that
        // provide more information than standard exceptions.
        //
    case SPINNAKER_EXCEPTION:
        try
        {
            causeSpinnakerException();
        }
        catch (Spinnaker::Exception& ex)
        {
            cout << endl << "Spinnaker exception caught." << endl << endl;

            //
            // Print Spinnaker exception
            //
            // *** NOTES ***
            // Spinnaker exceptions are still able to access information using
            // the standard what() method.
            //
            cout << "Error: " << ex.what() << endl;

            //
            // Print additional information available to Spinnaker exceptions
            //
            // *** NOTES ***
            // However, Spinnaker exceptions have additional information. The
            // message below prints out the error code, function name, and line
            // number of the exception; this functionality is not available for
            // standard exceptions.
            //
            cout << "Error code " << ex.GetError() << " raised in function " << ex.GetFunctionName() << " at line "
                 << ex.GetLineNumber() << "." << endl;
        }
        break;

        //
        // Catch a standard exception
        //
        // *** NOTES ***
        // Standard try-catch blocks can catch Spinnaker exceptions, but they
        // provide no access to the additional information available. Spinnaker
        // exceptions, on the other hand, provide
        //
    case STANDARD_EXCEPTION:
        try
        {
            causeStandardException();
        }
        catch (std::exception& ex)
        {
            cout << endl << "Standard exception caught." << endl << endl;

            //
            // Print more information
            //
            // *** NOTES ***
            // The simplest way to catch exceptions in Spinnaker is with
            // standard try-catch blocks. This will catch all exceptions, but
            // sacrifice extra functionality.
            //
            cout << "Error: " << ex.what() << endl;
        }
        break;

        //
        // Catch a standard exception; cast it to a Spinnaker exception.
        //
        // *** NOTES ***
        // Regular exceptions can be cast to Spinnaker exceptions by using a
        // dynamic_cast<>.
        //
    case STANDARD_CAST_TO_SPINNAKER:
        try
        {
            causeSpinnakerException();
        }
        catch (std::exception& ex)
        {
            cout << endl << "Standard exception caught; will be cast as Spinnaker exception." << endl << endl;

            //
            // Wrap the cast in a further try-catch
            //
            // *** NOTES ***
            // The cast needs to be wrapped in a further try-catch block
            // because standard exceptions will fail the cast.
            //
            try
            {
                //
                // Attempt to cast exception as Spinnaker exception
                //
                // *** NOTES ***
                // A successful cast means that the exception is particular to
                // Spinnaker and will keep the flow of control in the try block
                // while a failed cast means that the exception is standard and
                // will push the flow of control into the catch block.
                //
                Spinnaker::Exception& spinEx = dynamic_cast<Spinnaker::Exception&>(ex);

                // Print additional information if Spinnaker exception
                cout << "Error: " << spinEx.GetErrorMessage() << endl;
                cout << "Error code " << spinEx.GetError() << " raised in function " << spinEx.GetFunctionName()
                     << " at line " << spinEx.GetLineNumber() << "." << endl;
            }
            catch (std::exception& stdEx)
            {
                cout << "Cannot cast; not a Spinnaker exception: " << stdEx.what() << endl;

                // Print standard information if standard exception
                cout << "Standard error: " << ex.what() << endl;
            }
        }
    }

    cout << endl << "Done! Press Enter to exit..." << endl;
    getchar();

    return 0;
}
