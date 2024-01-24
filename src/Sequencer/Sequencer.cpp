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
 *	@example Sequencer.cpp
 *
 *	@brief Sequencer.cpp shows how to use the sequencer to grab images with
 *	various settings. It relies on information provided in the Enumeration,
 *	Acquisition, and NodeMapInfo examples.
 *
 *	It can also be helpful to familiarize yourself with the ImageFormatControl
 *	and Exposure examples as these examples provide a strong introduction to
 *	camera customization.
 *
 *	The sequencer is another very powerful tool, which can be used to create
 *	and store multiple states of customized image settings. A very useful
 *	application of the sequencer is creating high dynamic range images.
 *
 *	This example is probably the most complex and definitely the longest. As
 *	such, the configuration has been split between three functions. The first
 *	prepares the camera to set the sequences, the second sets the settings for
 *	a single state (it is run five times), and the third configures the
 *	camera to use the sequencer when it acquires images.
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

// This function handles the error prints when a node or entry is 
// not readable on the connected camera
void PrintRetrieveNodeFailure(string node, string name)
{
    cout << "Unable to get " << node << " (" << name << " " << node << " retrieval failed)." << endl << endl;
    cout << "The " << node << " may not be readable/writable on all camera models..." << endl;
    cout << "Please try a Blackfly S camera." << endl << endl;
}

// This function prepares the sequencer to accept custom configurations by
// ensuring sequencer mode is off (this is a requirement to the enabling of
// sequencer configuration mode), disabling automatic gain and exposure, and
// turning sequencer configuration mode on.
int ConfigureSequencerPartOne(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING SEQUENCER ***" << endl << endl;

    try
    {
        //
        // Ensure sequencer is off for configuration
        //
        // *** NOTES ***
        // In order to configure a new sequence, sequencer configuration mode
        // needs to be turned on. To do this, sequencer mode must be disabled.
        // However, simply disabling sequencer mode might throw an exception if
        // the current sequence is an invalid configuration.
        //
        // Thus, in order to ensure that sequencer mode is disabled, we first
        // check whether the current sequence is valid. If it
        // isn't, then we know that sequencer mode is off and we can move on;
        // if it is, then we can manually disable sequencer mode.
        //
        // Also note that sequencer configuration mode needs to be off in order
        // to manually disable sequencer mode. It should be off by default, so
        // the example skips checking this.
        //
        // Validate sequencer configuration
        CEnumerationPtr ptrSequencerConfigurationValid = nodeMap.GetNode("SequencerConfigurationValid");
        if (!IsReadable(ptrSequencerConfigurationValid))
        {
            PrintRetrieveNodeFailure("node", "SequencerConfigurationValid");
            return -1;
        }

        CEnumEntryPtr ptrSequencerConfigurationValidYes = ptrSequencerConfigurationValid->GetEntryByName("Yes");
        if (!IsReadable(ptrSequencerConfigurationValidYes))
        {
            PrintRetrieveNodeFailure("entry", "SequencerConfigurationValid 'Yes'");
            return -1;
        }

        // If valid, disable sequencer mode; otherwise, do nothing
        CEnumerationPtr ptrSequencerMode = nodeMap.GetNode("SequencerMode");
        if (ptrSequencerConfigurationValid->GetCurrentEntry()->GetValue() ==
            ptrSequencerConfigurationValidYes->GetValue())
        {
            if (!IsReadable(ptrSequencerMode) ||
                !IsWritable(ptrSequencerMode))
            {
                PrintRetrieveNodeFailure("node", "SequencerMode");
                return -1;
            }

            CEnumEntryPtr ptrSequencerModeOff = ptrSequencerMode->GetEntryByName("Off");
            if (!IsReadable(ptrSequencerModeOff))
            {
                PrintRetrieveNodeFailure("entry", "SequencerMode Off");
                return -1;
            }

            ptrSequencerMode->SetIntValue(static_cast<int64_t>(ptrSequencerModeOff->GetValue()));
        }

        cout << "Sequencer mode disabled..." << endl;

        //
        // Turn off automatic exposure
        //
        // *** NOTES ***
        // Automatic exposure prevents the manual configuration of exposure
        // times and needs to be turned off for this example.
        //
        // *** LATER ***
        // Automatic exposure is turned back on at the end of the example in
        // order to restore the camera to its default state.
        //
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (!IsReadable(ptrExposureAuto) ||
            !IsWritable(ptrExposureAuto))
        {
            PrintRetrieveNodeFailure("node", "ExposureAuto");
            return -1;
        }

        CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
        if (!IsReadable(ptrExposureAutoOff))
        {
            PrintRetrieveNodeFailure("entry", "ExposureAuto Off");
            return -1;
        }

        ptrExposureAuto->SetIntValue(static_cast<int64_t>(ptrExposureAutoOff->GetValue()));

        cout << "Automatic exposure disabled..." << endl;

        //
        // Turn off automatic gain
        //
        // *** NOTES ***
        // Automatic gain prevents the manual configuration of gain and needs
        // to be turned off for this example.
        //
        // *** LATER ***
        // Automatic gain is turned back on at the end of the example in
        // order to restore the camera to its default state.
        //
        CEnumerationPtr ptrGainAuto = nodeMap.GetNode("GainAuto");
        if (!IsReadable(ptrGainAuto) ||
            !IsWritable(ptrGainAuto))
        {
            PrintRetrieveNodeFailure("node", "GainAuto");
            return -1;
        }

        CEnumEntryPtr ptrGainAutoOff = ptrGainAuto->GetEntryByName("Off");
        if (!IsReadable(ptrGainAutoOff))
        {
            PrintRetrieveNodeFailure("entry", "GainAuto Off");
            return -1;
        }

        ptrGainAuto->SetIntValue(static_cast<int64_t>(ptrGainAutoOff->GetValue()));

        cout << "Automatic gain disabled..." << endl;

        //
        // Turn configuration mode on
        //
        // *** NOTES ***
        // Once sequencer mode is off, enabling sequencer configuration mode
        // allows for the setting of each state.
        //
        // *** LATER ***
        // Before sequencer mode is turned back on, sequencer configuration
        // mode must be turned back off.
        //
        CEnumerationPtr ptrSequencerConfigurationMode = nodeMap.GetNode("SequencerConfigurationMode");
        if (!IsReadable(ptrSequencerConfigurationMode) ||
            !IsWritable(ptrSequencerConfigurationMode))
        {
            PrintRetrieveNodeFailure("node", "SequencerConfigurationMode");
            return -1;
        }

        CEnumEntryPtr ptrSequencerConfigurationModeOn = ptrSequencerConfigurationMode->GetEntryByName("On");
        if (!IsReadable(ptrSequencerConfigurationModeOn))
        {
            PrintRetrieveNodeFailure("entry", "SequencerConfigurationMode On");
            return -1;
        }

        ptrSequencerConfigurationMode->SetIntValue(static_cast<int64_t>(ptrSequencerConfigurationModeOn->GetValue()));

        cout << "Sequencer configuration mode enabled..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function sets a single state. It sets the sequence number, applies
// custom settings, selects the trigger type and next state number, and saves
// the state. The custom values that are applied are all calculated in the
// function that calls this one, RunSingleCamera().
int SetSingleState(
    INodeMap& nodeMap,
    unsigned int sequenceNumber,
    int64_t widthToSet,
    int64_t heightToSet,
    double exposureTimeToSet,
    double gainToSet)
{
    int result = 0;

    try
    {
        //
        // Select the current sequence number
        //
        // *** NOTES ***
        // Select the index of the state to be set.
        //
        // *** LATER ***
        // The next state - i.e. the state to be linked to -
        // also needs to be set before saving the current state.
        //
        CIntegerPtr ptrSequencerSetSelector = nodeMap.GetNode("SequencerSetSelector");
        if (!IsWritable(ptrSequencerSetSelector))
        {
            cout << "Unable to set current state. Aborting..." << endl << endl;
            return -1;
        }

        ptrSequencerSetSelector->SetValue(sequenceNumber);

        cout << "Setting state " << sequenceNumber << "..." << endl;

        //
        // Set desired settings for the current state
        //
        // *** NOTES ***
        // Width, height, exposure time, and gain are set in this example. If
        // the sequencer isn't working properly, it may be important to ensure
        // that each feature is enabled on the sequencer. Features are enabled
        // by default, so this is not explored in this example.
        //
        // Changing the height and width for the sequencer is not available
        // for all camera models.
        //
        // Set width; width recorded in pixels
        CIntegerPtr ptrWidth = nodeMap.GetNode("Width");
        if (IsReadable(ptrWidth) && IsWritable(ptrWidth))
        {
            int64_t widthInc = ptrWidth->GetInc();

            if (widthToSet % widthInc != 0)
            {
                widthToSet = (widthToSet / widthInc) * widthInc;
            }

            ptrWidth->SetValue(widthToSet);

            cout << "\tWidth set to " << ptrWidth->GetValue() << "..." << endl;
        }
        else
        {
            cout << "\tUnable to get or set width; width for sequencer not readable or writable on all camera models..." << endl;
        }

        // Set height; height recorded in pixels
        CIntegerPtr ptrHeight = nodeMap.GetNode("Height");
        if (IsReadable(ptrHeight) && IsWritable(ptrHeight))
        {
            int64_t heightInc = ptrHeight->GetInc();

            if (heightToSet % heightInc != 0)
            {
                heightToSet = (heightToSet / heightInc) * heightInc;
            }

            ptrHeight->SetValue(heightToSet);

            cout << "\tHeight set to " << ptrHeight->GetValue() << "..." << endl;
        }
        else
        {
            cout << "\tUnable to get or set height; height for sequencer not readable or writable on all camera models..." << endl;
        }

        // Set exposure time; exposure time recorded in microseconds
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime) || !IsWritable(ptrExposureTime))
        {
            cout << "Unable to get or set exposure time (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        double exposureTimeMax = ptrExposureTime->GetMax();

        if (exposureTimeToSet > exposureTimeMax)
        {
            exposureTimeToSet = exposureTimeMax;
        }

        ptrExposureTime->SetValue(exposureTimeToSet);

        cout << "\tExposure set to " << ptrExposureTime->GetValue() << "..." << endl;

        // Set gain; gain recorded in decibels
        CFloatPtr ptrGain = nodeMap.GetNode("Gain");
        if (!IsReadable(ptrGain) || !IsWritable(ptrGain))
        {
            cout << "Unable to get or set gain (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        double gainMax = ptrGain->GetMax();

        if (gainToSet > gainMax)
        {
            gainToSet = gainMax;
        }

        ptrGain->SetValue(gainToSet);

        cout << "\tGain set to " << ptrGain->GetValue() << "..." << endl;

        //
        // Set the trigger type for the current state
        //
        // *** NOTES ***
        // It is a requirement of every state to have its trigger source set.
        // The trigger source refers to the moment when the sequencer changes
        // from one state to the next.
        //
        CEnumerationPtr ptrSequencerTriggerSource = nodeMap.GetNode("SequencerTriggerSource");
        if (!IsReadable(ptrSequencerTriggerSource) ||
            !IsWritable(ptrSequencerTriggerSource))
        {
            cout << "Unable to get or set trigger source (node retrieval). Aborting..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrSequencerTriggerSourceFrameStart = ptrSequencerTriggerSource->GetEntryByName("FrameStart");
        if (!IsReadable(ptrSequencerTriggerSourceFrameStart))
        {
            cout << "Unable to get trigger source (enum entry retrieval). Aborting..." << endl << endl;
            return -1;
        }

        ptrSequencerTriggerSource->SetIntValue(static_cast<int64_t>(ptrSequencerTriggerSourceFrameStart->GetValue()));

        cout << "\tTrigger source set to start of frame..." << endl;

        //
        // Set the next state in the sequence
        //
        // *** NOTES ***
        // When setting the next state in the sequence, ensure it does not
        // exceed the maximum and that the states loop appropriately.
        //
        static const int finalSequenceIndex = 4;

        CIntegerPtr ptrSequencerSetNext = nodeMap.GetNode("SequencerSetNext");
        if (!IsWritable(ptrSequencerSetNext))
        {
            cout << "Unable to select next state. Aborting..." << endl << endl;
            return -1;
        }

        if (sequenceNumber == finalSequenceIndex)
        {
            ptrSequencerSetNext->SetValue(0);
        }
        else
        {
            ptrSequencerSetNext->SetValue(sequenceNumber + 1);
        }

        cout << "\tNext state set to " << ptrSequencerSetNext->GetValue() << "..." << endl;

        //
        // Save current state
        //
        // *** NOTES ***
        // Once all appropriate settings have been configured, make sure to
        // save the state to the sequence. Notice that these settings will be
        // lost when the camera is power-cycled.
        //
        CCommandPtr ptrSequencerSetSave = nodeMap.GetNode("SequencerSetSave");
        if (!IsWritable(ptrSequencerSetSave))
        {
            cout << "Unable to save state. Aborting..." << endl << endl;
            return -1;
        }

        ptrSequencerSetSave->Execute();

        cout << "Current state saved..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// Now that the states have all been set, this function readies the camera
// to use the sequencer during image acquisition.
int ConfigureSequencerPartTwo(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn configuration mode off
        //
        // *** NOTES ***
        // Once all desired states have been set, turn sequencer
        // configuration mode off in order to turn sequencer mode on.
        //
        CEnumerationPtr ptrSequencerConfigurationMode = nodeMap.GetNode("SequencerConfigurationMode");
        if (!IsReadable(ptrSequencerConfigurationMode) ||
            !IsWritable(ptrSequencerConfigurationMode))
        {
            PrintRetrieveNodeFailure("node", "SequencerConfigurationMode");
            return -1;
        }

        CEnumEntryPtr ptrSequencerConfigurationModeOff = ptrSequencerConfigurationMode->GetEntryByName("Off");
        if (!IsReadable(ptrSequencerConfigurationModeOff))
        {
            PrintRetrieveNodeFailure("entry", "SequencerConfigurationMode Off");
            return -1;
        }

        ptrSequencerConfigurationMode->SetIntValue(static_cast<int64_t>(ptrSequencerConfigurationModeOff->GetValue()));

        cout << "Sequencer configuration mode disabled..." << endl;

        //
        // Turn sequencer mode on
        //
        // *** NOTES ***
        // After sequencer mode has been turned on, the camera will begin using the
        // saved states in the order that they were set.
        //
        // *** LATER ***
        // Once all images have been captured, disable the sequencer in order
        // to restore the camera to its initial state.
        //
        CEnumerationPtr ptrSequencerMode = nodeMap.GetNode("SequencerMode");
        if (!IsReadable(ptrSequencerMode) ||
            !IsWritable(ptrSequencerMode))
        {
            PrintRetrieveNodeFailure("node", "SequencerMode");
            return -1;
        }

        CEnumEntryPtr ptrSequencerModeOn = ptrSequencerMode->GetEntryByName("On");
        if (!IsReadable(ptrSequencerModeOn))
        {
            PrintRetrieveNodeFailure("entry", "SequencerMode On");
            return -1;
        }

        ptrSequencerMode->SetIntValue(static_cast<int64_t>(ptrSequencerModeOn->GetValue()));

        cout << "Sequencer mode enabled..." << endl;

        //
        // Validate sequencer settings
        //
        // *** NOTES ***
        // Once all states have been set, it is a good idea to
        // validate them. Although this node cannot ensure that the states
        // have been set up correctly, it does ensure that the states have
        // been set up in such a way that the camera can function.
        //
        CEnumerationPtr ptrSequencerConfigurationValid = nodeMap.GetNode("SequencerConfigurationValid");
        if (!IsReadable(ptrSequencerConfigurationValid))
        {
            PrintRetrieveNodeFailure("node", "SequencerConfigurationValid");
            return -1;
        }

        CEnumEntryPtr ptrSequencerConfigurationValidYes = ptrSequencerConfigurationValid->GetEntryByName("Yes");
        if (!IsReadable(ptrSequencerConfigurationValidYes))
        {
            PrintRetrieveNodeFailure("entry", "SequencerConfigurationValid Yes");
            return -1;
        }

        if (ptrSequencerConfigurationValid->GetCurrentEntry()->GetValue() !=
            ptrSequencerConfigurationValidYes->GetValue())
        {
            cout << "Sequencer configuration not valid. Aborting..." << endl << endl;
            return -1;
        }

        cout << "Sequencer configuration valid..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function restores the camera to its default state by turning sequencer mode
// off and re-enabling automatic exposure and gain.
int ResetSequencer(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn sequencer mode back off
        //
        // *** NOTES ***
        // The sequencer is turned off in order to return the camera to its default
        // state.
        //
        CEnumerationPtr ptrSequencerMode = nodeMap.GetNode("SequencerMode");
        if (!IsReadable(ptrSequencerMode) ||
            !IsWritable(ptrSequencerMode))
        {
            PrintRetrieveNodeFailure("node", "SequencerMode");
            return -1;
        }

        CEnumEntryPtr ptrSequencerModeOff = ptrSequencerMode->GetEntryByName("Off");
        if (!IsReadable(ptrSequencerModeOff))
        {
            PrintRetrieveNodeFailure("entry", "SequencerMode Off");
            return -1;
        }

        ptrSequencerMode->SetIntValue(static_cast<int64_t>(ptrSequencerModeOff->GetValue()));

        cout << "Turning off sequencer mode..." << endl;

        //
        // Turn automatic exposure back on
        //
        // *** NOTES ***
        // Automatic exposure is turned on in order to return the camera to its
        // default state.
        //
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (IsReadable(ptrExposureAuto) && IsWritable(ptrExposureAuto))
        {
            CEnumEntryPtr ptrExposureAutoContinuous = ptrExposureAuto->GetEntryByName("Continuous");
            if (IsReadable(ptrExposureAutoContinuous))
            {
                ptrExposureAuto->SetIntValue(static_cast<int64_t>(ptrExposureAutoContinuous->GetValue()));
                cout << "Turning automatic exposure back on..." << endl;
            }
        }

        //
        // Turn automatic gain back on
        //
        // *** NOTES ***
        // Automatic gain is turned on in order to return the camera to its
        // default state.
        //
        CEnumerationPtr ptrGainAuto = nodeMap.GetNode("GainAuto");
        if (IsReadable(ptrGainAuto) && IsWritable(ptrGainAuto))
        {
            CEnumEntryPtr ptrGainAutoContinuous = ptrGainAuto->GetEntryByName("Continuous");
            if (IsReadable(ptrGainAutoContinuous))
            {
                ptrGainAuto->SetIntValue(static_cast<int64_t>(ptrGainAutoContinuous->GetValue()));
                cout << "Turning automatic gain mode back on..." << endl << endl;
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

// This function acquires and saves 10 images from a device; please see
// Acquisition example for more in-depth comments on acquiring images.
int AcquireImages(CameraPtr pCam, INodeMap& nodeMap, INodeMap& nodeMapGenTL, uint64_t timeout)
{
    int result = 0;

    cout << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

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
            cout << "Unable to get acquisition mode to continuous (entry 'continuous' retrieval). Aborting..." << endl
                 << endl;
            return -1;
        }

        int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        cout << "Acquisition mode set to continuous..." << endl;

        // Begin acquiring images
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        // Retrieve device serial number for filename
        gcstring deviceSerialNumber("");

        CStringPtr ptrStringSerial = nodeMapGenTL.GetNode("DeviceSerialNumber");
        if (IsReadable(ptrStringSerial))
        {
            deviceSerialNumber = ptrStringSerial->GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        // Retrieve, convert, and save images
        const unsigned int k_numImages = 10;

        //
        // Create ImageProcessor instance for post processing images
        //
        ImageProcessor processor;

        //
        // Set default image processor color processing method
        //
        // *** NOTES ***
        // By default, if no specific color processing algorithm is set, the image
        // processor will default to NEAREST_NEIGHBOR method.
        //
        processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
        {
            try
            {
                // Retrieve next received image and ensure image completion
                ImagePtr pResultImage = pCam->GetNextImage(timeout);

                if (pResultImage->IsIncomplete())
                {
                    cout << "Image incomplete with image status " << pResultImage->GetImageStatus() << "..." << endl
                         << endl;
                }
                else
                {
                    // Print image information
                    cout << "Grabbed image " << imageCnt << ", width = " << pResultImage->GetWidth()
                         << ", height = " << pResultImage->GetHeight() << endl;

                    // Convert image to mono 8
                    ImagePtr convertedImage = processor.Convert(pResultImage, PixelFormat_Mono8);

                    // Create a unique filename
                    ostringstream filename;

                    filename << "Sequencer-";
                    if (deviceSerialNumber != "")
                    {
                        filename << deviceSerialNumber.c_str() << "-";
                    }
                    filename << imageCnt << ".jpg";

                    // Save image
                    convertedImage->Save(filename.str().c_str());

                    cout << "Image saved at " << filename.str() << endl;
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
        result = -1;
    }

    return result;
}

// This function acts very similarly to the RunSingleCamera() functions of other
// examples, except that the values for the sequences are also calculated here;
// please see NodeMapInfo example for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam)
{
    int result = 0;
    int err = 0;

    try
    {
        // Retrieve GenTL nodemap and print device information
        INodeMap& nodeMapGenTL = pCam->GetTLDeviceNodeMap();

        result = PrintDeviceInfo(nodeMapGenTL);

        // Initialize camera
        pCam->Init();

        // Retrieve GenICam nodemap
        INodeMap& nodeMap = pCam->GetNodeMap();

        // Configure sequencer to be ready to set sequences
        err = ConfigureSequencerPartOne(nodeMap);
        if (err < 0)
        {
            return err;
        }

        //
        // Set sequences
        //
        // *** NOTES ***
        // In the following section, the sequencer values are calculated. This
        // section does not appear in the configuration, as the values
        // calculated are somewhat arbitrary: width and height are both set to
        // 25% of their maximum values, incrementing by 10%; exposure time is
        // set to its minimum, also incrementing by 10% of its maximum; and gain
        // is set to its minimum, incrementing by 2% of its maximum.
        //
        const unsigned int k_numSequences = 5;

        // Retrieve maximum width; width recorded in pixels
        CIntegerPtr ptrWidth = nodeMap.GetNode("Width");
        if (!IsReadable(ptrWidth))
        {
            cout << "Unable to retrieve maximum width. Aborting..." << endl << endl;
            return -1;
        }

        const int64_t widthMax = ptrWidth->GetMax();

        // Retrieve maximum height; height recorded in pixels
        CIntegerPtr ptrHeight = nodeMap.GetNode("Height");
        if (!IsReadable(ptrHeight))
        {
            cout << "Unable to retrieve maximum height. Aborting..." << endl << endl;
            return -1;
        }

        const int64_t heightMax = ptrHeight->GetMax();

        // Retrieve maximum exposure time; exposure time recorded in microseconds
        const double k_exposureTimeMaxToSet = 2000000;

        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime))
        {
            cout << "Unable to retrieve maximum exposure time. Aborting..." << endl << endl;
            return -1;
        }

        double exposureTimeMax = ptrExposureTime->GetMax();

        if (exposureTimeMax > k_exposureTimeMaxToSet)
        {
            exposureTimeMax = k_exposureTimeMaxToSet;
        }

        // Retrieve maximum gain; gain recorded in decibels
        CFloatPtr ptrGain = nodeMap.GetNode("Gain");
        if (!IsReadable(ptrGain))
        {
            cout << "Unable to retrieve maximum gain. Aborting..." << endl << endl;
            return -1;
        }

        const double gainMax = ptrGain->GetMax();

        // Set initial values
        int64_t widthToSet = widthMax / 4;
        int64_t heightToSet = heightMax / 4;
        double exposureTimeToSet = ptrExposureTime->GetMin();
        double gainToSet = ptrGain->GetMin();

        // Set custom values of each sequence
        for (unsigned int sequenceNumber = 0; sequenceNumber < k_numSequences; sequenceNumber++)
        {
            err = SetSingleState(nodeMap, sequenceNumber, widthToSet, heightToSet, exposureTimeToSet, gainToSet);
            if (err < 0)
            {
                return err;
            }

            // Increment values
            widthToSet += widthMax / 10;
            heightToSet += heightMax / 10;
            exposureTimeToSet += exposureTimeMax / 10.0;
            gainToSet += gainMax / 50.0;
        }

        // Calculate appropriate acquisition grab timeout window based on exposure time
        // Note: exposureTimeToSet is in microseconds and needs to be converted to milliseconds
        uint64_t timeout = static_cast<uint64_t>((exposureTimeToSet / 1000) + 1000);

        // Configure sequencer to acquire images
        err = ConfigureSequencerPartTwo(nodeMap);
        if (err < 0)
        {
            return err;
        }

        // Acquire images
        result = result | AcquireImages(pCam, nodeMap, nodeMapGenTL, timeout);

        // Reset sequencer
        result = result | ResetSequencer(nodeMap);

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