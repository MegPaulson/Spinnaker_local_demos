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
 *  @example BufferHandling.cs
 *
 *  @brief BufferHandling_CSharp.cs shows how the different buffer handling modes work.
 *  It relies on information provided in the Acquisition and Trigger examples.
 *
 *  Buffer handling determines the ordering in which images are retrieved, and
 *  what occurs when an image is transmitted while the buffer is full.  There are
 *  four different buffer handling modes available; NewestFirst, NewestOnly,
 *  OldestFirst and OldestFirstOverwrite.
 *
 *  This example explores retrieving images in a set pattern; triggering the camera
 *  while not retrieving an image (letting the buffer fill up), and retrieving
 *  images while not triggering.  We cycle through the different buffer handling
 *  modes to see which images are retrieved, confirming their identites via their
 *  Frame ID values.
 *
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

using System;
using System.IO;
using System.Collections.Generic;
using System.Threading;
using SpinnakerNET.GenApi;
using SpinnakerNET;

namespace BufferHandling_CSharp
{
    class Program
    {
        static readonly int numBuffers = 3;
        static readonly int numTriggers = 6;
        static readonly int numLoops = 9;

        // This function configures the camera to use a trigger. First, trigger mode is
        // set to off in order to select the trigger source. Once the trigger source
        // has been selected, trigger mode is then enabled, which has the camera
        // capture only a single image upon the execution of the trigger.
        static int ConfigureTrigger(INodeMap nodeMap)
        {
            int result = 0;

            Console.WriteLine("\n*** CONFIGURING TRIGGER ***\n");

            try
            {
                //
                // Ensure trigger mode off
                //
                // *** NOTES ***
                // The trigger must be disabled in order to configure the
                // trigger source.
                //
                IEnum iTriggerMode = nodeMap.GetNode<IEnum>("TriggerMode");
                if (iTriggerMode == null || !iTriggerMode.IsReadable || !iTriggerMode.IsWritable)
                {
                    Console.WriteLine("Unable to disable trigger mode (enum retrieval). Aborting...");
                    return -1;
                }

                IEnumEntry iTriggerModeOff = iTriggerMode.GetEntryByName("Off");
                if (iTriggerModeOff == null || !iTriggerModeOff.IsReadable)
                {
                    Console.WriteLine("Unable to disable trigger mode (entry retrieval). Aborting...");
                    return -1;
                }

                iTriggerMode.Value = iTriggerModeOff.Value;

                Console.WriteLine("Trigger mode disabled...");

                // Set trigger source to software
                IEnum iTriggerSource = nodeMap.GetNode<IEnum>("TriggerSource");
                if (iTriggerSource == null || !iTriggerSource.IsReadable || !iTriggerSource.IsWritable)
                {
                    Console.WriteLine("Unable to set trigger mode (enum retrieval). Aborting...");
                    return -1;
                }

                // Set trigger mode to software
                IEnumEntry iTriggerSourceSoftware = iTriggerSource.GetEntryByName("Software");
                if (iTriggerSourceSoftware == null || !iTriggerSourceSoftware.IsReadable)
                {
                    Console.WriteLine("Unable to set software trigger mode (entry retrieval). Aborting...");
                    return -1;
                }

                iTriggerSource.Value = iTriggerSourceSoftware.Value;

                Console.WriteLine("Trigger source set to software...");

                // Turn trigger mode on
                IEnumEntry iTriggerModeOn = iTriggerMode.GetEntryByName("On");
                if (iTriggerModeOn == null || !iTriggerModeOn.IsReadable)
                {
                    Console.WriteLine("Unable to enable trigger mode (entry retrieval). Aborting...");
                    return -1;
                }

                iTriggerMode.Value = iTriggerModeOn.Value;

                Console.WriteLine("Trigger mode turned back on...");
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function retrieves a single image using the trigger. In this example,
        // only a single image is captured and made available for acquisition - as such,
        // attempting to acquire two images for a single trigger execution would cause
        // the example to hang. This is different from other examples, whereby a
        // constant stream of images are being captured and made available for image
        // acquisition.
        static int GrabNextImageByTrigger(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                // Execute software trigger
                ICommand softwareTriggerCommand = nodeMap.GetNode<ICommand>("TriggerSoftware");
                if (softwareTriggerCommand == null || !softwareTriggerCommand.IsWritable)
                {
                    Console.WriteLine("Unable to execute trigger. Aborting...");
                    return -1;
                }

                softwareTriggerCommand.Execute();
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function returns the camera to a normal state by turning off trigger
        // mode.
        static int ResetTrigger(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                //
                // Turn trigger mode back off
                //
                // *** NOTES ***
                // Once all images have been captured, turn trigger mode back off to
                // restore the camera to a clean state.
                //
                IEnum iTriggerMode = nodeMap.GetNode<IEnum>("TriggerMode");
                if (iTriggerMode == null || !iTriggerMode.IsWritable)
                {
                    Console.WriteLine("Unable to disable trigger mode (enum retrieval). Aborting...");
                    return -1;
                }

                IEnumEntry iTriggerModeOff = iTriggerMode.GetEntryByName("Off");
                if (iTriggerModeOff == null || !iTriggerModeOff.IsReadable)
                {
                    Console.WriteLine("Unable to disable trigger mode (entry retrieval). Aborting...");
                    return -1;
                }

                iTriggerMode.Value = iTriggerModeOff.Value;

                Console.WriteLine("Trigger mode disabled...");
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function prints the device information of the camera from the
        // transport layer; please see NodeMapInfo_CSharp example for more
        // in-depth comments on printing device information from the nodemap.
        static int PrintDeviceInfo(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                Console.WriteLine("\n*** DEVICE INFORMATION ***\n");

                ICategory category = nodeMap.GetNode<ICategory>("DeviceInformation");
                if (category != null && category.IsReadable)
                {
                    for (int i = 0; i < category.Children.Length; i++)
                    {
                        Console.WriteLine(
                            "{0}: {1}",
                            category.Children[i].Name,
                            (category.Children[i].IsReadable ? category.Children[i].ToString()
                             : "Node not available"));
                    }
                    Console.WriteLine();
                }
                else
                {
                    Console.WriteLine("Device control information not available.");
                }
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function acquires and saves 10 images from a device; please see
        // Acquisition_CSharp example for more in-depth comments on the
        // acquisition of images.
        static int AcquireImages(IManagedCamera cam, INodeMap nodeMap, INodeMap nodeMapTLDevice)
        {
            int result = 0;

            Console.WriteLine("\n*** IMAGE ACQUISITION ***\n");

            try
            {
                // Set acquisition mode to continuous
                IEnum iAcquisitionMode = nodeMap.GetNode<IEnum>("AcquisitionMode");
                if (iAcquisitionMode == null || !iAcquisitionMode.IsReadable || !iAcquisitionMode.IsWritable)
                {
                    Console.WriteLine("Unable to set acquisition mode to continuous (node retrieval). Aborting...\n");
                    return -1;
                }

                IEnumEntry iAcquisitionModeContinuous = iAcquisitionMode.GetEntryByName("Continuous");
                if (iAcquisitionModeContinuous == null || !iAcquisitionModeContinuous.IsReadable)
                {
                    Console.WriteLine(
                        "Unable to set acquisition mode to continuous (enum entry retrieval). Aborting...\n");
                    return -1;
                }

                iAcquisitionMode.Value = iAcquisitionModeContinuous.Symbolic;

                Console.WriteLine("Acquisition mode set to continuous...");

                // Retrieve device serial number for filename
                String deviceSerialNumber = "";

                IString iDeviceSerialNumber = nodeMapTLDevice.GetNode<IString>("DeviceSerialNumber");
                if (iDeviceSerialNumber != null && iDeviceSerialNumber.IsReadable)
                {
                    deviceSerialNumber = iDeviceSerialNumber.Value;

                    Console.WriteLine("Device serial number retrieved as {0}...", deviceSerialNumber);
                }
                Console.WriteLine();

                // Retrieve Stream Parameters device nodemap
                INodeMap sNodeMap = cam.GetTLStreamNodeMap();

                // Retrieve Buffer Handling Mode Information
                IEnum handlingMode = sNodeMap.GetNode<IEnum>("StreamBufferHandlingMode");
                if (handlingMode == null || !handlingMode.IsWritable)
                {
                    Console.WriteLine("Unable to set Buffer Handling mode (node retrieval). Aborting...");
                    return -1;
                }

                // Set stream buffer Count Mode to manual
                IEnum streamBufferCountMode = sNodeMap.GetNode<IEnum>("StreamBufferCountMode");
                if (streamBufferCountMode == null || !streamBufferCountMode.IsReadable || !streamBufferCountMode.IsWritable)
                {
                    Console.WriteLine("Unable to set Buffer Count Mode (node retrieval). Aborting...");
                    return -1;
                }

                IEnumEntry streamBufferCountModeManual = streamBufferCountMode.GetEntryByName("Manual");
                if (streamBufferCountModeManual == null || !streamBufferCountModeManual.IsReadable)
                {
                    Console.WriteLine("Unable to set Buffer Count Mode (entry retrieval). Aborting...");
                    return -1;
                }

                streamBufferCountMode.Value = streamBufferCountModeManual.Value;

                Console.WriteLine("Stream Buffer Count Mode set to manual...");

                // Retrieve and modify Stream Buffer Count
                IInteger bufferCount = sNodeMap.GetNode<IInteger>("StreamBufferCountManual");
                if (bufferCount == null || !bufferCount.IsWritable)
                {
                    Console.WriteLine("Unable to set Buffer Count (Integer node retrieval). Aborting...");
                    return -1;
                }

                // Display buffer info
                Console.WriteLine("Default Buffer Count: {0}", bufferCount.Value);
                Console.WriteLine("Maximum Buffer Count: {0}", bufferCount.Max);

                bufferCount.Value = numBuffers;

                Console.WriteLine("Buffer count now set to : {0}", bufferCount.Value);

                Console.WriteLine(
                    "Camera will be triggered {0} times in a row before {1} images will be retrieved",
                    numTriggers,
                    numLoops - numTriggers);
                Console.WriteLine("Note - Buffer behaviour is different for USB3 and GigE cameras");
                Console.WriteLine("     - USB3 cameras buffer images internally if no host buffers are available");
                Console.WriteLine("     - Once the stream buffer is released, the USB3 camera will fill that buffer");
                Console.WriteLine("     - GigE cameras do not buffer images");

                for (int i = 0; i < 4; i++)
                {
                    IEnumEntry handlingModeEntry = null;

                    switch (i)
                    {
                        case 0:
                            handlingModeEntry = handlingMode.GetEntryByName("NewestFirst");
                            break;
                        case 1:
                            handlingModeEntry = handlingMode.GetEntryByName("NewestOnly");
                            break;
                        case 2:
                            handlingModeEntry = handlingMode.GetEntryByName("OldestFirst");
                            break;
                        case 3:
                            handlingModeEntry = handlingMode.GetEntryByName("OldestFirstOverwrite");
                            break;
                    }

                    handlingMode.Value = handlingModeEntry.Value;

                    Console.WriteLine("Buffer handline mode has been set to {0}", handlingModeEntry.Value);

                    // Begin capturing images
                    cam.BeginAcquisition();

                    if (i == 0)
                    {
                        // Sleep for one second; only necessary when using non-BFS/ORX cameras on startup
                        Thread.Sleep(1000);
                    }

                    try
                    {
                        // Software trigger the camera and then save the images
                        for (int loopCnt = 0; loopCnt < numLoops; loopCnt++)
                        {
                            IManagedImage image = null;
                            if (loopCnt < numTriggers)
                            {
                                result = result | GrabNextImageByTrigger(nodeMap);
                                Console.WriteLine("Camera triggered. No image grabbed.");
                            }
                            else
                            {
                                Console.WriteLine("No trigger. Grabbing image {0}", loopCnt - numTriggers);
                                image = cam.GetNextImage(500);
                                if (image.IsIncomplete)
                                {
                                    Console.WriteLine("Image incomplete with image status {0} ...", image.ImageStatus);
                                }
                            }

                            if (loopCnt >= numTriggers)
                            {
                                // Retrieve Frame ID
                                Console.WriteLine("Frame ID: {0}", image.FrameID);

                                // Create a unique filename
                                String filename = handlingModeEntry.Symbolic + "-" + deviceSerialNumber + "-" +
                                                  (loopCnt - numTriggers) + ".jpg";

                                // Save image
                                image.Save(filename);
                                Console.WriteLine("Image saved at {0}", filename);

                                // Release image
                                image.Release();
                            }

                            Thread.Sleep(250);
                        }
                    }
                    catch (SpinnakerException ex)
                    {
                        Console.WriteLine("Error: {0}", ex.Message);

                        // For overwrite buffer handling modes such as NewestOnly and OldestFirstOverwrite, it is expected that
                        // two buffers are used to cycle images within spinnaker acquisition engine. So when buffer count is set
                        // to three, only one image is expected to return to the user so fetching another image without additional 
                        // triggers will return an error
                        if (handlingModeEntry.Symbolic == "NewestOnly" ||
                            handlingModeEntry.Symbolic == "OldestFirstOverwrite" )
                        {
                            Console.WriteLine(
                                "Error should occur when grabbing image 1 with handling mode set to NewestOnly or OldestFirstOverwrite");
                        }

                        result = -1;
                    }

                    cam.EndAcquisition();
                }
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function acts as the body of the example; please see
        // NodeMapInfo_CSharp example for more in-depth comments on setting up
        // cameras.
        int RunSingleCamera(IManagedCamera cam)
        {
            int result = 0;
            int err = 0;

            try
            {
                // Retrieve TL device nodemap and print device information
                INodeMap nodeMapTLDevice = cam.GetTLDeviceNodeMap();

                result = PrintDeviceInfo(nodeMapTLDevice);

                // Initialize camera
                cam.Init();

                // Retrieve GenICam nodemap
                INodeMap nodeMap = cam.GetNodeMap();

                // Configure trigger
                err = ConfigureTrigger(nodeMap);
                if (err < 0)
                {
                    return err;
                }

                // Acquire images
                result = result | AcquireImages(cam, nodeMap, nodeMapTLDevice);

                // Reset trigger
                result = result | ResetTrigger(nodeMap);

                // Deinitialize camera
                cam.DeInit();
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // Example entry point; please see Enumeration_CSharp example for more
        // in-depth comments on preparing and cleaning up the system.
        static int Main(string[] args)
        {
            int result = 0;

            Program program = new Program();

            // Since this application saves images in the current folder
            // we must ensure that we have permission to write to this folder.
            // If we do not have permission, fail right away.
            FileStream fileStream;
            try
            {
                fileStream = new FileStream(@"test.txt", FileMode.Create);
                fileStream.Close();
                File.Delete("test.txt");
            }
            catch
            {
                Console.WriteLine("Failed to create file in current folder. Please check permissions.");
                Console.WriteLine("Press enter to exit...");
                Console.ReadLine();
                return -1;
            }

            // Retrieve singleton reference to system object
            ManagedSystem system = new ManagedSystem();

            // Print out current library version
            LibraryVersion spinVersion = system.GetLibraryVersion();
            Console.WriteLine(
                "Spinnaker library version: {0}.{1}.{2}.{3}\n\n",
                spinVersion.major,
                spinVersion.minor,
                spinVersion.type,
                spinVersion.build);

            // Retrieve list of cameras from the system
            ManagedCameraList camList = system.GetCameras();

            Console.WriteLine("Number of cameras detected: {0}\n\n", camList.Count);

            // Finish if there are no cameras
            if (camList.Count == 0)
            {
                // Clear camera list before releasing system
                camList.Clear();

                // Release system
                system.Dispose();

                Console.WriteLine("Not enough cameras!");
                Console.WriteLine("Done! Press Enter to exit...");
                Console.ReadLine();

                return -1;
            }

            // Run example on each camera
            int i = 0;

            foreach(IManagedCamera managedCamera in camList) using(managedCamera)
            {
                Console.WriteLine("Running example for camera {0}...", i);

                try
                {
                    // Run example
                    result = result | program.RunSingleCamera(managedCamera);
                }
                catch (SpinnakerException ex)
                {
                    Console.WriteLine("Error: {0}", ex.Message);
                    result = -1;
                }

                Console.WriteLine("Camera {0} example complete...\n", i++);
            }

            // Clear camera list before releasing system
            camList.Clear();

            // Release system
            system.Dispose();

            Console.WriteLine("\nDone! Press Enter to exit...");
            Console.ReadLine();

            return result;
        }
    }
}
