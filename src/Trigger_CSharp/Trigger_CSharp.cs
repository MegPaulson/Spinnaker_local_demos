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
 *  @example Trigger_CSharp.cs
 *
 *  @brief Trigger_CSharp.cs shows how to trigger the camera. It relies on
 *  information provided in the Enumeration_CSharp, Acquisition_CSharp, and
 *  NodeMapInfo_CSharp examples.
 *
 *	It can also be helpful to familiarize yourself with the
 *  ImageFormatControl_CSharp and Exposure_CSharp examples. As they are somewhat
 *  shorter and simpler, either provides a strong introduction to camera
 *  customization.
 *
 *	This example shows the process of configuring, using, and cleaning up a
 *  camera for use with both a software and a hardware trigger.
 * 
 *  Please leave us feedback at: https://www.surveymonkey.com/r/TDYMVAPI
 *  More source code examples at: https://github.com/Teledyne-MV/Spinnaker-Examples
 *  Need help? Check out our forum at: https://teledynevisionsolutions.zendesk.com/hc/en-us/community/topics
 */

using System;
using System.IO;
using System.Collections.Generic;
using SpinnakerNET;
using SpinnakerNET.GenApi;

namespace Trigger_CSharp
{
    class Program
    {
        // Use the following enum and global static variable to select whether
        // a software or hardware trigger is used.
        enum triggerType
        {
            Software,
            Hardware
        }

        static triggerType chosenTrigger = triggerType.Software;

        // This function configures the camera to use a trigger. First, trigger
        // mode is set to off in order to select the trigger source. Once the
        // trigger source has been selected, trigger mode is then enabled,
        // which has the camera capture only a single image upon the execution
        // of the chosen trigger.
        int ConfigureTrigger(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                Console.WriteLine("\n\n*** CONFIGURING TRIGGER ***\n\n");

                Console.WriteLine(
                    "Note that if the application / user software triggers faster than frame time, the trigger may be dropped / skipped by the camera.");
                Console.WriteLine(
                    "If several frames are needed per trigger, a more reliable alternative for such case, is to use the multi-frame mode.\n");

                if (chosenTrigger == triggerType.Software)
                {
                    Console.WriteLine("Software trigger chosen...\n");
                }
                else if (chosenTrigger == triggerType.Hardware)
                {
                    Console.WriteLine("Hardware trigger chosen...\n");
                }

                //
                // Ensure trigger mode off
                //
                // *** NOTES ***
                // The trigger must be disabled in order to configure whether
                // the source is software or hardware.
                //
                IEnum iTriggerMode = nodeMap.GetNode<IEnum>("TriggerMode");
                if (iTriggerMode == null || !iTriggerMode.IsWritable || !iTriggerMode.IsReadable)
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

                //
                // Set TriggerSelector to FrameStart
                //
                // *** NOTES ***
                // For this example, the trigger selector should be set to frame start.
                // This is the default for most cameras.
                //
                IEnum iTriggerSelector = nodeMap.GetNode<IEnum>("TriggerSelector");
                if (iTriggerSelector == null || !iTriggerSelector.IsWritable || !iTriggerSelector.IsReadable)
                {
                    Console.WriteLine("Unable to set trigger selector (enum retrieval). Aborting...");
                    return -1;
                }

                // Set trigger mode to software
                IEnumEntry iTriggerSelectorFrameStarts = iTriggerSelector.GetEntryByName("FrameStart");
                if (iTriggerSelectorFrameStarts == null || !iTriggerSelectorFrameStarts.IsReadable)
                {
                    Console.WriteLine("Unable to set software trigger selector (entry retrieval). Aborting...");
                    return -1;
                }

                iTriggerSelector.Value = iTriggerSelectorFrameStarts.Value;

                Console.WriteLine("Trigger selector set to frame start...");

                //
                // Select trigger source
                //
                // *** NOTES ***
                // The trigger source must be set to hardware or software while
                // trigger mode is off.
                //
                IEnum iTriggerSource = nodeMap.GetNode<IEnum>("TriggerSource");
                if (iTriggerSource == null || !iTriggerSource.IsWritable || !iTriggerSource.IsReadable)
                {
                    Console.WriteLine("Unable to set trigger mode (enum retrieval). Aborting...");
                    return -1;
                }

                if (chosenTrigger == triggerType.Software)
                {
                    // Set trigger mode to software
                    IEnumEntry iTriggerSourceSoftware = iTriggerSource.GetEntryByName("Software");
                    if (iTriggerSourceSoftware == null || !iTriggerSourceSoftware.IsReadable)
                    {
                        Console.WriteLine("Unable to set software trigger mode (entry retrieval). Aborting...");
                        return -1;
                    }

                    iTriggerSource.Value = iTriggerSourceSoftware.Value;

                    Console.WriteLine("Trigger source set to software...");
                }
                else if (chosenTrigger == triggerType.Hardware)
                {
                    // Set trigger mode to hardware ('Line0')
                    IEnumEntry iTriggerSourceHardware = iTriggerSource.GetEntryByName("Line0");
                    if (iTriggerSourceHardware == null || !iTriggerSourceHardware.IsReadable)
                    {
                        Console.WriteLine("Unable to set hardware trigger mode (entry retrieval). Aborting...");
                        return -1;
                    }

                    iTriggerSource.Value = iTriggerSourceHardware.Value;

                    Console.WriteLine("Trigger source set to hardware...");
                }

                //
                // Turn trigger mode on
                //
                // *** LATER ***
                // Once the appropriate trigger source has been set, turn
                // trigger mode on in order to retrieve images using the
                // trigger.
                //

                IEnumEntry iTriggerModeOn = iTriggerMode.GetEntryByName("On");
                if (iTriggerModeOn == null || !iTriggerModeOn.IsReadable)
                {
                    Console.WriteLine("Unable to enable trigger mode (entry retrieval). Aborting...");
                    return -1;
                }

                iTriggerMode.Value = iTriggerModeOn.Value;

                // NOTE: Blackfly and Flea3 GEV cameras need 1 second delay after trigger mode is turned on

                Console.WriteLine("Trigger mode enabled...");
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function retrieves a single image using the trigger. In this
        // example, only a single image is captured and made available for
        // acquisition - as such, attempting to acquire two images for a single
        // trigger execution would cause the example to hang. This is different
        // from other examples, whereby a constant stream of images are being
        // captured and made available for image acquisition.
        int GrabNextImageByTrigger(INodeMap nodeMap, IManagedCamera cam)
        {
            int result = 0;

            try
            {
                //
                // Use trigger to capture image
                //
                // *** NOTES ***
                // The software trigger only feigns being executed by the Enter
                // key; what might not be immediately apparent is that there is
                // not a continuous stream of images being captured; in other
                // examples that acquire images, the camera captures a
                // continuous stream of images. When an image is retrieved, it
                // is plucked from the stream.
                //
                if (chosenTrigger == triggerType.Software)
                {
                    // Get user input
                    Console.WriteLine("Press the Enter key to initiate software trigger.");
                    Console.ReadLine();

                    // Execute software trigger
                    ICommand iTriggerSoftware = nodeMap.GetNode<ICommand>("TriggerSoftware");
                    if (iTriggerSoftware == null || !iTriggerSoftware.IsWritable)
                    {
                        Console.WriteLine("Unable to execute trigger. Aborting...");
                        return -1;
                    }

                    iTriggerSoftware.Execute();

                    // NOTE: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger
                }
                else if (chosenTrigger == triggerType.Hardware)
                {
                    // Execute hardware trigger
                    Console.WriteLine("Use the hardware to trigger image acquisition.");
                }
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function returns the camera to a normal state by turning off
        // trigger mode.
        int ResetTrigger(INodeMap nodeMap)
        {
            int result = 0;

            try
            {
                //
                // Turn trigger mode back off
                //
                // *** NOTES ***
                // Once all images have been captured, turn trigger mode back
                // off to restore the camera to a clean state.
                //
                IEnum iTriggerMode = nodeMap.GetNode<IEnum>("TriggerMode");
                if (iTriggerMode == null || !iTriggerMode.IsWritable || !iTriggerMode.IsReadable)
                {
                    Console.WriteLine("Unable to disable trigger mode (enum retrieval). Non-fatal error...");
                    return -1;
                }

                IEnumEntry iTriggerModeOff = iTriggerMode.GetEntryByName("Off");
                if (iTriggerModeOff == null || !iTriggerModeOff.IsReadable)
                {
                    Console.WriteLine("Unable to disable trigger mode (entry retrieval). Non-fatal error...");
                    return -1;
                }

                iTriggerMode.Value = iTriggerModeOff.Value;

                Console.WriteLine("Trigger mode disabled...\n");
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
        int AcquireImages(IManagedCamera cam, INodeMap nodeMap, INodeMap nodeMapTLDevice)
        {
            int result = 0;

            Console.WriteLine("\n*** IMAGE ACQUISITION ***\n");

            try
            {
                // Set acquisition mode to continuous
                IEnum iAcquisitionMode = nodeMap.GetNode<IEnum>("AcquisitionMode");
                if (iAcquisitionMode == null || !iAcquisitionMode.IsWritable || !iAcquisitionMode.IsReadable)
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

                // Begin acquiring images
                cam.BeginAcquisition();

                Console.WriteLine("Acquiring images...");

                // Retrieve device serial number for filename
                String deviceSerialNumber = "";

                IString iDeviceSerialNumber = nodeMapTLDevice.GetNode<IString>("DeviceSerialNumber");
                if (iDeviceSerialNumber != null && iDeviceSerialNumber.IsReadable)
                {
                    deviceSerialNumber = iDeviceSerialNumber.Value;

                    Console.WriteLine("Device serial number retrieved as {0}...", deviceSerialNumber);
                }
                Console.WriteLine();

                // Retrieve, convert, and save images
                const int NumImages = 10;

                //
                // Create ImageProcessor instance for post processing images
                //
                IManagedImageProcessor processor = new ManagedImageProcessor();

                //
                // Set default image processor color processing method
                //
                // *** NOTES ***
                // By default, if no specific color processing algorithm is set, the image
                // processor will default to NEAREST_NEIGHBOR method.
                //
                processor.SetColorProcessing(ColorProcessingAlgorithm.HQ_LINEAR);

                for (int imageCnt = 0; imageCnt < NumImages; imageCnt++)
                {
                    try
                    {
                        // Retrieve next received image and ensure image completion
                        result = result | GrabNextImageByTrigger(nodeMap, cam);

                        using(IManagedImage rawImage = cam.GetNextImage(1000))
                        {
                            if (rawImage.IsIncomplete)
                            {
                                Console.WriteLine("Image incomplete with image status {0}...", rawImage.ImageStatus);
                            }
                            else
                            {
                                // Print image information
                                Console.WriteLine(
                                    "Grabbed image {0}, width = {1}, height = {2}",
                                    imageCnt,
                                    rawImage.Width,
                                    rawImage.Height);

                                // Convert image to mono 8
                                using(
                                    IManagedImage convertedImage = processor.Convert(rawImage, PixelFormatEnums.Mono8))
                                {
                                    // Create unique file name
                                    String filename = "Trigger-CSharp-";
                                    if (deviceSerialNumber != "")
                                    {
                                        filename = filename + deviceSerialNumber + "-";
                                    }
                                    filename = filename + imageCnt + ".jpg";

                                    // Save image
                                    convertedImage.Save(filename);

                                    Console.WriteLine("Image saved at {0}\n", filename);
                                }
                            }
                        }
                    }
                    catch (SpinnakerException ex)
                    {
                        Console.WriteLine("Error: {0}", ex.Message);
                        result = -1;
                    }
                }

                // End acquisition
                cam.EndAcquisition();
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
            int index = 0;

            foreach(IManagedCamera managedCamera in camList) using(managedCamera)
            {
                Console.WriteLine("Running example for camera {0}...", index);

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

                Console.WriteLine("Camera {0} example complete...\n", index++);
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