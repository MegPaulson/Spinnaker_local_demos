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
 *	@example DeviceEvents_CSharp.cs
 *
 *	@brief DeviceEvents_CSharp.cs shows how to create a handler to access device
 *	events. It relies on information provided in the Enumeration_CSharp,
 *  Acquisition_CSharp, and NodeMapInfo_CSharp examples.
 *
 *	It can also be helpful to familiarize yourself with the
 *  NodeMapCallback_CSharp example, as callbacks follow the same general
 *  procedure as events, but with a few less steps.
 *
 *	Device events can be thought of as camera-related events. This example
 *	creates a user-defined class, DeviceEventHandler, that inherits from the
 *	Spinnaker C# class, ManagedDeviceEventHandler. DeviceEventHandler, the child class,
 *  allows the user to define any properties, parameters, and the event itself
 *  while ManagedDeviceEventHandler, the parent class, allows the child class to
 *  appropriately interface with the C# Spinnaker SDK.
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

namespace DeviceEvents_CSharp
{
    class Program
    {
        // Use the following enum and global static variable to select whether
        // device events will be registered universally to all events or
        // specifically to exposure end events.
        enum eventType
        {
            Generic,
            Specific
        }

        static eventType chosenEvent = eventType.Generic;

        // This class defines the properties, parameters, and the event handler itself.
        // Take a moment to notice what parts of the class are mandatory, and
        // what have been added for demonstration purposes. First, any class
        // used to define device events must inherit from DeviceEvent. Second,
        // the method signature of OnDeviceEvent() must also be consistent.
        // Everything else - including the constructor, properties, and body of
        // OnDeviceEvent() - are particular to the example.
        public class DeviceEventListener : ManagedDeviceEventHandler
        {
            private string specificEvent;
            private int count;

            // This constructor registers an event name to be used on device
            // events.
            public DeviceEventListener(string eventName)
            {
                specificEvent = eventName;
                count = 0;
            }

            // This method is designed to demonstrate the difference between
            // registering all device events generally versus registering only a
            // specific device event. If registered specifically, the else-block
            // will never run because only the specific event will call this
            // method.
            protected override void OnDeviceEvent(string eventName)
            {
                // Check that device event is registered
                if (eventName == specificEvent)
                {
                    // Print information on specified device event
                    Console.WriteLine(
                        "\tDevice event {0} with ID {1} number {2}...",
                        GetDeviceEventName(),
                        GetDeviceEventID(),
                        ++count);
                }
                else
                {
                    // Print no information on non-specified information
                    Console.WriteLine("\tDevice event occurred; not {0}; ignoring...", specificEvent);
                }
            }
        }

        // This function configures the example to execute device events by
        // enabling all types of device events, and then creating and registering
        // a device event that only concerns itself with an end of exposure event.
        int ConfigureDeviceEvents(INodeMap nodeMap, IManagedCamera cam, ref DeviceEventListener deviceEventListener)
        {
            int result = 0;

            Console.WriteLine("\n\n**** CONFIGURING DEVICE EVENTS***\n");

            try
            {
                //
                // Retrieve device event selector
                //
                // *** NOTES ***
                // Each type of device event must be enabled individually. This
                // is done by retrieving "EventSelector" (an enumeration node)
                // and then enabling the device event on "EventNotification"
                // (another enumeration node).
                //
                // This example only deals with exposure end events. However,
                // instead of only enabling exposure end events with a simpler
                // device event function, all device events are enabled while
                // the device event handler deals with ensuring that only
                // exposure end events are considered. A more standard use-case
                // might be to enable only the events of interest.
                //
                IEnum iEventSelector = nodeMap.GetNode<IEnum>("EventSelector");
                if (iEventSelector == null || !iEventSelector.IsReadable)
                {
                    Console.WriteLine("Unable to fetch event enumeration entries. Aborting...");
                    return -1;
                }

                var entries = iEventSelector.Entries;

                Console.WriteLine("Enabling event selector entries...");

                //
                // Enable device events
                //
                // *** NOTES ***
                // In order to enable a device event, the event selector and
                // event notification nodes (both of type enumeration) must work
                // in unison. The desired event must first be selected on the
                // event selector node and then enabled on the event
                // notification node.
                //
                for (int i = 0; i < entries.Length; i++)
                {
                    // Select entry on selector node
                    IEnumEntry iEnumEntry = entries[i];
                    if (!iEnumEntry.IsReadable)
                    {
                        // Skip if node fails
                        result = -1;
                        continue;
                    }

                    if (!iEventSelector.IsWritable)
                    {
                        Console.WriteLine("Unable to write to event selector node. Aborting...");
                        return -1;
                    }

                    iEventSelector.Value = iEnumEntry.Value;

                    // Retrieve event notification node (an enumeration node)
                    IEnum iEventNotification = nodeMap.GetNode<IEnum>("EventNotification");
                    if (iEventNotification == null || !iEventNotification.IsWritable)
                    {
                        // Skip if node fails
                        result = -1;
                        continue;
                    }

                    // Retrieve entry node to enable device event
                    IEnumEntry iEventNotificationOn = iEventNotification.GetEntryByName("On");
                    if (iEventNotificationOn == null || !iEventNotification.IsReadable)
                    {
                        // Skip if node fails
                        result = -1;
                        continue;
                    }

                    iEventNotification.Value = iEventNotificationOn.Value;

                    Console.WriteLine("\t{0}: enabled...", iEnumEntry.DisplayName);
                }

                //
                // Create device event handler
                //
                // *** NOTES ***
                // The class has been designed to take in the name of an event.
                // If all event handlers are registered generically, all event types
                // will trigger a device event; on the other hand, if an event
                // is registered specifically, only that event will trigger an
                // event.
                //
                deviceEventListener = new DeviceEventListener("EventExposureEnd");

                //
                // Register device event handler
                //
                // *** NOTES ***
                // Device events are registered to cameras. If there are multiple
                // cameras, each camera must have any device event handlers registered to
                // it separately. Also, multiple device event handlers may be registered
                // to a single camera.
                //
                // *** LATER ***
                // Device event handlers need to be unregistered manually. This must be
                // done prior to releasing the system and while the device events
                // are still in scope.
                //
                if (chosenEvent == eventType.Generic)
                {
                    // Device event listeners registered generally will be
                    // triggered by any device events.
                    cam.RegisterEventHandler(deviceEventListener);

                    Console.WriteLine("Device event listener registered generally...");
                }
                else if (chosenEvent == eventType.Specific)
                {
                    // Device event listeners registered to a specific event
                    // will only be triggered by the type of event that is
                    // registered.
                    cam.RegisterEventHandler(deviceEventListener, "EventExposureEnd");

                    Console.WriteLine("Device event listener registered specifically to EventExposureEnd events...");
                }
            }
            catch (SpinnakerException ex)
            {
                Console.WriteLine("Error: {0}", ex.Message);
                result = -1;
            }

            return result;
        }

        // This function resets the example by unregistering the device event.
        int ResetDeviceEvents(IManagedCamera cam, ref DeviceEventListener deviceEventListener)
        {
            int result = 0;

            try
            {
                //
                // Unregister device event handler
                //
                // *** NOTES ***
                // It is important to unregister all device event handlers from all
                // cameras that they are registered to.
                //
                cam.UnregisterEventHandler(deviceEventListener);

                Console.WriteLine("Device event listener unregistered...\n");
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
                if (iAcquisitionMode == null || !iAcquisitionMode.IsWritable)
                {
                    Console.WriteLine("Unable to set acquisition mode to continuous (node retrieval). Aborting...\n");
                    return -1;
                }

                IEnumEntry iAcquisitionModeContinuous = iAcquisitionMode.GetEntryByName("Continuous");
                if (iAcquisitionModeContinuous == null || !iAcquisitionModeContinuous.IsReadable)
                {
                    Console.WriteLine(
                        "Unable to get acquisition mode to continuous (enum entry retrieval). Aborting...\n");
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
                                    "Grabbed image {0}, width = {1}, height = {1}",
                                    imageCnt,
                                    rawImage.Width,
                                    rawImage.Height);

                                // Convert image to mono 8
                                using(
                                    IManagedImage convertedImage = processor.Convert(rawImage, PixelFormatEnums.Mono8))
                                {
                                    // Create unique file name
                                    String filename = "DeviceEvents-CSharp-";
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

                // Configure device events
                DeviceEventListener deviceEventListener = null;

                err = ConfigureDeviceEvents(nodeMap, cam, ref deviceEventListener);
                if (err < 0)
                {
                    return err;
                }

                // Acquire images
                result = result | AcquireImages(cam, nodeMap, nodeMapTLDevice);

                // Reset device events
                result = result | ResetDeviceEvents(cam, ref deviceEventListener);

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
