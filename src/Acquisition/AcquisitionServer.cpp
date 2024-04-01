#include <opencv2/opencv.hpp>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include <iostream>
#include <fstream>
#include <cstdio>  
#include <sstream>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <windows.h>
#include "focusMeasure.h"

#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;
using namespace cv;



#ifdef _DEBUG

// This function configures a custom exposure time. Automatic exposure is turned
// off in order to allow for the customization, and then the custom setting is
// applied.
int ConfigureExposure(INodeMap& nodeMap)
{
    int result = 0;

    cout << endl << endl << "*** CONFIGURING EXPOSURE ***" << endl << endl;

    try
    {
        //
        // Turn off automatic exposure mode
        //
        // *** NOTES ***
        // Automatic exposure prevents the manual configuration of exposure
        // time and needs to be turned off. Some models have auto-exposure
        // turned off by default
        //
        // *** LATER ***
        // Exposure time can be set automatically or manually as needed. This
        // example turns automatic exposure off to set it manually and back
        // on in order to return the camera to its default state.
        //
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (IsReadable(ptrExposureAuto) &&
            IsWritable(ptrExposureAuto))
        {
            CEnumEntryPtr ptrExposureAutoOff = ptrExposureAuto->GetEntryByName("Off");
            if (IsReadable(ptrExposureAutoOff))
            {
                ptrExposureAuto->SetIntValue(ptrExposureAutoOff->GetValue());
                cout << "Automatic exposure disabled..." << endl;
            }
        }
        else 
        {
            CEnumerationPtr ptrAutoBright = nodeMap.GetNode("autoBrightnessMode");
            if (!IsReadable(ptrAutoBright) ||
                !IsWritable(ptrAutoBright))
            {
                cout << "Unable to get or set exposure time. Aborting..." << endl << endl;
                return -1;
            }
            cout << "Unable to disable automatic exposure. Expected for some models... " << endl;
            cout << "Proceeding..." << endl;
            result = 1;
        }

        //
        // Set exposure time manually; exposure time recorded in microseconds
        //
        // *** NOTES ***
        // The node is checked for availability and writability prior to the
        // setting of the node. Further, it is ensured that the desired exposure
        // time does not exceed the maximum. Exposure time is counted in
        // microseconds. This information can be found out either by
        // retrieving the unit with the GetUnit() method or by checking SpinView.
        //
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime) ||
            !IsWritable(ptrExposureTime))
        {
            cout << "Unable to get or set exposure time. Aborting..." << endl << endl;
            return -1;
        }

        // Ensure desired exposure time does not exceed the maximum
        const double exposureTimeMax = ptrExposureTime->GetMax();
        //const double exposureTimecurr = ptrExposureTime->GetValue();
        double exposureTimeToSet = 16689.78; // At least 1/60th of a second- 1 projector duty cycle to avoid dither

        if (exposureTimeToSet > exposureTimeMax)
        {
            exposureTimeToSet = exposureTimeMax;
        }

        ptrExposureTime->SetValue(exposureTimeToSet);

        cout << std::fixed << "Exposure time set to " << exposureTimeToSet << " us..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// This function returns the camera to its default state by re-enabling automatic
// exposure.
int ResetExposure(INodeMap& nodeMap)
{
    int result = 0;

    try
    {
        //
        // Turn automatic exposure back on
        //
        // *** NOTES ***
        // Automatic exposure is turned on in order to return the camera to its
        // default state.
        //
        CEnumerationPtr ptrExposureAuto = nodeMap.GetNode("ExposureAuto");
        if (!IsReadable(ptrExposureAuto) ||
            !IsWritable(ptrExposureAuto))
        {
            cout << "Unable to enable automatic exposure (node retrieval). Non-fatal error..." << endl << endl;
            return -1;
        }

        CEnumEntryPtr ptrExposureAutoContinuous = ptrExposureAuto->GetEntryByName("Continuous");
        if (!IsReadable(ptrExposureAutoContinuous))
        {
            cout << "Unable to enable automatic exposure (enum entry retrieval). Non-fatal error..." << endl << endl;
            return -1;
        }

        ptrExposureAuto->SetIntValue(ptrExposureAutoContinuous->GetValue());

        cout << "Automatic exposure enabled..." << endl << endl;
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

// Disables heartbeat on GEV cameras so debugging does not incur timeout errors
int DisableHeartbeat(INodeMap & nodeMap, INodeMap & nodeMapTLDevice)
{
    cout << "Checking device type to see if we need to disable the camera's heartbeat..." << endl << endl;

    //
    // Write to boolean node controlling the camera's heartbeat
    //
    // *** NOTES ***
    // This applies only to GEV cameras and only applies when in DEBUG mode.
    // GEV cameras have a heartbeat built in, but when debugging applications the
    // camera may time out due to its heartbeat. Disabling the heartbeat prevents
    // this timeout from occurring, enabling us to continue with any necessary debugging.
    // This procedure does not affect other types of cameras and will prematurely exit
    // if it determines the device in question is not a GEV camera.
    //
    // *** LATER ***
    // Since we only disable the heartbeat on GEV cameras during debug mode, it is better
    // to power cycle the camera after debugging. A power cycle will reset the camera
    // to its default settings.
    //
    CEnumerationPtr ptrDeviceType = nodeMapTLDevice.GetNode("DeviceType");
    if (!IsAvailable(ptrDeviceType) || !IsReadable(ptrDeviceType))
    {
        cout << "Error with reading the device's type. Aborting..." << endl << endl;
        return -1;
    }
    else
    {
        if (ptrDeviceType->GetIntValue() == DeviceType_GigEVision) //== DeviceType_GEV
        {
            cout << "Working with a GigE camera. Attempting to disable heartbeat before continuing..." << endl << endl;
            CBooleanPtr ptrDeviceHeartbeat = nodeMap.GetNode("GevGVCPHeartbeatDisable");
            if (!IsAvailable(ptrDeviceHeartbeat) || !IsWritable(ptrDeviceHeartbeat))
            {
                cout << "Unable to disable heartbeat on camera. Continuing with execution as this may be non-fatal..." << endl << endl;
            }
            else
            {
                ptrDeviceHeartbeat->SetValue(true);
                cout << "WARNING: Heartbeat on GigE camera disabled for the rest of Debug Mode." << endl;
                cout << "         Power cycle camera when done debugging to re-enable the heartbeat..." << endl << endl;
            }
        }
        else
        {
            cout << "Camera does not use GigE interface. Resuming normal execution..." << endl << endl;
        }
    }
    return 0;
}
#endif


int get_camera_number(CameraPtr pCam)
{
    int ip = 0;

    INodeMap& nodeMap = pCam->GetTLDeviceNodeMap();

    cout << endl << "*** retrieving cam # ***" << endl << endl;

    try
    {
        FeatureList_t features;
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsAvailable(category) && IsReadable(category))
        {
            category->GetFeatures(features);

            for (auto it = features.begin(); it != features.end(); ++it)
            {
                const CNodePtr pfeatureNode = *it;
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);

                // Get camera number from last digit of ip address
                if (IsReadable(pValue) && pfeatureNode->GetName() == "GevDeviceIPAddress")
                {
                    string hex_num = pValue->ToString();
                    ip = stoi(hex_num, nullptr, 16) % 10;
                    cout << "GevDeviceIPAddress dec :" << ip << endl;

                }
            }
        }
        else
        {
            cout << "IP not available." << endl;
        }
    }
    catch (Spinnaker::Exception& e)
    {
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return ip;
}
// All buffers in the input pool and output queue will be discarded when EndAcquisition() is called, but image must be released beforehand
// last possible use of image pointer is SaveImage, so EndAcquisition() must be called afterwards
int SaveImage(CameraPtr pCam, ImagePtr Image, INodeMap & nodeMapTLDevice, string inputFilename, string outputFolder){

    int result = 0;
    cout << endl << "SAVING IMAGE" << endl <<endl;

    try
    {
        gcstring deviceSerialNumber("");
        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");

        if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
        {
            deviceSerialNumber = ptrStringSerial->GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        else {
            cout << "serial number not readable" << endl;
            return -1;
        }

        cout << endl;

        int serialNumber = 0;
        if (!deviceSerialNumber.empty())
        {
            serialNumber = stoi(deviceSerialNumber.c_str())%10;
        }
       
        ImageProcessor processor;

        processor.SetColorProcessing(SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

        ImagePtr convertedImage = processor.Convert(Image, PixelFormat_Mono8);

        cout << convertedImage <<endl;

        ostringstream filename;

        ostringstream full_path; 

        // Append last digit of SERIAL NUMBER to keep track of camera number (if last ip digit is wanted, use: << get_camera_number(pCam) <<)
        // Append filename of test pattern to keep track of horizontal/vertical codes and pattern #
        filename << "Cam_" << serialNumber << "_" << inputFilename << ".jpg";

        full_path << outputFolder << "/" << filename.str();

        // Save the current image using imwrite
        convertedImage->Save(full_path.str().c_str());

        cout << "Image saved (with SaveImage) at " << full_path.str() << endl;

        // Image->Release();

        // pCam->EndAcquisition();

        cout << endl;
        }
        catch (Spinnaker::Exception& e)
        {
            cout << "Error: " << e.what() << endl;
            result = -1;
        }

    return result;
}

// This function acquires (but does not save) 1 image from a device. (no longer 50- modify k_num)
ImagePtr AcquireImages(CameraPtr pCam, INodeMap & nodeMap, INodeMap & nodeMapTLDevice)
{
    ImagePtr pResultImage = nullptr;

    cout << endl << endl << "*** IMAGE ACQUISITION ***" << endl << endl;

    try
    {
        //
        // Set acquisition mode to continuous
        //
        // *** NOTES ***
        // Because the example acquires and saves 50 images, setting acquisition
        // mode to continuous lets the example finish. If set to single frame
        // or multiframe (at a lower number of images), the example would just
        // hang. This would happen because the example has been written to
        // acquire 10 images while the camera would have been programmed to
        // retrieve less than that.
        //
        // Setting the value of an enumeration node is slightly more complicated
        // than other node types. Two nodes must be retrieved: first, the
        // enumeration node is retrieved from the nodemap; and second, the entry
        // node is retrieved from the enumeration node. The integer value of the
        // entry node is then set as the new value of the enumeration node.
        //
        // Notice that both the enumeration and the entry nodes are checked for
        // availability and readability/writability. Enumeration nodes are
        // generally readable and writable whereas their entry nodes are only
        // ever readable.
        //
        // Set Acquisition mode to continuous (detailed)
        //Retrieve enumeration node from nodemap
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
        {   
            cout << IsAvailable(ptrAcquisitionMode) << endl;
            cout << IsWritable(ptrAcquisitionMode) << endl;
            cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << endl << endl;
            return nullptr;
        }

        // Retrieve entry node from enumeration node
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
        {
            cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << endl << endl;
            return nullptr;
        }

        // Retrieve integer value from entry node
        const int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        // Set integer value from entry node as new value of enumeration node
        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        cout << "Acquisition mode set to continuous..." << endl;

        // Set acquisition mode to singleframe
        //if (!IsWritable(pCam->AcquisitionMode))
        //{
        //    cout << "Unable to set acquisition mode. Aborting..." << endl << endl;
        //    return -1;
        //}
        //pCam->AcquisitionMode.SetValue(AcquisitionMode_SingleFrame);
        //cout << "Acquisition mode set to Single Frame..." << endl; 



#ifdef _DEBUG
        cout << endl << endl << "*** DEBUG ***" << endl << endl;

        // If using a GEV camera and debugging, should disable heartbeat first to prevent further issues
        if (DisableHeartbeat(nodeMap, nodeMapTLDevice) != 0)
        {
            return nullptr;
        }

        cout << endl << endl << "*** END OF DEBUG ***" << endl << endl;
#endif

        //
        // Begin acquiring images
        //
        // *** NOTES ***
        // What happens when the camera begins acquiring images depends on the
        // acquisition mode. Single frame captures only a single image, multi
        // frame captures a set number of images, and continuous captures a
        // continuous stream of images. Because the example calls for the
        // retrieval of 50 images, continuous mode has been set.
        //
        // *** LATER ***
        // Image acquisition must be ended when no more images are needed.
        //
        pCam->BeginAcquisition();

        cout << "Acquiring images..." << endl;

        //
        // Retrieve device serial number for filename
        //
        // *** NOTES ***
        // The device serial number is retrieved in order to keep cameras from
        // overwriting one another. Grabbing image IDs could also accomplish
        // this. --> We're using last digit of serial # to differentiate
        //
        gcstring deviceSerialNumber("");
        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
        if (IsAvailable(ptrStringSerial) && IsReadable(ptrStringSerial))
        {
            deviceSerialNumber = ptrStringSerial->GetValue();

            cout << "Device serial number retrieved as " << deviceSerialNumber << "..." << endl;
        }
        cout << endl;

        // Get the value of exposure time to set an appropriate timeout for GetNextImage
        CFloatPtr ptrExposureTime = nodeMap.GetNode("ExposureTime");
        if (!IsReadable(ptrExposureTime))
        {
            cout << "Unable to read exposure time. Aborting..." << endl << endl;
            return -1;
        }
        // The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit
        // being used in GetNextImage
        cout << "current exposure time:" << (ptrExposureTime->GetValue()) << endl;
        uint64_t timeout = static_cast<uint64_t>(ptrExposureTime->GetValue() / 1000 + 1000);

        // Create a highgui window to display incomming images
   /*     namedWindow(WINDOW_NAME);
        moveWindow(WINDOW_NAME, 0, 0);*/

        // Retrieve, convert, and save images
        const unsigned int k_numImages = 1; // Just save one image!!

        for (unsigned int imageCnt = 0; imageCnt < k_numImages; imageCnt++)
        {
            try
            {
                //
                // Retrieve next received image
                //
                // *** NOTES ***
                // Capturing an image houses images on the camera buffer. Trying
                // to capture an image that does not exist will hang the camera.
                //
                // *** LATER ***
                // Once an image from the buffer is saved and/or no longer
                // needed, the image must be released in order to keep the
                // buffer from filling up.
                //
                pResultImage = pCam->GetNextImage(timeout);

                //
                // Ensure image completion
                //
                // *** NOTES ***
                // Images can easily be checked for completion. This should be
                // done whenever a complete image is expected or required.
                // Further, check image status for a little more insight into
                // why an image is incomplete.
                //
                if (pResultImage->IsIncomplete())
                {
                    // Retrieve and print the image status description
                    cout << "Image incomplete: "
                        << Image::GetImageStatusDescription(pResultImage->GetImageStatus())
                        << "..." << endl << endl;
                }
                else
                {
                    //
                    // Print image information; height and width recorded in pixels
                    //
                    // *** NOTES ***
                    // Images have quite a bit of available metadata including
                    // things such as CRC, image status, and offset values, to
                    // name a few.
                    
                    const size_t width = pResultImage->GetWidth(); //const size_t

                    //unsigned int width = static_cast<unsigned int>(pResultImage->GetWidth());

                    const size_t height = pResultImage->GetHeight();

                   // unsigned int height = static_cast<unsigned int>(pResultImage->GetHeight());

                    cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;

                    // Convert the Image to OpenCV Mat and display the result in a window.
                    // Note that if the camera is streaming in BayerRG8 format you will need to debayer the image to
                    // convert it into a 3-channel color image. You can do this before passing the spinnaker image data
                    // to the Mat object, using pResultImage->Convert(PixelFormat_RGB8, HQ_LINEAR) or you can
                    // use OpenCV to debayer the image after the BayerRG8 image data is passed to the Mat object.
                    // To preform the debayering (demosaicing) on the Mat object with OpenCV you can use cvtColor as follow:
                    // cvtColor(current_frame, demosaiced_frame, COLOR_BayerRG2RGB); ///

                    unsigned int rows = static_cast<unsigned int>(pResultImage->GetHeight()); //unsigned int rows
                    //unsigned int rows = height;

                    unsigned int cols = static_cast<unsigned int>(pResultImage->GetWidth());
                    //unsigned int cols = width;

                    unsigned int num_channels = static_cast<unsigned int>(pResultImage->GetNumChannels());

                    cout << "channels: " << num_channels << endl;

                    void *image_data = pResultImage->GetData();
                    unsigned int stride = static_cast<unsigned int>(pResultImage->GetStride());
                    //const size_t stride = pResultImage->GetStride();
                    static_cast<unsigned int>(stride);

                    Mat current_frame = cv::Mat(rows, cols, (num_channels == 3) ? CV_8UC3 : CV_8UC1, image_data, stride);

                }

                //
                // Release image
                //
                // *** NOTES ***
                // Images retrieved directly from the camera (i.e. non-converted
                // images) need to be released in order to keep from filling the
                // buffer.
                //

                //pResultImage->Release(); // only release image after it is used by SaveImage or ProcessImage 

                cout << endl;
            }
            catch (Spinnaker::Exception &e)
            {
                cout << "Error: " << e.what() << endl;
                pResultImage = nullptr;
            }
        }

        //
        // End acquisition
        //
        // *** NOTES ***
        // Ending acquisition appropriately helps ensure that devices clean up
        // properly and do not need to be power-cycled to maintain integrity.
        //

        //pCam->EndAcquisition();

        // Destroy the image display window
        //destroyWindow(WINDOW_NAME);
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        return nullptr;
    }

    return pResultImage;
}

bool isHexadecimal(const std::string& str) {
    // Check if the string starts with "0x"
    if (str.size() < 3 || str.substr(0, 2) != "0x")
        return false;

    // Iterate over characters starting from index 2
    for (size_t i = 2; i < str.size(); ++i) {
        if (!std::isxdigit(str[i])) {
            return false; // Not a valid hexadecimal digit
        }
    }
    return true;
}

int WriteToCSV(const std::vector<int>& results, const std::string& outputLocation) {
    try {
        //get current system time
        auto currentTime = chrono::system_clock::now();

        // Convert the system time to a time_point representing the epoch
        time_t currentTimeT = chrono::system_clock::to_time_t(currentTime);

        // Convert the time_point to a struct tm for further formatting
        tm* localTime = localtime(&currentTimeT);

        std::ofstream outputFile(outputLocation, std::ios::app); // Open the CSV file in append mode

        if (outputFile.is_open()) {
            outputFile << put_time(localTime, "%H:%M:%S")<< ","; // Write the time in the first column
            for (size_t i = 0; i < results.size(); ++i) {
                outputFile << results[i]; // Write each result to a separate cell in the row, with time in the first column

                if (i != results.size() - 1) {
                    outputFile << ","; // Add a comma separator between values
                }
            }

            outputFile << std::endl; // Add a new line after writing the row

            outputFile.close(); // Close the CSV file

            return 0;
        } else {
            std::cout << "Error: Failed to open the CSV file." << std::endl;
            return -1;
        }
    }
    catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return -1;
    }
}

int ApplyFocusMetric(uchar* buffer, unsigned int w, unsigned int h, string outputLocation) {

try
{    
    std::vector<int> results;
    int num_cases = 28;
    FocusMeasure focus; 

    ofstream fp;
    fp.open(outputLocation, fstream::app);
    
    for (int apply = 0; apply < num_cases; ++apply) {
    int result;
    switch( apply ) {
        case  0: result = static_cast<int>(focus.firstorder3x3( buffer, w, h ));	break;
        case  1: result = static_cast<int>(focus.roberts3x3( buffer, w, h ));	break;
        case  2: result = static_cast<int>(focus.prewitt3x3( buffer, w, h ));	break;
        case  3: result = static_cast<int>(focus.scharr3x3( buffer, w, h ));	break;
        case  4: result = static_cast<int>(focus.sobel3x3( buffer, w, h ));	break;
        case  5: result = static_cast<int>(focus.sobel5x5( buffer, w, h ));	break;
        case  6: result = static_cast<int>(focus.laplacian3x3( buffer, w, h ));	break;
        case  7: result = static_cast<int>(focus.laplacian5x5( buffer, w, h ));	break;
        case  8: result = static_cast<int>(focus.sobel3x3so( buffer, w, h ));	break;
        case  9: result = static_cast<int>(focus.sobel5x5so( buffer, w, h ));	break;
        case 10: result = static_cast<int>(focus.brenner( buffer, w, h ));		break;
        case 11: result = static_cast<int>(focus.thresholdGradient( buffer, w, h )); break;
        case 12: result = static_cast<int>(focus.squaredGradient( buffer, w, h )); break;
        case 13: result = static_cast<int>(focus.MMHistogram( buffer, w, h ));	break;
        case 14: result = static_cast<int>(focus.rangeHistogram( buffer, w, h ));	break;
        case 15: result = static_cast<int>(focus.MGHistogram( buffer, w, h ));	break;
        case 16: result = static_cast<int>(focus.entropyHistogram( buffer, w, h )); break;
        case 17: result = static_cast<int>(focus.th_cont( buffer, w, h ));		break;
        case 18: result = static_cast<int>(focus.num_pix( buffer, w, h ));		break;
        case 19: result = static_cast<int>(focus.power( buffer, w, h ));		break;
        case 20: result = static_cast<int>(focus.var( buffer, w, h ));		break;
        case 21: result = static_cast<int>(focus.nor_var( buffer, w, h ));		break;
        case 22: result = static_cast<int>(focus.vollath4( buffer, w, h ));	break;
        case 23: result = static_cast<int>(focus.vollath5( buffer, w, h ));	break;
        case 24: result = static_cast<int>(focus.autoCorrelation( buffer, w, h, 2 )); break;
        case 25: result = static_cast<int>(focus.sobel3x3soCross( buffer, w, h )); break;
        case 26: result = static_cast<int>(focus.sobel5x5soCross( buffer, w, h )); break;
        // case 27: result = static_cast<int>(focus.firstDerivGaussian( buffer, w, h )); break;
        // case 28: result = static_cast<int>(focus.LoG( buffer, w, h )); break;
        case 27: result = static_cast<int>(focus.curvature( buffer, w, h )); break;
        }

        results.push_back(result);
        
    }   

        WriteToCSV(results, outputLocation);

        // // Set the column width
        // int columnWidth = 10;

        // // Write results to the output file
        // fp << put_time(localTime, "%H-%M-%S") << std::setw(10) << " ";
        // for (size_t i = 0; i < results.size(); ++i) {
        //     fp << std::left << std::setw(columnWidth) << results[i];
        // }
        // fp << std::endl; // Move to the next line after writing all results

        // // Close the output file
        // fp.close();

    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}

// Process captured image and save focus metric with current system time to output file
int ProcessImage(ImagePtr Image, string outputLocation) {
    try
    {   
    
        
        unsigned int width = static_cast<unsigned int>(Image->GetWidth());
        unsigned int height = static_cast<unsigned int>(Image->GetHeight());

        void *image_data = Image->GetData();
        unsigned int stride = static_cast<unsigned int>(Image->GetStride());

        Mat current_frame = cv::Mat(height, width, CV_8UC1, image_data, stride);

        uchar* buffer = current_frame.data;
        //int v = focus.brenner(buffer, width, height);

        // //get current system time
        // auto currentTime = chrono::system_clock::now();

        // // Convert the system time to a time_point representing the epoch
        // time_t currentTimeT = chrono::system_clock::to_time_t(currentTime);

        // // Convert the time_point to a struct tm for further formatting
        // tm* localTime = localtime(&currentTimeT);
        //cout << "Current system time: " << put_time(localTime, "%H-%M-%S") << endl;

        // ofstream fp;
        // fp.open(outputLocation, fstream::app);
        // fp << put_time(localTime, "%H-%M-%S") << " " << v << endl; // write current time and focus metric to output file
        // fp.close();

        // cout << "image metric:" << v << endl;

        ApplyFocusMetric(buffer, width, height, outputLocation);

    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        return -1;
    }
    
    return 0;
}

// This function prints the device information of the camera from the transport
// layer; please see NodeMapInfo example for more in-depth comments on printing
// device information from the nodemap.
int PrintDeviceInfo(INodeMap & nodeMap)
{
    cout << endl << "*** DEVICE INFORMATION ***" << endl << endl;

    try
    {
        FeatureList_t features;
        const CCategoryPtr category = nodeMap.GetNode("DeviceInformation");
        if (IsAvailable(category) && IsReadable(category))
        {
            category->GetFeatures(features);

            for (auto it = features.begin(); it != features.end(); ++it)
            {
                const CNodePtr pfeatureNode = *it;
                cout << pfeatureNode->GetName() << " : ";
                CValuePtr pValue = static_cast<CValuePtr>(pfeatureNode);
                cout << (IsReadable(pValue) ? pValue->ToString() : "Node not readable");
                cout << endl;

                // Get camera number from ip address
                if (pfeatureNode->GetName() == "GevDeviceIPAddress")
                {
                    string hex_num = pValue->ToString();
                    cout << "GevDeviceIPAddress dec :" << stoi(hex_num, nullptr, 16)%10 << endl;
                }
            }
        }
        else
        {
            cout << "Device control information not available." << endl;
        }
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        return -1;
    }

    return 0;
}

// This function acts as the body of the example; please see NodeMapInfo example
// for more in-depth comments on setting up cameras.
int RunSingleCamera(CameraPtr pCam, string inputFilename, string outputFolder, string dataOutputFile)
{   
    // Camera flow: 
    // if a data output file has been passed, we are in focus mode and will save focus data to the file
    // if an output folder has been passed, we will save the image to the folder

    int result = 0;
    int err = 0;
    try
    {
        // Retrieve TL device nodemap and print device information
        INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();

        //result = PrintDeviceInfo(nodeMapTLDevice); 
        INodeMap & nodeMap = pCam->GetNodeMap();
        

        // Acquire images
        //result = result | AcquireImages(pCam, nodeMap, nodeMapTLDevice, inputFilename, outputFolder);
        ImagePtr imageResult = AcquireImages(pCam, nodeMap, nodeMapTLDevice);

        result = result | ((imageResult == nullptr) ? -1 : 0); // AcquireImages now returns ImagePtr object, convert to int to perform bitwise OR

        // if a data output file has been passed, process the image and save the metric to the file
        if (!dataOutputFile.empty()) {
            result = result | ProcessImage(imageResult, dataOutputFile);
        }

        // if an output folder has been passed, save the image to the folder
        if (!outputFolder.empty()) {
			result = result | SaveImage(pCam, imageResult, nodeMapTLDevice, inputFilename, outputFolder);
		}

        // use Image->Release() and pCam->EndAcquisition() at the END of each camera run, regardless of acquisition "mode" 
        imageResult->Release();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        result = -1;
    }

    return result;
}

int main(int argc, char* argv[])
{   
    // Parse command line arguments to get the input filename, output folder, and data output file
    vector<string> args(argv + 1, argv + argc);

    // where we store focus data + timestamps (full file path), how we name the saved image file, and where we store the saved images 
    string dataOutputFile, inputFilename, outputFolder; 

    for (auto i = args.begin(); i != args.end(); ++i) {
        if (*i == "-h" || *i == "--help") {
            cout << "Syntax: -f <pattern_filename> -i <image_output_folder> -d <data_output_file>" << endl;
            return 0;
        } 
        else if (*i == "-f") {
            inputFilename = *++i;
        } 
        else if (*i == "-i") {
            outputFolder= *++i;
        }
        else if (*i == "-d") {
            dataOutputFile= *++i;
            FILE *tempFile = fopen(dataOutputFile.c_str(), "r");
            // check if output file is valid
            if (tempFile == nullptr)
            {
                cout << "Failed to open test data file" << endl;
                cout << "Press Enter to exit..." << endl;
                getchar();
                return -1;
            }
            fclose(tempFile);
        }
    }


    // Initialize Winsock
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0)
    {
        cout << "Failed to initialize winsock" << endl;
        return -1;
    }

    // Create a socket
    SOCKET serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == INVALID_SOCKET)
    {
        cout << "Failed to create socket" << endl;
        WSACleanup();
        return -1;
    }

    // Bind the socket to an IP address and port
    sockaddr_in serverAddress;
    serverAddress.sin_family = AF_INET;
    serverAddress.sin_port = htons(1234); // Change the port number if needed
    serverAddress.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSocket, (struct sockaddr*)&serverAddress, sizeof(serverAddress)) == SOCKET_ERROR)
    {
        cout << "Failed to bind socket" << endl;
        closesocket(serverSocket);
        WSACleanup();
        return -1;
    }

    // Listen for incoming connections
    if (listen(serverSocket, SOMAXCONN) == SOCKET_ERROR)
    {
        cout << "Failed to listen for connections" << endl;
        closesocket(serverSocket);
        WSACleanup();
        return -1;
    }

    cout << "Server started. Waiting for connections..." << endl;

    // Accept a client connection
    SOCKET clientSocket = accept(serverSocket, NULL, NULL);
    if (clientSocket == INVALID_SOCKET)
    {
        cout << "Failed to accept client connection" << endl;
        closesocket(serverSocket);
        WSACleanup();
        return -1;
    }

    cout << "Client connected" << endl;

    // set up cameras

    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
    cout << "Spinnaker library version: "
        << spinnakerLibraryVersion.major << "."
        << spinnakerLibraryVersion.minor << "."
        << spinnakerLibraryVersion.type << "."
        << spinnakerLibraryVersion.build << endl << endl;

    // Retrieve list of cameras from the system
    CameraList camList = system->GetCameras();

    const unsigned int numCameras = camList.GetSize();

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

    //based on the number of cameras located, we will create a camera object for each camera
    // vector<CameraPtr> cameras;
    // for (unsigned int i = 0; i < numCameras; i++)
    // {
    //     // Select camera
    //     CameraPtr pCam = camList.GetByIndex(i);
    //     cameras.push_back(pCam);
    // }

    //create an output csv file for each camera, named after the last digit of the camera's serial number
    vector<string> outputFiles;
    vector<CameraPtr> cameras;
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera, add to cameras vector
        CameraPtr pCam = camList.GetByIndex(i);
        cameras.push_back(pCam);
        
        INodeMap & nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
        CStringPtr ptrStringSerial = nodeMapTLDevice.GetNode("DeviceSerialNumber");
        string deviceSerialNumber = ptrStringSerial->GetValue();
        string outputLocation = "DataOutput_Cam_" +  to_string(stoi(deviceSerialNumber) % 10) + ".csv";
        outputFiles.push_back(outputLocation);
    }


    // Initialize cameras, set exposure time, and begin waiting for commands from the client
    for (unsigned int i = 0; i < numCameras; i++)
    {
       CameraPtr pCam = cameras[i];
        pCam->Init();
        // Retrieve GenICam nodemap
        INodeMap & nodeMap = pCam->GetNodeMap();

        // Configure exposure- custom timing
        int err = ConfigureExposure(nodeMap);
        if (err < 0)
        {
            cout << "Camera " << i << "Exposure Config Error: " << err << endl;
        }
   
    // Main loop to receive and process commands from the client
    while (true)
    {
        char command[256];
        memset(command, 0, sizeof(command));

        // Receive command from the client
        int bytesReceived = recv(clientSocket, command, sizeof(command), 0);
        if (bytesReceived <= 0)
        {
            cout << "Client disconnected" << endl;
            break;
        }

        // Process the command
        if (strcmp(command, "capture") == 0)
        {
            // Capture an image with each created camera object

            for (unsigned int i = 0; i < numCameras; i++)
            {
                // Select camera object
                CameraPtr pCam = cameras[i];

                cout << endl << "Running example for camera " << i << "..." << endl;

                //set the output file for the camera
                string dataOutputFile = outputFiles[i];

                // Run camera
                int result = RunSingleCamera(pCam, inputFilename, outputFolder, dataOutputFile);

                if (result < 0)
                {
                    cout << "Camera " << i << "Error: " << result << endl;
                }
                else
                {
                    cout << "Camera " << i << " example complete..." << endl << endl;
                }

            }
        }
        else if (strcmp(command, "exit") == 0)
        {
            // Exit the server
            break;
        }
        else
        {
            cout << "Invalid command: " << command << endl;
        }
    }

    // End acquisition, and deinitialize each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        CameraPtr pCam = cameras[i];
        INodeMap & nodeMap = pCam->GetNodeMap();
        pCam->EndAcquisition();
        ResetExposure(nodeMap);
        pCam->DeInit();
    }


    // Clear camera list before releasing system
    camList.Clear();

    // Release system
    system->ReleaseInstance();

    // Close the client socket
    closesocket(clientSocket);

    // Close the server socket
    closesocket(serverSocket);

    // Cleanup Winsock
    WSACleanup();

    return 0;
}



// Example entry point; please see Enumeration example for more in-depth
// comments on preparing and cleaning up the system.
int main(int argc, char* argv[]) //int main(int /*argc*/, char** /*argv*/)
{   
    vector<string> args(argv + 1, argv + argc);

    string dataOutputFile, inputFilename, outputFolder; // where we store focus data + timestamps (full file path), how we name the saved image file, and where we store the saved images 

    for (auto i = args.begin(); i != args.end(); ++i) {
        if (*i == "-h" || *i == "--help") {
            cout << "Syntax: -f <pattern_filename> -i <image_output_folder> -d <data_output_file>" << endl;
            return 0;
        } 
        else if (*i == "-f") {
            inputFilename = *++i;
        } 
        else if (*i == "-i") {
            outputFolder= *++i;
        }
        else if (*i == "-d") {
            dataOutputFile= *++i;
            FILE *tempFile = fopen(dataOutputFile.c_str(), "r");
            // check if output file is valid
            if (tempFile == nullptr)
            {
                cout << "Failed to open test data file" << endl;
                cout << "Press Enter to exit..." << endl;
                getchar();
                return -1;
            }
            fclose(tempFile);
        }
    }


    // Since this application saves images in the current folder
    // we must ensure that we have permission to write to this folder.
    // If we do not have permission, fail right away.
    FILE *tempFile = fopen("test.txt", "w+");
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

    // Print application build information
    cout << "Application build date: " << __DATE__ << " " << __TIME__ << endl << endl;

    // Retrieve singleton reference to system object
    SystemPtr system = System::GetInstance();

    // Print out current library version
    const LibraryVersion spinnakerLibraryVersion = system->GetLibraryVersion();
    cout << "Spinnaker library version: "
        << spinnakerLibraryVersion.major << "."
        << spinnakerLibraryVersion.minor << "."
        << spinnakerLibraryVersion.type << "."
        << spinnakerLibraryVersion.build << endl << endl;

    // Retrieve list of cameras from the system
    CameraList camList = system->GetCameras();

    const unsigned int numCameras = camList.GetSize();

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

    int result = 0;

    // Run example on each camera
    for (unsigned int i = 0; i < numCameras; i++)
    {
        // Select camera
        pCam = camList.GetByIndex(i);

        cout << endl << "Running example for camera " << i << "..." << endl;

        // Run example
        result = result | RunSingleCamera(pCam, inputFilename, outputFolder, dataOutputFile);

        cout << "Camera " << i << " example complete..." << endl << endl;
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

    //cout << endl << "Done! Press Enter to exit..." << endl;
    //getchar();

    return result;
}