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
 *  @example SpinSimpleGUI_DirectShow.cpp
 *
 *  @brief SpinSimpleGUI_DirectShow.cpp is written based off Microsoft's
 *  DirectShow sample playcap application. This example demonstrates how to
 *  connect to FLIR's DirectShow video capture source filter, setup camera
 *  settings through the IID_IPtGreyDevice interface, and renders and
 *  previews video capture data.
 *
 *  Also check out the NodeMapInfo example if you haven't already. NodeMapInfo
 *  demonstrates how to retrieve node map information and explores retrieving
 *  information from all major node types on the camera. The various node type
 *  include string, integer, float, boolean, command, enumeration, category,
 *  and value types. Only enumeration and integer type retrieval are covered in
 *  this example.
 *
 *  Please make sure Microsoft Windows SDK 7.1 is installed prior to building
 *  this example. Windows SDK 7.1 is the last version shipped with a set of
 *  DirectShow samples, and the DirectShow BaseClasses. DirectShow runtime is
 *  now a part of the operating system so no DirectShow redistributables are
 *  required.
 *
 *  Microsoft Windows SDK 7.1 can be found here:
 *  https://www.microsoft.com/en-us/download/details.aspx?id=8279
 *
 *  Note: You may run into problems installing Windows SDK 7.1 if you have a newer
 *  version of Visual Studio C++ 2010 Redistributable installed. To resolve the
 *  issue, you must uninstall all versions of the Visual C++ 2010 Redistributable
 *  before installing the Windows 7 SDK.
 *
 *  Reference:
 *  https://support.microsoft.com/en-us/topic/windows-sdk-fails-to-install-with-return-code-5100-56c17a69-4b37-77a6-aee5-9fd35ac534c4
 */

#ifndef _WIN32_WINNT        // Allow use of features specific to Windows XP or later.
#define _WIN32_WINNT 0x0501 // Change this to the appropriate value to target other versions of Windows.
#endif

#include <iostream>
#include <dshow.h>

#include "resource.h"

#include "SpinSimpleGUI_DirectShow.h"
#include "SpinnakerDirectShow.h"

//
// Global data
//
HWND ghApp = 0;
DWORD g_dwGraphRegister = 0;

IVideoWindow* g_pVW = nullptr;
IMediaControl* g_pMC = nullptr;
IMediaEventEx* g_pME = nullptr;
IGraphBuilder* g_pGraph = nullptr;
ICaptureGraphBuilder2* g_pCapture = nullptr;
PLAYSTATE g_psCurrent = Stopped;

HRESULT CaptureVideo()
{
    HRESULT hr;
    IBaseFilter* pSrcFilter = nullptr;

    // Get DirectShow interfaces
    hr = GetInterfaces();
    if (FAILED(hr))
    {
        MsgError(TEXT("Failed to get video interfaces!  hr=0x%x"), hr);
        return hr;
    }

    // Attach the filter graph to the capture graph
    hr = g_pCapture->SetFiltergraph(g_pGraph);
    if (FAILED(hr))
    {
        MsgError(TEXT("Failed to set capture filter graph!  hr=0x%x"), hr);
        return hr;
    }

    // Use the system device enumerator and class enumerator to find
    // a FLIR video capture/preview device.
    hr = FindCaptureDevice(&pSrcFilter);
    if (FAILED(hr))
    {
        // Don't display a message because FindCaptureDevice will handle it
        return hr;
    }

    // Add Capture filter to our graph.
    hr = g_pGraph->AddFilter(pSrcFilter, L"Video Capture");
    if (FAILED(hr))
    {
        MsgError(
            TEXT("Couldn't add the capture filter to the graph!  hr=0x%x\r\n\r\n")
                TEXT("If you have a working video capture device, please make sure\r\n")
                    TEXT("that it is connected and is not being used by another application.\r\n\r\n")
                        TEXT("The sample will now close."),
            hr);
        pSrcFilter->Release();
        return hr;
    }

    // Setup FLIR camera
    hr = SetupCamera(pSrcFilter);
    if (FAILED(hr))
    {
        // Don't display a message because FindCaptureDevice will handle it
        pSrcFilter->Release();
        return hr;
    }

    // Render the preview pin on the video capture filter
    // Use this instead of g_pGraph->RenderFile
    hr = g_pCapture->RenderStream(&PIN_CATEGORY_PREVIEW, &MEDIATYPE_Video, pSrcFilter, nullptr, nullptr);
    if (FAILED(hr))
    {
        MsgError(
            TEXT("Couldn't render the video capture stream.  hr=0x%x\r\n")
                TEXT("The capture device may already be in use by another application.\r\n\r\n")
                    TEXT("The sample will now close."),
            hr);
        pSrcFilter->Release();
        return hr;
    }

    // Now that the filter has been added to the graph and we have
    // rendered its stream, we can release this reference to the filter.
    pSrcFilter->Release();

    // Set video window style and position
    hr = SetupVideoWindow();
    if (FAILED(hr))
    {
        MsgError(TEXT("Couldn't initialize video window!  hr=0x%x"), hr);
        return hr;
    }

    // Start previewing video data
    hr = g_pMC->Run();
    if (FAILED(hr))
    {
        MsgError(TEXT("Couldn't run the graph!  hr=0x%x"), hr);
        return hr;
    }

    // Remember current state
    g_psCurrent = Running;

    return S_OK;
}

HRESULT FindCaptureDevice(IBaseFilter** ppSrcFilter)
{
    HRESULT hr = S_OK;
    IBaseFilter* pSrc = nullptr;
    IMoniker* pMoniker = nullptr;
    ICreateDevEnum* pDevEnum = nullptr;
    IEnumMoniker* pClassEnum = nullptr;

    if (!ppSrcFilter)
    {
        return E_POINTER;
    }

    // Create the system device enumerator
    hr = CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC, IID_ICreateDevEnum, (void**)&pDevEnum);
    if (FAILED(hr))
    {
        MsgError(TEXT("Couldn't create system enumerator!  hr=0x%x"), hr);
    }

    // Create an enumerator for the video capture devices
    if (SUCCEEDED(hr))
    {
        hr = pDevEnum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &pClassEnum, 0);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't create class enumerator!  hr=0x%x"), hr);
        }
    }

    if (SUCCEEDED(hr))
    {
        // If there are no enumerators for the requested type, then
        // CreateClassEnumerator will succeed, but pClassEnum will be nullptr.
        if (pClassEnum == nullptr)
        {
            MessageBox(
                ghApp,
                TEXT("No video capture device was detected.\r\n\r\n")
                    TEXT("This sample requires a video capture device, such as a USB WebCam,\r\n")
                        TEXT("to be installed and working properly.  The sample will now close."),
                TEXT("No Video Capture Hardware"),
                MB_OK | MB_ICONINFORMATION);
            hr = E_FAIL;
        }
    }

    // Enumerate FLIR device
    // Note that if the Next() call succeeds but there are no monikers,
    // it will return S_FALSE (which is not a failure).  Therefore, we
    // check that the return code is S_OK instead of using SUCCEEDED() macro.
    if (SUCCEEDED(hr))
    {
        while (pClassEnum->Next(1, &pMoniker, nullptr) == S_OK)
        {
            IPropertyBag* pPropBag = nullptr;
            hr = pMoniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)(&pPropBag));

            if (FAILED(hr))
            {
                SAFE_RELEASE(pMoniker);
                continue; // Skip this one, maybe the next one will work.
            }

            // Find the description or friendly name.
            VARIANT varName;
            VariantInit(&varName);

            hr = pPropBag->Read(L"FriendlyName", &varName, 0);
            if (SUCCEEDED(hr))
            {
                if (wcscmp(varName.bstrVal, L"PtGrey Camera") == 0)
                {
                    hr = pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&pSrc);
                    if (FAILED(hr))
                    {
                        MsgError(TEXT("Couldn't bind moniker to filter object!  hr=0x%x"), hr);
                    }

                    SAFE_RELEASE(pPropBag);
                    SAFE_RELEASE(pMoniker);
                    VariantClear(&varName);
                    break;
                }

                VariantClear(&varName);
            }

            SAFE_RELEASE(pPropBag);
            SAFE_RELEASE(pMoniker);
        }
    }

    // Copy the found filter pointer to the output parameter.
    if (SUCCEEDED(hr) && pSrc)
    {
        *ppSrcFilter = pSrc;
        (*ppSrcFilter)->AddRef();
    }

    if (!pSrc)
    {
        MessageBox(
            ghApp,
            TEXT("Unable to detect Spinnaker video capture source.\r\n")
                TEXT("Please make sure Spinnaker DirectShow DLL is registered properly."),
            TEXT("Spinnaker DirectShow Source Not Found"),
            MB_OK | MB_ICONINFORMATION);
        hr = E_FAIL;
    }

    SAFE_RELEASE(pSrc);
    SAFE_RELEASE(pDevEnum);
    SAFE_RELEASE(pClassEnum);

    return hr;
}

HRESULT GetInterfaces(void)
{
    HRESULT hr;

    // Create the filter graph
    hr = CoCreateInstance(CLSID_FilterGraph, nullptr, CLSCTX_INPROC, IID_IGraphBuilder, (void**)&g_pGraph);
    if (FAILED(hr))
    {
        return hr;
    }

    // Create the capture graph builder
    hr = CoCreateInstance(
        CLSID_CaptureGraphBuilder2, nullptr, CLSCTX_INPROC, IID_ICaptureGraphBuilder2, (void**)&g_pCapture);
    if (FAILED(hr))
    {
        return hr;
    }

    // Obtain interfaces for media control and Video Window
    hr = g_pGraph->QueryInterface(IID_IMediaControl, (LPVOID*)&g_pMC);
    if (FAILED(hr))
    {
        return hr;
    }

    hr = g_pGraph->QueryInterface(IID_IVideoWindow, (LPVOID*)&g_pVW);
    if (FAILED(hr))
    {
        return hr;
    }

    hr = g_pGraph->QueryInterface(IID_IMediaEventEx, (LPVOID*)&g_pME);
    if (FAILED(hr))
    {
        return hr;
    }

    // Set the window handle used to process graph events
    hr = g_pME->SetNotifyWindow((OAHWND)ghApp, WM_GRAPHNOTIFY, 0);

    return hr;
}

void CloseInterfaces(void)
{
    // Stop previewing data
    if (g_pMC)
    {
        g_pMC->StopWhenReady();
    }

    g_psCurrent = Stopped;

    // Stop receiving events
    if (g_pME)
    {
        g_pME->SetNotifyWindow(NULL, WM_GRAPHNOTIFY, 0);
    }

    // Relinquish ownership (IMPORTANT!) of the video window.
    // Failing to call put_Owner can lead to assert failures within
    // the video renderer, as it still assumes that it has a valid
    // parent window.
    if (g_pVW)
    {
        g_pVW->put_Visible(OAFALSE);
        g_pVW->put_Owner(NULL);
    }

    // Release DirectShow interfaces
    SAFE_RELEASE(g_pMC);
    SAFE_RELEASE(g_pME);
    SAFE_RELEASE(g_pVW);
    SAFE_RELEASE(g_pGraph);
    SAFE_RELEASE(g_pCapture);
}

HRESULT SetupCamera(IBaseFilter* pSrcFilter)
{
    //
    // Get a pointer to the IPtGreyDevice interface
    //
    ISpinnakerInterface* pProps = nullptr;
    HRESULT hr = pSrcFilter->QueryInterface(IID_ISpinnakerInterface, reinterpret_cast<void**>(&pProps));
    if (FAILED(hr))
    {
        MsgError(TEXT("Couldn't Query the Capture Interface!  hr=0x%x"), hr);
        return E_FAIL;
    }

    //
    // Get a pointer to the PtGreyDevice
    //
    ISpinDevice* pDevice = pProps->GetDevice();
    if (pDevice == nullptr)
    {
        MsgError(TEXT("Couldn't Query the Capture Device!"));
        return E_FAIL;
    }

    try
    {
        //
        // Query basic device information
        //
        char model[MAX_LENGTH];
        char sensor[MAX_LENGTH];
        char serial[MAX_LENGTH];

        unsigned int numDevices = 0;
        hr = pDevice->GetNumCameras(&numDevices);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't query number of devices!"));
            return hr;
        }

        unsigned int selectedDeviceIndex = 0;
        hr = pDevice->GetSelectedCameraIndex(&selectedDeviceIndex);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't query currently selected camera index!"));
            return hr;
        }

        hr = pDevice->GetCameraInfo(selectedDeviceIndex, model, sensor, serial, MAX_LENGTH);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't query camera information!"));
            return hr;
        }

        //
        // Query list of supported nodes
        //
        std::vector<std::string> nodes;

        size_t numNodes = 0;
        hr = pDevice->NodeMapGetNumNodes(&numNodes);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't query number of available nodes!"));
            return hr;
        }

        for (unsigned int index = 0; index < numNodes; index++)
        {
            char nodeName[MAX_LENGTH];
            hr = pDevice->NodeMapGetNodeAtIndex(index, nodeName, MAX_LENGTH);
            if (SUCCEEDED(hr))
            {
                nodes.push_back(nodeName);
            }
        }

        //
        // Query image settings
        //
        char pixelValue[MAX_LENGTH];
        int64_t widthValue = 0;
        int64_t heightValue = 0;

        //
        // Query and set PixelFormat enumeration node
        //
        std::string nodeName = "PixelFormat";
        bool isAvailable = false;
        hr = pDevice->NodeIsAvailable(nodeName.c_str(), &isAvailable);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't check whether PixelFormat is available!"));
            return hr;
        }

        bool isReadable = false;
        hr = pDevice->NodeIsReadable(nodeName.c_str(), &isReadable);
        if (FAILED(hr))
        {
            MsgError(TEXT("Couldn't check whether PixelFormat is readable!"));
            return hr;
        }

        if (isReadable)
        {
            // Get list of available entries
            size_t numEntries = 0;
            hr = pDevice->EnumerationGetNumEntries(nodeName.c_str(), &numEntries);
            if (FAILED(hr))
            {
                MsgError(TEXT("Couldn't query number of enumeration entries for PixelFormat!"));
                return hr;
            }

            std::vector<std::string> enumEntries;
            for (unsigned int index = 0; index < numNodes; index++)
            {
                char enumEntryName[MAX_LENGTH];
                hr = pDevice->EnumerationGetEntryAtIndex(nodeName.c_str(), index, enumEntryName, MAX_LENGTH);
                if (SUCCEEDED(hr))
                {
                    enumEntries.push_back(enumEntryName);
                }
            }

            // Check if entry exist and set to mono8
            bool isWritable = false;
            hr = pDevice->NodeIsWritable(nodeName.c_str(), &isWritable);
            if (FAILED(hr))
            {
                MsgError(TEXT("Couldn't check whether PixelFormat is writable!"));
                return hr;
            }

            bool entryExist = false;
            hr = pDevice->EnumerationEntryExists(nodeName.c_str(), "Mono8", &entryExist);
            if (FAILED(hr))
            {
                MsgError(TEXT("Couldn't check whether Mono8 entry is available for PixelFormat!"));
                return hr;
            }

            if (entryExist && isWritable)
            {
                pDevice->EnumerationSetEntry(nodeName.c_str(), "Mono8");
            }

            // Get current enumeration entry
            hr = pDevice->EnumerationGetEntry(nodeName.c_str(), pixelValue, MAX_LENGTH);
            if (FAILED(hr))
            {
                MsgError(TEXT("Couldn't get current values for PixelFormat!"));
                return hr;
            }
        }

        //
        // Query and read Width integer node
        //
        nodeName = "Width";
        isAvailable = false;
        pDevice->NodeIsAvailable(nodeName.c_str(), &isAvailable);
        isReadable = false;
        pDevice->NodeIsReadable(nodeName.c_str(), &isReadable);
        if (isReadable)
        {
            pDevice->IntegerGetValue(nodeName.c_str(), &widthValue);
        }

        //
        // Query and read Height integer node
        //
        nodeName = "Height";
        isAvailable = false;
        pDevice->NodeIsAvailable(nodeName.c_str(), &isAvailable);
        isReadable = false;
        pDevice->NodeIsReadable(nodeName.c_str(), &isReadable);
        if (isReadable)
        {
            pDevice->IntegerGetValue(nodeName.c_str(), &heightValue);
        }

        //
        // Display camera information
        //
        std::wstringstream infoString;
        infoString << L"Number of cameras found: " << numDevices << std::endl
                   << L"Selected Index : " << selectedDeviceIndex << std::endl
                   << L"Camera Model : " << model << std::endl
                   << L"Camera Sensor : " << sensor << std::endl
                   << L"Camera Serial : " << serial << std::endl
                   << L"Number of nodes: " << nodes.size() << std::endl
                   << L"Current Pixel Format: " << pixelValue << std::endl
                   << L"Width : " << widthValue << std::endl
                   << L"Height : " << heightValue << std::endl;
        MsgInfo(TEXT("%s"), infoString.str().c_str());
    }
    catch (Spinnaker::Exception& se)
    {
        std::basic_ostringstream<TCHAR> ex;
        ex << se.what();
        MsgError(TEXT("Unexpected exception: %s\n\r"), ex.str().c_str());
    }

    SAFE_RELEASE(pProps);

    return S_OK;
}

HRESULT SetupVideoWindow(void)
{
    HRESULT hr;

    // Set the video window to be a child of the main window
    hr = g_pVW->put_Owner((OAHWND)ghApp);
    if (FAILED(hr))
    {
        return hr;
    }

    // Set video window style
    hr = g_pVW->put_WindowStyle(WS_CHILD | WS_CLIPCHILDREN);
    if (FAILED(hr))
    {
        return hr;
    }

    // Use helper function to position video window in client rect
    // of main application window
    ResizeVideoWindow();

    // Make the video window visible, now that it is properly positioned
    hr = g_pVW->put_Visible(OATRUE);
    if (FAILED(hr))
    {
        return hr;
    }

    return hr;
}

void ResizeVideoWindow(void)
{
    // Resize the video preview window to match owner window size
    if (g_pVW)
    {
        RECT rc;

        // Make the preview video fill our window
        GetClientRect(ghApp, &rc);
        g_pVW->SetWindowPosition(0, 0, rc.right, rc.bottom);
    }
}

HRESULT ChangePreviewState(int nShow)
{
    HRESULT hr = S_OK;

    // If the media control interface isn't ready, don't call it
    if (!g_pMC)
    {
        return S_OK;
    }

    if (nShow)
    {
        if (g_psCurrent != Running)
        {
            // Start previewing video data
            hr = g_pMC->Run();
            g_psCurrent = Running;
        }
    }
    else
    {
        // Stop previewing video data
        hr = g_pMC->StopWhenReady();
        g_psCurrent = Stopped;
    }

    return hr;
}

void MsgError(TCHAR* szFormat, ...)
{
    TCHAR szBuffer[1024]; // Large buffer for long filenames or URLs
    const size_t NUMCHARS = sizeof(szBuffer) / sizeof(szBuffer[0]);
    const int LASTCHAR = NUMCHARS - 1;

    // Format the input string
    va_list pArgs;
    va_start(pArgs, szFormat);

    // Use a bounded buffer size to prevent buffer overruns.  Limit count to
    // character size minus one to allow for a nullptr terminating character.
    (void)StringCchVPrintf(szBuffer, NUMCHARS - 1, szFormat, pArgs);
    va_end(pArgs);

    // Ensure that the formatted string is nullptr-terminated
    szBuffer[LASTCHAR] = TEXT('\0');

    MessageBox(nullptr, szBuffer, TEXT("SpinSimpleGUI_DirectShow Error"), MB_OK | MB_ICONERROR);
}

void MsgInfo(TCHAR* szFormat, ...)
{
    TCHAR szBuffer[1024]; // Large buffer for long filenames or URLs
    const size_t NUMCHARS = sizeof(szBuffer) / sizeof(szBuffer[0]);
    const int LASTCHAR = NUMCHARS - 1;

    // Format the input string
    va_list pArgs;
    va_start(pArgs, szFormat);

    // Use a bounded buffer size to prevent buffer overruns.  Limit count to
    // character size minus one to allow for a nullptr terminating character.
    (void)StringCchVPrintf(szBuffer, NUMCHARS - 1, szFormat, pArgs);
    va_end(pArgs);

    // Ensure that the formatted string is nullptr-terminated
    szBuffer[LASTCHAR] = TEXT('\0');

    MessageBox(nullptr, szBuffer, TEXT("SpinSimpleGUI_DirectShow Info"), MB_OK | MB_ICONINFORMATION);
}

HRESULT HandleGraphEvent(void)
{
    LONG evCode;
    LONG_PTR evParam1, evParam2;
    HRESULT hr = S_OK;

    if (!g_pME)
    {
        return E_POINTER;
    }

    while (SUCCEEDED(g_pME->GetEvent(&evCode, &evParam1, &evParam2, 0)))
    {
        //
        // Free event parameters to prevent memory leaks associated with
        // event parameter data.  While this application is not interested
        // in the received events, applications should always process them.
        //
        hr = g_pME->FreeEventParams(evCode, evParam1, evParam2);

        // Insert event processing code here, if desired
    }

    return hr;
}

LRESULT CALLBACK WndMainProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_CREATE:
    {
        HINSTANCE hInstance = reinterpret_cast<LPCREATESTRUCT>(lParam)->hInstance;
        HICON hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_ICON1));
        if (hIcon != nullptr)
        {
            SendMessage(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(hIcon));
        }
        break;
    }
    case WM_GRAPHNOTIFY:
        HandleGraphEvent();
        break;

    case WM_SIZE:
        ResizeVideoWindow();
        break;

    case WM_WINDOWPOSCHANGED:
        ChangePreviewState(!(IsIconic(hwnd)));
        break;

    case WM_CLOSE:
        // Hide the main window while the graph is destroyed
        ShowWindow(ghApp, SW_HIDE);
        CloseInterfaces(); // Stop capturing and release interfaces
        break;

    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }

    // Pass this message to the video window for notification of system changes
    if (g_pVW)
    {
        g_pVW->NotifyOwnerMessage((LONG_PTR)hwnd, message, wParam, lParam);
    }

    return DefWindowProc(hwnd, message, wParam, lParam);
}

int PASCAL WinMain(HINSTANCE hInstance, HINSTANCE hInstP, LPSTR lpCmdLine, int nCmdShow)
{
    MSG msg = {0};
    WNDCLASS wc;

    // Initialize COM
    if (FAILED(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)))
    {
        MsgError(TEXT("CoInitialize Failed!\r\n"));
        exit(1);
    }

    // Register the window class
    ZeroMemory(&wc, sizeof wc);
    wc.lpfnWndProc = WndMainProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASSNAME;
    wc.lpszMenuName = nullptr;
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hIcon = nullptr;
    if (!RegisterClass(&wc))
    {
        MsgError(TEXT("RegisterClass Failed! Error=0x%x\r\n"), GetLastError());
        CoUninitialize();
        exit(1);
    }

    // Create the main window.  The WS_CLIPCHILDREN style is required.
    ghApp = CreateWindow(
        CLASSNAME,
        APPLICATIONNAME,
        WS_OVERLAPPEDWINDOW | WS_CAPTION | WS_CLIPCHILDREN,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        DEFAULT_VIDEO_WIDTH,
        DEFAULT_VIDEO_HEIGHT,
        0,
        0,
        hInstance,
        0);

    if (ghApp)
    {
        HRESULT hr;

        // Create DirectShow graph and start capturing video
        hr = CaptureVideo();
        if (FAILED(hr))
        {
            CloseInterfaces();
            DestroyWindow(ghApp);
        }
        else
        {
            // Don't display the main window until the DirectShow
            // preview graph has been created.  Once video data is
            // being received and processed, the window will appear
            // and immediately have useful video data to display.
            // Otherwise, it will be black until video data arrives.
            ShowWindow(ghApp, nCmdShow);
        }

        // Main message loop
        while (GetMessage(&msg, nullptr, 0, 0))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    // Release COM
    CoUninitialize();

    return (int)msg.wParam;
}