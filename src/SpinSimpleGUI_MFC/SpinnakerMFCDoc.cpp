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

#include "stdafx.h"
#include "SpinnakerMFC.h"
#include "SpinnakerMFCDoc.h"
#include "SpinnakerMFCView.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CSpinnakerMFCDoc

IMPLEMENT_DYNCREATE(CSpinnakerMFCDoc, CDocument)
BEGIN_MESSAGE_MAP(CSpinnakerMFCDoc, CDocument)
ON_COMMAND(ID_CAMERACONTROL_TOGGLECAMERACONTROL, &CSpinnakerMFCDoc::OnToggleCameraControl)
ON_COMMAND(ID_FILE_SAVE_AS, &CSpinnakerMFCDoc::OnFileSaveAs)
END_MESSAGE_MAP()

CSpinnakerMFCDoc::CSpinnakerMFCDoc()
{
    InitBitmapStruct(_DEFAULT_WINDOW_X, _DEFAULT_WINDOW_Y);

    m_continueGrabThread = false;
    m_heventThreadDone = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    m_uiFilterIndex = 0;

    m_beingSaved = false;

    m_pCamera = nullptr;

    m_pGridWnd = nullptr;
    m_pCamSelWnd = nullptr;
}

CSpinnakerMFCDoc::~CSpinnakerMFCDoc()
{
    CloseHandle(m_heventThreadDone);
}

void CSpinnakerMFCDoc::InitBitmapStruct(const int cols, const int rows)
{
    BITMAPINFOHEADER* pheader = &m_bitmapInfo.bmiHeader;

    // Initialize permanent data in the bitmap info header.
    pheader->biSize = sizeof(BITMAPINFOHEADER);
    pheader->biPlanes = 1;
    pheader->biCompression = BI_RGB;
    pheader->biXPelsPerMeter = 100;
    pheader->biYPelsPerMeter = 100;
    pheader->biClrUsed = 0;
    pheader->biClrImportant = 0;

    // Set a default window size.
    pheader->biWidth = cols;
    pheader->biHeight = -rows;
    pheader->biBitCount = 32;

    m_bitmapInfo.bmiHeader.biSizeImage = 0;
}

BOOL CSpinnakerMFCDoc::OnNewDocument()
{
    if (!CDocument::OnNewDocument())
    {
        return FALSE;
    }

    CWinApp* theApp = AfxGetApp();
    if (theApp)
    {
        CWnd* mainWnd = theApp->m_pMainWnd;
        if (mainWnd)
        {
            mainWnd->ShowWindow(SW_HIDE);
        }
    }

    // If entering this function from File->New Camera, stop the grab thread
    // first before doing anything else
    if (m_continueGrabThread)
    {
        m_continueGrabThread = false;

        const DWORD dwRet = WaitForSingleObject(m_heventThreadDone, 5000);
        if (dwRet == WAIT_TIMEOUT)
        {
            // Timed out while waiting for thread to exit
            CString csMessage;
            csMessage.Format(L"Failed to stop current grab thread.");
            AfxMessageBox(csMessage, MB_ICONSTOP);
        }
    }

    try
    {
        // Clean up Grid Window
        if (m_pGridWnd)
        {
            m_pGridWnd->Disconnect();
            m_pGridWnd->Hide();
        }

        m_pCamSelWnd = new Spinnaker::GUI_WPF::CameraSelectionWindow();

        if (m_pCamSelWnd->ShowModal(&m_devInfo) == false)
        {
            // User closed camera selection dialog
            return FALSE;
        }

        // An interface is selected
        if (!m_devInfo.isCamera)
        {
            if (m_devInfo.pInterface != nullptr)
            {
                CString csMessage;
                csMessage.Format(L"Connect Failure: Failed to get camera from camera selection dialog.");
                AfxMessageBox(csMessage, MB_ICONSTOP);

                delete m_devInfo.pInterface;
                m_devInfo.pInterface = nullptr;
            }
            return FALSE;
        }

        // A camera is selected
        m_pCamera = *m_devInfo.pCamera;

        // Initialize camera.
        m_pCamera->Init();

        m_pGridWnd = new Spinnaker::GUI_WPF::PropertyGridWindow();
        m_pGridWnd->Connect(&m_pCamera);

        m_processedImage = Image::Create();
        m_saveImage = Image::Create();
    }
    catch (Exception& ex)
    {
        CString csMessage;
        csMessage.Format(L"Connect Failure: %s (Error %s)", "Failed to get camera from index 0", ex.what());
        AfxMessageBox(csMessage, MB_ICONSTOP);

        return FALSE;
    }

    // Start the grab thread
    m_continueGrabThread = true;
    AfxBeginThread(ThreadGrabImage, this);

    return TRUE;
}

void CSpinnakerMFCDoc::OnCloseDocument()
{
    if (m_pGridWnd != nullptr && m_pGridWnd->IsVisible())
    {
        m_pGridWnd->Hide();
    }

    if (m_pGridWnd != nullptr && m_pGridWnd->IsConnected())
    {
        m_pGridWnd->Disconnect();
    }

    m_continueGrabThread = false;

    const DWORD dwRet = WaitForSingleObject(m_heventThreadDone, 5000);
    if (dwRet == WAIT_TIMEOUT)
    {
        // Timed out while waiting for thread to exit
    }

    if (m_pCamera != nullptr)
    {
        m_pCamera->DeInit();
    }

    delete m_pCamSelWnd;
    delete m_pGridWnd;

    CDocument::OnCloseDocument();
}

UINT CSpinnakerMFCDoc::ThreadGrabImage(void* pparam)
{
    auto pDoc = static_cast<CSpinnakerMFCDoc*>(pparam);
    const UINT uiRetval = pDoc->DoGrabLoop();
    if (uiRetval != 0)
    {
        CString csMessage;
        csMessage.Format(L"The grab thread has encountered a problem and had to terminate.");
        AfxMessageBox(csMessage, MB_ICONSTOP);

        //
        // Signal that the thread has died.
        //
        SetEvent(pDoc->m_heventThreadDone);
    }

    return uiRetval;
}

UINT CSpinnakerMFCDoc::DoGrabLoop()
{
    CString csMessage;

    try
    {
        m_pCamera->BeginAcquisition();
    }
    catch (std::exception& e)
    {
        csMessage.Format(L"StartCapture Failure: %s", e.what());
        AfxMessageBox(csMessage, MB_ICONSTOP);
        return 0;
    }

    //
    // Start of main grab loop
    //
    while (m_continueGrabThread)
    {
        try
        {
            ImagePtr rawImage = m_pCamera->GetNextImage();

            // Keep a reference to image so that we can save it
            // Doing this only when m_saveImage is not being saved
            if (!m_beingSaved)
            {
                m_saveImage->DeepCopy(rawImage);
            }

            //
            // Check to see if the thread should die.
            //
            if (!m_continueGrabThread)
            {
                break;
            }

            //
            // Update current frame rate.
            //
            m_processedFrameRate.NewFrame();

            CSingleLock dataLock(&m_csData);
            dataLock.Lock();

            try
            {
                m_processedImage = m_processor.Convert(rawImage, PixelFormat_BGRa8);
            }
            catch (std::exception& e)
            {
                csMessage.Format(L"Convert Failure: %s", e.what());
                continue;
            }

            dataLock.Unlock();

            const auto rows = static_cast<unsigned int>(rawImage->GetHeight());
            const auto cols = static_cast<unsigned int>(rawImage->GetWidth());

            InitBitmapStruct(cols, rows);

            RedrawAllViews();

            // Release image
            rawImage->Release();
        }
        catch (std::exception& e)
        {
            csMessage.Format(L"Grab loop Failure: %s", e.what());
            AfxMessageBox(csMessage, MB_ICONSTOP);
            return 0;
        }
    }

    try
    {
        m_pCamera->EndAcquisition();
    }
    catch (std::exception& e)
    {
        csMessage.Format(L"Stop Failure: %s", e.what());
        AfxMessageBox(csMessage, MB_ICONSTOP);
        return 0;
    }

    //
    // End of main grab loop
    //
    SetEvent(m_heventThreadDone);

    return 0;
}

// CSpinnakerMFCDoc diagnostics

#ifdef _DEBUG
void CSpinnakerMFCDoc::AssertValid() const
{
    CDocument::AssertValid();
}

void CSpinnakerMFCDoc::Dump(CDumpContext& dc) const
{
    CDocument::Dump(dc);
}
#endif //_DEBUG

// CSpinnakerMFCDoc commands

void CSpinnakerMFCDoc::RedrawAllViews()
{
    POSITION pos = GetFirstViewPosition();
    while (pos != nullptr)
    {
        InvalidateRect(GetNextView(pos)->GetSafeHwnd(), nullptr, FALSE);
    }
}

double CSpinnakerMFCDoc::GetProcessedFrameRate()
{
    return m_processedFrameRate.GetFrameRate();
}

unsigned char* CSpinnakerMFCDoc::GetProcessedPixels()
{
    return static_cast<unsigned char*>(m_processedImage->GetData());
}

void CSpinnakerMFCDoc::GetImageSize(unsigned int* pWidth, unsigned int* pHeight)
{
    *pWidth = abs(m_bitmapInfo.bmiHeader.biWidth);
    *pHeight = abs(m_bitmapInfo.bmiHeader.biHeight);
}

void CSpinnakerMFCDoc::OnToggleCameraControl()
{
    // TODO:
    // Toggle PropertyGrid through GUI Factory
    if (m_pGridWnd->IsVisible())
    {
        m_pGridWnd->Hide();
    }
    else
    {
        m_pGridWnd->Show();
    }
}

void CSpinnakerMFCDoc::OnFileSaveAs()
{
    CString csMessage;
    JPEGOption JPEG_Save_Option;
    PNGOption PNG_Save_Option;

    m_beingSaved = true;

    // Define the list of filters to include in the SaveAs dialog.
    const unsigned int uiNumFilters = 8;
    const CString arcsFilter[uiNumFilters] = {L"Windows Bitmap (*.bmp)|*.bmp",
                                              L"Portable Pixelmap (*.ppm)|*.ppm",
                                              L"Portable Greymap (raw image) (*.pgm)|*.pgm",
                                              L"Independent JPEG Group (*.jpg, *.jpeg)|*.jpg; *.jpeg",
                                              L"Tagged Image File Format (*.tiff)|*.tiff",
                                              L"Portable Network Graphics (*.png)|*.png",
                                              L"Raw data (*.raw)|*.raw",
                                              L"All Files (*.*)|*.*"};

    CString csFilters;

    // Keep track of which filter should be selected as default.
    // m_uiFilterIndex is set to what was previously used (0 if this is first time).
    for (unsigned int i = 0; i < (uiNumFilters - 1); i++)
    {
        csFilters += arcsFilter[(m_uiFilterIndex + i) % (uiNumFilters - 1)];
        csFilters += "|";
    }
    // Always finish with All Files and a ||.
    csFilters += arcsFilter[uiNumFilters - 1];
    csFilters += "||";

    time_t rawtime;
    time(&rawtime);
    struct tm* timeinfo = localtime(&rawtime);

    char timestamp[64];
    strftime(timestamp, 64, "%Y-%m-%d-%H%M%S", timeinfo);

    // Retrieve Device ID
    Spinnaker::GenApi::CStringPtr ptrStringSerial = m_pCamera->GetTLDeviceNodeMap().GetNode("DeviceSerialNumber");
    char tempFilenameCStr[128];
    TCHAR tempFilename[128];

    sprintf(tempFilenameCStr, "%s-%s", ptrStringSerial->GetValue().c_str(), timestamp);
    mbstowcs(tempFilename, tempFilenameCStr, sizeof(tempFilenameCStr));

    CFileDialog fileDialog(
        FALSE, L"bmp", tempFilename, OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT, csFilters, AfxGetMainWnd());

    if (fileDialog.DoModal() == IDOK)
    {
        ImageFileFormat saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_FROM_FILE_EXT;
        CString csExt = fileDialog.GetFileExt();

        // Check file extension
        if (csExt.CompareNoCase(L"bmp") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_BMP;
        }
        else if (csExt.CompareNoCase(L"ppm") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_PPM;
        }
        else if (csExt.CompareNoCase(L"pgm") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_PGM;
        }
        else if (csExt.CompareNoCase(L"jpeg") == 0 || csExt.CompareNoCase(L"jpg") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_JPEG;
            JPEG_Save_Option.progressive = false;
            JPEG_Save_Option.quality = 100; // Superb quality.
        }
        else if (csExt.CompareNoCase(L"tiff") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_TIFF;
        }
        else if (csExt.CompareNoCase(L"png") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_PNG;
            PNG_Save_Option.interlaced = false;
            PNG_Save_Option.compressionLevel = 9; // Best compression
        }
        else if (csExt.CompareNoCase(L"raw") == 0)
        {
            saveImageFormat = SPINNAKER_IMAGE_FILE_FORMAT_RAW;
        }
        else
        {
            AfxMessageBox(L"Invalid file type");
        }

        if (saveImageFormat == SPINNAKER_IMAGE_FILE_FORMAT_RAW)
        {
            try
            {
                m_saveImage->Save((CStringA)fileDialog.GetPathName(), SPINNAKER_IMAGE_FILE_FORMAT_RAW);
            }
            catch (Exception& ex)
            {
                csMessage.Format(L"Failed to save image (Error: %s)", (CString)ex.what());
                AfxMessageBox(csMessage, MB_ICONSTOP);
            }
        }
        else if (saveImageFormat == SPINNAKER_IMAGE_FILE_FORMAT_PGM)
        {
            const PixelFormatEnums tempPixelFormat = m_saveImage->GetPixelFormat();
            if (tempPixelFormat == PixelFormat_Mono8 || tempPixelFormat == PixelFormat_Mono12 ||
                tempPixelFormat == PixelFormat_Mono16 || tempPixelFormat == PixelFormat_Raw8 ||
                tempPixelFormat == PixelFormat_Raw16)
            {
                try
                {
                    m_saveImage->Save((CStringA)fileDialog.GetPathName(), saveImageFormat);
                }
                catch (Exception& ex)
                {
                    csMessage.Format(L"Failed to convert image (Error: %s)", (CString)ex.what());
                    AfxMessageBox(csMessage, MB_ICONSTOP);
                }
            }
            else
            {
                AfxMessageBox(L"Invalid file format.\r\nNon mono / raw images cannot be saved as PGM.", MB_ICONSTOP);
            }
        }
        else
        {
            ImagePtr convertedImage = Image::Create();

            try
            {
                convertedImage = m_processor.Convert(m_saveImage, PixelFormat_RGB8);
            }
            catch (Exception& ex)
            {
                csMessage.Format(L"Failed to convert image (Error: %s)", (CString)ex.what());
                AfxMessageBox(csMessage, MB_ICONSTOP);
            }

            try
            {
                convertedImage->Save((CStringA)fileDialog.GetPathName(), saveImageFormat);
            }
            catch (Exception& ex)
            {
                csMessage.Format(L"Failed to save image (Error: %s)", (CString)ex.what());
                AfxMessageBox(csMessage, MB_ICONSTOP);
            }
        }
    }
    m_beingSaved = false;
}
