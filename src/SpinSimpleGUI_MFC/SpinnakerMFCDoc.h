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

#ifndef FLIR_SPINNAKER_MFCDOC_H
#define FLIR_SPINNAKER_MFCDOC_H

#include "FrameRateCounter.h"
#include "GUI/SpinnakerGUI.h"

//
// Size of the window when it the application first starts.
//
#define _DEFAULT_WINDOW_X 640
#define _DEFAULT_WINDOW_Y 480

#pragma once

class CSpinnakerMFCDoc : public CDocument
{
  protected: // create from serialization only
    CSpinnakerMFCDoc();
    DECLARE_DYNCREATE(CSpinnakerMFCDoc)

  public:
    virtual ~CSpinnakerMFCDoc();
#ifdef _DEBUG
    virtual void AssertValid() const;
    virtual void Dump(CDumpContext& dc) const;
#endif

    // Critical section to protect access to the processed image
    CCriticalSection m_csData;

    // Structure used to draw to the screen.
    BITMAPINFO m_bitmapInfo;

    // Get the processed frame rate
    double GetProcessedFrameRate();

    // Get the data pointer to the image
    unsigned char* GetProcessedPixels();

    // Get the dimensions of the image
    void GetImageSize(unsigned int* pWidth, unsigned int* pHeight);

    // Initialize the bitmap struct used for drawing.
    void InitBitmapStruct(int cols, int rows);

    // The image grab thread.
    static UINT ThreadGrabImage(void* pparam);

    // The object grab image loop.  Only executed from within the grab thread.
    UINT DoGrabLoop();

    // Redraw all the views in the application
    void RedrawAllViews();

    virtual BOOL OnNewDocument();
    virtual void OnCloseDocument(void);

  protected:
    Spinnaker::CameraPtr m_pCamera;
    Spinnaker::GUI_WPF::CameraSelectionWindow* m_pCamSelWnd;
    Spinnaker::GUI_WPF::PropertyGridWindow* m_pGridWnd;

    Spinnaker::ImagePtr m_saveImage;
    Spinnaker::ImagePtr m_processedImage;

    Spinnaker::ImageProcessor m_processor;

    bool m_continueGrabThread;
    bool m_beingSaved;

    HANDLE m_heventThreadDone;
    FrameRateCounter m_processedFrameRate;

  private:
    // Keeps track of the last filter index used for image saving.
    unsigned int m_uiFilterIndex;

    Spinnaker::GUI_WPF::DeviceInformationStruct m_devInfo;

    // Generated message map functions
  protected:
    DECLARE_MESSAGE_MAP()
  public:
    afx_msg void OnToggleCameraControl();
    afx_msg void OnFileSaveAs();
};

#endif // FLIR_SPINNAKER_MFCDOC_H