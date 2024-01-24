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
#include "MainFrm.h"

#include "SpinnakerMFCDoc.h"
#include "SpinnakerMFCView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CSpinnakerMFCApp
BEGIN_MESSAGE_MAP(CSpinnakerMFCApp, CWinApp)
ON_COMMAND(ID_APP_ABOUT, &CSpinnakerMFCApp::OnAppAbout)
// Standard file based document commands
ON_COMMAND(ID_FILE_NEW, &CWinApp::OnFileNew)
ON_COMMAND(ID_FILE_OPEN, &CWinApp::OnFileOpen)
END_MESSAGE_MAP()

// CSpinnakerMFCApp construction
CSpinnakerMFCApp::CSpinnakerMFCApp()
{
    // Place all significant initialization in InitInstance
}

// The one and only CSpinnakerMFCApp object
CSpinnakerMFCApp theApp;

// CSpinnakerMFCApp initialization
BOOL CSpinnakerMFCApp::InitInstance()
{
    // *** NOTES ***
    // Dialog based MFC applications may incorrectly initialize the threading model to MTA.
    // To prevent this, initialize the threading model to STA with the following code block.

    const HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr))
    {
        AfxMessageBox(L"CoInitializeEx initialization failed");
        return FALSE;
    }

    // InitCommonControlsEx() is required on Windows XP if an application
    // manifest specifies use of ComCtl32.dll version 6 or later to enable
    // visual styles.  Otherwise, any window creation will fail.
    INITCOMMONCONTROLSEX InitCtrls;
    InitCtrls.dwSize = sizeof(InitCtrls);
    // Set this to include all the common control classes you want to use
    // in your application.
    InitCtrls.dwICC = ICC_WIN95_CLASSES;
    InitCommonControlsEx(&InitCtrls);

    CWinApp::InitInstance();

    // Initialize OLE libraries
    if (!AfxOleInit())
    {
        AfxMessageBox(IDP_OLE_INIT_FAILED);
        return FALSE;
    }
    AfxEnableControlContainer();
    // Standard initialization
    // If you are not using these features and wish to reduce the size
    // of your final executable, you should remove from the following
    // the specific initialization routines you do not need
    // Change the registry key under which our settings are stored
    SetRegistryKey(_T("FLIR Systems, Inc."));
    LoadStdProfileSettings(4); // Load standard INI file options (including MRU)
                               // Register the application's document templates.  Document templates
                               //  serve as the connection between documents, frame windows and views
    auto* pDocTemplate = new CSingleDocTemplate(
        IDR_MAINFRAME,
        RUNTIME_CLASS(CSpinnakerMFCDoc),
        RUNTIME_CLASS(CMainFrame), // main SDI frame window
        RUNTIME_CLASS(CSpinnakerMFCView));
    if (!pDocTemplate)
    {
        return FALSE;
    }
    AddDocTemplate(pDocTemplate);

    // Parse command line for standard shell commands, DDE, file open
    CCommandLineInfo cmdInfo;
    ParseCommandLine(cmdInfo);

    // Dispatch commands specified on the command line.  Will return FALSE if
    // app was launched with /RegServer, /Register, /Unregserver or /Unregister.
    if (!ProcessShellCommand(cmdInfo))
    {
        return FALSE;
    }
    // The one and only window has been initialized, so show and update it
    m_pMainWnd->ShowWindow(SW_SHOW);
    m_pMainWnd->UpdateWindow();
    // call DragAcceptFiles only if there's a suffix
    //  In an SDI app, this should occur after ProcessShellCommand
    return TRUE;
}

int CSpinnakerMFCApp::ExitInstance()
{
    CoUninitialize(); // COM library will be unloaded

    return CWinApp::ExitInstance();
}

// CAboutDlg dialog used for App About
class CAboutDlg : public CDialog
{
  public:
    CAboutDlg();

    // Dialog Data
    enum
    {
        IDD = IDD_ABOUTBOX
    };

  protected:
    virtual void DoDataExchange(CDataExchange* pDX); // DDX/DDV support

    // Implementation
  protected:
    DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialog(CAboutDlg::IDD)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
    CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialog)
END_MESSAGE_MAP()

// App command to run the dialog
void CSpinnakerMFCApp::OnAppAbout()
{
    CAboutDlg aboutDlg;
    aboutDlg.DoModal();
}

// CSpinnakerMFCApp message handlers
