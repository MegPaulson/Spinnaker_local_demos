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

// CMainFrame

IMPLEMENT_DYNCREATE(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
ON_WM_CREATE()
ON_WM_TIMER()
ON_WM_DESTROY()
END_MESSAGE_MAP()

static UINT indicators[] = {
    ID_SEPARATOR, // status line indicator
                  // ID_INDICATOR_CAPS,
                  // ID_INDICATOR_NUM,
                  // ID_INDICATOR_SCRL,
};

// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
    m_timer = NULL;
}

CMainFrame::~CMainFrame()
{
}

// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
    CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
    CFrameWnd::Dump(dc);
}

#endif //_DEBUG

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
    if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
    {
        return -1;
    }

    // Remove the maximize button
    ModifyStyle(WS_MAXIMIZEBOX, 0);

    // Remove the ability to resize the window
    ModifyStyle(WS_THICKFRAME, 0);

    if (!m_wndStatusBar.Create(this) || !m_wndStatusBar.SetIndicators(indicators, sizeof(indicators) / sizeof(UINT)))
    {
        TRACE0("Failed to create status bar\n");
        return -1;
    }

    if (m_timer == NULL)
    {
        m_timer = ::SetTimer(m_hWnd, 123456, 100, static_cast<TIMERPROC>(nullptr));
    }

    return 0;
}

void CMainFrame::OnDestroy()
{
    if (m_timer != NULL)
    {
        ::KillTimer(m_hWnd, m_timer);
    }

    CFrameWnd::OnDestroy();
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
    if (!CFrameWnd::PreCreateWindow(cs))
        return FALSE;
    // TODO: Modify the Window class or styles here by modifying
    //  the CREATESTRUCT cs

    return TRUE;
}

void CMainFrame::ResizeToFit()
{
    auto pDoc = static_cast<CSpinnakerMFCDoc*>(GetActiveDocument());
    if (pDoc == nullptr)
    {
        return;
    }

    unsigned int width = 0;
    unsigned int height = 0;

    pDoc->GetImageSize(&width, &height);

    RECT rect;
    rect.top = 0;
    rect.left = 0;
    rect.bottom = height;
    rect.right = width;

    GetActiveView()->CalcWindowRect(&rect, adjustOutside);

    static bool first = true;
    if (first)
    {
        first = false;
        m_wndStatusBar.GetWindowRect(&m_rectStatus);
    }

    if (m_wndStatusBar.IsWindowVisible())
    {
        rect.bottom += m_rectStatus.bottom - m_rectStatus.top;
    }

    AdjustWindowRectEx(&rect, GetStyle(), TRUE, GetExStyle());

    SetWindowPos(
        nullptr, 0, 0, rect.right - rect.left, rect.bottom - rect.top, SWP_NOMOVE | SWP_NOACTIVATE | SWP_NOZORDER);
}

void CMainFrame::UpdateStatusBar()
{
    CString status;
    auto pDoc = static_cast<CSpinnakerMFCDoc*>(GetActiveDocument());
    auto pView = static_cast<CSpinnakerMFCView*>(GetActiveView());

    if (pDoc == nullptr || pView == nullptr)
    {
        status = "N/A";
        m_wndStatusBar.SetPaneText(0, status);
        return;
    }

    unsigned int width;
    unsigned int height;
    pDoc->GetImageSize(&width, &height);

    const double processedFPS = pDoc->GetProcessedFrameRate();
    const double displayedFPS = pView->GetDisplayedFrameRate();

    status.Format(
        L"Processed FPS: %3.2fHz Displayed FPS: %3.2fHz Image dimensions: (%u x %u)",
        processedFPS,
        displayedFPS,
        width,
        height);
    m_wndStatusBar.SetPaneText(0, status);
}

void CMainFrame::OnTimer(const UINT_PTR idEvent)
{
    UpdateStatusBar();

    CFrameWnd::OnTimer(idEvent);
}
