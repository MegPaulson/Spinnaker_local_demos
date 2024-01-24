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
#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

int GetMinimumPowerOfTwo(int in)
{
    int i = 1;
    while (i < in)
    {
        i *= 2;
    }

    return i;
}

// CSpinnakerMFCView

IMPLEMENT_DYNCREATE(CSpinnakerMFCView, CView)

BEGIN_MESSAGE_MAP(CSpinnakerMFCView, CView)
ON_WM_CREATE()
ON_WM_DESTROY()
ON_WM_SIZE()
ON_WM_ERASEBKGND()
END_MESSAGE_MAP()

// CSpinnakerMFCView construction/destruction

CSpinnakerMFCView::CSpinnakerMFCView()
{
    m_pDC = nullptr;
}

CSpinnakerMFCView::~CSpinnakerMFCView()
{
}

#ifdef _DEBUG
void CSpinnakerMFCView::AssertValid() const
{
    CView::AssertValid();
}

void CSpinnakerMFCView::Dump(CDumpContext& dc) const
{
    CView::Dump(dc);
}

CSpinnakerMFCDoc* CSpinnakerMFCView::GetDocument() const // non-debug version is inline
{
    ASSERT(m_pDocument->IsKindOf(RUNTIME_CLASS(CSpinnakerMFCDoc)));
    return (CSpinnakerMFCDoc*)m_pDocument;
}

#endif //_DEBUG

BOOL CSpinnakerMFCView::PreCreateWindow(CREATESTRUCT& cs)
{
    return CView::PreCreateWindow(cs);
}

CSpinnakerMFCView* CSpinnakerMFCView::GetView()
{
    CFrameWnd* pFrame = (CFrameWnd*)(AfxGetApp()->m_pMainWnd);

    CView* pView = pFrame->GetActiveView();

    if (!pView)
        return nullptr;

    // Fail if view is of wrong kind
    // (this could occur with splitter windows, or additional
    // views on a single document
    if (!pView->IsKindOf(RUNTIME_CLASS(CSpinnakerMFCView)))
        return nullptr;

    return (static_cast<CSpinnakerMFCView*>(pView));
}

void CSpinnakerMFCView::OnDraw(CDC* pDC)
{
    CSpinnakerMFCDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);
    if (!pDoc)
        return;

    (static_cast<CMainFrame*>(GetParentFrame()))->ResizeToFit();
    (static_cast<CMainFrame*>(GetParentFrame()))->UpdateStatusBar();

    // Transfer the RGB buffer to graphics card.
    CSingleLock dataLock(&pDoc->m_csData);
    dataLock.Lock();

    unsigned char* pImagePixels = pDoc->GetProcessedPixels();
    if (pImagePixels == nullptr)
    {
        return;
    }

    if (::SetDIBitsToDevice(
            pDC->GetSafeHdc(),
            0,
            0,
            pDoc->m_bitmapInfo.bmiHeader.biWidth,
            ::abs(pDoc->m_bitmapInfo.bmiHeader.biHeight),
            0,
            0,
            0,
            ::abs(pDoc->m_bitmapInfo.bmiHeader.biHeight),
            pDoc->GetProcessedPixels(),
            &pDoc->m_bitmapInfo,
            DIB_RGB_COLORS) == 0)
    {
        // error.
    }

    m_displayedFrameRate.NewFrame();
    dataLock.Unlock();
}

double CSpinnakerMFCView::GetDisplayedFrameRate()
{
    return m_displayedFrameRate.GetFrameRate();
}

void CSpinnakerMFCView::OnInitialUpdate()
{
    CView::OnInitialUpdate();

    CSpinnakerMFCDoc* pDoc = GetDocument();
    ASSERT_VALID(pDoc);

    RECT clientRect;
    GetClientRect(&clientRect);

    // Resize the window to properly display the image
    GetParentFrame()->SetWindowPos(
        nullptr, 0, 0, clientRect.right, clientRect.right / 2, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOZORDER);
}

int CSpinnakerMFCView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
    if (CView::OnCreate(lpCreateStruct) == -1)
        return -1;

    return 0;
}

void CSpinnakerMFCView::OnDestroy()
{
    CView::OnDestroy();
}

void CSpinnakerMFCView::OnSize(UINT nType, int cx, int cy)
{
    CView::OnSize(nType, cx, cy);

    if ((cx <= 0) || (cy <= 0))
    {
        return;
    }
}

BOOL CSpinnakerMFCView::OnEraseBkgnd(CDC* /*pDC*/)
{
    return 0;
}
