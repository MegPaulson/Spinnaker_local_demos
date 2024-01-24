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

#ifndef FLIR_SPINNAKER_MFCVIEW_H
#define FLIR_SPINNAKER_MFCVIEW_H

#include "FrameRateCounter.h"

class CSpinnakerMFCView : public CView
{
  public:
    CSpinnakerMFCDoc* GetDocument() const;
    double GetDisplayedFrameRate();
    virtual void OnDraw(CDC* pDC);
    virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
    virtual ~CSpinnakerMFCView();
    static CSpinnakerMFCView* GetView();

#ifdef _DEBUG
    virtual void AssertValid() const;
    virtual void Dump(CDumpContext& dc) const;
#endif

  protected:
    FrameRateCounter m_displayedFrameRate;

    /** Device context for drawing. */
    CDC* m_pDC;

    CSpinnakerMFCView();
    DECLARE_DYNCREATE(CSpinnakerMFCView)

    // Generated message map functions
  protected:
    DECLARE_MESSAGE_MAP()
  public:
    virtual void OnInitialUpdate();
    afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
    afx_msg void OnDestroy();
    afx_msg void OnSize(UINT nType, int cx, int cy);
    afx_msg BOOL OnEraseBkgnd(CDC* pDC);
};

#ifndef _DEBUG // debug version in SpinnakerMFCView.cpp
inline CSpinnakerMFCDoc* CSpinnakerMFCView::GetDocument() const
{
    return reinterpret_cast<CSpinnakerMFCDoc*>(m_pDocument);
}
#endif

#endif // FLIR_SPINNAKER_MFCVIEW_H