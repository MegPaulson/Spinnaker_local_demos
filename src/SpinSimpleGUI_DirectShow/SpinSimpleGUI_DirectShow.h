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

#ifndef FLIR_SPINSIMPLEGUI_DIRECTSHOW_H
#define FLIR_SPINSIMPLEGUI_DIRECTSHOW_H

//
// Function prototypes
//
int PASCAL WinMain(HINSTANCE hInstance, HINSTANCE hInstP, LPSTR lpCmdLine, int nCmdShow);
LRESULT CALLBACK WndMainProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam);

HRESULT GetInterfaces(void);
HRESULT CaptureVideo();
HRESULT FindCaptureDevice(IBaseFilter** ppSrcFilter);
HRESULT SetupVideoWindow(void);
HRESULT ChangePreviewState(int nShow);
HRESULT HandleGraphEvent(void);
HRESULT SetupCamera(IBaseFilter* pSrcFilter);

void MsgError(TCHAR* szFormat, ...);
void MsgInfo(TCHAR* szFormat, ...);
void CloseInterfaces(void);
void ResizeVideoWindow(void);

enum PLAYSTATE
{
    Stopped,
    Paused,
    Running,
    Init
};

//
// Macros
//
#define SAFE_RELEASE(x)                                                                                                \
    {                                                                                                                  \
        if (x)                                                                                                         \
            x->Release();                                                                                              \
        x = NULL;                                                                                                      \
    }

#define JIF(x)                                                                                                         \
    if (FAILED(hr = (x)))                                                                                              \
    {                                                                                                                  \
        Msg(TEXT("FAILED(hr=0x%x) in ") TEXT(#x) TEXT("\n\0"), hr);                                                    \
        return hr;                                                                                                     \
    }

//
// Constants
//
#define DEFAULT_VIDEO_WIDTH  320
#define DEFAULT_VIDEO_HEIGHT 320

#define APPLICATIONNAME TEXT("SpinSimpleGUI_DirectShow\0")
#define CLASSNAME       TEXT("SpinSimpleGUI_DirectShow\0")

// Application-defined message to notify app of filtergraph events
#define WM_GRAPHNOTIFY WM_APP + 1

#endif // FLIR_SPINSIMPLEGUI_DIRECTSHOW_H
