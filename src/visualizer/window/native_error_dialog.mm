/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "window_manager.hpp"
#include "core/logger.hpp"
#include <SDL3/SDL.h>
#import <Cocoa/Cocoa.h>

namespace lfs::vis {
    void WindowManager::showErrorDialog(const char* title, const char* message) {
        auto* parent = static_cast<NSWindow*>(SDL_GetPointerProperty(
            SDL_GetWindowProperties(window_), SDL_PROP_WINDOW_COCOA_WINDOW_POINTER, nullptr));
        if (!parent) {
            LOG_ERROR("Could not display GPU failure dialog: native window unavailable");
            return;
        }
        NSAlert* alert = [[NSAlert alloc] init];
        alert.alertStyle = NSAlertStyleCritical;
        alert.messageText = [NSString stringWithUTF8String:title];
        alert.informativeText = [NSString stringWithUTF8String:message];
        [alert addButtonWithTitle:@"OK"];
        // A sheet leaves the main loop free to settle writes and serve MCP.
        [alert beginSheetModalForWindow:parent completionHandler:^(NSModalResponse) {
            [alert release];
        }];
    }
}
