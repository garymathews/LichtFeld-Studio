/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "training/progress.hpp"
#include <gtest/gtest.h>

TEST(TrainingProgress, FailedBeforeFirstIterationDoesNotReportCompletion) {
    testing::internal::CaptureStdout();
    {
        lfs::training::TrainingProgress progress(100);
        progress.complete(false, 0, true);
    }
    const auto output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("Failed at 0/100"), std::string::npos);
    EXPECT_EQ(output.find("Training completed"), std::string::npos);
}

TEST(TrainingProgress, UserStopDoesNotPromiseAnUnsavedCheckpoint) {
    testing::internal::CaptureStdout();
    {
        lfs::training::TrainingProgress progress(100);
        progress.print_final_summary(1000, 4, true);
    }
    const auto output = testing::internal::GetCapturedStdout();
    EXPECT_NE(output.find("stopped by user at iteration 4"), std::string::npos);
    EXPECT_EQ(output.find("checkpoint saved"), std::string::npos);
}

TEST(TrainingProgress, FailureWhilePausedStillReportsTheTerminalIteration) {
    testing::internal::CaptureStdout();
    {
        lfs::training::TrainingProgress progress(100);
        progress.update(4, 0.1f, 1000);
        progress.pause();
        progress.complete(false, 4, true);
    }
    const auto output = testing::internal::GetCapturedStdout();
    const auto failure = output.find("Failed at 4/100");
    EXPECT_NE(failure, std::string::npos);
    if (failure != std::string::npos)
        EXPECT_EQ(output.find("Failed at 4/100", failure + 1), std::string::npos);
    EXPECT_EQ(output.find("Training completed"), std::string::npos);
}
