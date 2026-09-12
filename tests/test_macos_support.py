# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run in the macOS app's console with runpy.run_path(..., run_name='__main__')."""

import sys

if sys.platform != "darwin":
    import unittest
    raise unittest.SkipTest("macOS integration smoke checks")

import lichtfeld as lf


def test_vulkan_python_device_roundtrip():
    from lfs_plugins.props import TensorProperty

    assert lf.get_gpu_backend() == "vulkan"
    tensor = lf.Tensor.full([3], 2.5, device="gpu")
    assert TensorProperty(device="gpu").validate(tensor) is tensor
    tensor.sync()
    assert tensor.device == "vulkan" and not tensor.is_cuda
    assert tensor.cpu().tolist() == [2.5, 2.5, 2.5]
    assert lf.Tensor.zeros([1], device=tensor.device).device == tensor.device


def test_histogram_preserves_gpu_device():
    from lfs_plugins.histogram_panel import HistogramPanel

    tensor = lf.Tensor.ones([3], device="gpu")
    device = HistogramPanel._device_string(tensor)
    assert device == "gpu"
    assert HistogramPanel._to_device(tensor.cpu(), device).device == "vulkan"
    assert HistogramPanel._to_device(tensor, "cpu").device == "cpu"


if __name__ == "__main__":
    test_vulkan_python_device_roundtrip()
    test_histogram_preserves_gpu_device()
    print("macOS support: 2 smoke checks passed")
