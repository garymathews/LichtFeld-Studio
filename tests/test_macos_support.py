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
    assert tensor.device == tensor.backend == "vulkan" and not tensor.is_cuda
    assert tensor.cpu().tolist() == [2.5, 2.5, 2.5]
    assert lf.Tensor.zeros([1], device=tensor.device).device == tensor.device


def test_histogram_preserves_gpu_device():
    from lfs_plugins.histogram_panel import HistogramPanel

    tensor = lf.Tensor.ones([3], device="gpu")
    device = HistogramPanel._device_string(tensor)
    assert device == "gpu"
    assert HistogramPanel._to_device(tensor.cpu(), device).device == "vulkan"
    assert HistogramPanel._to_device(tensor, "cpu").device == "cpu"


def test_vulkan_bool_preserves_special_values():
    import numpy as np

    values = np.array([
        0.0, -0.0, np.nextafter(np.float32(0), np.float32(1)),
        -np.nextafter(np.float32(0), np.float32(1)),
        2.0 ** -24, -(2.0 ** -24), 1.0, -1.0, np.inf, -np.inf, np.nan,
    ], dtype=np.float32)
    tensor = lf.Tensor.from_numpy(values)
    for dtype in ("float32", "float16"):
        source = tensor.to(dtype)
        expected = source.to("bool").tolist()
        gpu = source.gpu()
        assert gpu.to("bool").cpu().tolist() == expected, dtype
        if dtype == "float32":
            assert gpu.count_nonzero() == sum(expected)
            assert gpu.nonzero().numel == sum(expected)
            assert (~gpu).cpu().tolist() == [not value for value in expected]

    for dtype in (np.int32, np.int64):
        limits = np.iinfo(dtype)
        values = [0, 1, -1, limits.min, limits.max]
        if dtype == np.int64:
            values.extend([1 << 32, -(1 << 32)])
        source = lf.Tensor.from_numpy(np.array(values, dtype=dtype))
        assert source.gpu().to("bool").cpu().tolist() == [value != 0 for value in values], dtype


def test_vulkan_writer_submits_pending_reader():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    source = lf.Tensor.ones([4096], device="gpu")
    source.sync()
    # Reserve an older batch on this thread before the worker records its read.
    earlier = lf.Tensor.full([4096], 2.0, device="gpu")
    ready, release = Event(), Event()
    result = []

    def read_source():
        result.append(source.clone())
        ready.set()
        assert release.wait(10), "Reader was not released"

    with ThreadPoolExecutor(max_workers=1) as executor:
        reader = executor.submit(read_source)
        try:
            assert ready.wait(10), "Reader did not record its copy"
            source.fill_(7.0)
            assert result[0].cpu().tolist() == [1.0] * 4096
            earlier.sync()
        finally:
            release.set()
            reader.result(timeout=10)


if __name__ == "__main__":
    test_vulkan_python_device_roundtrip()
    test_histogram_preserves_gpu_device()
    test_vulkan_bool_preserves_special_values()
    test_vulkan_writer_submits_pending_reader()
    print("macOS support: 4 smoke checks passed")
