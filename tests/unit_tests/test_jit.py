# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression coverage for import-time fuser selection."""

import importlib.util
from pathlib import Path

import pytest
import torch

from megatron.core import utils


@pytest.mark.parametrize("script_available", [True, False])
def test_modern_fuser_import_without_torchscript(
    monkeypatch: pytest.MonkeyPatch, script_available: bool
) -> None:
    """Modern PyTorch must select compile without reading the optional script API."""
    if script_available and not hasattr(torch.jit, "script"):
        pytest.skip("This PyTorch build no longer exposes torch.jit.script")
    if not script_available:
        monkeypatch.delattr(torch.jit, "script", raising=False)
    # Load a fresh module: the regular package import may already have cached jit.
    path = Path(utils.__file__).with_name("jit.py")
    spec = importlib.util.spec_from_file_location("jit_import_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.jit_fuser is torch.compile
    assert hasattr(torch.jit, "script") == script_available
    module.disable_jit_fuser()

    def function(x: torch.Tensor) -> torch.Tensor:
        """A small operation to verify the disabled fuser remains executable."""
        return x + 1

    assert module.jit_fuser(function)(torch.tensor(2)).item() == 3
    module.enable_jit_fuser()
    assert module.jit_fuser is torch.compile
