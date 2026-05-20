"""Deployment capability detection without overclaiming external validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _status(available: bool) -> str:
    return "available" if available else "missing"


def _find_onnxruntime_cpp(root: str | None) -> dict[str, Any]:
    if not root:
        return {
            "status": "missing",
            "headers": False,
            "libs": False,
            "root": None,
            "detail": "ONNXRUNTIME_ROOT is not set",
        }
    root_path = Path(root)
    header = root_path / "include" / "onnxruntime_cxx_api.h"
    lib_candidates = list(root_path.glob("lib/**/onnxruntime.lib")) + list(
        root_path.glob("lib/**/libonnxruntime.*")
    )
    headers = header.exists()
    libs = any(path.is_file() for path in lib_candidates)
    return {
        "status": "available" if headers and libs else "missing",
        "headers": headers,
        "libs": libs,
        "root": str(root_path),
        "detail": "C++ headers/libs detected" if headers and libs else "C++ headers/libs incomplete",
    }


def _cuda_status(which: Callable[[str], str | None]) -> dict[str, Any]:
    nvcc = which("nvcc")
    nvidia_smi = which("nvidia-smi")
    torch_cuda = False
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
    except Exception:
        torch_cuda = False
    available = torch_cuda or nvcc is not None or nvidia_smi is not None
    return {
        "status": _status(available),
        "torch_cuda_available": torch_cuda,
        "nvcc": nvcc,
        "nvidia_smi": nvidia_smi,
    }


def _tensorrt_status(which: Callable[[str], str | None]) -> dict[str, Any]:
    python_module = _module_available("tensorrt")
    trtexec = which("trtexec")
    available = python_module or trtexec is not None
    return {
        "status": _status(available),
        "python_module": python_module,
        "trtexec": trtexec,
    }


def detect_deployment_capabilities(
    *,
    env: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    env = env or os.environ
    which = which or shutil.which
    cmake_path = which("cmake")
    onnxruntime_root = env.get("ONNXRUNTIME_ROOT")
    onnxruntime_cpp = _find_onnxruntime_cpp(onnxruntime_root)
    cuda = _cuda_status(which)
    tensorrt = _tensorrt_status(which)
    cpp_build_verifiable = cmake_path is not None and onnxruntime_cpp["status"] == "available"
    tensorrt_verifiable = cuda["status"] == "available" and tensorrt["status"] == "available"
    torch_available = _module_available("torch")
    onnxruntime_available = _module_available("onnxruntime")
    return {
        "python_onnx_export": {
            "status": _status(torch_available),
            "available": torch_available,
            "detail": "torch.onnx export path",
        },
        "python_onnxruntime": {
            "status": _status(onnxruntime_available),
            "available": onnxruntime_available,
        },
        "cmake": {
            "status": _status(cmake_path is not None),
            "available": cmake_path is not None,
            "path": cmake_path,
        },
        "onnxruntime_root": {
            "status": _status(bool(onnxruntime_root)),
            "available": bool(onnxruntime_root),
            "path": onnxruntime_root,
        },
        "onnxruntime_cpp": onnxruntime_cpp,
        "cuda": cuda,
        "tensorrt": tensorrt,
        "cpp_build_verifiable": cpp_build_verifiable,
        "tensorrt_verifiable": tensorrt_verifiable,
        "cpp_inference_verified": False,
        "tensorrt_verified": False,
        "cpp_inference_verification_status": "skipped",
        "tensorrt_verification_status": "skipped",
    }


def write_status(output: str | Path) -> dict[str, Any]:
    status = detect_deployment_capabilities()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    status = write_status(args.output)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
