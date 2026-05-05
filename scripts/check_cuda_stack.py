#!/usr/bin/env python3
"""
Quick CUDA smoke test for cluster nodes.

Use this on Unity after you land on a GPU node and before you launch the full
Report 6 pipeline. It checks that PyTorch can actually launch CUDA kernels on
the allocated GPU.
"""

import sys

import torch


def main() -> int:
    print(f"torch: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA is not available in this environment.")
        return 1

    device_index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)
    major, minor = torch.cuda.get_device_capability(device_index)
    print(f"device: cuda:{device_index}")
    print(f"gpu: {name}")
    print(f"compute capability: sm_{major}{minor}")

    try:
        x = torch.ones((4, 4), device="cuda")
        y = x @ x
        print("cuda tensor check: OK")
        print(f"sample value: {y[0, 0].item()}")
    except Exception as exc:
        print("cuda tensor check: FAILED")
        print(f"{type(exc).__name__}: {exc}")
        print(
            "This often means the PyTorch/CUDA build does not support the GPU "
            "architecture on this node. On Unity, request a modern GPU "
            "constraint such as --constraint=a100|a40|l40s."
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
