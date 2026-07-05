"""Small environment check for the next RLBench RGB-D experiments."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path


PACKAGES = ("rlbench", "pyrep", "peract_colab", "yarr", "h5py", "numpy")


def import_status(module_name: str) -> str:
    if not importlib.util.find_spec(module_name):
        return "missing"
    try:
        importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        return f"broken ({type(exc).__name__}: {exc})"
    return "ok"


def main() -> None:
    print("RLBench environment check")
    for package in PACKAGES:
        print(f"  {package:12s}: {import_status(package)}")
    try:
        importlib.import_module("pyrep.backend._sim_cffi")
        print("  pyrep_cffi  : ok")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  pyrep_cffi  : broken ({type(exc).__name__}: {exc})")
    try:
        from peract_colab.rlbench.utils import get_stored_demo  # noqa: F401

        print("  get_demo    : ok")
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  get_demo    : broken ({type(exc).__name__}: {exc})")

    for env_name in ("COPPELIASIM_ROOT", "LD_LIBRARY_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH", "DISPLAY"):
        value = os.environ.get(env_name)
        if env_name == "LD_LIBRARY_PATH" and value:
            value = value[:160] + ("..." if len(value) > 160 else "")
        print(f"  {env_name:28s}: {value or 'unset'}")

    bridgevla_rlbench = Path("/root/autodl-tmp/BridgeVLA/finetune/RLBench")
    print(f"  BridgeVLA RLBench helper: {'found' if bridgevla_rlbench.exists() else 'missing'}")
    if bridgevla_rlbench.exists():
        print(f"    {bridgevla_rlbench}")


if __name__ == "__main__":
    main()
