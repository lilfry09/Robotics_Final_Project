#!/usr/bin/env python3
"""Check whether the ManiSkill3 adapter prerequisites are available."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass


@dataclass
class PackageStatus:
    name: str
    available: bool
    version: str | None = None
    error: str | None = None


def check_package(module_name: str, version_attr: str = "__version__") -> PackageStatus:
    if importlib.util.find_spec(module_name) is None:
        return PackageStatus(name=module_name, available=False, error="module not found")
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, version_attr, None)
        return PackageStatus(name=module_name, available=True, version=str(version) if version else None)
    except Exception as exc:  # pragma: no cover - environment diagnostics
        return PackageStatus(name=module_name, available=False, error=f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    packages = [
        check_package("mani_skill"),
        check_package("gymnasium"),
        check_package("h5py"),
        check_package("numpy"),
        check_package("torch"),
    ]
    payload = {
        "ok": all(pkg.available for pkg in packages),
        "packages": [asdict(pkg) for pkg in packages],
        "next_action": (
            "ManiSkill3 environment is ready for an adapter smoke."
            if all(pkg.available for pkg in packages)
            else "Install ManiSkill3 and rerun this checker before writing the converter."
        ),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("ManiSkill3 adapter environment check")
        for pkg in packages:
            status = "ok" if pkg.available else "missing"
            suffix = f" ({pkg.version})" if pkg.version else ""
            detail = f": {pkg.error}" if pkg.error else ""
            print(f"- {pkg.name}: {status}{suffix}{detail}")
        print(f"next: {payload['next_action']}")

    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
