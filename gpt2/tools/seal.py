#!/usr/bin/env python3
"""Update the package checksums after preparing/capturing or editing code."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from checks import digest


def main():
    records = {}
    for path in sorted(ROOT.rglob("*")):
        name = path.relative_to(ROOT)
        if not path.is_file() or any(part in (".venv", "__pycache__", ".git") for part in name.parts):
            continue
        if str(name) in ("checksums.json", "validation-local.json"):
            continue
        records[str(name)] = digest(path)
    (ROOT / "checksums.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Sealed {len(records)} files")


if __name__ == "__main__":
    main()
