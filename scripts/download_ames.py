"""Download the published Ames Housing file into data/local and verify it.

Usage: ``uv run python scripts/download_ames.py``
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

SOURCES = {
    "AmesHousing.txt": (
        "https://jse.amstat.org/v19n3/decock/AmesHousing.txt",
        "6cfe6cb525ba437de428653a1040e2aed7d696640bf75203786a6d7a0e67cfcc",
    ),
    "DataDocumentation.txt": (
        "https://jse.amstat.org/v19n3/decock/DataDocumentation.txt",
        "93d741d17422cca39584c6f03fd3f45194501101790bd8241e33d21c991e3eed",
    ),
}
DESTINATION = Path(__file__).resolve().parent.parent / "data" / "local"


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str, url: str, expected: str) -> None:
    """Download one file unless a verified copy already exists."""

    target = DESTINATION / name
    if target.is_file() and sha256(target) == expected:
        print(f"Verified existing {target}")
        return
    partial = target.with_suffix(".partial")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        partial.write_bytes(response.read())
    actual = sha256(partial)
    if actual != expected:
        partial.unlink()
        raise RuntimeError(
            f"Checksum mismatch for {name}: expected {expected}, got {actual}. "
            "The published file changed; update the dataset card before use."
        )
    partial.replace(target)
    print(f"Downloaded and verified {target}")


def main() -> int:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        fetch(name, url, expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
