"""Portable project root; optionally override with VOLTKD_ROOT."""
import os
from pathlib import Path

ROOT = str(Path(os.environ.get("VOLTKD_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve())
