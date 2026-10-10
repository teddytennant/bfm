"""Make `bfm` importable and set nix-ld before JAX loads."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bfm._host as _host

_host.ensure_nix_ld()
