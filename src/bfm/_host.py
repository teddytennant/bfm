"""Set the nix-ld library path before JAX is imported."""

import os

NIX_LD_LIB = "/run/current-system/sw/share/nix-ld/lib"


def ensure_nix_ld() -> None:
    """Prepend the nix-ld lib dir if LD_LIBRARY_PATH does not already mention it."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if "nix-ld" not in current:
        os.environ["LD_LIBRARY_PATH"] = (
            NIX_LD_LIB if not current else f"{NIX_LD_LIB}:{current}"
        )


ensure_nix_ld()
